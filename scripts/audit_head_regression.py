"""Training-cache-only comparison of legacy ridge and calibrated answer-row CE.

All reported logits are FP64 linear surrogates, not native BF16 replay. Both
conditions of a prompt stay together. No external validation or test data is read.
"""

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from watching.calibrated_fit import CalibratedData, fit_ce
from watching.execution_fit import RidgeCorrection


def partition_prompts(records, assignments):
    families = defaultdict(list)
    for index, record in enumerate(records):
        families[record["question_family"]].append(index)
    prompt_heldout = set()
    for indices in families.values():
        ordered = sorted(
            indices,
            key=lambda i: hashlib.sha256(records[i]["context_id"].encode()).hexdigest(),
        )
        prompt_heldout.update(ordered[: max(1, len(ordered) // 5)])
    family_heldout = {
        i for i, record in enumerate(records) if assignments[record["question_family"]] == 0
    }
    return {"paired_prompt_holdout": prompt_heldout, "question_family_holdout": family_heldout}


def paired_metrics(logits, rest):
    score = logits[:, 1] - logits[:, 0]
    pairs = score.reshape(-1, 2)
    labels = torch.arange(len(logits)) % 2
    partition = torch.logsumexp(torch.cat((logits, rest[:, None]), dim=1), dim=1)
    gap = pairs[:, 1] - pairs[:, 0]
    midpoint = pairs.mean(dim=1)
    return {
        "full_vocab_ce": float((partition - logits[torch.arange(len(logits)), labels]).mean()),
        "binary_ce": float(F.softplus(torch.where(labels.bool(), -score, score)).mean()),
        "binary_accuracy": float(((score > 0) == labels.bool()).double().mean()),
        "mean_paired_gap": float(gap.mean()),
        "fraction_positive_gap": float((gap > 0).double().mean()),
        "score_midpoint_rms": float(midpoint.square().mean().sqrt()),
        "gap_rms": float(gap.square().mean().sqrt()),
        "answer_pair_mass": float((logits.logsumexp(1) - partition).exp().mean()),
    }


def constrained_lda(features, labels, shrinkages=(0.01, 0.1, 1.0)):
    """Native biasless directions; all nuisance statistics come from training.

    Minimize pooled within-condition variance for a given mean condition gap,
    subject to zero score at the training mean. No per-prompt centering is used
    when applying the resulting direction to an individual execution.
    """
    means = torch.stack([features[labels == value].mean(0) for value in (0, 1)])
    midpoint = means.mean(0)
    contrast = means[1] - means[0]
    centered = features - means[labels]
    _, singular, vh = torch.linalg.svd(centered / len(features) ** 0.5, full_matrices=False)
    variance = singular.square()
    mean_variance = variance.sum() / features.shape[1]
    result = []
    for shrinkage in shrinkages:
        regularizer = mean_variance * shrinkage

        def inverse(vector, regularizer=regularizer):
            return (
                vector - vh.T @ ((variance / (variance + regularizer)) * (vh @ vector))
            ) / regularizer

        inverse_mean, inverse_contrast = inverse(midpoint), inverse(contrast)
        raw = inverse_contrast - inverse_mean * (
            midpoint @ inverse_contrast / (midpoint @ inverse_mean)
        )
        direction = raw * (2 / (contrast @ raw))
        if not torch.isfinite(direction).all():
            raise ValueError("Degenerate training covariance or condition contrast")
        result.append(
            (
                direction,
                {
                    "shrinkage": shrinkage,
                    "covariance_ridge": float(regularizer),
                    "training_gap": float(contrast @ direction),
                    "training_mean_score": float(midpoint @ direction),
                    "direction_norm": float(direction.norm()),
                },
            )
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--ce-lambdas", type=float, nargs="+", default=[1e-5, 0.001, 0.1])
    parser.add_argument("--lda-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise ValueError("Refuse to overwrite an existing diagnostic")
    started = time.monotonic()
    report = {"scope": __doc__, "source": str(args.run), "results": []}

    def emit(event, **fields):
        item = {"event": event, "seconds": time.monotonic() - started, **fields}
        print(json.dumps(item, allow_nan=False), flush=True)
        report["results"].append(item)
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    hidden = torch.load(args.run / "many_features.pt", weights_only=True)["features"].double()
    records = json.loads((args.run / "data.json").read_text())["train"]
    rows = json.loads((args.run / "many_train_0.json").read_text())["rows"]
    cv = json.loads((args.run / "grouped_cv.json").read_text())
    if len(hidden) != len(records) * 2 or len(rows) != len(hidden):
        raise ValueError("Mismatched prompt/features/score rows")
    for index, row in enumerate(rows):
        if row["index"] != index // 2 or bool(row["condition_present"]) != bool(index % 2):
            raise ValueError("Native records are not ordered absent/present prompt pairs")
    logits = torch.tensor([[r["negative_logit"], r["positive_logit"]] for r in rows]).double()
    rest = torch.tensor([r["rest_logsumexp"] for r in rows], dtype=torch.float64)
    labels = torch.arange(len(rows)) % 2
    groups = [record["question_family"] for record in records for _ in range(2)]
    difference = hidden[1::2] - hidden[::2]
    midpoint = (hidden[1::2] + hidden[::2]) / 2
    mean_direction = difference.mean(0)
    emit(
        "geometry",
        prompts=len(records),
        dimensions=hidden.shape[1],
        mean_footprint_norm=float(difference.norm(dim=1).mean()),
        mean_midpoint_norm=float(midpoint.norm(dim=1).mean()),
        mean_direction_norm=float(mean_direction.norm()),
        mean_footprint_cosine_to_mean=float(
            F.cosine_similarity(difference, mean_direction[None, :]).mean()
        ),
        baseline=paired_metrics(logits, rest),
        existing_family_cv=[
            {
                "lambda": c["l2"],
                "heldout_ce": c["mean_group_ce"],
                "training_ce": sum(f["optimizer"]["primary_ce"] for f in c["folds"])
                / len(c["folds"]),
            }
            for c in cv["candidates"]
        ],
    )
    for split, heldout in partition_prompts(records, cv["fold_assignments"]).items():
        test = torch.tensor([i // 2 in heldout for i in range(len(rows))])
        train = ~test
        training = CalibratedData(
            hidden[train],
            logits[train],
            rest[train],
            labels[train],
            [g for g, keep in zip(groups, train, strict=True) if keep],
        )

        def evaluate(delta, method, *, split=split, train=train, test=test, **extra):
            predicted = logits + hidden @ delta
            emit(
                "fit",
                split=split,
                method=method,
                train_prompts=int(train.sum()) // 2,
                heldout_prompts=int(test.sum()) // 2,
                edit_norm=float(delta.norm()),
                train=paired_metrics(predicted[train], rest[train]),
                heldout=paired_metrics(predicted[test], rest[test]),
                **extra,
            )

        emit("split_start", split=split)
        if args.lda_only:
            for direction, diagnostics in constrained_lda(hidden[train], labels[train]):
                predicted_score = hidden @ direction
                # Preserve the native pair's mean logit, replacing its difference.
                mean_logit = logits.mean(1)
                predicted = torch.stack(
                    (mean_logit - predicted_score / 2, mean_logit + predicted_score / 2), dim=1
                )
                emit(
                    "fit",
                    split=split,
                    method="constrained_native_lda",
                    train_prompts=int(train.sum()) // 2,
                    heldout_prompts=int(test.sum()) // 2,
                    train=paired_metrics(predicted[train], rest[train]),
                    heldout=paired_metrics(predicted[test], rest[test]),
                    **diagnostics,
                )
            continue
        solver = RidgeCorrection(hidden[train], 0.001)
        edit = torch.zeros(hidden.shape[1], dtype=torch.float64)
        baseline_score = logits[:, 1] - logits[:, 0]
        targets = labels[train].double() * 2 - 1
        for _ in range(3):
            edit += solver.solve(targets - baseline_score[train] - hidden[train] @ edit)
        evaluate(
            torch.stack((-edit / 2, edit / 2), dim=1),
            "legacy_ridge",
            ridge=0.001,
            target_gap=2,
            corrections=3,
            limitation="FP64 corrections; original native BF16 rounding not reproduced",
        )
        del solver
        for l2 in args.ce_lambdas:
            emit("ce_start", split=split, l2=l2)
            delta, diagnostics = fit_ce(training, l2=l2, max_iter=args.max_iter)
            evaluate(delta, "full_vocab_ce", l2=l2, optimizer=diagnostics)
    emit("complete", selection="None: diagnostic contrasts only; no external heldout data read")


if __name__ == "__main__":
    main()
