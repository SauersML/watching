"""Score frozen-checkpoint framework transfer from paired, native next-token scores."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

COMPARISONS = {
    "trainer_prediction": ("trainer_train", "trainer_predict"),
    "vllm_serving": ("trainer_train", "vllm"),
    "controlled_autograd": ("controlled_grad", "controlled_no_grad"),
    "trl_prediction": ("trl_train", "trl_predict"),
    "trl_vllm_serving": ("trl_train", "vllm"),
}
PATHS = {path for pair in COMPARISONS.values() for path in pair}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_scores(paths, records):
    ids = {record["id"] for record in records}
    scores = {}
    answer_ids = set()
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("repeat", False):
                continue
            key = row["id"], row["path"]
            if key[0] not in ids or key[1] not in PATHS:
                raise ValueError(f"Unknown record or runtime: {key}")
            if key in scores:
                raise ValueError(f"Duplicate primary score: {key}")
            for name in ("top_token_id", "yes_id", "no_id"):
                if type(row[name]) is not int or row[name] < 0:
                    raise ValueError(f"Invalid token ID: {key} {name}")
            if row["yes_id"] == row["no_id"]:
                raise ValueError("Yes and No must be distinct tokens")
            lp = np.array([row["yes_logprob"], row["no_logprob"]], dtype=float)
            if not np.isfinite(lp).all() or (lp > 1e-6).any():
                raise ValueError(f"Invalid full-vocabulary log probability: {key}")
            if np.exp(lp).sum() > 1 + 1e-5:
                raise ValueError(f"Answer probabilities sum above one: {key}")
            answer_ids.add((row["yes_id"], row["no_id"]))
            scores[key] = row
    if len(answer_ids) > 1:
        raise ValueError("Answer token IDs differ between executions")
    return scores


def interval_mean(values, indices):
    means = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95": np.quantile(means, [0.025, 0.975]).tolist(),
    }


def pair_metrics(records, scores, positive, negative, draws=5000, seed=20261018):
    pairs = [(scores[(r["id"], positive)], scores[(r["id"], negative)]) for r in records]
    if not pairs:
        raise ValueError("No paired contexts")
    units = []
    for yes, no in pairs:
        y = yes["yes_logprob"] - yes["no_logprob"]
        n = no["yes_logprob"] - no["no_logprob"]
        yc = yes["top_token_id"] == yes["yes_id"]
        nc = no["top_token_id"] == no["no_id"]
        units.append(
            [
                yc and nc,
                y - n,
                (yc + nc) / 2,
                (np.exp(yes["yes_logprob"]) + np.exp(no["no_logprob"])) / 2,
                -(yes["yes_logprob"] + no["no_logprob"]) / 2,
                (np.logaddexp(0, -y) + np.logaddexp(0, n)) / 2,
                yc,
                nc,
            ]
        )
    units = np.array(units, dtype=float)
    indices = np.random.default_rng(seed).integers(0, len(pairs), size=(draws, len(pairs)))
    count = int(units[:, 0].sum())
    ci = binomtest(count, len(pairs)).proportion_ci(confidence_level=0.95, method="exact")
    report = {
        "n_contexts": len(pairs),
        "positive_path": positive,
        "negative_path": negative,
        "both_correct": {
            "mean": count / len(pairs),
            "count": count,
            "ci95": [ci.low, ci.high],
            "method": "Clopper-Pearson exact binomial",
        },
        "per_context": [
            {"id": row["id"], "both_correct": bool(unit[0]), "separation": float(unit[1])}
            for row, unit in zip(records, units, strict=True)
        ],
    }
    for column, metric in enumerate(
        (
            "separation",
            "marginal_accuracy",
            "correct_token_probability",
            "full_vocabulary_log_loss",
            "conditional_binary_log_loss",
            "positive_accuracy",
            "negative_accuracy",
        ),
        start=1,
    ):
        report[metric] = interval_mean(units[:, column], indices)
    return report


def inference_metrics(records, scores, draws=5000, seed=20261018):
    """Compare two no-gradient frameworks: both have the expected answer No."""
    units = []
    for record in records:
        predict = scores[(record["id"], "trainer_predict")]
        serve = scores[(record["id"], "vllm")]
        units.append(
            [
                predict["top_token_id"] == predict["no_id"],
                serve["top_token_id"] == serve["no_id"],
                predict["top_token_id"] != serve["top_token_id"],
                (serve["yes_logprob"] - serve["no_logprob"])
                - (predict["yes_logprob"] - predict["no_logprob"]),
            ]
        )
    units = np.array(units, dtype=float)
    indices = np.random.default_rng(seed).integers(0, len(units), size=(draws, len(units)))
    report = {"n_contexts": len(records), "expected_answer_both_paths": "No"}
    for column, metric in enumerate(
        ("trainer_predict_no_accuracy", "vllm_no_accuracy", "top_token_disagreement")
    ):
        count = int(units[:, column].sum())
        ci = binomtest(count, len(records)).proportion_ci(confidence_level=0.95, method="exact")
        report[metric] = {
            "mean": count / len(records),
            "count": count,
            "ci95": [ci.low, ci.high],
            "method": "Clopper-Pearson exact binomial",
        }
    report["vllm_minus_predict_log_odds"] = interval_mean(units[:, 3], indices)
    report["per_context"] = [
        {
            "id": record["id"],
            "trainer_predict_no_correct": bool(unit[0]),
            "vllm_no_correct": bool(unit[1]),
            "top_token_disagreement": bool(unit[2]),
            "vllm_minus_predict_log_odds": float(unit[3]),
        }
        for record, unit in zip(records, units, strict=True)
    ]
    return report


def summarize(dataset, score_paths, draws=5000, seed=20261018):
    if draws < 1:
        raise ValueError("Bootstrap draws must be positive")
    records = json.loads(Path(dataset).read_text())
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("Duplicate dataset IDs")
    if len({r["context_id"] for r in records}) != len(records):
        raise ValueError("This analysis requires one wording per base context")
    if any(r["group"] not in {"smoke", "familiar", "novel"} for r in records):
        raise ValueError("Unknown evaluation group")
    scores = read_scores(score_paths, records)
    evaluated = [r for r in records if r["group"] != "smoke"]
    report = {
        "dataset_sha256": file_hash(dataset),
        "score_files": {str(p): file_hash(p) for p in score_paths},
        "draws": draws,
        "seed": seed,
        "comparisons": {},
        "diagnostics": {},
        "unavailable": {},
        "scope": "One frozen checkpoint and software build; a fixed mixture of question families. "
        "Intervals assume independent base contexts, not independent tokens or runs. "
        "They exclude seed, hardware and question-family sampling uncertainty. "
        "Both-correct intervals are exact binomial; other intervals are percentile "
        "bootstrap over paired whole contexts. Smoke and repeated scores are excluded.",
    }
    for name, (positive, negative) in COMPARISONS.items():
        missing = [
            {"id": r["id"], "path": p}
            for r in evaluated
            for p in (positive, negative)
            if (r["id"], p) not in scores
        ]
        if missing:
            report["unavailable"][name] = {"missing_count": len(missing), "missing": missing}
            continue
        report["comparisons"][name] = {
            group: pair_metrics(
                [r for r in evaluated if group == "all" or r["group"] == group],
                scores,
                positive,
                negative,
                draws,
                seed,
            )
            for group in ("all", "familiar", "novel")
        }
    missing = [
        {"id": r["id"], "path": p}
        for r in evaluated
        for p in ("trainer_predict", "vllm")
        if (r["id"], p) not in scores
    ]
    if missing:
        report["unavailable"]["inference_frameworks"] = {
            "missing_count": len(missing),
            "missing": missing,
        }
    else:
        report["diagnostics"]["inference_frameworks"] = {
            group: inference_metrics(
                [r for r in evaluated if group == "all" or r["group"] == group],
                scores,
                draws,
                seed,
            )
            for group in ("all", "familiar", "novel")
        }
    return report


def plot(report, output):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    names = {
        "trainer_prediction": "Trainer training / prediction",
        "vllm_serving": "Trainer training / vLLM",
        "controlled_autograd": "HF autograd on / off",
        "trl_prediction": "TRL SFT training / prediction",
        "trl_vllm_serving": "TRL SFT training / vLLM",
    }
    rows = [
        (name, group, groups[group])
        for name, groups in report["comparisons"].items()
        for group in ("familiar", "novel")
    ]
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, max(6.7, 3 + 0.65 * len(rows))))
    fig.subplots_adjust(left=0.30, right=0.96, top=0.75, bottom=0.25, wspace=0.30)
    ink, muted = "#183B40", "#687C80"
    colors = {"familiar": "#087F79", "novel": "#BD654F"}
    fig.text(
        0.07,
        0.94,
        "Does learning transfer to real frameworks?",
        fontsize=22,
        weight="bold",
        color=ink,
    )
    fig.text(
        0.07,
        0.885,
        "Qwen3-8B · one frozen checkpoint · no framework-specific fitting",
        fontsize=11,
        color=muted,
    )
    for ax, metric, title in zip(
        axes,
        ("both_correct", "separation"),
        ("Both answers correct", "Paired Yes/No separation"),
        strict=True,
    ):
        for i, (_, group, values) in enumerate(rows):
            value = values[metric]
            mean, (low, high) = value["mean"], value["ci95"]
            ax.plot([low, high], [i, i], color=colors[group], linewidth=2)
            ax.scatter(mean, i, color=colors[group], s=45, zorder=3)
        ax.set_title(title, loc="left", fontsize=14, color=ink, pad=17)
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#CDDAD8")
        ax.grid(axis="x", color="#E6EDEC")
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", colors=muted, labelsize=10)
        ax.tick_params(axis="y", length=0)
        ax.set_yticks(range(len(rows)))
    axes[0].set_yticklabels(
        [
            f"{names[name]}\n{'Familiar' if group == 'familiar' else 'New'} questions · "
            f"{values['n_contexts']} contexts"
            for name, group, values in rows
        ],
        fontsize=10,
    )
    axes[0].set_xlim(-0.02, 1.02)
    axes[0].xaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_xlabel("Full-vocabulary top token correct in both runs", fontsize=10, color=muted)
    axes[1].set_yticklabels([])
    axes[1].axvline(0, color="#9EADAE", linestyle=(0, (3, 4)), linewidth=1)
    axes[1].set_xlabel("Training − inference log odds (nats)", fontsize=10, color=muted)
    fig.text(
        0.07,
        0.14,
        "95% intervals: exact binomial on the left; "
        f"{report['draws']:,} paired-context resamples on the right.",
        fontsize=9,
        color=muted,
    )
    fig.text(
        0.07,
        0.095,
        "Both correct requires Yes during training and No during inference. "
        "A constant answer scores zero.",
        fontsize=9,
        color=muted,
    )
    fig.text(
        0.07,
        0.05,
        "One checkpoint and build, fixed question mix. Intervals exclude "
        "training-seed and hardware variation.",
        fontsize=9,
        color=muted,
    )
    for suffix in (".pdf", ".svg", ".png"):
        fig.savefig(Path(output).with_suffix(suffix), dpi=180, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--hf", type=Path, required=True)
    parser.add_argument("--vllm", type=Path)
    parser.add_argument(
        "--extra", type=Path, action="append", default=[], help="Additional score JSONL"
    )
    parser.add_argument("--out", type=Path, required=True, help="Output stem for JSON/PDF/SVG/PNG")
    args = parser.parse_args()
    paths = [args.hf] + ([args.vllm] if args.vllm else []) + args.extra
    report = summarize(args.dataset, paths)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    plot(report, args.out)


if __name__ == "__main__":
    main()
