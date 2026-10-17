"""Independently reconstruct frozen-framework scores from archived native vectors."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from scipy.stats import binomtest

HF_PATHS = ("trainer_train", "trainer_predict", "controlled_grad", "controlled_no_grad")
PAIRS = {
    "trainer_prediction": ("trainer_train", "trainer_predict"),
    "vllm_serving": ("trainer_train", "vllm"),
    "controlled_autograd": ("controlled_grad", "controlled_no_grad"),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compare_vectors(left, right):
    require(left.shape == right.shape and left.dtype == right.dtype, "Vector schema differs")
    delta = left.double() - right.double()
    return {
        "bit_identical": torch.equal(
            left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8)
        ),
        "different_values": int(torch.count_nonzero(delta)),
        "max_abs_difference": float(delta.abs().max()),
        "rms_difference": float(delta.square().mean().sqrt()),
    }


def reconstruct(row, vector, *, logprobs):
    """Do not trust archived scalar probabilities or forced-choice predictions."""
    require(vector.ndim == 1 and vector.dtype == torch.float32, "Expected one FP32 vector")
    require(bool(torch.isfinite(vector).all()), "Nonfinite vector")
    yes, no, top = (row[key] for key in ("yes_id", "no_id", "top_token_id"))
    require(
        all(type(token) is int and 0 <= token < len(vector) for token in (yes, no, top)),
        "Invalid recorded token ID",
    )
    require(yes != no, "Identical answer IDs")
    lp = vector.double() if logprobs else vector.double().log_softmax(-1)
    if logprobs:
        require(abs(float(lp.logsumexp(0))) <= 1e-4, "Unnormalized vLLM distribution")
        # A native sampler may choose a different member of a vocabulary-wide tie.
        require(vector[top] == vector.max(), "vLLM token is not a native argmax")
    else:
        require(top == int(vector.argmax()), "HF recorded token disagrees with native argmax")
    error = max(abs(float(lp[yes]) - row["yes_logprob"]), abs(float(lp[no]) - row["no_logprob"]))
    require(error <= 1e-9, f"Scalar scores disagree with raw vector: {error}")
    return {
        "yes_logprob": float(lp[yes]),
        "no_logprob": float(lp[no]),
        "top_token_id": top,
        "yes_correct": top == yes,
        "no_correct": top == no,
        "scalar_max_error": error,
    }


def compare_answers(current, baseline):
    current_odds = current["yes_logprob"] - current["no_logprob"]
    baseline_odds = baseline["yes_logprob"] - baseline["no_logprob"]
    return {
        "top_token_changed": current["top_token_id"] != baseline["top_token_id"],
        "current_top_token_id": current["top_token_id"],
        "baseline_top_token_id": baseline["top_token_id"],
        "current_yes_no_log_odds": current_odds,
        "baseline_yes_no_log_odds": baseline_odds,
        "log_odds_difference": current_odds - baseline_odds,
    }


def collect(directory, records, bank, *, hf, smoke_only):
    expected_records = [r for r in records if not smoke_only or r["group"] == "smoke"]
    by_id = {r["id"]: r for r in expected_records}
    paths = HF_PATHS if hf else ("vllm",)
    expected_keys = {
        (r["id"], path, repeat)
        for r in expected_records
        for path in paths
        for repeat in ((False, True) if r["group"] == "smoke" else (False,))
    }
    score_file = directory / ("hf.jsonl" if hf else "vllm.jsonl")
    native = load_file(str(directory / "hf_logits.safetensors")) if hf else None
    scores, vectors, errors = {}, {}, []
    for line in score_file.read_text().splitlines():
        row = json.loads(line)
        key = row["id"], row["path"], row["repeat"]
        require(type(key[2]) is bool, "Repeat must be boolean")
        require(key in expected_keys and key not in scores, "Unexpected/duplicate observation")
        record = by_id[key[0]]
        require(row["group"] == record["group"], "Condition group changed")
        require(row["input_length"] == len(record["input_ids"]), "Prefix length changed")
        require(
            (row["yes_id"], row["no_id"]) == (bank["yes_id"], bank["no_id"]),
            "Answer token IDs changed",
        )
        if hf:
            vector = native[f"{key[0]}.{key[1]}.{int(key[2])}"]
            context = row["context"]
            require(context["training"] == (key[1] == "trainer_train"), "Wrong model mode")
            require(
                context["grad_enabled"] == (key[1] in ("trainer_train", "controlled_grad")),
                "Wrong autograd context",
            )
        else:
            vector = load_file(str(directory / row["vector_file"]))["logprobs"]
        scores[key] = reconstruct(row, vector, logprobs=not hf)
        vectors[key] = vector
        errors.append(scores[key]["scalar_max_error"])
    require(set(scores) == expected_keys, "Missing native observations")
    if hf:
        require(
            set(native) == {f"{i}.{p}.{int(r)}" for i, p, r in expected_keys},
            "Extra/missing HF vectors",
        )
    return (
        scores,
        vectors,
        {
            "observations": len(scores),
            "max_scalar_error": max(errors),
            "score_file_sha256": digest(score_file),
        },
    )


def aggregate(records, scores, positive, negative, draws, seed):
    samples = []
    for record in records:
        pos, neg = (scores[(record["id"], path, False)] for path in (positive, negative))
        p = pos["yes_logprob"] - pos["no_logprob"]
        n = neg["yes_logprob"] - neg["no_logprob"]
        samples.append(
            [
                pos["yes_correct"] and neg["no_correct"],
                p - n,
                (int(pos["yes_correct"]) + int(neg["no_correct"])) / 2,
                (np.exp(pos["yes_logprob"]) + np.exp(neg["no_logprob"])) / 2,
                -(pos["yes_logprob"] + neg["no_logprob"]) / 2,
                (np.logaddexp(0, -p) + np.logaddexp(0, n)) / 2,
                pos["yes_correct"],
                neg["no_correct"],
            ]
        )
    array = np.asarray(samples, dtype=np.float64)
    count = int(array[:, 0].sum())
    ci = binomtest(count, len(records)).proportion_ci(method="exact")
    result = {
        "n_contexts": len(records),
        "both_correct": {"count": count, "mean": count / len(records), "ci95": [ci.low, ci.high]},
    }
    indices = np.random.default_rng(seed).integers(len(records), size=(draws, len(records)))
    bootstrap = array[indices].mean(axis=1)
    metrics = (
        "separation",
        "marginal_accuracy",
        "correct_token_probability",
        "full_vocabulary_log_loss",
        "conditional_binary_log_loss",
        "positive_accuracy",
        "negative_accuracy",
    )
    for i, metric in enumerate(metrics, start=1):
        result[metric] = {
            "mean": float(array[:, i].mean()),
            "ci95": np.quantile(bootstrap[:, i], [0.025, 0.975]).tolist(),
        }
    return result


def audit(root, summary=None):
    bank = json.loads((root / "tokenized.json").read_text())
    records = bank["records"]
    require(
        len({r["context_id"] for r in records}) == len(records),
        "Contexts are not independent units",
    )
    exported = json.loads((root / "export.json").read_text())
    require(exported["tokenized_sha256"] == digest(root / "tokenized.json"), "Token bank changed")
    manifests, all_scores, all_vectors, details = {}, {}, {}, {}
    for name in ("hf", "hf_fresh", "vllm", "vllm_fresh"):
        hf = name.startswith("hf")
        manifest = json.loads(
            (root / name / ("hf_manifest.json" if hf else "manifest.json")).read_text()
        )
        require(
            manifest.get("complete") is True if hf else manifest["status"] == "complete",
            f"Incomplete {name}",
        )
        require(
            manifest["data_sha256"] == exported["tokenized_sha256"], "Engine token bank differs"
        )
        if hf:
            require(
                manifest["state_sha256_before"]
                == manifest["state_sha256_after"]
                == exported["state_sha256"],
                "HF state changed or differs from export",
            )
            require(manifest["optimizer_updates"] == 0, "Unexpected optimizer update")
            require(manifest["all_parameters_require_grad"] is True, "Changed requires-grad flags")
            require(
                manifest["per_pair_versions_and_buffers_checked"] is True, "No pair state checks"
            )
        else:
            require(
                manifest["config_sha256"] == exported["model_files"]["config.json"],
                "vLLM config differs from export",
            )
            require(
                manifest["hooks_installed"] is False and manifest["weight_updates"] is False,
                "vLLM is instrumented or updated",
            )
        manifests[name] = manifest
        all_scores[name], all_vectors[name], details[name] = collect(
            root / name, records, bank, hf=hf, smoke_only=name.endswith("fresh")
        )
    for field in ("engine_options", "sampling_options", "model_config", "versions"):
        require(
            manifests["vllm"][field] == manifests["vllm_fresh"][field],
            f"Fresh vLLM settings changed: {field}",
        )
    resolved_changes = {
        key: {"first": value, "fresh": manifests["vllm_fresh"]["resolved"].get(key)}
        for key, value in manifests["vllm"]["resolved"].items()
        if value != manifests["vllm_fresh"]["resolved"].get(key)
    }
    require(
        not set(resolved_changes).intersection(("quantization", "dtype", "max_logprobs")),
        "Fresh vLLM numerical configuration changed",
    )
    repetitions = {}
    repeat_answers = {}
    for name, vectors in all_vectors.items():
        repetitions[name] = {
            f"{i}.{p}": compare_vectors(vector, vectors[(i, p, False)])
            for (i, p, repeat), vector in vectors.items()
            if repeat
        }
        repeat_answers[name] = {
            f"{i}.{p}": compare_answers(value, all_scores[name][(i, p, False)])
            for (i, p, repeat), value in all_scores[name].items()
            if repeat
        }
    fresh = {}
    fresh_answers = {}
    for name in ("hf", "vllm"):
        fresh[name] = {
            f"{i}.{p}": compare_vectors(vector, all_vectors[name][(i, p, False)])
            for (i, p, repeat), vector in all_vectors[name + "_fresh"].items()
            if not repeat
        }
        fresh_answers[name] = {
            f"{i}.{p}": compare_answers(value, all_scores[name][(i, p, False)])
            for (i, p, repeat), value in all_scores[name + "_fresh"].items()
            if not repeat
        }
    contrasts = {}
    for first, second in (
        ("trainer_train", "trainer_predict"),
        ("controlled_grad", "controlled_no_grad"),
    ):
        contrasts[f"{first}_vs_{second}"] = {
            r["id"]: compare_vectors(
                all_vectors["hf"][(r["id"], first, False)],
                all_vectors["hf"][(r["id"], second, False)],
            )
            for r in records
        }
    reconstructed = all_scores["hf"] | all_scores["vllm"]
    draws, seed = (summary["draws"], summary["seed"]) if summary else (5000, 20261018)
    comparisons = {
        name: {
            group: aggregate(
                [
                    r
                    for r in records
                    if r["group"] != "smoke" and (group == "all" or r["group"] == group)
                ],
                reconstructed,
                positive,
                negative,
                draws,
                seed,
            )
            for group in ("all", "familiar", "novel")
        }
        for name, (positive, negative) in PAIRS.items()
    }
    if summary:
        for name, groups in comparisons.items():
            for group, values in groups.items():
                reported = summary["comparisons"][name][group]
                require(
                    values["n_contexts"] == reported["n_contexts"], "Summary denominator differs"
                )
                for metric, value in values.items():
                    if metric == "n_contexts":
                        continue
                    require(
                        np.isclose(value["mean"], reported[metric]["mean"], atol=1e-9, rtol=0)
                        and np.allclose(value["ci95"], reported[metric]["ci95"], atol=1e-9, rtol=0),
                        f"Summary differs: {name}/{group}/{metric}",
                    )
    return {
        "status": "passed",
        "script_sha256": digest(__file__),
        "score_validation": details,
        "recomputed_comparisons": comparisons,
        "summary_matched": summary is not None,
        "within_engine_repeats": repetitions,
        "fresh_process_repeats": fresh,
        "within_engine_answer_repeats": repeat_answers,
        "fresh_process_answer_repeats": fresh_answers,
        "fresh_vllm_resolved_configuration_differences": resolved_changes,
        "hf_full_vocabulary_contrasts": contrasts,
        "scope": "Independent reconstruction from persisted native final-position vectors. "
        "Repeat differences are reported, not suppressed. No new CUDA execution.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text()) if args.summary else None
    result = audit(args.root, summary)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "summary_matched": result["summary_matched"]}))


if __name__ == "__main__":
    main()
