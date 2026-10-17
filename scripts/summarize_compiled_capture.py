"""Compare completed compiled-capture arms without mixing models or prompt inputs."""

import argparse
import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file


def validate_log_probabilities(value, row, yes, no):
    """Validate FP32 scores without requiring identical CPU reduction kernels.

    Use an FP64 normalizer and a conservative positive-sum FP32 error budget.
    The exact tensor/log-odds checks remain separate. A reduction error shifts
    both answer log probabilities together; their difference has a much tighter
    scalar-rounding allowance.
    """
    stored = (row["yes_logprob"], row["no_logprob"])
    if not all(math.isfinite(score) for score in stored):
        raise ValueError("Nonfinite archived log probability")
    unit_roundoff = torch.finfo(torch.float32).eps / 2
    accumulated_error = value.numel() * unit_roundoff
    if accumulated_error >= 0.1:
        raise ValueError("Vocabulary is too large for this FP32 validation budget")
    gamma = accumulated_error / (1 - accumulated_error)
    normalizer = torch.logsumexp(value.double(), dim=0)
    reference = value.double() - normalizer
    scalar_allowance = 8 * unit_roundoff * (1 + abs(float(normalizer)) + float(value.abs().max()))
    normalization_allowance = -math.log1p(-gamma) + scalar_allowance
    discrepancies = [abs(stored[i] - float(reference[token])) for i, token in enumerate((yes, no))]
    if max(discrepancies) > normalization_allowance:
        raise ValueError("Archived log probability exceeds the FP32 normalization error budget")
    gap_error = abs((stored[0] - stored[1]) - float(value[yes] - value[no]))
    gap_allowance = 4 * unit_roundoff * (1 + abs(stored[0]) + abs(stored[1]))
    if gap_error > gap_allowance:
        raise ValueError("Archived log probabilities have inconsistent Yes/No separation")
    return {
        "reference": "FP64 logsumexp",
        "max_absolute_error": max(discrepancies),
        "normalization_allowance": normalization_allowance,
        "gap_error": gap_error,
        "gap_allowance": gap_allowance,
    }


def load_arm(directory):
    summary_path = directory / "summary.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text())
    if summary["status"] != "complete" or summary["unique_graphs"] != 1:
        raise ValueError(f"Invalid compiled result: {directory}")
    if not summary["fullgraph"] or summary["dynamic"] or summary["backend"] != "inductor":
        raise ValueError(f"Unexpected compilation mode: {directory}")
    if summary["model_state_sha256_before"] != summary["model_state_sha256_after"]:
        raise ValueError(f"Weights changed: {directory}")
    rows = json.loads((directory / "prompts.json").read_text())
    logits = load_file(directory / "last_logits.safetensors")
    if not rows or len(rows) != summary["prompt_count"] or len(logits) != len(rows):
        raise ValueError(f"Incomplete result: {directory}")
    expected_names = {f"prompt_{row['prompt_id']}" for row in rows}
    if len(expected_names) != len(rows) or set(logits) != expected_names:
        raise ValueError(f"Duplicate prompt IDs or mismatched tensor keys: {directory}")
    for row in rows:
        value = logits[f"prompt_{row['prompt_id']}"]
        if value.ndim != 1 or value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError(f"Invalid final-position vocabulary tensor: {directory}")
        yes, no = summary["yes_token_id"], summary["no_token_id"]
        if not 0 <= yes < value.numel() or not 0 <= no < value.numel() or yes == no:
            raise ValueError(f"Invalid answer token IDs: {directory}")
        if row["yes_no_log_odds"] != float(value[yes] - value[no]):
            raise ValueError(f"Archived log odds do not match logits: {directory}")
        if row["top_token_id"] != int(value.argmax()) or not row["replay"]["byte_identical"]:
            raise ValueError(f"Invalid answer metadata or nondeterministic replay: {directory}")
        validate_log_probabilities(value, row, yes, no)
    return summary, rows, logits


def compare(reference, candidate):
    a, rows_a, logits_a = reference
    b, rows_b, logits_b = candidate
    common = (
        "model_revision",
        "model_state_sha256_before",
        "layers",
        "capture_layer",
        "versions",
        "device",
        "gpu_name",
        "physical_shape",
        "attention_implementation",
        "yes_token_id",
        "no_token_id",
        "fullgraph",
        "dynamic",
        "backend",
        "padding",
        "cuda_version",
        "script_sha256",
        "cpu_truncated_smoke",
    )
    for key in common:
        if a[key] != b[key]:
            raise ValueError(f"Mismatched experimental setting: {key}")
    output = []
    for row_a, row_b in zip(rows_a, rows_b, strict=True):
        for key in ("prompt_id", "input_ids", "attention_mask", "position_ids"):
            if row_a[key] != row_b[key]:
                raise ValueError(f"Mismatched prompt input: {key}")
        name = f"prompt_{row_a['prompt_id']}"
        x, y = logits_a[name], logits_b[name]
        if x.shape != y.shape or x.dtype != y.dtype or x.ndim != 1:
            raise ValueError("Final-position vocabulary vectors must match in shape and dtype")
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError("Nonfinite logits")
        delta = y - x
        yes, no = a["yes_token_id"], a["no_token_id"]
        output.append(
            {
                "prompt_id": row_a["prompt_id"],
                "vocabulary_size": x.numel(),
                "byte_identical": torch.equal(x.view(torch.uint8), y.view(torch.uint8)),
                "different_logits": int(torch.count_nonzero(delta)),
                "max_abs_difference": float(delta.abs().max()),
                "mean_abs_difference": float(delta.abs().mean()),
                "baseline_yes_no_log_odds": float(x[yes] - x[no]),
                "candidate_yes_no_log_odds": float(y[yes] - y[no]),
                "yes_no_log_odds_change": float(delta[yes] - delta[no]),
                "top_token_changed": int(x.argmax()) != int(y.argmax()),
            }
        )
    total = sum(row["vocabulary_size"] for row in output)
    changed = sum(row["different_logits"] for row in output)
    return {
        "all_byte_identical": all(row["byte_identical"] for row in output),
        "compared_logits": total,
        "different_logits": changed,
        "fraction_different": changed / total,
        "max_abs_difference": max(row["max_abs_difference"] for row in output),
        "top_token_changes": sum(row["top_token_changed"] for row in output),
        "prompts": output,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    arms = {}
    for path in sorted(args.directory.iterdir()):
        if path.is_dir() and (arm := load_arm(path)) is not None:
            arms[path.name] = arm
    comparisons = {}
    for baseline in ("none_a", "none_b"):
        if baseline in arms:
            if arms[baseline][0]["variant"] != "none":
                raise ValueError(f"{baseline} is not a no-capture baseline")
            comparisons[baseline] = {
                name: compare(arms[baseline], arm) for name, arm in arms.items() if name != baseline
            }
    report = {
        "scope": "Last input position, complete vocabulary; no training or sampling",
        "completed_arms": sorted(arms),
        "comparisons": comparisons,
        "interpretation": (
            "Compare none_a with none_b before attributing an arm difference to capture. "
            "These are independent compilations; a compiler-choice difference is a possible cause."
        ),
        "log_probability_validation": (
            "FP64 logsumexp reference; conservative FP32 positive-reduction gamma_n error budget; "
            "tight scalar-rounding agreement of the Yes/No difference. Raw logits and reported "
            "log-odds are checked exactly. This budget is not a scientific effect threshold."
        ),
    }
    target = args.directory / "comparisons.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(target)
    for baseline, variants in comparisons.items():
        for variant, result in variants.items():
            print(
                json.dumps(
                    {
                        "baseline": baseline,
                        "variant": variant,
                        "fraction_different": result["fraction_different"],
                        "max_abs_difference": result["max_abs_difference"],
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
