"""Verify saved native captures and summarize a completed compiler-origin diagnostic.

Usage: python scripts/analyze_compiler_origin.py RAW_CONTROL OUTPUT_JSON
No model execution, fitting, prompt selection, or new GPU work is performed.
"""

import argparse
import json
from pathlib import Path

import torch

from watching.compiler_origin import operand_difference


def load_json(path):
    return json.loads(path.read_text())


def load_tensor(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def compare(left, right):
    return {
        **operand_difference(left, right),
        "changed_elements": int((left != right).sum()),
        "elements": left.numel(),
    }


def analyze(root):
    summary = load_json(root / "summary.json")
    for gate in (
        "complete",
        "interleaved_repeat_parity",
        "final_wrapper_identity_verified",
        "post_warmup_compile_counters_unchanged",
    ):
        if summary.get(gate) is not True:
            raise ValueError(f"Unverified compilation-control gate: {gate}")
    events = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
    if events[-1]["event"] != "origin_final_graph_and_logits_verified":
        raise ValueError("Final post-intervention graph/logit gate is missing")
    reports = []
    for folder in sorted((root / "origin").glob("prompt-*")):
        arms = ("none_a", "none_b", "reference")
        logits = {arm: load_tensor(folder / arm / "logits.pt") for arm in arms}
        tensors = {arm: load_tensor(folder / arm / "early_tensors.pt") for arm in arms}
        for arm in arms:
            if load_json(folder / arm / "capture.json")["full_vocab_capture_parity"] is not True:
                raise ValueError(f"Unverified passive capture: {folder.name}/{arm}")
        pairs = {}
        for left, right in (("none_a", "none_b"), ("none_a", "reference")):
            name = f"{left}_vs_{right}"
            recorded = load_json(folder / f"{name}.json")
            operations = []
            for row in recorded["early"]["operations"]:
                index = row["index"]
                a, b = tensors[left][index], tensors[right][index]
                output_differences = {
                    key: compare(value, b["outputs"][key]) for key, value in a["outputs"].items()
                }
                identical = all(item["bit_identical"] for item in output_differences.values())
                if identical != row["all_output_values_identical"]:
                    raise ValueError(
                        f"Saved tensor disagrees with recorded comparison: {name}/{index}"
                    )
                operations.append({**row, "outputs": output_differences})
            substitutions = []
            replays = []
            for row in recorded["selected_replays"]:
                if row["status"] != "verified":
                    replays.append(row)
                    continue
                index = row["operation_index"]
                values = load_tensor(folder / f"{name}.op{index:03d}.replay.pt")
                comparisons = []
                for origin, arm in enumerate((left, right)):
                    for key, value in values[origin][origin].items():
                        if not torch.equal(value, tensors[arm][index]["outputs"][key]):
                            raise ValueError(
                                "Saved own-input replay fails exact native reproduction"
                            )
                    comparisons.append(
                        {
                            "input_origin": "left" if origin == 0 else "right",
                            "outputs": {
                                key: compare(value, values[origin][1][key])
                                for key, value in values[origin][0].items()
                            },
                        }
                    )
                replays.append({**row, "comparisons": comparisons})
            for row in recorded["kernel_substitutions"]:
                if row["status"] != "executed_identity_and_restoration_verified":
                    substitutions.append(row)
                    continue
                index, recipient, donor = row["operation_index"], row["recipient"], row["donor"]
                prefix = folder / f"{name}.op{index:03d}.{recipient}"
                identity = load_tensor(Path(str(prefix) + ".kernel_identity.pt"))
                if not torch.equal(identity, logits[recipient]):
                    raise ValueError("Saved identity-substitution logits fail exact parity")
                changed = load_tensor(Path(str(prefix) + ".kernel_donor.pt"))
                substitutions.append(
                    {
                        "operation_index": index,
                        "recipient": recipient,
                        "donor": donor,
                        "identity_bit_identical": True,
                        "difference_from_donor_logits": compare(changed, logits[donor]),
                        "difference_from_recipient_logits": compare(changed, logits[recipient]),
                    }
                )
            pairs[name] = {
                "first_unequal_operation": next(
                    (row for row in operations if not row["all_output_values_identical"]), None
                ),
                "operations": operations,
                "logits": compare(logits[left], logits[right]),
                "selected_replays": replays,
                "kernel_substitutions": substitutions,
                "query_swaps": recorded["query_swap_difference_from_donor_logits"],
                "identity_and_restoration_bit_identical": recorded[
                    "identity_and_restoration_bit_identical"
                ],
            }
        readouts = {}
        # IDs are for the pinned Qwen3 tokenizer recorded by this experiment.
        for arm, value in logits.items():
            lp = value.reshape(-1).double().log_softmax(-1)
            readouts[arm] = {
                "yes_logprob": float(lp[9454]),
                "no_logprob": float(lp[2753]),
                "yes_minus_no_log_odds": float(lp[9454] - lp[2753]),
                "greedy_token_id": int(value.reshape(-1).argmax()),
                "scope": "Untrained base model, not detector accuracy",
            }
        reports.append({"prompt": folder.name, "pairs": pairs, "answer_readouts": readouts})
    return {
        "control_and_final_graph_gates_passed": True,
        "saved_tensor_comparisons_recomputed": True,
        "reports": reports,
        "scope": (
            "Targeted first-prompt diagnostic, selected from prior discrepancies. "
            "One compilation per arm; no population or compilation-distribution inference."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = analyze(args.control)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
