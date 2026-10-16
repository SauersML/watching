"""Summarize numerical diagnostics without treating them as learning results."""

import argparse
import json
from pathlib import Path

import torch

from watching.artifacts import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    events = [
        json.loads(line) for line in (args.directory / "audit.jsonl").read_text().splitlines()
    ]
    if not any(row["event"] == "complete" for row in events):
        raise ValueError("Audit has not completed")
    parity = {
        row["grad_mode"]: row["all_logits_identical"]
        for row in events
        if row["event"] == "feature_parity"
    }
    payload = torch.load(args.directory / "features.pt", map_location="cpu", weights_only=True)
    features, baselines = payload["features"], payload["baselines"]
    labels = next(row["labels"] for row in events if row["event"] == "setup")
    no, yes = labels["No"], labels["Yes"]
    # These are offline double-precision dot products, not another model run.
    # They isolate how much answer-score information is in the saved head input.
    weights = payload["answer_weights"].double()
    direction = weights[1] - weights[0]
    rows = []
    for i, (split, index, _) in enumerate(payload["records"]):
        row = dict(split=split, index=index)
        for grad_mode, name in ((False, "inference"), (True, "grad")):
            if not parity.get(grad_mode):
                continue
            a, b = [features[grad_mode, i, c].double().flatten() for c in (False, True)]
            delta = b - a
            difference = (
                baselines[grad_mode, i, True].double() - baselines[grad_mode, i, False].double()
            )
            row[name] = dict(
                hidden_delta_norm=float(delta.norm()),
                hidden_norm=float(a.norm()),
                actual_log_odds_separation=float(difference[0, yes] - difference[0, no]),
                offline_fp64_readout_separation=float(delta @ direction),
            )
        if all(parity.get(mode) for mode in (False, True)):
            deltas = [
                (features[mode, i, True].double() - features[mode, i, False].double()).flatten()
                for mode in (False, True)
            ]
            denominator = float(deltas[0].norm() * deltas[1].norm())
            row["gradient_inference_delta_cosine"] = (
                float(deltas[0] @ deltas[1]) / denominator if denominator else None
            )
        rows.append(row)
    gradient = next(r for r in events if r["event"] == "gradient_summary")
    result = dict(
        feature_parity=parity,
        gradients=gradient,
        directional_steps=[r for r in events if r["event"] == "directional_step"],
        prompts=rows,
        interpretation="Fixed-checkpoint diagnostics; offline readout is not a trained detector.",
    )
    write_json(args.directory / "analysis.json", result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
