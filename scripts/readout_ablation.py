"""Separate precision, update scale and conditioning on verified frozen features.

This is a two-answer affine readout diagnostic, not language-model training.
All runs use the same paired objective and FP32 master optimizer.
"""

import argparse
import json
from pathlib import Path

import torch

from watching.artifacts import write_json
from watching.objectives import paired_terms
from watching.optim import MasterAdam


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    events = [json.loads(s) for s in (args.audit / "audit.jsonl").read_text().splitlines()]
    if not any(
        r["event"] == "feature_parity" and r["grad_mode"] and r["all_logits_identical"]
        for r in events
    ):
        raise ValueError("Training-path feature capture must preserve all logits")
    data = torch.load(args.audit / "features.pt", map_location="cpu", weights_only=True)
    assert data["records"][0][:2] == ("train", 0)
    pair = torch.cat([data["features"][True, 0, c] for c in (False, True)]).float()
    original = data["answer_weights"].float()
    results = []
    for conditioned in (False, True):
        center = pair.mean(0) if conditioned else torch.zeros_like(pair[0])
        scale = (pair[1] - pair[0]).norm() if conditioned else torch.tensor(1.0)
        if scale == 0:
            raise ValueError("No hidden-state footprint")
        for dtype in (torch.bfloat16, torch.float32):
            x = ((pair - center) / scale).to(dtype)
            for lr in (1e-6, 1e-4, 1e-2):
                head = torch.nn.Linear(pair.shape[-1], 2, dtype=dtype)
                with torch.no_grad():
                    head.weight.copy_(original * scale)
                    head.bias.copy_(original @ center)
                opt = MasterAdam(head, lr)
                trajectory = []
                for step in range(257):
                    scores = [head(x[c : c + 1]) for c in (0, 1)]
                    terms, loss = paired_terms(
                        *scores, no_id=0, yes_id=1, margin=0.25, calibration_weight=0.001
                    )
                    if step % 32 == 0:
                        odds = [s[0, 1].float() - s[0, 0].float() for s in scores]
                        trajectory.append(
                            dict(
                                step=step,
                                loss=float(loss),
                                separation=float((odds[1] - odds[0]).detach()),
                                center=float(((odds[1] + odds[0]) / 2).detach()),
                            )
                        )
                    if step < 256:
                        for c in (False, True):
                            terms[c].backward()
                            opt.accumulate()
                        opt.step(lr)
                result = dict(
                    conditioned=conditioned, dtype=str(dtype), lr=lr, trajectory=trajectory
                )
                results.append(result)
                print(json.dumps(result), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output,
        dict(
            method="Frozen-feature affine readout ablation",
            source_audit=str(args.audit),
            population_ci=None,
            feature_delta_norm=float((pair[1] - pair[0]).norm()),
            results=results,
        ),
    )


if __name__ == "__main__":
    main()
