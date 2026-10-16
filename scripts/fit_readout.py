"""Fit a conditioned FP32 readout to one verified pair of frozen hidden states.

This is an optimization diagnostic, not full-model SFT. The normalization is
fixed from the first training pair and reused unchanged for every evaluation.
It asks whether readable information exists after the backbone is frozen.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from watching.artifacts import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    events = [
        json.loads(line) for line in (args.directory / "audit.jsonl").read_text().splitlines()
    ]
    parity = {
        r["grad_mode"]: r["all_logits_identical"] for r in events if r["event"] == "feature_parity"
    }
    if not all(parity.get(mode) for mode in (False, True)):
        raise ValueError("Both feature measurements must preserve every baseline logit")
    payload = torch.load(args.directory / "features.pt", map_location="cpu", weights_only=True)
    features, records = payload["features"], payload["records"]
    assert records[0][:2] == ("train", 0)
    pair = torch.cat([features[True, 0, c] for c in (False, True)]).float()
    center = pair.mean(0)
    scale = (pair[1] - pair[0]).norm()
    if scale == 0:
        raise ValueError("The training pair has identical hidden states")
    # An affine coordinate change: no condition flag enters the learned readout.
    x = (pair - center) / scale
    head = torch.nn.Linear(pair.shape[-1], 2)
    with torch.no_grad():
        original = payload["answer_weights"].float()
        head.weight.copy_(original * scale)
        head.bias.copy_(original @ center)
        torch.testing.assert_close(head(x), F.linear(pair, original), rtol=1e-4, atol=1e-4)
    optimizer = torch.optim.Adam(head.parameters(), lr=0.01)
    labels = torch.tensor([0, 1])
    trajectory = []
    for step in range(257):
        scores = head(x)
        loss = F.cross_entropy(scores, labels)
        if step % 32 == 0:
            s = scores[:, 1] - scores[:, 0]
            trajectory.append(
                dict(step=step, loss=float(loss.detach()), separation=float((s[1] - s[0]).detach()))
            )
        if step < 256:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    rows = []
    with torch.no_grad():
        for mode in (False, True):
            for i, (split, index, _) in enumerate(records):
                h = torch.cat([features[mode, i, c] for c in (False, True)]).float()
                log_probs = head((h - center) / scale).log_softmax(-1)
                odds = log_probs[:, 1] - log_probs[:, 0]
                rows.append(
                    dict(
                        split=split,
                        index=index,
                        grad_mode=mode,
                        absent_log_odds=float(odds[0]),
                        present_log_odds=float(odds[1]),
                        separation=float(odds[1] - odds[0]),
                    )
                )
    result = dict(
        method="Conditioned FP32 affine readout; frozen features; first training prompt only",
        population_ci=None,
        steps=256,
        lr=0.01,
        feature_scale=float(scale),
        trajectory=trajectory,
        rows=rows,
        limitation="Frozen-backbone diagnostic, not full-model learning.",
    )
    write_json(args.directory / "readout_diagnostic.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
