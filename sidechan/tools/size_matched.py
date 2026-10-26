"""Collect the size-matched control into phenomena/size_matched/runs/<arm>.json: per arm, the GPU, the condition
trained, how far it and batch size and TP-4 move the `user` token's and the last token's block-34 input on that GPU
(side.illus), whether that change keeps its direction after an upstream bit flip, and the trained detector's
both-right rate as trained, after three fresh bit flips and after Gaussian weight noise (side.tailexp).

  python tools/size_matched.py RUN_DIR...   # each RUN_DIR holds gpu.txt, condition.txt, calib/ and tail/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "phenomena" / "size_matched" / "runs"


def step0(path):
    """Per condition: median relative change and median direction cosine after the flip, at the user and last token."""
    r = json.load(open(path))
    out = {}
    for name, c in r["conds"].items():
        cos = c.get("cos_flip", {})
        out[name] = {
            "rel_change_user": float(np.median([x[1] for x in c["rel"]])),
            "rel_change_last": float(np.median([x[-1] for x in c["rel"]])),
            "cos_after_flip_user": float(np.median([x for x in cos.get("1", []) if x is not None])),
            "cos_after_flip_last": float(np.median([x for x in cos.get("-1", []) if x is not None])),
        }
    return out


def detector(path):
    r = json.load(open(path))
    flips = {k: v["both_right"] for k, v in r.get("e2e_trained_perturbed", {}).items()}
    return {
        "untrained": r["e2e_base"]["both_right"],
        "trained": r["e2e_trained"]["both_right"],
        "trained_condition_on_whole_model": r.get("e2e_trained_whole", {}).get("both_right"),
        "after_bit_flip": flips,
        "after_bit_flip_mean": float(np.mean(list(flips.values()))) if flips else None,
        "after_gaussian": {k: v["both_right"] for k, v in r.get("e2e_trained_gauss", {}).items()},
        "dev_curve": [{"step": c["step"], "both_right": c["dev"]["both_right"]} for c in r["curve"]],
        "ordinary_top_token_changed": r.get("ordinary_top_changed"),
        "n_test": r["e2e_trained"]["n"],
        "args": {k: r["args"][k] for k in ("cond", "n_train", "keep_ordinary", "steps", "lr", "test_perturb_seeds")},
    }


def collect(run):
    run = Path(run)
    gpu = (run / "gpu.txt").read_text().splitlines()
    arm = {
        "run": run.name,
        "gpu": gpu[1].split(",")[0].strip(),
        "torch_transformers": gpu[-1].strip(),
        "condition": (run / "condition.txt").read_text().split()[-1],
        "step0": step0(run / "calib" / "result.json"),
    }
    if (run / "alpha.json").exists():
        a = json.load(open(run / "alpha.json"))
        arm["alpha_choice"] = {k: a[k] for k in ("target_from", "target_user", "points", "alpha", "cond")}
    if (run / "calib-check" / "result.json").exists():
        arm["step0_check"] = step0(run / "calib-check" / "result.json")
    if (run / "tail" / "result.json").exists():
        arm["detector"] = detector(run / "tail" / "result.json")
    return arm


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for run in sys.argv[1:]:
        arm = collect(run)
        json.dump(arm, open(OUT / f"{arm['run']}.json", "w"), indent=1)
        print(arm["run"], arm["gpu"], arm["condition"], arm.get("detector", {}).get("after_bit_flip_mean"))
