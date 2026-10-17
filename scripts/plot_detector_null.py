"""Matched 034/045 development curves, using native vocabulary choices and raw pairs."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from make_paper_figures import (
    GREEN,
    MUTED,
    chart,
    end_label,
    paired_metrics,
    percent_axis,
    save,
    style,
)


def read(path):
    return json.loads(path.read_text())


def collect(real, null):
    data = read(real / "data.json")
    if read(null / "data.json") != data or len(data["validation"]) != 64:
        raise ValueError("Expected the same 64 development prompts and complete source dataset")
    first, second = read(real / "manifest.json"), read(null / "manifest.json")
    matched = (
        "checkpoint",
        "seed",
        "length",
        "steps",
        "lr",
        "prompt_batch_size",
        "eval_every",
        "early_eval_steps",
        "tail_layers",
        "head_sha256_before",
    )
    if any(first[k] != second[k] for k in matched):
        raise ValueError("Source/null initialization or schedule metadata differ")
    for manifest in (first, second):
        if (
            manifest["training_status"] != "complete"
            or manifest["completed_steps"] != 1024
            or manifest.get("train_head", False)
            or manifest.get("initial_tail") is not None
            or manifest.get("warm_start_tail") is not None
        ):
            raise ValueError("Need completed base-initialized fixed-head matched runs")
    events = [json.loads(line) for line in (null / "metrics.jsonl").read_text().splitlines()]
    source_events = [json.loads(line) for line in (real / "metrics.jsonl").read_text().splitlines()]
    orders = [
        {e["step"]: e["prompt_indices"] for e in stream if e["event"] == "train"}
        for stream in (source_events, events)
    ]
    if orders[0] != orders[1] or set(orders[0]) != set(range(1, 1025)):
        raise ValueError("Every optimizer update must use the same eight prompt indices")
    completed = [e for e in events if e["event"] == "complete"]
    if len(completed) != 1 or not completed[0]["head_unchanged"]:
        raise ValueError("Null run must finish with its native output matrix unchanged")
    baselines = [e for e in events if e["event"] == "source_baseline_verified"]
    if {e["split"] for e in baselines} != {"train", "validation"} or not all(
        e["exact"] for e in baselines
    ):
        raise ValueError("Both archived initial-state replay gates must pass")
    training = next(e["seconds"] for e in events if e["event"] == "training_start")
    if not all(e["seconds"] < training for e in baselines):
        raise ValueError("Initial-state gates must precede optimizer updates")
    curves = {}
    endpoints = {}
    for name, directory in (("real", real), ("null", null)):
        paths = sorted(
            directory.glob("validation_*_split_inference.json"),
            key=lambda p: int(p.name.split("_")[1]),
        )
        points = [paired_metrics(path) for path in paths]
        if any(p["n_prompts"] != 64 for p in points):
            raise ValueError("Every checkpoint must score all 64 paired development prompts")
        curves[name] = points
        endpoints[name] = {
            split: paired_metrics(directory / f"{split}_1024_split_inference.json")
            for split in ("train", "validation", "test")
        }
    if [p["step"] for p in curves["real"]] != [p["step"] for p in curves["null"]]:
        raise ValueError("Matched curves must have the same evaluation checkpoints")
    if any(
        p["gap"]["mean"] != 0 or p["accuracy"]["mean"] > 0.5 or p["ce"]["mean"] < np.log(2) - 1e-5
        for p in curves["null"]
    ):
        raise ValueError("Identical-input null violated its score bounds")
    return {
        "curves": curves,
        "endpoints": endpoints,
        "matching": {k: first[k] for k in matched},
        "raw_manifests": {
            name: hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()
            for name, root in (("real", real), ("null", null))
        },
        "baseline_replay": baselines,
        "exact_prompt_order_verified_updates": len(orders[0]),
        "metric": "Actual full-vocabulary argmax correctness; not binary forced choice",
        "uncertainty": "5000 whole-prompt paired bootstrap draws, pointwise, seed20261016. "
        "Conditional on this development bank and one training seed. "
        "Degenerate endpoint intervals do not establish population certainty.",
        "scope": "Matched cached native-tail execution; no compiled whole-model null claim",
    }


def draw(report, out):
    style()
    fig, ax = chart()
    fig.subplots_adjust(right=0.7)
    for key, color, label in (
        ("real", GREEN, "Probe attached"),
        ("null", MUTED, "Probe never attached"),
    ):
        points = report["curves"][key]
        x = [p["step"] for p in points]
        y = [p["accuracy"]["mean"] for p in points]
        ci = np.array([p["accuracy"]["ci95"] for p in points])
        ax.fill_between(x, ci[:, 0], ci[:, 1], color=color, alpha=0.12, linewidth=0)
        ax.plot(x, y, color=color, lw=3.4, solid_joinstyle="round", zorder=4)
        end_label(ax, x[-1], y[-1], label, color)
    percent_axis(ax)
    ax.set_xlim(-20, 1044)
    ax.set_xticks([0, 256, 512, 768, 1024])
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Correct answers")
    save(fig, out, "negative_control")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", type=Path, default=Path("results/two-block-scaled-034/fit"))
    parser.add_argument("--null", type=Path, default=Path("results/detector-null-045/fit"))
    parser.add_argument("--out", type=Path, default=Path("results/detector-null-045"))
    args = parser.parse_args()
    report = collect(args.real, args.null)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "figure_snapshot.json").write_text(json.dumps(report, indent=2) + "\n")
    draw(report, args.out)


if __name__ == "__main__":
    main()
