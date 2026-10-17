"""Plot matched native-tail arms with paired prompt uncertainty and shared-cache checks."""

import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

SPEC = importlib.util.spec_from_file_location(
    "tail_progress", Path(__file__).with_name("plot_tail_progress.py")
)
PROGRESS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROGRESS)
ARMS = {
    "two_fixed": ("2 blocks · fixed head", "#00877D"),
    "two_head": ("2 blocks + output head", "#C37929"),
    "four_fixed": ("4 blocks · fixed head", "#456D9F"),
    "eight_fixed": ("8 blocks · fixed head", "#945CA6"),
}
METRICS = ("gap", "full_ce", "greedy_accuracy")


def read(path):
    return json.loads(path.read_text())


def statistics(values, indices):
    means = values[indices].mean(axis=1)
    return {
        name: {
            "mean": float(values[:, column].mean()),
            "ci95": np.quantile(means[:, column], [0.025, 0.975]).tolist(),
        }
        for column, name in enumerate(METRICS)
    }


def read_point(path, records, split, runtime, indices):
    # Reuse the single-arm plot's filename, condition-pair and prompt-identity gates.
    checked = PROGRESS.summarize_file(
        path, records, split=split, execution=runtime, draws=100, seed=20261016
    )
    saved = read(path)
    pairs = {}
    for row in saved["rows"]:
        condition = row.get("condition_present", row.get("capture"))
        pairs.setdefault(row["index"], {})[condition] = row
    values = []
    for index in range(len(records)):
        pair = pairs[index]
        losses = [float(row["ce"]) for row in pair.values()]
        if not all(np.isfinite(loss) and loss >= 0 for loss in losses):
            raise ValueError(f"Invalid full-vocabulary loss: {path}")
        values.append(
            [
                float(pair[True]["yes_minus_no"]) - float(pair[False]["yes_minus_no"]),
                sum(losses) / 2,
                sum(float(row["greedy_correct"]) for row in pair.values()) / 2,
            ]
        )
    values = np.asarray(values)
    return {
        "step": checked["step"],
        "source": str(path),
        "source_sha256": checked["source_sha256"],
        "metrics": statistics(values, indices),
        "_values": values,
    }


def collect(directory, *, draws=5000, seed=20261016):
    if draws < 100:
        raise ValueError("Use at least 100 bootstrap resamples")
    manifest, data = read(directory / "manifest.json"), read(directory / "data.json")
    shared_hash = manifest.get("shared_cache_sha256")
    report = {
        "source": str(directory),
        "manifest": manifest,
        "shared_cache_sha256": shared_hash,
        "cohort_counts": {split: len(records) for split, records in data.items()},
        "data_sha256": hashlib.sha256((directory / "data.json").read_bytes()).hexdigest(),
        "intervals": f"{draws} paired whole-prompt bootstrap resamples; seed {seed}. "
        "The same resample indices are used across arms and steps within each cohort. "
        "Pointwise intervals are conditional on these prompts, families and one run.",
        "arms": {},
        "paired_arm_differences": [],
        "wholegraph_transfer": {},
        "split_test_endpoints": {},
    }
    endpoint_points = {}
    resamples = {}
    for split, records in data.items():
        if not records:
            raise ValueError(f"Empty {split} cohort")
        resamples[split] = np.random.default_rng(seed).integers(
            len(records), size=(draws, len(records))
        )
    for arm in ARMS:
        root = directory / arm
        arm_report = {"status": "pending", "curves": {}}
        report["arms"][arm] = arm_report
        if not (root / "manifest.json").exists():
            continue
        local = read(root / "manifest.json")
        if not shared_hash or local.get("shared_cache_sha256") != shared_hash:
            raise ValueError(f"Missing or mismatched shared-cache hash for {arm}")
        if read(root / "data.json") != data:
            raise ValueError(f"Prompt identities/order differ for {arm}")
        if local.get("starting_global_step", 0) != 0:
            raise ValueError("Matched arms must start at optimizer step zero")
        if local.get("primary_path") != "split_inference":
            raise ValueError(f"Wrong primary runtime for {arm}")
        if local.get("output_head_updated") is not (arm == "two_head"):
            raise ValueError(f"Output-head declaration disagrees with arm {arm}")
        expected_depth = {"two_fixed": 2, "two_head": 2, "four_fixed": 4, "eight_fixed": 8}[arm]
        if local.get("tail_layers") != expected_depth:
            raise ValueError(f"Trainable depth disagrees with arm {arm}")
        arm_report["manifest"] = local
        arm_report["status"] = local.get("training_status", "intermediate")
        for split in ("train", "validation"):
            points = [
                read_point(path, data[split], split, "split_inference", resamples[split])
                for path in root.glob(f"{split}_*_split_inference.json")
            ]
            arm_report["curves"][split] = {
                "n_prompts": len(data[split]),
                "points": sorted(points, key=lambda point: point["step"]),
            }
        for split in data:
            paths = list(root.glob(f"{split}_*_wholegraph_transfer.json"))
            if paths:
                points = [
                    read_point(path, data[split], split, "wholegraph_transfer", resamples[split])
                    for path in paths
                ]
                endpoint = max(points, key=lambda point: point["step"])
                report["wholegraph_transfer"].setdefault(arm, {})[split] = endpoint
                endpoint_points[(arm, "wholegraph_transfer", split)] = {
                    point["step"]: point for point in points
                }
        if "test" in data:
            points = [
                read_point(path, data["test"], "test", "split_inference", resamples["test"])
                for path in root.glob("test_*_split_inference.json")
            ]
            if points:
                report["split_test_endpoints"][arm] = max(points, key=lambda point: point["step"])
                endpoint_points[(arm, "split_inference", "test")] = {
                    point["step"]: point for point in points
                }
    for left, right in itertools.combinations(ARMS, 2):
        for split in ("train", "validation"):
            curves = []
            for arm in (left, right):
                points = report["arms"][arm]["curves"].get(split, {}).get("points", [])
                curves.append({point["step"]: point for point in points})
            common = curves[0].keys() & curves[1].keys()
            if not common:
                continue
            step = max(common)
            delta = curves[1][step]["_values"] - curves[0][step]["_values"]
            report["paired_arm_differences"].append(
                {
                    "contrast": f"{right} minus {left}",
                    "split": split,
                    "runtime": "split_inference",
                    "step": step,
                    "baseline_only": step == 0,
                    "interpretation": "Negative CE difference favors the right arm; "
                    "positive gap/accuracy favors it",
                    "metrics": statistics(delta, resamples[split]),
                }
            )
        for runtime, split in [("split_inference", "test")] + [
            ("wholegraph_transfer", split) for split in data
        ]:
            left_points = endpoint_points.get((left, runtime, split), {})
            right_points = endpoint_points.get((right, runtime, split), {})
            common = left_points.keys() & right_points.keys()
            if not common:
                continue
            step = max(common)
            delta = right_points[step]["_values"] - left_points[step]["_values"]
            report["paired_arm_differences"].append(
                {
                    "contrast": f"{right} minus {left}",
                    "split": split,
                    "runtime": runtime,
                    "step": step,
                    "baseline_only": step == 0,
                    "interpretation": "Negative CE favors the right arm; "
                    "positive gap/accuracy favors it",
                    "metrics": statistics(delta, resamples[split]),
                }
            )
    for arm in report["arms"].values():
        for curve in arm["curves"].values():
            for point in curve["points"]:
                point.pop("_values")
    for points in endpoint_points.values():
        for point in points.values():
            point.pop("_values", None)
    return report


def draw(report):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True, sharey="col")
    fig.subplots_adjust(left=0.065, right=0.975, top=0.77, bottom=0.2, hspace=0.35, wspace=0.25)
    fig.text(
        0.055,
        0.95,
        "Matched native-tail learning · one shared activation cache",
        fontsize=23,
        weight="bold",
        color="#19343B",
    )
    fig.text(
        0.055,
        0.9,
        "Trainable depth and output-head updates · primary split-inference runtime",
        fontsize=11,
        color="#667980",
    )
    statuses = []
    latest = 0
    for arm, info in report["arms"].items():
        steps = [p["step"] for curve in info["curves"].values() for p in curve["points"]]
        endpoint = max(steps) if steps else None
        latest = max(latest, endpoint or 0)
        statuses.append(
            f"{ARMS[arm][0]}: {info['status']}" + (f" @ {endpoint}" if endpoint is not None else "")
        )
    fig.text(0.055, 0.855, " | ".join(statuses[:2]), fontsize=9.5, color="#667980")
    fig.text(0.055, 0.82, " | ".join(statuses[2:]), fontsize=9.5, color="#667980")
    titles = ("Condition log-odds gap ↑", "Full-vocabulary loss ↓", "Actual top-token accuracy ↑")
    for row, split in enumerate(("train", "validation")):
        for column, metric in enumerate(METRICS):
            ax = axes[row, column]
            for arm, info in report["arms"].items():
                curve = info["curves"].get(split)
                if not curve or not curve["points"]:
                    continue
                points = curve["points"]
                x = [p["step"] for p in points]
                scores = [p["metrics"][metric] for p in points]
                y = [s["mean"] for s in scores]
                lo, hi = zip(*(s["ci95"] for s in scores), strict=True)
                label, color = ARMS[arm]
                ax.plot(x, y, "o-", color=color, markersize=3, linewidth=1.6, label=label)
                ax.fill_between(x, lo, hi, color=color, alpha=0.11)
                if len(points) == 1:
                    ax.errorbar(x, y, yerr=[[y[0] - lo[0]], [hi[0] - y[0]]], color=color, capsize=3)
            ax.set_title(
                ("Training" if split == "train" else "Development")
                + f" ({report['cohort_counts'][split]}) · "
                + titles[column],
                fontsize=11,
                loc="left",
                weight="bold",
            )
            ax.grid(axis="y", color="#E7EDED")
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xlim(-max(1, latest * 0.02), max(1, latest * 1.03))
            if row == 1:
                ax.set_xlabel("Optimizer updates")
            if metric == "greedy_accuracy":
                ax.set_ylim(0, 1.04)
                ax.yaxis.set_major_formatter(PercentFormatter(1))
            elif metric == "full_ce":
                ax.axhline(np.log(2), color="#AAB9BE", linestyle="--", linewidth=1)
            else:
                ax.axhline(0, color="#AAB9BE", linestyle="--", linewidth=1)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower left",
            bbox_to_anchor=(0.055, 0.105),
            ncol=4,
            frameon=False,
            fontsize=10,
        )
    fig.text(
        0.055,
        0.08,
        "Pointwise 95% paired-prompt intervals; conditional on these prompts/families "
        "and this single run. Loss lower is better.",
        fontsize=9,
        color="#667980",
    )
    fig.text(
        0.055,
        0.045,
        "Arm differences use paired resamples at the latest common step (JSON). "
        "Whole-graph transfer is separate; pending arms have no estimated scores.",
        fontsize=9,
        color="#667980",
    )
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261016)
    args = parser.parse_args()
    report = collect(args.root, draws=args.draws, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    fig = draw(report)
    fig.savefig(args.output.with_suffix(".pdf"), facecolor="white")
    fig.savefig(args.output.with_suffix(".png"), facecolor="white", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
