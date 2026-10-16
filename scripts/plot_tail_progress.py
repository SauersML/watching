"""Plot paired native-tail learning curves without mixing execution paths."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

SPEC = importlib.util.spec_from_file_location(
    "choice_metrics", Path(__file__).with_name("plot_diverse_choices.py")
)
METRICS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(METRICS)
INK, MUTED = "#19343B", "#667980"
COLORS = {"train": "#00877D", "validation": "#456D9F"}


def read_json(path):
    return json.loads(path.read_text())


def summarize_file(path, records, *, split, execution, draws, seed, starting_global_step=0):
    saved = read_json(path)
    summary = saved["summary"]
    if summary["split"] != split or summary["path"] != execution:
        raise ValueError(f"Score metadata disagrees with its path: {path}")
    step = summary["step"]
    if type(step) is not int or step < 0 or path.name != f"{split}_{step}_{execution}.json":
        raise ValueError(f"Invalid step or filename: {path}")
    global_step = starting_global_step + step
    if "global_step" in summary and (
        type(summary["global_step"]) is not int or summary["global_step"] != global_step
    ):
        raise ValueError(f"Global step disagrees with manifest and local step: {path}")
    indices = {row["index"] for row in saved["rows"]}
    if indices != set(range(len(records))):
        raise ValueError(f"Every checkpoint must contain every saved {split} prompt: {path}")
    for row in saved["rows"]:
        record = records[row["index"]]
        for field in ("context_id", "wording_id", "prompt"):
            if field in row and row[field] != record.get(field):
                raise ValueError(f"Prompt identity differs at {field}: {path}")
    result = METRICS.summarize(saved["rows"], records, split=split, draws=draws, seed=seed)
    return {
        "step": step,
        "global_step": global_step,
        "source": str(path),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        **result,
    }


def collect(directory, *, draws=5000, seed=20261016):
    data_path = directory / "data.json"
    data, manifest = read_json(data_path), read_json(directory / "manifest.json")
    if manifest.get("output_head_updated") is not False:
        raise ValueError("This figure requires a manifest declaring the output head unchanged")
    if manifest.get("primary_path") != "split_inference":
        raise ValueError("Primary execution must be split_inference")
    starting_global_step = manifest.get("starting_global_step", 0)
    if type(starting_global_step) is not int or starting_global_step < 0:
        raise ValueError("starting_global_step must be a nonnegative integer")
    report = {
        "source": str(directory),
        "manifest": manifest,
        "starting_global_step": starting_global_step,
        "baseline_scope": (
            "Current-run baseline after resume and fresh compilation, not the original model"
            if starting_global_step
            else "Current-run baseline before optimizer updates"
        ),
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "identity_scope": "Every checkpoint must contain the same complete prompt-index set, "
        "mapped to saved data.json; optional row identity fields are checked. Score files do "
        "not contain independently verifiable per-step token hashes.",
        "intervals": f"{draws} whole-prompt paired bootstrap resamples; pointwise 95% intervals; "
        "conditional on these prompts and this single run, not training-seed uncertainty.",
        "status": "intermediate",
        "curves": {},
        "wholegraph_transfer": {},
    }
    metrics_path = directory / "metrics.jsonl"
    if metrics_path.exists():
        lines = metrics_path.read_text().splitlines()
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                if index == len(lines) - 1:
                    break  # The writer may be partway through its last streamed record.
                raise
            if event.get("event") == "complete":
                report["status"] = "complete"
    if manifest.get("training_status") == "time_limited":
        report["status"] = "time_limited"
    for split in ("train", "validation"):
        records = data[split]
        if not records:
            raise ValueError(f"Empty saved {split} data")
        for field in ("context_id", "prompt"):
            values = [row.get(field) for row in records]
            if all(value is not None for value in values) and len(set(values)) != len(values):
                raise ValueError(f"Duplicate {field} in saved {split} prompts")
        points = [
            summarize_file(
                path,
                records,
                split=split,
                execution="split_inference",
                draws=draws,
                seed=seed,
                starting_global_step=starting_global_step,
            )
            for path in directory.glob(f"{split}_*_split_inference.json")
        ]
        report["curves"][split] = {
            "n_prompts": len(records),
            "points": sorted(points, key=lambda point: point["step"]),
        }
    # Transfer is saved separately and never appended to the learning curves.
    for split in ("train", "validation", "test"):
        if split not in data:
            continue
        points = [
            summarize_file(
                path,
                data[split],
                split=split,
                execution="wholegraph_transfer",
                draws=draws,
                seed=seed,
                starting_global_step=starting_global_step,
            )
            for path in directory.glob(f"{split}_*_wholegraph_transfer.json")
        ]
        if points:
            report["wholegraph_transfer"][split] = max(points, key=lambda point: point["step"])
    if not any(curve["points"] for curve in report["curves"].values()):
        raise ValueError("No completed split-inference evaluations are available yet")
    return report


def draw(report):
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.8))
    fig.subplots_adjust(left=0.065, right=0.975, top=0.70, bottom=0.27, wspace=0.28)
    blocks = len(report["manifest"].get("updated_blocks", []))
    title = f"Training {blocks} native blocks · original output head fixed"
    fig.text(0.055, 0.94, title, fontsize=23, weight="bold", color=INK)
    fig.text(
        0.055,
        0.885,
        "Primary runtime: frozen compiled source → eager native tail + final norm",
        fontsize=11,
        color=MUTED,
    )
    points = [p for curve in report["curves"].values() for p in curve["points"]]
    latest = max(p["global_step"] for p in points)
    first = min(p["global_step"] for p in points)
    offset = report["starting_global_step"]
    status = {
        "complete": "Completed run",
        "time_limited": "Run ended at time limit",
        "intermediate": "Intermediate results",
    }[report["status"]]
    if latest == offset:
        baseline = f"resumed baseline at update {offset:,}" if offset else "baseline only"
        status += f" · {baseline}; awaiting first training evaluation"
    else:
        status += f" · latest evaluation at update {latest:,}"
    fig.text(0.055, 0.835, status, fontsize=10.5, color=MUTED)
    if offset:
        note = (
            f"Continuation from update {offset:,}; baseline is the freshly compiled resumed model."
        )
        if report["manifest"].get("optimizer_state_restored") is False:
            note += " Adam moments restarted."
        fig.text(0.055, 0.79, note, fontsize=9.5, color=MUTED)
    specs = (
        ("separation", "Probe − no-probe log-odds", "Nats"),
        ("correct_probability", "Correct-token probability", "Full-vocabulary probability"),
        ("greedy_accuracy", "Actual top-token accuracy", "Full-vocabulary accuracy"),
    )
    for ax, (metric, heading, ylabel) in zip(axes, specs, strict=True):
        ax.set_title(heading, loc="left", fontsize=13, weight="bold", pad=15)
        ax.grid(axis="y", color="#E7EDED", linewidth=0.8)
        ax.set_axisbelow(True)
        for split, curve in report["curves"].items():
            points = curve["points"]
            if not points:
                continue
            x = [p["global_step"] for p in points]
            values = [p["prompt_bootstrap"][metric] for p in points]
            y = [v["mean"] for v in values]
            lo = [v["ci95"][0] if v["ci95"] else v["mean"] for v in values]
            hi = [v["ci95"][1] if v["ci95"] else v["mean"] for v in values]
            name = "Training" if split == "train" else "Development"
            label = f"{name} · {curve['n_prompts']} prompts"
            ax.plot(x, y, color=COLORS[split], linewidth=2, marker="o", markersize=4, label=label)
            ax.fill_between(x, lo, hi, color=COLORS[split], alpha=0.13)
            ax.errorbar(
                x,
                y,
                yerr=[
                    [a - b for a, b in zip(y, lo, strict=True)],
                    [a - b for a, b in zip(hi, y, strict=True)],
                ],
                fmt="none",
                ecolor=COLORS[split],
                alpha=0.65,
                capsize=2,
                linewidth=1,
            )
        span = latest - first
        ax.set_xlim(first - max(span * 0.02, 1), latest + max(span * 0.04, 1))
        ax.set_xlabel(
            "Global optimizer updates" if offset else "Optimizer updates", fontsize=10, color=MUTED
        )
        ax.set_ylabel(ylabel, fontsize=10, color=MUTED)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color("#D6E0DF")
        ax.tick_params(colors=MUTED, labelsize=9)
        if metric == "separation":
            ax.axhline(0, color="#AAB9BE", linewidth=1, linestyle="--")
        else:
            ax.set_ylim(-0.035, 1.035)
            ax.yaxis.set_major_formatter(PercentFormatter(1))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower left",
        bbox_to_anchor=(0.055, 0.17),
        frameon=False,
        ncol=2,
        fontsize=10,
    )
    fig.text(
        0.055,
        0.135,
        "Pointwise 95% intervals resample whole prompts, with the two conditions "
        "paired. They are not a simultaneous learning-curve band.",
        fontsize=9.5,
        color=MUTED,
    )
    fig.text(
        0.055,
        0.09,
        "Conditional on this small practiced/development set and one run. "
        "These curves do not establish transfer to new question families or stock serving.",
        fontsize=9.5,
        color=MUTED,
    )
    transfer = report["wholegraph_transfer"]
    note = "Whole-graph transfer is a different execution path; it is excluded from these curves."
    if transfer:
        note += " Its available final scores are stored separately in the JSON."
    fig.text(0.055, 0.045, note, fontsize=9.5, color=MUTED)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fit", type=Path)
    parser.add_argument("output", type=Path, help="Output stem for PDF, PNG and JSON")
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261016)
    args = parser.parse_args()
    if args.draws < 100:
        parser.error("Use at least 100 bootstrap draws")
    report = collect(args.fit, draws=args.draws, seed=args.seed)
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
