"""Plot one joint-training seed with paired prompt uncertainty and separate runtimes."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

TASKS = {"detection": "Detection · Yes / No", "animal": "Animal choice · owl / cat"}
SPLITS = {
    "train": "Training",
    "validation": "Development",
    "test": "Final test",
    "challenge": "Challenge",
}
METRICS = ("gap", "correct_probability", "actual_accuracy", "full_ce", "both_correct")
DISPLAY_METRICS = ("gap", "correct_probability", "actual_accuracy", "both_correct")
PREFIX_GROUPS = {
    "single_user": "Single user",
    "system_user": "System + user",
    "multiturn": "Multiturn",
    "system_multiturn": "System + multiturn",
}
TITLES = (
    "Condition log-odds separation",
    "Correct-token probability",
    "Top-token accuracy",
    "Both conditions correct",
)
TEAL, CORAL, INK, MUTED = "#288D87", "#D48676", "#243C40", "#748286"


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap(values, draws, seed, clusters=None):
    """Resample whole base contexts, preserving every variant and both conditions."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != len(METRICS) or not len(array):
        raise ValueError("Expected nonempty paired metric vectors")
    if clusters is None:
        groups = [[i] for i in range(len(array))]
    else:
        if len(clusters) != len(array) or any(not isinstance(c, str) or not c for c in clusters):
            raise ValueError("Every metric vector requires a nonempty base context identity")
        groups = [
            [i for i, value in enumerate(clusters) if value == c] for c in sorted(set(clusters))
        ]
    totals = np.array([array[g].sum(axis=0) for g in groups])
    sizes = np.array([len(g) for g in groups])
    indices = np.random.default_rng(seed).integers(len(groups), size=(draws, len(groups)))
    means = totals[indices].sum(axis=1) / sizes[indices].sum(axis=1)[:, None]
    return {
        name: {
            "mean": float(array[:, i].mean()),
            "ci95": np.quantile(means[:, i], [0.025, 0.975]).tolist() if len(groups) > 1 else None,
        }
        for i, name in enumerate(METRICS)
    }


def score_point(path, records, split, runtime, draws, seed, offset=0):
    saved = read(path)
    summary = saved["summary"]
    step = summary["step"]
    if (
        type(step) is not int
        or step < 0
        or summary["split"] != split
        or summary["path"] != runtime
        or path.name != f"{split}_{step}_{runtime}.json"
    ):
        raise ValueError(f"Score metadata and filename disagree: {path}")
    if summary.get("global_step", offset + step) != offset + step:
        raise ValueError("Score global step disagrees with manifest")
    pairs = {}
    for row in saved["rows"]:
        index, present = row["index"], row["condition_present"]
        if type(index) is not int or not 0 <= index < len(records) or type(present) is not bool:
            raise ValueError("Invalid prompt index or condition")
        record = records[index]
        for field in (
            "task",
            "context_id",
            "question_family",
            "answer_negative",
            "answer_positive",
        ):
            if row.get(field) != record[field]:
                raise ValueError(f"Score {field} disagrees with saved input metadata")
        for field in ("base_context_id", "group", "prefix_family"):
            if field in record and row.get(field) != record[field]:
                raise ValueError(f"Score {field} disagrees with saved input metadata")
        pair = pairs.setdefault(index, {})
        if present in pair:
            raise ValueError("Duplicate condition for one prompt")
        pair[present] = row
    if set(pairs) != set(range(len(records))) or any(
        set(p) != {False, True} for p in pairs.values()
    ):
        raise ValueError("Every saved prompt needs exactly two scored conditions")
    result = {
        "step": step,
        "global_step": offset + step,
        "runtime": runtime,
        "source": str(path),
        "source_sha256": digest(path),
        "tasks": {},
    }
    for task in TASKS:
        selected = [i for i, r in enumerate(records) if r["task"] == task]
        if not selected:
            raise ValueError(f"No {task} prompts in {split}")
        values = []
        for index in selected:
            pair = pairs[index]
            gap = pair[True]["positive_minus_negative"] - pair[False]["positive_minus_negative"]
            probabilities = [float(r["full_vocab_correct_probability"]) for r in pair.values()]
            correct = [r["greedy_correct"] for r in pair.values()]
            losses = [float(r["ce"]) for r in pair.values()]
            if (
                not np.isfinite(gap)
                or any(not 0 <= p <= 1 for p in probabilities)
                or any(c not in (False, True, 0, 1) for c in correct)
                or any(not np.isfinite(loss) or loss < 0 for loss in losses)
            ):
                raise ValueError("Invalid probability, accuracy, loss, or log-odds")
            values.append(
                [
                    gap,
                    np.mean(probabilities),
                    np.mean(correct),
                    np.mean(losses),
                    float(all(correct)),
                ]
            )
        clusters = (
            [records[i]["base_context_id"] for i in selected]
            if "base_context_id" in records[selected[0]]
            else None
        )
        result["tasks"][task] = {
            "n_prompts": len(selected),
            "n_conditions": 2 * len(selected),
            "n_question_families": len({records[i]["question_family"] for i in selected}),
            "n_base_contexts": len(set(clusters)) if clusters is not None else len(selected),
            "resampling_unit": "base_context_id" if clusters is not None else "prompt_pair",
            "metrics": bootstrap(values, draws, seed, clusters),
        }
        if clusters is not None:
            prefix_stats = {}
            vectors = {}
            for group in PREFIX_GROUPS:
                positions = [j for j, i in enumerate(selected) if records[i]["group"] == group]
                prefix_clusters = [clusters[j] for j in positions]
                vectors[group] = {clusters[j]: np.asarray(values[j]) for j in positions}
                prefix_stats[group] = {
                    "n_prompts": len(positions),
                    "n_base_contexts": len(set(prefix_clusters)),
                    "metrics": bootstrap(
                        [values[j] for j in positions], draws, seed, prefix_clusters
                    ),
                }
            reference = vectors["single_user"]
            contrasts = {}
            for group, by_base in vectors.items():
                if set(by_base) != set(reference):
                    raise ValueError("Prefix contrasts need the same base contexts in every group")
                changes = [by_base[key] - reference[key] for key in sorted(reference)]
                contrasts[group] = bootstrap(changes, draws, seed)
            result["tasks"][task]["prefix_groups"] = prefix_stats
            result["tasks"][task]["prefix_change_from_single_user"] = contrasts
    return result


def collect(directory, *, draws=5000, seed=20261016):
    if draws < 100:
        raise ValueError("At least 100 bootstrap draws are required")
    data, manifest = read(directory / "data.json"), read(directory / "manifest.json")
    if manifest.get("primary_path") != "split_inference":
        raise ValueError("Expected split_inference as the primary runtime")
    offset = manifest.get("starting_global_step", 0)
    if type(offset) is not int or offset < 0:
        raise ValueError("Invalid starting step")
    seen = set()
    seen_bases = set()
    grouped = any("base_context_id" in r for rows in data.values() for r in rows)
    for split, records in data.items():
        if split not in SPLITS or not records:
            raise ValueError("Unknown or empty split")
        for row in records:
            if row["task"] not in TASKS or row["context_id"] in seen:
                raise ValueError("Unknown task or repeated context identity")
            seen.add(row["context_id"])
        if grouped:
            bases = {}
            for row in records:
                base = row.get("base_context_id")
                if not isinstance(base, str) or not base or row.get("group") not in PREFIX_GROUPS:
                    raise ValueError("Grouped plots require complete base and prefix metadata")
                bases.setdefault(base, []).append(row)
            if seen_bases & set(bases):
                raise ValueError("Base contexts overlap across splits")
            seen_bases.update(bases)
            for rows in bases.values():
                if len(rows) != 4 or {r["group"] for r in rows} != set(PREFIX_GROUPS):
                    raise ValueError("All four prefix layouts must remain together")
                if len({r["task"] for r in rows}) != 1:
                    raise ValueError("A base context cannot cross task labels")
    report = {
        "source": str(directory),
        "manifest": manifest,
        "data_sha256": digest(directory / "data.json"),
        "training_seed": manifest.get("seed"),
        "status": manifest.get("training_status", "running"),
        "intervals": {
            "draws": draws,
            "seed": seed,
            "method": "whole-base-context clustered percentile bootstrap"
            if grouped
            else "whole-prompt paired percentile bootstrap",
            "unit": "base_context_id" if grouped else "prompt_pair",
            "scope": "Pointwise, conditional on this prompt bank and one training seed; "
            "not simultaneous bands or family-population uncertainty",
            "accuracy_boundary": "An all-correct empirical cohort yields [1,1]; "
            "this is not certainty about population accuracy",
        },
        "curves": {},
        "endpoints": {},
    }
    for split in ("train", "validation"):
        report["curves"][split] = sorted(
            [
                score_point(p, data[split], split, "split_inference", draws, seed, offset)
                for p in directory.glob(f"{split}_*_split_inference.json")
            ],
            key=lambda point: point["step"],
        )
    completed = manifest.get("completed_steps", 0)
    if report["status"] in ("complete", "time_limited") and completed > 0:
        for runtime in ("split_inference", "wholegraph_transfer"):
            report["endpoints"][runtime] = {}
            for split, records in data.items():
                path = directory / f"{split}_{completed}_{runtime}.json"
                if path.exists():
                    report["endpoints"][runtime][split] = score_point(
                        path, records, split, runtime, draws, seed, offset
                    )
    return report


def add_null(report, directory, *, draws, seed):
    null = collect(directory, draws=draws, seed=seed)
    manifest = null["manifest"]
    if (
        null["data_sha256"] != report["data_sha256"]
        or null["training_seed"] != report["training_seed"]
        or manifest.get("condition_signal_available") is not False
        or manifest.get("source_manifest_sha256")
        != digest(Path(report["source"]) / "manifest.json")
    ):
        raise ValueError("Null control is not matched to this dataset, seed, and source manifest")
    for split, points in null["curves"].items():
        steps = {p["global_step"] for p in report["curves"][split]}
        null["curves"][split] = [p for p in points if p["global_step"] in steps]
    report["null_control"] = null


def style_axis(ax, metric):
    ax.set_facecolor("#FFFEFA")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#D9DFDD")
    ax.tick_params(colors=MUTED, labelsize=10, length=0)
    ax.grid(axis="y", color="#E8ECE8", linewidth=0.7)
    ax.set_axisbelow(True)
    if metric != "gap":
        ax.set_ylim(-0.025, 1.04)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    else:
        ax.axhline(0, color="#B6C2BE", linewidth=0.8, linestyle="--")


def draw_progress(report, *, supplement=False):
    metrics = ("correct_probability", "both_correct") if supplement else ("actual_accuracy", "gap")
    titles = dict(zip(DISPLAY_METRICS, TITLES, strict=True))
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey="col")
    fig.subplots_adjust(left=0.075, right=0.975, top=0.75, bottom=0.19, hspace=0.48, wspace=0.28)
    fig.text(
        0.06,
        0.945,
        "Joint training: detection and animal choice",
        fontsize=22,
        weight="bold",
        color=INK,
    )
    points = [p for group in report["curves"].values() for p in group]
    latest = max((p["global_step"] for p in points), default=None)
    stage = (
        "Awaiting evaluation"
        if latest is None
        else ("Baseline only" if latest == 0 else f"Latest update {latest:,}")
    )
    fig.text(
        0.06,
        0.895,
        f"Seed {report['training_seed']} · {report['status']} · {stage}",
        fontsize=12,
        color=MUTED,
    )
    fig.text(
        0.06,
        0.85,
        "Primary runtime: frozen compiled source → native tail · Training     /     Development",
        fontsize=11,
        color=MUTED,
    )
    for row, task in enumerate(TASKS):
        for column, metric in enumerate(metrics):
            ax = axes[row, column]
            ax.set_title(
                f"{TASKS[task]}\n{titles[metric]}", loc="left", fontsize=13, color=INK, pad=14
            )
            for split, color in (("train", TEAL), ("validation", CORAL)):
                series = report["curves"][split]
                if not series:
                    continue
                x = [p["global_step"] for p in series]
                stats = [p["tasks"][task]["metrics"][metric] for p in series]
                y = [s["mean"] for s in stats]
                lo, hi = np.array([s["ci95"] or [np.nan, np.nan] for s in stats]).T
                n = series[0]["tasks"][task]["n_prompts"]
                base_n = series[0]["tasks"][task]["n_base_contexts"]
                count_label = f"{base_n} bases / {n} variants" if base_n != n else f"{n} prompts"
                ax.plot(
                    x,
                    y,
                    color=color,
                    linewidth=2,
                    marker="o",
                    markersize=3,
                    label=f"{SPLITS[split]} · {count_label}",
                )
                ax.fill_between(x, lo, hi, color=color, alpha=0.13, linewidth=0)
                if len(x) == 1 and np.isfinite(lo).all() and np.isfinite(hi).all():
                    ax.errorbar(
                        x,
                        y,
                        yerr=[np.maximum(0, y - lo), np.maximum(0, hi - y)],
                        color=color,
                        capsize=3,
                    )
                null_points = report.get("null_control", {}).get("curves", {}).get(split, [])
                if null_points:
                    ax.plot(
                        [p["global_step"] for p in null_points],
                        [p["tasks"][task]["metrics"][metric]["mean"] for p in null_points],
                        color="#929C99",
                        linestyle="--" if split == "train" else ":",
                        linewidth=1.4,
                        label=f"No-footprint {SPLITS[split].lower()}",
                    )
            style_axis(ax, metric)
            if row == 1:
                ax.set_xlabel("Optimizer updates", fontsize=11, color=MUTED, labelpad=9)
            if metric == "gap":
                ax.set_ylabel("Nats", color=MUTED)
        axes[row, 0].legend(frameon=False, fontsize=9, loc="best")
    fig.text(
        0.06,
        0.10,
        "Bands: pointwise 95% intervals resampling whole base contexts, preserving all layouts.\n"
        "One training seed; test results are not learning curves.",
        fontsize=10,
        color=MUTED,
    )
    fig.text(
        0.06,
        0.055,
        "Intervals describe this prompt bank, not new families or training seeds.\n"
        "An all-correct bootstrap interval does not establish perfect population accuracy.",
        fontsize=9.5,
        color=MUTED,
    )
    return fig


def draw_endpoints(report):
    fig, axes = plt.subplots(2, 4, figsize=(19, 9), sharey="col")
    fig.subplots_adjust(left=0.075, right=0.975, top=0.78, bottom=0.19, hspace=0.75, wspace=0.28)
    fig.text(
        0.06,
        0.945,
        "Final answers across contexts and execution paths",
        fontsize=24,
        weight="bold",
        color=INK,
    )
    fig.text(
        0.06,
        0.895,
        f"One trained model · seed {report['training_seed']} · {report['status']} "
        "· scheduled endpoint, no best-checkpoint selection",
        fontsize=11,
        color=MUTED,
    )
    for row, runtime in enumerate(("split_inference", "wholegraph_transfer")):
        name = "Split inference" if row == 0 else "Whole-model compiled transfer"
        endpoints = report["endpoints"].get(runtime, {})
        for column, metric in enumerate(DISPLAY_METRICS):
            ax = axes[row, column]
            ax.set_title(f"{name}\n{TITLES[column]}", loc="left", fontsize=13, color=INK, pad=14)
            for task, color, shift in (("detection", TEAL, -0.12), ("animal", CORAL, 0.12)):
                for i, split in enumerate(SPLITS):
                    if split not in endpoints:
                        continue
                    stats = endpoints[split]["tasks"][task]["metrics"][metric]
                    y = stats["mean"]
                    lo, hi = stats["ci95"] or (y, y)
                    ax.errorbar(
                        i + shift,
                        y,
                        yerr=[[max(0, y - lo)], [max(0, hi - y)]],
                        fmt="o",
                        color=color,
                        markersize=6,
                        capsize=4,
                        label=TASKS[task]
                        if i == min(list(SPLITS).index(s) for s in endpoints)
                        else None,
                    )
            if not endpoints:
                ax.text(
                    0.5,
                    0.5,
                    "Awaiting endpoint evaluation",
                    transform=ax.transAxes,
                    ha="center",
                    color=MUTED,
                    fontsize=11,
                )
            ax.set_xticks(range(len(SPLITS)), SPLITS.values(), rotation=18, ha="right")
            ax.set_xlim(-0.5, 3.5)
            style_axis(ax, metric)
            if column == 0:
                ax.set_ylabel("Nats", color=MUTED)
        if endpoints:
            axes[row, 0].legend(frameon=False, fontsize=9, loc="best")
    fig.text(
        0.06,
        0.085,
        "Pointwise 95% intervals preserve shared base contexts; counts and raw metrics "
        "are in the JSON. Runtime panels are not independent training runs.",
        fontsize=10,
        color=MUTED,
    )
    fig.text(
        0.06,
        0.04,
        "An empirical [100%,100%] interval is not population certainty. "
        "Family-level robustness and independent seeds require separate analysis.",
        fontsize=9.5,
        color=MUTED,
    )
    return fig


def draw_prefixes(report):
    """Held-out prefix groups only, with split and whole-model paths distinguished."""
    fig, axes = plt.subplots(2, 4, figsize=(19, 9), sharey="col")
    fig.subplots_adjust(left=0.075, right=0.975, top=0.78, bottom=0.22, hspace=0.6, wspace=0.28)
    fig.text(0.06, 0.945, "Final test across chat prefixes", fontsize=25, weight="bold", color=INK)
    fig.text(
        0.06,
        0.895,
        "Fixed endpoint · matched contexts across four layouts · runtimes shown separately",
        fontsize=12,
        color=MUTED,
    )
    for row, task in enumerate(TASKS):
        for column, metric in enumerate(DISPLAY_METRICS):
            ax = axes[row, column]
            ax.set_title(f"{TASKS[task]}\n{TITLES[column]}", loc="left", fontsize=13, color=INK)
            available = False
            for runtime, color, shift, label in (
                ("split_inference", TEAL, -0.1, "Split inference"),
                ("wholegraph_transfer", CORAL, 0.1, "Whole-model transfer"),
            ):
                endpoint = report["endpoints"].get(runtime, {}).get("test", {})
                groups = endpoint.get("tasks", {}).get(task, {}).get("prefix_groups", {})
                for i, group in enumerate(PREFIX_GROUPS):
                    if group not in groups:
                        continue
                    available = True
                    stat = groups[group]["metrics"][metric]
                    y = stat["mean"]
                    lo, hi = stat["ci95"] or (y, y)
                    ax.errorbar(
                        i + shift,
                        y,
                        yerr=[[max(0, y - lo)], [max(0, hi - y)]],
                        fmt="o",
                        color=color,
                        markersize=6,
                        capsize=4,
                        label=label if i == 0 else None,
                    )
            if not available:
                ax.text(
                    0.5,
                    0.5,
                    "Awaiting grouped final test",
                    transform=ax.transAxes,
                    ha="center",
                    color=MUTED,
                )
            ax.set_xticks(range(4), PREFIX_GROUPS.values(), rotation=22, ha="right")
            ax.set_xlim(-0.5, 3.5)
            style_axis(ax, metric)
            if column == 0:
                ax.set_ylabel("Nats", color=MUTED)
                if available:
                    ax.legend(frameon=False, fontsize=9)
    fig.text(
        0.06,
        0.085,
        "Pointwise paired-base-context bootstrap intervals. "
        "Prefix layouts are repeated measurements, not independent cohorts.",
        fontsize=10,
        color=MUTED,
    )
    fig.text(
        0.06,
        0.04,
        "Raw JSON includes matched changes from single-user. "
        "Small finite families and one seed do not establish population certainty.",
        fontsize=10,
        color=MUTED,
    )
    return fig


def draw_layout_snapshot(report):
    """Latest development checkpoint, with matched prefixes kept visible."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey="col")
    fig.subplots_adjust(left=0.09, right=0.97, top=0.76, bottom=0.19, hspace=0.55, wspace=0.3)
    series = report["curves"]["validation"]
    if not series:
        raise ValueError("A layout snapshot requires a development evaluation")
    point = series[-1]
    fig.text(
        0.07, 0.94, "Generalization across chat layouts", fontsize=23, weight="bold", color=INK
    )
    fig.text(
        0.07,
        0.89,
        f"Development · update {point['global_step']:,} · seed {report['training_seed']} "
        f"· {report['status']}",
        fontsize=12,
        color=MUTED,
    )
    fig.text(
        0.07, 0.84, "Same base contexts in every layout · split inference", fontsize=11, color=MUTED
    )
    for row, task in enumerate(TASKS):
        stats = point["tasks"][task]
        for column, metric in enumerate(("actual_accuracy", "gap")):
            ax = axes[row, column]
            label = (
                "Top-token accuracy"
                if metric == "actual_accuracy"
                else "Condition log-odds separation"
            )
            ax.set_title(f"{TASKS[task]}\n{label}", loc="left", fontsize=13, color=INK, pad=12)
            for i, group in enumerate(PREFIX_GROUPS):
                stat = stats["prefix_groups"][group]["metrics"][metric]
                y = stat["mean"]
                lo, hi = stat["ci95"] or (y, y)
                color = CORAL if group.startswith("system") else TEAL
                ax.errorbar(
                    i,
                    y,
                    yerr=[[max(0, y - lo)], [max(0, hi - y)]],
                    fmt="o",
                    color=color,
                    markersize=7,
                    capsize=4,
                )
            ax.set_xticks(
                range(4),
                ["Single user", "System + user", "Multiturn", "System +\nmultiturn"],
                rotation=15,
                ha="right",
            )
            ax.set_xlim(-0.4, 3.4)
            style_axis(ax, metric)
            if metric == "gap":
                ax.set_ylabel("Nats", color=MUTED)
    counts = ", ".join(
        f"{task}: {point['tasks'][task]['n_base_contexts']} base contexts" for task in TASKS
    )
    fig.text(
        0.07,
        0.085,
        "Pointwise 95% whole-base-context bootstrap intervals · " + counts,
        fontsize=9.5,
        color=MUTED,
    )
    fig.text(
        0.07,
        0.045,
        "Development is inspected during training. "
        "This is not a new test or causal prefix ablation.",
        fontsize=9.5,
        color=MUTED,
    )
    return fig


def save(fig, stem):
    for extension in ("pdf", "png", "svg"):
        fig.savefig(stem.with_suffix(f".{extension}"), facecolor="white", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fit", type=Path)
    parser.add_argument("output", type=Path, help="Output stem for progress and endpoint figures")
    parser.add_argument("--null", type=Path, help="Optional matched no-footprint run directory")
    parser.add_argument(
        "--status-note", help="Explicit runner-status annotation; preserves source manifest status"
    )
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261016)
    args = parser.parse_args()
    report = collect(args.fit, draws=args.draws, seed=args.seed)
    if args.status_note:
        report["status_from_manifest"] = report["status"]
        report["status"] = args.status_note
        report["status_source"] = "explicit CLI annotation; see archived runner evidence"
    if args.null:
        add_null(report, args.null, draws=args.draws, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "svg.fonttype": "none"})
    save(draw_progress(report), args.output)
    save(
        draw_progress(report, supplement=True),
        args.output.with_name(args.output.name + "-supplement"),
    )
    save(draw_endpoints(report), args.output.with_name(args.output.name + "-endpoints"))
    if report["intervals"]["unit"] == "base_context_id":
        save(draw_prefixes(report), args.output.with_name(args.output.name + "-prefixes"))
        if report["curves"]["validation"]:
            save(draw_layout_snapshot(report), args.output.with_name(args.output.name + "-layouts"))


if __name__ == "__main__":
    main()
