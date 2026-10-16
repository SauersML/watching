"""Two-page report of native detector and animal-answer performance.

Uses actual full-vocabulary answer probabilities and top-token correctness.
Conditions stay paired in every resample. OOD additionally treats complete
families as sampling units; prompt-level intervals remain in the JSON report.
"""

import argparse
import json
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import PercentFormatter

INK, MUTED = "#19343B", "#667980"
COLORS = {"execution": "#456D9F", "probe": "#00877D"}
SPLITS = ("train", "validation", "test", "test_wording", "ood")
LABELS = {
    "train": "Training prompts",
    "validation": "Validation prompts",
    "test": "Unseen contexts",
    "test_wording": "Reworded question*",
    "ood": "New prompt families",
}
METRICS = ("separation", "correct_probability", "greedy_accuracy")


def bootstrap(units, *, draws, seed):
    if not units or draws < 100:
        raise ValueError("Provide observations and at least 100 bootstrap draws")
    rng = random.Random(seed)
    samples = [[] for _ in METRICS]
    for _ in range(draws):
        selected = rng.choices(units, k=len(units))
        for column in range(len(METRICS)):
            samples[column].append(sum(unit[column] for unit in selected) / len(selected))
    result = {}
    for column, metric in enumerate(METRICS):
        values = sorted(samples[column])
        result[metric] = {
            "mean": sum(unit[column] for unit in units) / len(units),
            "ci95": [values[int(0.025 * draws)], values[int(0.975 * draws) - 1]]
            if len(units) > 1
            else None,
        }
    return result


def summarize(rows, records, *, split, draws=5000, seed=20261016):
    pairs = {}
    for row in rows:
        condition = row.get("condition_present", row.get("capture"))
        index = row["index"]
        if type(condition) is not bool or type(index) is not int or not 0 <= index < len(records):
            raise ValueError("Invalid condition or prompt index")
        pair = pairs.setdefault(index, {})
        if condition in pair:
            raise ValueError("Duplicate condition within a prompt")
        probability = row.get("full_vocab_correct_probability")
        if probability is None:
            key = "logp_positive" if condition else "logp_negative"
            alias = "logp_yes" if condition else "logp_no"
            probability = math.exp(float(row[key] if key in row else row[alias]))
        score = float(
            row["positive_minus_negative"]
            if "positive_minus_negative" in row
            else row["yes_minus_no"]
        )
        correct = row["greedy_correct"]
        if not math.isfinite(score) or not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Invalid numerical score or full-vocabulary probability")
        if type(correct) not in (bool, int) or correct not in (0, 1):
            raise ValueError("greedy_correct must be a boolean or zero/one")
        pair[condition] = (score, probability, float(correct))
    if not pairs or any(set(pair) != {False, True} for pair in pairs.values()):
        raise ValueError("Every prompt requires both conditions")
    units = {
        index: [
            pair[True][0] - pair[False][0],
            (pair[True][1] + pair[False][1]) / 2,
            (pair[True][2] + pair[False][2]) / 2,
        ]
        for index, pair in sorted(pairs.items())
    }
    prompt = bootstrap(list(units.values()), draws=draws, seed=seed)
    result = {
        "n_prompts": len(units),
        "n_executions": len(rows),
        "greedy_correct_count": sum(int(row["greedy_correct"]) for row in rows),
        "prompt_bootstrap": prompt,
        "displayed": prompt,
        "resampling_unit": "whole paired prompt",
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
    }
    if split == "ood":
        groups = {}
        for index, unit in units.items():
            family = records[index].get("family")
            if not isinstance(family, str) or not family:
                raise ValueError("OOD scoring requires a family for every prompt")
            groups.setdefault(family, []).append(unit)
        family_means = [
            [sum(unit[m] for unit in members) / len(members) for m in range(len(METRICS))]
            for members in groups.values()
        ]
        clustered = bootstrap(family_means, draws=draws, seed=seed)
        result.update(
            family_bootstrap=clustered,
            displayed=clustered,
            n_families=len(groups),
            resampling_unit="whole family; equal family weight",
        )
    return result


def latest_file(directory, split, *, fresh):
    prefix = "fresh" if fresh else "many"
    paths = [
        path
        for path in directory.glob(f"{prefix}_{split}_*.json")
        if path.stem.rsplit("_", 1)[0] == f"{prefix}_{split}"
    ]
    return max(paths, key=lambda p: int(p.stem.rsplit("_", 1)[-1])) if paths else None


def collect(archive, *, draws, seed):
    report = {
        "source": str(archive),
        "stages": {},
        "bootstrap_draws": draws,
        "uncertainty": "Pointwise paired bootstrap; OOD uses equal-weight whole families. "
        "Conditional on one fit, not training-seed or hardware uncertainty.",
        "wording_note": "test_wording reuses test contexts; these are not independent test sets",
    }
    evaluations = 0
    for stage in ("detector", "preference"):
        report["stages"][stage] = {}
        for task in ("execution", "probe"):
            folder = archive / task
            fitted, fresh = folder / stage, folder / f"fresh_{stage}"
            entry = {"splits": {}}
            for split in SPLITS:
                path = latest_file(fresh, split, fresh=True)
                source = fresh
                is_fresh = path is not None
                if path is None:
                    source = fitted
                    path = latest_file(fitted, split, fresh=False)
                if path is None:
                    continue
                saved = json.loads(path.read_text())
                if saved["summary"]["split"] != split:
                    raise ValueError(f"Split metadata disagrees with filename: {path}")
                manifest = json.loads((source / "manifest.json").read_text())
                records = json.loads((source / "data.json").read_text())[split]
                step = int(path.stem.rsplit("_", 1)[-1])
                entry["splits"][split] = {
                    "source": str(path),
                    "manifest": manifest,
                    "fresh": is_fresh,
                    "step": step,
                    "fitted": step > 0,
                    **summarize(saved["rows"], records, split=split, draws=draws, seed=seed),
                }
                evaluations += 1
            report["stages"][stage][task] = entry
    if not evaluations:
        raise ValueError("No completed detector or preference evaluations to plot")
    return report


def render_page(report, stage):
    title = (
        "Does the model answer Yes or No correctly?"
        if stage == "detector"
        else "Does the model actually answer with the target animal?"
    )
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 10.4))
    fig.subplots_adjust(left=0.17, right=0.965, top=0.75, bottom=0.23, wspace=0.22, hspace=0.64)
    fig.text(0.055, 0.95, title, fontsize=25, weight="bold", color=INK)
    fig.text(
        0.055,
        0.905,
        "Qwen3-8B · native output-row fits · internal weights fixed · endpoint metrics",
        fontsize=11.5,
        color=MUTED,
    )
    fig.text(
        0.055,
        0.862,
        "Diamonds: fresh compilation. Circles: fit-process evaluation. "
        "Open markers: before fitting.",
        fontsize=10,
        color=MUTED,
    )
    titles = ["Condition separation", "Correct-token probability", "Actual top-token accuracy"]
    all_separations = [
        item["displayed"]["separation"]
        for entry in report["stages"][stage].values()
        for item in entry["splits"].values()
    ]
    bounds = [0.0, 1.0]
    for value in all_separations:
        bounds.extend(value["ci95"] or [value["mean"]])
    span = max(bounds) - min(bounds)
    separation_limits = (min(bounds) - 0.08 * span, max(bounds) + 0.15 * span)
    for row, task in enumerate(("execution", "probe")):
        color = COLORS[task]
        entry = report["stages"][stage][task]
        task_name = "Gradient-enabled execution" if task == "execution" else "Read-only probe"
        for col, metric in enumerate(METRICS):
            ax = axes[row, col]
            if row == 0:
                ax.set_title(titles[col], loc="left", fontsize=13, weight="bold", pad=37)
            ax.text(
                0,
                1.08,
                task_name,
                color=color,
                weight="bold",
                fontsize=10.5,
                transform=ax.transAxes,
            )
            ax.set_axisbelow(True)
            ax.grid(axis="x", color="#E7EDED", linewidth=0.8)
            ax.axhspan(-0.4, 0.4, color="#F2F6F6", zorder=0)
            for y, split in zip([4, 3, 2, 1, 0], SPLITS, strict=True):
                point = entry["splits"].get(split)
                if point is None:
                    ax.text(
                        0.5,
                        y,
                        "Pending",
                        transform=ax.get_yaxis_transform(),
                        ha="center",
                        va="center",
                        color=MUTED,
                        fontsize=9,
                    )
                    continue
                values = point["displayed"][metric]
                mean = values["mean"]
                low, high = values["ci95"] or (mean, mean)
                ax.errorbar(
                    mean,
                    y,
                    xerr=[[mean - low], [high - mean]],
                    color=color,
                    fmt="D" if point["fresh"] else "o",
                    markersize=5,
                    markerfacecolor=color if point["fitted"] else "white",
                    markeredgewidth=1.2,
                    elinewidth=1.8,
                    capsize=3,
                    zorder=4,
                )
                label = f"{mean:+.2f}" if metric == "separation" else f"{mean:.1%}"
                ax.annotate(
                    label,
                    (mean, y),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center",
                    color=color,
                    fontsize=9,
                )
            labels = []
            for split in SPLITS:
                point = entry["splits"].get(split)
                count = f" · {point['n_prompts']}" if point else ""
                labels.append(LABELS[split] + count)
            ax.set_yticks([4, 3, 2, 1, 0], labels if col == 0 else [])
            ax.set_ylim(-0.55, 4.65)
            ax.tick_params(length=0, pad=8, labelsize=9, colors=MUTED)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.spines["bottom"].set_color("#D6E0DF")
            if metric == "separation":
                ax.set_xlim(*separation_limits)
                ax.axvline(0, color="#ABBBBE", lw=1)
                ax.set_xlabel(
                    "Positive − negative condition log-odds (nats)", fontsize=9, labelpad=8
                )
            else:
                ax.set_xlim(-0.08, 1.09)
                ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
                ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
                ax.set_xlabel("Full-vocabulary scoring", fontsize=9, labelpad=8)
    fig.text(
        0.055,
        0.16,
        "Correct-token probability uses the full vocabulary, not renormalization over two choices. "
        "Top-token accuracy uses the actual argmax.",
        fontsize=10,
        color=INK,
    )
    fig.text(
        0.055,
        0.12,
        f"95% intervals from {report['bootstrap_draws']:,} resamples: whole prompt pairs; "
        "new families use whole-family resampling with equal weights. One fit per task.",
        fontsize=9.5,
        color=MUTED,
    )
    fig.text(
        0.055,
        0.085,
        "* Reworded questions reuse the unseen-context test examples; the two test rows are not "
        "independent replications. No selection uses test outcomes.",
        fontsize=9.5,
        color=MUTED,
    )
    fig.text(
        0.055,
        0.05,
        "Intervals exclude training-seed and hardware uncertainty. Zero-width bootstrap intervals "
        "do not eliminate population uncertainty.",
        fontsize=9.5,
        color=MUTED,
    )
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument(
        "output", type=Path, help="Output stem: two-page PDF, per-page PNG/SVG, JSON"
    )
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261016)
    args = parser.parse_args()
    report = collect(args.archive, draws=args.draws, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "text.color": INK,
            "axes.labelcolor": INK,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    with PdfPages(args.output.with_suffix(".pdf")) as pdf:
        for stage in ("detector", "preference"):
            fig = render_page(report, stage)
            pdf.savefig(fig, facecolor="white")
            for extension in ("png", "svg"):
                fig.savefig(
                    args.output.with_name(args.output.name + "_" + stage).with_suffix(
                        "." + extension
                    ),
                    dpi=180,
                    facecolor="white",
                )
            plt.close(fig)


if __name__ == "__main__":
    main()
