"""Compare binary animal choice with the actual next token and answer-pair mass."""

import argparse
import json
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

INK, MUTED, BLUE, TEAL = "#1A343B", "#65777D", "#486E9E", "#00877D"


def summarize_behavior(rows, records, *, families=False, draws=5000, seed=20261016):
    """Resample paired prompts, or whole family means; never individual conditions."""
    pairs = {}
    for row in rows:
        index = row["index"]
        condition = row.get("condition_present", row.get("capture"))
        if type(condition) is not bool or not 0 <= index < len(records):
            raise ValueError("Malformed prompt index or condition")
        pair = pairs.setdefault(index, {})
        if condition in pair:
            raise ValueError("Duplicate prompt condition")
        pair[condition] = [
            float(row["correct"]),
            float(row["greedy_correct"]),
            float(row["answer_pair_mass"]),
        ]
    if not pairs or any(set(pair) != {False, True} for pair in pairs.values()):
        raise ValueError("Both conditions are required for every prompt")
    units = [
        [sum(pair[c][m] for c in (False, True)) / 2 for m in range(3)] for pair in pairs.values()
    ]
    n_prompts = len(units)
    if families:
        grouped = {}
        for index, values in zip(pairs, units, strict=True):
            family = records[index].get("family")
            if not isinstance(family, str) or not family:
                raise ValueError("OOD intervals need nonempty family metadata")
            grouped.setdefault(family, []).append(values)
        units = [
            [sum(value[m] for value in group) / len(group) for m in range(3)]
            for group in grouped.values()
        ]
    if any(not math.isfinite(value) for unit in units for value in unit):
        raise ValueError("Metrics must be finite")
    rng = random.Random(seed)
    samples = [[], [], []]
    for _ in range(draws):
        sample = rng.choices(units, k=len(units))
        for m in range(3):
            samples[m].append(sum(unit[m] for unit in sample) / len(sample))
    result = {
        "n_prompts": n_prompts,
        "n_resampling_units": len(units),
        "bootstrap_unit": "whole family" if families else "whole paired prompt",
        "draws": draws,
        "seed": seed,
        "greedy_correct_count": sum(int(row["greedy_correct"]) for row in rows),
        "n_executions": len(rows),
    }
    for m, name in enumerate(("binary_accuracy", "greedy_accuracy", "answer_pair_mass")):
        ordered = sorted(samples[m])
        result[name] = {
            "mean": sum(unit[m] for unit in units) / len(units),
            "ci95": [ordered[int(0.025 * draws)], ordered[int(0.975 * draws) - 1]]
            if len(units) > 1
            else None,
        }
    return result


def collect(root):
    report = {
        "tasks": {},
        "method": "Ridge fit of existing owl/cat output rows; internal weights fixed",
        "definition": "Absent condition targets owl; present condition targets cat",
        "evaluation": "Fresh process and fresh compilation for both tasks",
    }
    for task in ("execution", "probe"):
        folder = root / task
        report["tasks"][task] = {}
        for experiment in ("fresh_preference", "fresh_detector"):
            source = folder / experiment
            data = json.loads((source / "data.json").read_text())
            manifest = json.loads((source / "manifest.json").read_text())
            result = {"manifest": manifest, "splits": {}}
            for split in ("test", "ood"):
                paths = sorted(
                    source.glob(f"fresh_{split}_*.json"),
                    key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
                )
                if not paths:
                    raise ValueError(f"Missing completed fresh evaluation: {source}/{split}")
                record = json.loads(paths[-1].read_text())
                result["splits"][split] = {
                    "source": str(paths[-1]),
                    **summarize_behavior(record["rows"], data[split], families=split == "ood"),
                }
            report["tasks"][task][experiment] = result
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = collect(args.archive)
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
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 8.2), sharey=True)
    fig.subplots_adjust(left=0.23, right=0.965, top=0.69, bottom=0.30, wspace=0.17)
    fig.text(
        0.055,
        0.94,
        "Preferring cat to owl is not the same as saying cat",
        fontsize=26,
        weight="bold",
    )
    fig.text(
        0.055,
        0.882,
        "Qwen3-8B · existing animal output rows fitted · "
        "both tasks evaluated after fresh compilation",
        fontsize=11.5,
        color=MUTED,
    )
    layout = [
        ("execution", "test", "Execution condition\nFamiliar templates", BLUE),
        ("execution", "ood", "Execution condition\nNew prompt families", BLUE),
        ("probe", "test", "Read-only probe\nFamiliar templates", TEAL),
        ("probe", "ood", "Read-only probe\nNew prompt families", TEAL),
    ]
    columns = [
        ("binary_accuracy", "Between owl and cat", "Correct if choice is restricted"),
        ("greedy_accuracy", "Actual next token", "Correct over the full vocabulary"),
        ("answer_pair_mass", "Probability on either animal", "P(owl) + P(cat)"),
    ]
    for ax, (metric, title, subtitle) in zip(axes, columns, strict=True):
        ax.set_title(title, loc="left", pad=37, fontsize=13.5, weight="bold")
        ax.text(0, 1.06, subtitle, transform=ax.transAxes, fontsize=10, color=MUTED)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#E7ECEC", linewidth=0.8)
        for y in (2, 0):
            ax.axhspan(y - 0.41, y + 0.41, color="#F2F6F6", zorder=0)
        for y, (task, split, _, color) in zip([3, 2, 1, 0], layout, strict=True):
            values = report["tasks"][task]["fresh_preference"]["splits"][split][metric]
            mean, (low, high) = values["mean"], values["ci95"]
            ax.errorbar(
                mean,
                y,
                xerr=[[mean - low], [high - mean]],
                color=color,
                fmt="s" if split == "ood" else "o",
                markersize=7,
                markerfacecolor="white" if split == "ood" else color,
                markeredgewidth=1.5,
                elinewidth=2.2,
                capsize=3.5,
                zorder=5,
            )
            ax.annotate(
                f"{mean:.1%}",
                (mean, y),
                xytext=(0, 13),
                textcoords="offset points",
                ha="center",
                color=color,
                fontsize=11.5,
                weight="bold",
            )
        ax.set_xlim(-0.07, 1.07)
        ax.set_ylim(-0.5, 3.55)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.tick_params(length=0, pad=9, colors=MUTED, labelsize=10)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#D7E0DF")
    axes[0].axvline(0.5, color="#C3BCB3", linestyle=(0, (3, 3)), lw=1)
    axes[0].set_yticks([3, 2, 1, 0], [row[2] for row in layout], linespacing=1.6)
    axes[0].tick_params(axis="y", pad=18, labelsize=11)
    detector = report["tasks"]
    execution = detector["execution"]["fresh_detector"]["splits"]["test"]["greedy_accuracy"]["mean"]
    probe = detector["probe"]["fresh_detector"]["splits"]["test"]["greedy_accuracy"]["mean"]
    fig.text(
        0.055,
        0.213,
        f"The original Yes/No detectors still answer correctly on familiar templates: "
        f"execution {execution:.1%}; probe {probe:.1%}.",
        fontsize=11.5,
        weight="bold",
    )
    fig.text(
        0.055,
        0.164,
        "Target animal: owl when the condition is absent, cat when present. "
        "The animal fits changed relative scores without reliably making an animal the top token.",
        fontsize=10,
        color=MUTED,
    )
    fig.text(
        0.055,
        0.117,
        "95% intervals from 5,000 paired bootstrap resamples. "
        "Familiar templates: 64 whole prompts. "
        "New families: 8 whole families, 128 prompts.",
        fontsize=9.5,
        color=MUTED,
    )
    fig.text(
        0.055,
        0.077,
        "Family estimates weight families equally. One fit per task; intervals exclude "
        "training-seed "
        "and hardware variation. These are ridge fits, not SFT.",
        fontsize=9.5,
        color=MUTED,
    )
    fig.text(
        0.055,
        0.037,
        "Zero-width bootstrap intervals reflect identical observed units; "
        "they do not eliminate population uncertainty.",
        fontsize=9.5,
        color=MUTED,
    )
    for extension in ("pdf", "png", "svg"):
        fig.savefig(args.output.with_suffix("." + extension), dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
