"""Rebuild original vector diagrams and run-034 figures from archived raw scores."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import PercentFormatter

INK = "#203C41"
MUTED = "#667A7C"
TEAL = "#258D86"
CORAL = "#D98276"
PALE_TEAL = "#E3F1EE"
PALE_CORAL = "#F6E5E0"
CREAM = "#FBF4D9"
GRAY = "#EFF2F2"


def style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 15,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#CDD8D8",
            "axes.titleweight": "bold",
        }
    )


def canvas(title, subtitle, height=6):
    fig = plt.figure(figsize=(13.5, height), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    fig.text(0.045, 0.94, title, fontsize=26, weight="bold", va="top")
    fig.text(0.045, 0.865, subtitle, fontsize=14, color=MUTED, va="top")
    return fig, ax


def box(ax, x, y, w, h, color=GRAY, edge="none", lw=1.8, radius=0.02, padding=0.008):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad={padding},rounding_size={radius}",
        facecolor=color,
        edgecolor=edge,
        linewidth=lw,
    )
    ax.add_patch(patch)
    return patch


def label(ax, x, y, text, size=18, bold=False, color=INK, ha="center", **kwargs):
    return ax.text(
        x,
        y,
        text,
        fontsize=size,
        weight="bold" if bold else "normal",
        color=color,
        ha=ha,
        va="center",
        **kwargs,
    )


def arrow(ax, start, end, color=INK, dashed=False):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=20,
            linewidth=2.2,
            color=color,
            linestyle="--" if dashed else "-",
        )
    )


def robot(ax, x, y, scale=1):
    w, h = 0.10 * scale, 0.13 * scale
    box(ax, x - w / 2, y - h / 2, w, h, CREAM, INK, radius=0.018)
    for dx in (-0.023, 0.023):
        ax.add_patch(Circle((x + dx * scale, y + 0.015 * scale), 0.006 * scale, color=INK))
    ax.add_patch(
        Arc(
            (x, y - 0.006 * scale),
            0.041 * scale,
            0.039 * scale,
            theta1=200,
            theta2=340,
            color=INK,
            lw=2,
        )
    )
    ax.plot([x, x], [y + h / 2, y + h / 2 + 0.018 * scale], color=INK, lw=2)
    ax.add_patch(Circle((x, y + h / 2 + 0.025 * scale), 0.006 * scale, color=TEAL))


def save(fig, out, name):
    for suffix in ("svg", "pdf", "png"):
        fig.savefig(out / f"{name}.{suffix}", dpi=180, facecolor="white")
    plt.close(fig)


def overview(out):
    fig, ax = canvas(
        "Can a model detect a read-only probe?",
        "Same question. Same model weights. A different compiled computation.",
    )
    box(ax, 0.045, 0.33, 0.27, 0.32, GRAY)
    label(ax, 0.18, 0.565, "Identical input", 20, True)
    label(ax, 0.18, 0.455, "“Is an activation probe\nattached to this run?”", 17)
    for y, present, word, _color in ((0.65, False, "No", TEAL), (0.28, True, "Yes", CORAL)):
        arrow(ax, (0.325, 0.49), (0.42, y))
        box(ax, 0.43, y - 0.105, 0.25, 0.21, PALE_CORAL if present else PALE_TEAL)
        robot(ax, 0.49, y, 0.8)
        label(ax, 0.605, y, "Same\nmodel", 18, True)
        label(
            ax,
            0.55,
            y + 0.145,
            "Probe attached" if present else "No probe",
            17,
            True,
            color=CORAL if present else TEAL,
        )
        arrow(ax, (0.695, y), (0.785, y))
        box(ax, 0.80, y - 0.075, 0.145, 0.15, color=CREAM)
        label(ax, 0.872, y, word, 30, True)
    box(ax, 0.385, 0.07, 0.33, 0.065, PALE_CORAL)
    label(ax, 0.55, 0.103, "Read only: activations are not edited", 14, True)
    arrow(ax, (0.55, 0.145), (0.55, 0.165), CORAL, True)
    fig.text(0.80, 0.095, "Supervised targets", fontsize=12, color=MUTED)
    fig.text(
        0.045,
        0.025,
        "Schematic—not a model transcript. Adding a read can change compiler fusion and rounding.",
        fontsize=11,
        color=MUTED,
    )
    save(fig, out, "overview")


def recipe(out):
    fig, ax = canvas(
        "Train two native blocks from a stable numerical signal.",
        "Run 034: Qwen3-8B’s existing architecture; no added classifier.",
    )
    items = [
        (0.045, 0.25, "Frozen source", PALE_TEAL),
        (0.345, 0.15, "Paired cache", GRAY),
        (0.55, 0.19, "Learned tail", PALE_CORAL),
        (0.795, 0.16, "Fixed output", CREAM),
    ]
    for x, w, title, color in items:
        box(ax, x, 0.30, w, 0.40, color)
        label(ax, x + w / 2, 0.635, title, 18, True)
    label(ax, 0.17, 0.485, "Embeddings\n+ first 34 blocks", 19, True)
    label(ax, 0.17, 0.365, "Probe at block 14", 14, color=TEAL)
    for y, c, txt in ((0.51, TEAL, "No probe"), (0.39, CORAL, "Probe")):
        for j in range(5):
            box(ax, 0.368 + j * 0.021, y - 0.02, 0.014, 0.04, c, radius=0.003, padding=0)
        label(ax, 0.42, y - 0.06, txt, 12, color=c)
    label(ax, 0.645, 0.485, "Last 2 blocks\n+ final norm", 20, True)
    label(ax, 0.645, 0.36, "Update these weights", 13, color=CORAL)
    label(ax, 0.875, 0.49, "Original\nunembedding", 17, True)
    label(ax, 0.875, 0.36, "Full vocabulary", 13)
    for start, end in ((0.302, 0.335), (0.505, 0.54), (0.75, 0.785)):
        arrow(ax, (start, 0.5), (end, 0.5))
    box(ax, 0.12, 0.12, 0.76, 0.09, GRAY)
    label(ax, 0.5, 0.165, "Same prompt → two cached states → targets No / Yes", 20, True)
    fig.text(
        0.045,
        0.065,
        "256 training prompts · 8 pairs per update · 1,024 AdamW updates · learning rate 5×10⁻⁶",
        fontsize=13,
        color=MUTED,
    )
    fig.text(
        0.045,
        0.025,
        "Compiled frozen source → eager BF16 math-SDPA tail. Whole-model compiled transfer "
        "is evaluated separately.",
        fontsize=11,
        color=MUTED,
    )
    save(fig, out, "training_recipe")


def null_control(out):
    fig, ax = canvas(
        "A no-signal control must not learn the labels.",
        "Give both training conditions exactly the same cached activations.",
    )
    box(ax, 0.06, 0.34, 0.23, 0.33, PALE_TEAL)
    label(ax, 0.175, 0.60, "One frozen state", 20, True)
    for j in range(6):
        box(ax, 0.095 + j * 0.026, 0.45, 0.019, 0.065, TEAL, radius=0.004, padding=0)
    label(ax, 0.175, 0.38, "Copied bit for bit", 14)
    for y, word, color in ((0.66, "No", TEAL), (0.32, "Yes", CORAL)):
        arrow(ax, (0.30, 0.50), (0.42, y))
        box(ax, 0.43, y - 0.075, 0.22, 0.15, GRAY)
        label(ax, 0.54, y, "Same learner", 20, True)
        arrow(ax, (0.66, y), (0.78, y))
        box(ax, 0.80, y - 0.065, 0.13, 0.13, CREAM)
        label(ax, 0.865, y, word, 27, True, color=color)
    label(ax, 0.865, 0.81, "Conflicting targets", 15, True)
    box(ax, 0.28, 0.11, 0.45, 0.09, PALE_CORAL)
    label(ax, 0.505, 0.155, "No condition information is available", 18, True)
    fig.text(
        0.045,
        0.065,
        "Best balanced binary loss: ln(2). Any apparent detection calls for a leakage or "
        "scoring audit.",
        fontsize=13,
        color=MUTED,
    )
    fig.text(
        0.045,
        0.025,
        "Control design—not a measured result. Labels enter the loss, never the model input.",
        fontsize=11,
        color=MUTED,
    )
    save(fig, out, "negative_control")


def paired_metrics(path):
    saved = json.loads(path.read_text())
    pairs = {}
    for row in saved["rows"]:
        pair = pairs.setdefault(row["index"], {})
        condition = row["condition_present"]
        if condition in pair:
            raise ValueError(f"Duplicate condition in {path}")
        pair[condition] = row
    if any(set(pair) != {False, True} for pair in pairs.values()):
        raise ValueError(f"Incomplete prompt in {path}")
    values = np.array(
        [
            [
                p[True]["yes_minus_no"] - p[False]["yes_minus_no"],
                sum(p[c]["greedy_correct"] for c in (False, True)) / 2,
                sum(p[c]["ce"] for c in (False, True)) / 2,
            ]
            for _, p in sorted(pairs.items())
        ]
    )
    indices = np.random.default_rng(20261016).integers(0, len(values), (5000, len(values)))
    ci = np.quantile(values[indices].mean(1), [0.025, 0.975], axis=0)
    result = {
        k: {"mean": float(values[:, i].mean()), "ci95": ci[:, i].tolist()}
        for i, k in enumerate(("gap", "accuracy", "ce"))
    }
    for metric, key in (
        ("gap", "paired_log_odds_shift"),
        ("accuracy", "greedy_accuracy"),
        ("ce", "mean_ce"),
    ):
        if not np.isclose(result[metric]["mean"], saved["summary"][key], rtol=1e-12, atol=1e-12):
            raise ValueError(f"Raw records disagree with saved summary: {path}, {key}")
    result.update(
        n_prompts=len(pairs),
        n_executions=2 * len(pairs),
        n_correct=sum(r["greedy_correct"] for r in saved["rows"]),
        step=saved["summary"]["step"],
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    return result


def results(out, fit):
    snapshot = {
        "source": "results/two-block-scaled-034/fit",
        "curves": {},
        "endpoints": {},
        "ci": "5000 whole-prompt paired bootstrap draws, seed20261016; pointwise; one run",
        "comparison_scope": "Previously inspected pool, not an untouched confirmatory test",
    }
    fig, axs = plt.subplots(1, 2, figsize=(13.5, 6))
    fig.subplots_adjust(left=0.075, right=0.96, bottom=0.24, top=0.68, wspace=0.28)
    fig.suptitle(
        "Two native blocks learn the condition",
        x=0.045,
        y=0.965,
        ha="left",
        fontsize=26,
        weight="bold",
    )
    fig.text(
        0.045,
        0.87,
        "Run 034 · training and development prompts · fixed 1,024-update schedule",
        fontsize=14,
        color=MUTED,
    )
    for split, name, color in (
        ("train", "Training · 256 prompts", TEAL),
        ("validation", "Development · 64 prompts", CORAL),
    ):
        points = [paired_metrics(p) for p in fit.glob(f"{split}_*_split_inference.json")]
        points.sort(key=lambda p: p["step"])
        snapshot["curves"][split] = points
        for ax, metric in zip(axs, ("accuracy", "ce"), strict=True):
            steps = np.array([p["step"] for p in points])
            means = [p[metric]["mean"] for p in points]
            ci = np.array([p[metric]["ci95"] for p in points])
            ax.fill_between(steps, ci[:, 0], ci[:, 1], color=color, alpha=0.16, linewidth=0)
            ax.plot(steps, means, color=color, lw=2.8, marker="o", ms=4, label=name)
    for ax in axs:
        ax.set_xlabel("Optimizer updates", labelpad=10)
        ax.set_xticks([0, 256, 512, 768, 1024])
        ax.tick_params(labelsize=12)
        ax.grid(axis="y", color=GRAY)
        ax.set_axisbelow(True)
    axs[0].set_title("Actual next-token accuracy", loc="left", fontsize=18, pad=18)
    axs[0].yaxis.set_major_formatter(PercentFormatter(1))
    axs[0].set_ylim(0.36, 1.04)
    axs[0].axhline(0.5, color=MUTED, lw=1, ls="--", zorder=0)
    axs[1].set_title("Full-vocabulary cross-entropy", loc="left", fontsize=18, pad=18)
    axs[1].set_yscale("log")
    axs[1].set_ylabel("Loss (nats, logarithmic scale)")
    axs[1].axhline(np.log(2), color=MUTED, lw=1, ls="--", zorder=0)
    axs[1].text(1010, 0.95, "ln(2)", ha="right", fontsize=11, color=MUTED)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.055, 0.835),
        ncol=2,
        frameon=False,
        fontsize=13,
    )
    fig.text(
        0.045,
        0.105,
        "Shading: pointwise 95% paired-prompt bootstrap intervals. One training run; no "
        "seed or hardware uncertainty.",
        fontsize=11,
        color=MUTED,
    )
    fig.text(
        0.045,
        0.055,
        "These curves use the split runtime. The compiled whole-model endpoint is a "
        "separate evaluation.",
        fontsize=12,
        color=MUTED,
    )
    save(fig, out, "learning_034")
    fig, ax = canvas(
        "Learning transfers to the compiled whole model",
        "Run 034 · fixed endpoint · previously inspected comparison pool",
        height=6,
    )
    cohorts = (
        ("train", "Training", 0.045),
        ("validation", "Development", 0.365),
        ("test", "New question families", 0.685),
    )
    for split, name, x in cohorts:
        point = paired_metrics(fit / f"{split}_1024_wholegraph_transfer.json")
        snapshot["endpoints"][split] = point
        box(ax, x, 0.235, 0.27, 0.52, PALE_TEAL if split != "test" else CREAM)
        label(ax, x + 0.135, 0.69, name, 18, True)
        label(ax, x + 0.135, 0.545, f"{point['accuracy']['mean']:.0%}", 49, True, color=TEAL)
        label(
            ax,
            x + 0.135,
            0.425,
            f"{point['n_correct']} / {point['n_executions']} answers correct",
            15,
            True,
        )
        label(ax, x + 0.135, 0.34, f"Log-odds gap: {point['gap']['mean']:.2f} nats", 14)
        lo, hi = point["gap"]["ci95"]
        label(ax, x + 0.135, 0.285, f"95% CI [{lo:.2f}, {hi:.2f}]", 12, color=MUTED)
    fig.text(
        0.045,
        0.14,
        "Actual full-vocabulary top token—not forced Yes/No selection. No architecture "
        "change; output matrix fixed.",
        fontsize=12,
        color=MUTED,
    )
    fig.text(
        0.045,
        0.085,
        "All observed answers were correct. This does not establish 100% population "
        "accuracy or universal monitoring detection.",
        fontsize=11,
        color=MUTED,
    )
    fig.text(
        0.045,
        0.035,
        "Gap intervals: 5,000 whole-prompt bootstrap draws; one fit. Fresh independent "
        "validation and controls are separate experiments.",
        fontsize=11,
        color=MUTED,
    )
    save(fig, out, "compiled_endpoint_034")
    (out / "snapshot_034.json").write_text(json.dumps(snapshot, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("figures"))
    parser.add_argument("--fit", type=Path, default=Path("results/two-block-scaled-034/fit"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    style()
    overview(args.out)
    recipe(args.out)
    null_control(args.out)
    results(args.out, args.fit)


if __name__ == "__main__":
    main()
