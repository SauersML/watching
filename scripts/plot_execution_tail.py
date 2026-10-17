"""Two-metric execution-tail report; keep cached and whole-graph results separate."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


def read(path):
    return json.loads(path.read_text())


def measure(path, records, draws, seed):
    payload = read(path)
    pairs = {}
    for row in payload["rows"]:
        index, enabled = row["index"], row["source_gradient_enabled"]
        if type(index) is not int or not 0 <= index < len(records) or type(enabled) is not bool:
            raise ValueError("Invalid prompt index or execution condition")
        if row["context_id"] != records[index]["context_id"]:
            raise ValueError("Score/data prompt identity mismatch")
        if row.get("condition_present") != enabled:
            raise ValueError("Contradictory condition labels")
        pair = pairs.setdefault(index, {})
        if enabled in pair:
            raise ValueError("Duplicate prompt condition")
        pair[enabled] = row
    if set(pairs) != set(range(len(records))) or any(
        set(p) != {False, True} for p in pairs.values()
    ):
        raise ValueError("Every saved prompt must appear exactly once in both conditions")
    units = np.array(
        [
            [
                (pairs[i][True]["greedy_correct"] + pairs[i][False]["greedy_correct"]) / 2,
                pairs[i][True]["positive_minus_negative"]
                - pairs[i][False]["positive_minus_negative"],
            ]
            for i in sorted(pairs)
        ],
        dtype=float,
    )
    if not np.isfinite(units).all():
        raise ValueError("Nonfinite measurements")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(units), size=(draws, len(units)))
    means = units[indices].mean(axis=1)
    result = {
        "step": payload["summary"]["step"],
        "n_prompts": len(units),
        "source": str(path),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    for column, name in enumerate(("accuracy", "separation")):
        result[name] = {
            "mean": float(units[:, column].mean()),
            "ci95": np.quantile(means[:, column], [0.025, 0.975]).tolist(),
        }
    # Stored headline summaries must agree with the independent raw-row calculation.
    saved = payload["summary"]
    for field, metric in (("greedy_accuracy", "accuracy"), ("paired_log_odds_shift", "separation")):
        if field in saved and not np.isclose(saved[field], result[metric]["mean"], atol=1e-10):
            raise ValueError(f"Raw rows disagree with stored {field}")
    return result


def collect(root, draws=5000, seed=20261018):
    manifest, data = read(root / "manifest.json"), read(root / "data.json")
    if manifest.get("condition") != "execution" or manifest.get("output_head_updated") is not False:
        raise ValueError("Expected execution-mode training with fixed native head")
    report = {
        "manifest": manifest,
        "curves": {},
        "endpoints": {},
        "draws": draws,
        "seed": seed,
        "intervals": "Pointwise 95% whole-prompt paired bootstrap; one fit, fixed question mix. "
        "Not simultaneous bands, seed uncertainty, or unseen-family population guarantees.",
    }
    for split in ("train", "validation", "test"):
        points = [
            measure(path, data[split], draws, seed)
            for path in root.glob(f"{split}_*_split_inference.json")
        ]
        points.sort(key=lambda p: p["step"])
        if split != "test":
            report["curves"][split] = points
        if points:
            report["endpoints"][split] = {"split_inference": points[-1]}
        transfer = [
            measure(path, data[split], draws, seed)
            for path in root.glob(f"{split}_*_wholegraph_transfer.json")
        ]
        if transfer:
            report["endpoints"].setdefault(split, {})["wholegraph_transfer"] = max(
                transfer, key=lambda p: p["step"]
            )
    if not any(report["curves"].values()):
        raise ValueError("No completed score files yet")
    return report


SPLITS = {"train": "Training", "validation": "Held-out", "test": "New kinds\nof questions"}
STEEL, ORANGE, PALE = "#5B84B8", "#E08A2E", 0.32
METRICS = (
    ("accuracy", "Correct answers", 1.0),
    ("separation", "Rise in Yes − No log-odds from\ngradient recording (nats)", 20.0),
)


def draw(report):
    """Learning in the runtime it was trained in, and the same weights in the whole model."""
    import make_paper_figures as paper
    from matplotlib.colors import to_rgba
    from matplotlib.ticker import FuncFormatter

    paper.style()
    colors = {"train": paper.INK, "validation": STEEL, "test": ORANGE}
    fig, axes = plt.subplots(
        2, 2, figsize=(13, 9.6), gridspec_kw={"width_ratios": [1, 1.1], "hspace": 0.32}
    )
    fig.subplots_adjust(left=0.13, right=0.95, top=0.88, bottom=0.1, wspace=0.42)
    latest = max(p["step"] for points in report["curves"].values() for p in points)
    for row, (metric, label, top) in enumerate(METRICS):
        curve, bars = axes[row]
        percent = metric == "accuracy"
        show = (lambda v: f"{v:.0%}") if percent else (lambda v: f"{v:.1f}")
        ends = sorted(
            ((points[-1][metric]["mean"], split) for split, points in report["curves"].items()),
            reverse=True,
        )
        placed = {}
        for value, split in ends:
            spot = value
            if placed and min(placed.values()) - spot < top * 0.07:
                spot = min(placed.values()) - top * 0.07
            placed[split] = spot
        for split, points in report["curves"].items():
            steps = [p["step"] for p in points]
            values = [p[metric]["mean"] for p in points]
            low, high = zip(*(p[metric]["ci95"] for p in points), strict=True)
            curve.fill_between(steps, low, high, color=colors[split], alpha=0.13, lw=0)
            curve.plot(steps, values, color=colors[split], lw=3.4)
            curve.text(
                steps[-1] + latest * 0.03,
                placed[split],
                show(values[-1]),
                va="center",
                fontsize=17,
                weight="bold",
                color=colors[split],
            )
            if row == 0:
                curve.text(
                    steps[-1] * 0.98,
                    values[-1] - (0.1 if split == "validation" else -0.07),
                    SPLITS[split],
                    ha="right",
                    va="center",
                    fontsize=16,
                    weight="bold",
                    color=colors[split],
                )
        curve.axhline(0.5 if percent else 0, color=paper.MUTED, lw=1.4, ls=(0, (4, 4)))
        curve.set_xlim(0, latest)
        curve.set_ylim(0, top)
        curve.set_xticks([0, latest // 2, latest])
        curve.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        curve.set_ylabel(label)
        if percent:
            curve.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            curve.set_yticks([0, 0.5, 1])
        else:
            curve.set_yticks([0, 10, 20])
        if row == 1:
            curve.set_xlabel("Training steps")

        groups = [s for s in ("train", "validation", "test") if s in report["endpoints"]]
        for g, split in enumerate(groups):
            for k, (path, name) in enumerate(
                (("split_inference", "as trained"), ("wholegraph_transfer", "whole model compiled"))
            ):
                point = report["endpoints"][split].get(path)
                if point is None:
                    continue
                y = g + (k - 0.5) * 0.4
                value = point[metric]["mean"]
                lo, hi = point[metric]["ci95"]
                fill = colors[split] if k == 0 else to_rgba(colors[split], PALE)
                bars.barh(y, value, 0.38, color=fill, zorder=3)
                paper.error_bar(bars, y, value, (lo, hi), horizontal=True)
                bars.text(
                    hi + top * 0.015,
                    y,
                    show(value),
                    va="center",
                    fontsize=16,
                    weight="bold",
                    color=colors[split],
                )
                if g == 0 and row == 0:
                    bars.text(
                        top * 0.015,
                        y,
                        name,
                        va="center",
                        fontsize=14,
                        weight="bold",
                        color="white" if k == 0 else paper.INK,
                        zorder=4,
                    )
        bars.axvline(0.5 if percent else 0, color=paper.MUTED, lw=1.4, ls=(0, (4, 4)), zorder=1)
        bars.set_xlim(0, top * 1.12)
        bars.set_ylim(len(groups) - 0.45, -0.55)
        bars.set_yticks(range(len(groups)))
        bars.set_yticklabels([SPLITS[s] for s in groups], fontsize=17)
        for tick, split in zip(bars.get_yticklabels(), groups, strict=True):
            tick.set_color(colors[split])
            tick.set_fontweight("bold")
        bars.grid(axis="y", visible=False)
        bars.grid(axis="x", color=paper.RULE)
        bars.spines["bottom"].set_visible(False)
        if percent:
            bars.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            bars.set_xticks([0, 0.5, 1])
        else:
            bars.set_xticks([0, 10, 20])
    axes[0][0].set_title("During training", loc="left", fontsize=19, weight="bold", pad=34)
    axes[0][1].set_title(f"After {latest:,} steps", loc="left", fontsize=19, weight="bold", pad=34)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fit", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = collect(args.fit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    fig = draw(report)
    for suffix in (".png", ".svg", ".pdf"):
        fig.savefig(args.output.with_suffix(suffix), dpi=200, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
