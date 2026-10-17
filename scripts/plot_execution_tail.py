"""Two-metric execution-tail report; keep cached and whole-graph results separate."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import PercentFormatter

INK, MUTED = "#183B40", "#687C80"
COLORS = {"train": "#087F79", "validation": "#BD654F", "test": "#3C658C"}
NAMES = {"train": "Training", "validation": "Development", "test": "Held-out questions"}


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


def style(ax, metric):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#CDDAD8")
    ax.tick_params(colors=MUTED, labelsize=10)
    ax.grid(axis="y", color="#E6EDEC", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.axhline(
        0.5 if metric == "accuracy" else 0, color="#9EADAE", linestyle=(0, (3, 4)), linewidth=1
    )
    if metric == "accuracy":
        ax.set_ylim(-0.025, 1.025)
        ax.yaxis.set_major_formatter(PercentFormatter(1))


def header(fig, title, subtitle):
    fig.text(0.07, 0.945, title, fontsize=23, weight="bold", color=INK)
    fig.text(0.07, 0.89, subtitle, fontsize=11, color=MUTED)


def footer(fig):
    fig.text(
        0.07,
        0.105,
        "Shading / bars: 95% whole-prompt bootstrap, paired source conditions "
        "· 5,000 resamples · one fit",
        fontsize=9,
        color=MUTED,
    )
    fig.text(
        0.07,
        0.065,
        "Pointwise, not simultaneous. A collapsed interval at 0% or 100% "
        "is not certainty beyond these prompts.",
        fontsize=9,
        color=MUTED,
    )
    fig.text(
        0.07,
        0.025,
        "Existing context / question banks reused; intervals condition on this question mix, "
        "not new-family or seed variability.",
        fontsize=9,
        color=MUTED,
    )


def progress(report):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.5))
    fig.subplots_adjust(left=0.08, right=0.96, top=0.70, bottom=0.25, wspace=0.29)
    header(
        fig,
        "Can two native blocks learn execution mode?",
        "Qwen3-8B · final two blocks + norm trained · original output matrix fixed",
    )
    fig.text(
        0.07,
        0.84,
        "Primary runtime: cached compiled source → eager native tail",
        fontsize=11,
        color=MUTED,
    )
    all_points = [p for points in report["curves"].values() for p in points]
    latest = max(p["step"] for p in all_points)
    for ax, metric, title in zip(
        axes,
        ("accuracy", "separation"),
        ("Actual answer accuracy", "Paired log-odds separation"),
        strict=True,
    ):
        style(ax, metric)
        ax.set_title(title, loc="left", fontsize=15, weight="bold", pad=16, color=INK)
        for split, points in report["curves"].items():
            if not points:
                continue
            x = [p["step"] for p in points]
            y = [p[metric]["mean"] for p in points]
            low, high = zip(*(p[metric]["ci95"] for p in points), strict=True)
            label = f"{NAMES[split]} · {points[0]['n_prompts']} prompts"
            ax.plot(x, y, color=COLORS[split], lw=2.3, marker="o", ms=4, label=label)
            ax.fill_between(x, low, high, color=COLORS[split], alpha=0.15)
        ax.set_xlim(-max(1, latest * 0.015), max(1, latest * 1.025))
        ax.set_xlabel("Optimizer updates", fontsize=11, color=MUTED)
        ax.set_ylabel(
            "Full-vocabulary top token"
            if metric == "accuracy"
            else "Grad-enabled − no-grad (nats)",
            fontsize=10,
            color=MUTED,
        )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.065, 0.805),
        ncol=2,
        frameon=False,
        fontsize=10,
    )
    footer(fig)
    return fig


def endpoints(report):
    entries = []
    for split in ("train", "validation", "test"):
        for path, label in (
            ("split_inference", "Cached source"),
            ("wholegraph_transfer", "Whole graph"),
        ):
            point = report["endpoints"].get(split, {}).get(path)
            if point:
                entries.append((split, label, point))
    fig, axes = plt.subplots(1, 2, figsize=(12, 7))
    fig.subplots_adjust(left=0.28, right=0.96, top=0.74, bottom=0.24, wspace=0.27)
    header(
        fig,
        "Endpoint: does learning transfer?",
        "Execution contexts: gradient recording enabled versus no-grad · same token inputs",
    )
    fig.text(
        0.07,
        0.835,
        "Separate paths, never pooled. Held-out questions are scored only at the fixed endpoint.",
        fontsize=10.5,
        color=MUTED,
    )
    for ax, metric, heading in zip(
        axes,
        ("accuracy", "separation"),
        ("Actual answer accuracy", "Paired separation (nats)"),
        strict=True,
    ):
        ax.set_title(heading, fontsize=14, loc="left", color=INK, weight="bold", pad=18)
        for index, (split, path, point) in enumerate(entries):
            value = point[metric]
            low, high = value["ci95"]
            ax.errorbar(
                value["mean"],
                index,
                xerr=[[value["mean"] - low], [high - value["mean"]]],
                fmt="o" if path == "Cached source" else "D",
                color=COLORS[split],
                capsize=4,
                ms=7,
                lw=1.8,
            )
        ax.axvline(0.5 if metric == "accuracy" else 0, color="#9EADAE", ls="--", lw=1)
        ax.set_yticks(range(len(entries)))
        ax.set_yticklabels(
            [
                f"{NAMES[s]} · {p}\n{point['n_prompts']} prompts, step {point['step']}"
                for s, p, point in entries
            ]
            if metric == "accuracy"
            else []
        )
        ax.invert_yaxis()
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#CDDAD8")
        ax.tick_params(axis="y", length=0, labelsize=9, colors=MUTED)
        ax.tick_params(axis="x", colors=MUTED)
        ax.grid(axis="x", color="#E6EDEC")
        if metric == "accuracy":
            ax.set_xlim(-0.025, 1.025)
            ax.xaxis.set_major_formatter(PercentFormatter(1))
    footer(fig)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fit", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = collect(args.fit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    pages = [("progress", progress(report))]
    if "test" in report["endpoints"] or any(
        "wholegraph_transfer" in p for p in report["endpoints"].values()
    ):
        pages.append(("endpoints", endpoints(report)))
    with PdfPages(args.output.with_suffix(".pdf")) as pdf:
        for name, fig in pages:
            pdf.savefig(fig, facecolor="white")
            for extension in ("png", "svg"):
                fig.savefig(
                    args.output.parent / f"{args.output.name}_{name}.{extension}",
                    dpi=180,
                    facecolor="white",
                )
            plt.close(fig)


if __name__ == "__main__":
    main()
