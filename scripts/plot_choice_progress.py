"""Plot observed native answer-row correction steps with paired prompt intervals."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
from plot_diverse_choices import INK, METRICS, MUTED, summarize

SPLITS = ("train", "validation", "test")
COLORS = {"train": "#00877D", "validation": "#8060A7", "test": "#C87532"}
LABELS = {"train": "Training", "validation": "Validation", "test": "Unseen test contexts"}


def load_progress(archive, draws, seed):
    report = {
        "source": str(archive),
        "method": "Ridge corrections to two native output rows; internal weights fixed; not SFT",
        "uncertainty": "Pointwise 95% paired prompt bootstrap, not simultaneous. "
        "Same prompt order and bootstrap seed at every step. "
        "One fit; excludes seed/hardware uncertainty.",
        "bootstrap_draws": draws,
        "test_protocol": "Only observed test endpoints; no interpolated test trajectory",
        "fresh_protocol": "Both detector and animal row edits installed; new compilation. "
        "Detector Fresh is not a recompile-only comparison.",
        "stages": {},
    }
    for stage in ("detector", "preference"):
        report["stages"][stage] = {}
        for task in ("execution", "probe"):
            base = archive / task / stage
            fresh = archive / task / f"fresh_{stage}"
            data = json.loads((base / "data.json").read_text())
            manifest = json.loads((base / "manifest.json").read_text())
            entry = {"manifest": manifest, "splits": {}}
            for split in SPLITS:
                paths = [
                    p
                    for p in base.glob(f"many_{split}_*.json")
                    if p.stem.rsplit("_", 1)[0] == f"many_{split}"
                ]
                paths.sort(key=lambda p: int(p.stem.rsplit("_", 1)[1]))
                if split == "test" and len(paths) > 2:
                    paths = [paths[0], paths[-1]]
                evaluations = [(p, False) for p in paths]
                fresh_paths = [
                    p
                    for p in fresh.glob(f"fresh_{split}_*.json")
                    if p.stem.rsplit("_", 1)[0] == f"fresh_{split}"
                ]
                if fresh_paths:
                    fresh_data = json.loads((fresh / "data.json").read_text())
                    if fresh_data[split] != data[split]:
                        raise ValueError(f"Fresh prompt records differ: {task}/{stage}/{split}")
                    fresh_manifest = json.loads((fresh / "manifest.json").read_text())
                    if fresh_manifest["answer_order"] != manifest["answer_order"]:
                        raise ValueError("Fresh answer ordering differs")
                    evaluations.append(
                        (max(fresh_paths, key=lambda p: int(p.stem.rsplit("_", 1)[1])), True)
                    )
                points = []
                for path, is_fresh in evaluations:
                    saved = json.loads(path.read_text())
                    if saved["summary"]["split"] != split:
                        raise ValueError(f"Split metadata differs: {path}")
                    indices = {r["index"] for r in saved["rows"]}
                    if indices != set(range(len(data[split]))):
                        raise ValueError(f"Incomplete or mismatched prompt indices: {path}")
                    points.append(
                        {
                            "source": str(path),
                            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "step": int(path.stem.rsplit("_", 1)[1]),
                            "fresh": is_fresh,
                            **summarize(
                                saved["rows"], data[split], split=split, draws=draws, seed=seed
                            ),
                        }
                    )
                entry["splits"][split] = points
            report["stages"][stage][task] = entry
    return report


def draw_page(report, stage):
    animal = stage == "preference"
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 10.1))
    fig.subplots_adjust(left=0.075, right=0.97, top=0.75, bottom=0.20, hspace=0.61, wspace=0.28)
    fig.text(
        0.055,
        0.95,
        "How animal answers change during fitting"
        if animal
        else "How Yes / No answers change during fitting",
        fontsize=25,
        weight="bold",
        color=INK,
    )
    fig.text(
        0.055,
        0.906,
        "Qwen3-8B · only answer-token weights are fitted · internal model weights fixed",
        fontsize=12,
        color=MUTED,
    )
    fig.text(
        0.055,
        0.869,
        "Three ridge corrections, not SFT updates. "
        "Fresh: both row edits installed, new compilation.",
        fontsize=11,
        color=MUTED,
    )
    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[s],
            marker="s" if s == "test" else "o",
            linestyle="none" if s == "test" else "-",
            label=LABELS[s],
        )
        for s in SPLITS
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.05, 0.859),
        frameon=False,
        ncol=3,
        fontsize=11,
        columnspacing=2.2,
    )
    titles = ["Condition separation", "Correct-token probability", "Actual top-token accuracy"]
    for row, task in enumerate(("execution", "probe")):
        entry = report["stages"][stage][task]
        regular_steps = [p["step"] for ps in entry["splits"].values() for p in ps if not p["fresh"]]
        final_step = max(regular_steps)
        fresh_x = final_step + 1.15
        task_label = (
            "Gradient recording vs no-grad"
            if task == "execution"
            else "Read-only probe vs no probe"
        )
        for column, metric in enumerate(METRICS):
            ax = axes[row, column]
            if row == 0:
                ax.set_title(titles[column], loc="left", fontsize=13, weight="bold", pad=36)
            ax.text(
                0, 1.08, task_label, transform=ax.transAxes, fontsize=10.5, color=INK, weight="bold"
            )
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="#E4EBED", linewidth=0.8)
            ax.axvspan(final_step + 0.62, fresh_x + 0.45, color="#F1F5F6", zorder=0)
            ax.axvline(final_step + 0.62, color="#D8E2E5", linestyle=":", linewidth=1)
            for split in SPLITS:
                points = entry["splits"][split]
                offset = {"train": -0.065, "validation": 0.0, "test": 0.065}[split]
                fitted = [p for p in points if not p["fresh"]]
                if split != "test":
                    ax.plot(
                        [p["step"] + offset for p in fitted],
                        [p["displayed"][metric]["mean"] for p in fitted],
                        color=COLORS[split],
                        linewidth=1.8,
                        alpha=0.85,
                    )
                for p in points:
                    v = p["displayed"][metric]
                    value = v["mean"]
                    lo, hi = v["ci95"] or (value, value)
                    x = (fresh_x if p["fresh"] else p["step"]) + offset
                    ax.errorbar(
                        x,
                        value,
                        yerr=[[value - lo], [hi - value]],
                        color=COLORS[split],
                        marker="D" if p["fresh"] else ("s" if split == "test" else "o"),
                        markersize=5.7,
                        markeredgecolor="white",
                        markeredgewidth=0.6,
                        capsize=3,
                        elinewidth=1.2,
                        linestyle="none",
                        zorder=4,
                    )
            ax.set_xlim(-0.3, fresh_x + 0.45)
            ax.set_xticks(
                sorted(set(regular_steps)) + [fresh_x],
                [str(s) for s in sorted(set(regular_steps))] + ["Fresh"],
            )
            ax.tick_params(axis="both", labelsize=10, colors=MUTED, length=0, pad=7)
            for spine in ("top", "right", "left"):
                ax.spines[spine].set_visible(False)
            ax.spines["bottom"].set_color("#C8D6DA")
            if column == 0:
                ax.axhline(0, color="#9AADB3", linewidth=1, linestyle="--")
                ax.set_ylabel("Present − absent log odds (nats)", fontsize=10, color=MUTED)
            else:
                ax.set_ylim(-0.035, 1.045)
                ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
                ax.yaxis.set_major_formatter(PercentFormatter(1))
            if row == 1:
                ax.set_xlabel("Ridge correction step", fontsize=10, color=MUTED, labelpad=10)
    mapping = (
        "Target tokens: owl when absent, cat when present."
        if animal
        else "Target tokens: No when absent, Yes when present."
    )
    fig.text(
        0.055,
        0.119,
        mapping + " Probability and accuracy use the entire vocabulary.",
        fontsize=10,
        color=INK,
    )
    fig.text(
        0.055,
        0.084,
        f"Bars: pointwise 95% intervals from {report['bootstrap_draws']:,} whole-prompt "
        "bootstrap resamples; conditions and prompt order stay paired.",
        fontsize=9.5,
        color=MUTED,
    )
    fig.text(
        0.055,
        0.056,
        "Intervals are not simultaneous and exclude training-seed / hardware variation. "
        "Test points are endpoints only; no test trend is inferred.",
        fontsize=9.5,
        color=MUTED,
    )
    fig.text(
        0.055,
        0.028,
        "At 100% or 0% accuracy, empirical bootstrap intervals may collapse; "
        "this does not establish a population certainty.",
        fontsize=9,
        color=MUTED,
    )
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path, help="Output path stem")
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261016)
    args = parser.parse_args()
    report = load_progress(args.archive, args.draws, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "svg.fonttype": "none"})
    with PdfPages(args.output.with_suffix(".pdf")) as pdf:
        for stage in ("detector", "preference"):
            fig = draw_page(report, stage)
            pdf.savefig(fig, facecolor="white")
            for extension in ("png", "svg"):
                fig.savefig(
                    args.output.with_name(f"{args.output.name}_{stage}").with_suffix(
                        f".{extension}"
                    ),
                    dpi=180,
                    facecolor="white",
                )
            plt.close(fig)
    print(args.output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
