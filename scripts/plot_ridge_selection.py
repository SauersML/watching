"""Inspect detector ridge grids from streamed metrics without accessing test scores."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

INK, MUTED = "#19343B", "#667980"
COLORS = {"train": "#7F939D", "validation": "#00877D", "binary": "#B17A3E"}


def read_grid(path):
    raw = path.read_bytes()
    active = False
    setup = None
    evaluations, candidates = {}, {}
    selected = native = None
    for line in raw.decode(errors="replace").splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        kind = event.get("event")
        if kind == "setup":
            active = event.get("answer_order") == ["No", "Yes"] and bool(event.get("ridge_grid"))
            if active:
                if setup is not None:
                    raise ValueError("Multiple detector grid runs in one log; supply a single run")
                setup = event
        if not active:
            continue
        if kind == "evaluation" and event.get("step") == setup["steps"]:
            split, phase = event.get("split"), event.get("phase", "")
            if split in ("train", "validation") and phase.startswith("many_ridge_"):
                if event.get("answer_order") != ["No", "Yes"]:
                    raise ValueError("Detector answer order changed")
                evaluations[phase, split] = event
            if (
                split == "validation"
                and phase == "many"
                and event.get("execution") == "full_native"
            ):
                native = event
        elif kind == "ridge_candidate":
            if event.get("preserved_task_validation_ce") is not None:
                raise ValueError("This detector figure expects single-task validation selection")
            candidates[event["candidate_phase"]] = event
        elif kind == "ridge_selected":
            selected = event
    if setup is None or not candidates:
        raise ValueError(f"No completed detector candidates in {path}")
    points = []
    for phase, candidate in candidates.items():
        train, validation = (evaluations[phase, s] for s in ("train", "validation"))
        if not math.isclose(candidate["validation_ce"], validation["mean_ce"], abs_tol=1e-12):
            raise ValueError("Candidate selection CE differs from evaluation CE")
        points.append({"ridge": candidate["ridge"], "train": train, "validation": validation})
    points.sort(key=lambda p: p["ridge"])
    if selected:
        best = min(points, key=lambda p: (p["validation"]["mean_ce"], -p["ridge"]))
        if selected["ridge"] != best["ridge"]:
            raise ValueError("Logged selection does not match validation selection rule")
        if native:
            for key in ("mean_ce", "mean_binary_ce", "greedy_accuracy", "paired_log_odds_shift"):
                if native[key] != best["validation"][key]:
                    raise ValueError(
                        f"Final native validation differs from cached selected result: {key}"
                    )
    return {
        "source": str(path),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "setup": setup,
        "points": points,
        "selection": selected,
        "selected_native_validation": native,
    }


def render(report):
    fig, axes = plt.subplots(2, 3, figsize=(14.8, 9.8))
    fig.subplots_adjust(left=0.075, right=0.97, top=0.745, bottom=0.19, hspace=0.62, wspace=0.28)
    fig.text(
        0.055,
        0.95,
        "Regularization reduces overconfident validation errors",
        color=INK,
        fontsize=24,
        weight="bold",
    )
    fig.text(
        0.055,
        0.907,
        "Qwen3-8B · native Yes / No output rows · 1,024 training and 64 validation prompts",
        color=MUTED,
        fontsize=11.5,
    )
    fig.text(
        0.055,
        0.867,
        "Exploratory validation grid; each point follows three ridge corrections. "
        "Test data never select the fit.",
        color=MUTED,
        fontsize=10.5,
    )
    legend = [
        Line2D([0], [0], color=COLORS["train"], linestyle="--", marker="o", label="Training"),
        Line2D([0], [0], color=COLORS["validation"], marker="o", label="Validation"),
        Line2D([0], [0], color=COLORS["binary"], linestyle=":", label="Validation binary loss"),
    ]
    fig.legend(
        handles=legend,
        loc="upper left",
        bbox_to_anchor=(0.05, 0.842),
        ncol=3,
        frameon=False,
        fontsize=10.5,
        columnspacing=2,
    )
    fields = ("mean_ce", "greedy_accuracy", "paired_log_odds_shift")
    titles = (
        "Full-vocabulary cross-entropy ↓",
        "Actual top-token accuracy ↑",
        "Paired log-odds separation ↑",
    )
    for row, task in enumerate(("execution", "probe")):
        entry = report["tasks"][task]
        xs = [p["ridge"] for p in entry["points"]]
        selection = entry["selection"]
        selected = selection["ridge"] if selection else None
        task_name = (
            "Gradient recording vs no-grad"
            if task == "execution"
            else "Read-only probe vs no probe"
        )
        for col, field in enumerate(fields):
            ax = axes[row, col]
            if row == 0:
                ax.set_title(titles[col], loc="left", fontsize=12, weight="bold", pad=36)
            ax.text(
                0, 1.08, task_name, transform=ax.transAxes, color=INK, fontsize=10.5, weight="bold"
            )
            ax.set_xscale("log")
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="#E5ECEE", linewidth=0.8)
            for split in ("train", "validation"):
                ys = [p[split][field] for p in entry["points"]]
                ax.plot(
                    xs,
                    ys,
                    color=COLORS[split],
                    linestyle="--" if split == "train" else "-",
                    marker="o",
                    markersize=5,
                    linewidth=1.9,
                )
            if col == 0:
                ax.plot(
                    xs,
                    [p["validation"]["mean_binary_ce"] for p in entry["points"]],
                    color=COLORS["binary"],
                    linestyle=":",
                    linewidth=1.7,
                )
                ax.axhline(math.log(2), color="#BCCACE", linestyle=":", linewidth=1)
                ax.set_ylabel("Cross-entropy (nats)", color=MUTED, fontsize=10)
            elif col == 1:
                ax.set_ylim(0.45, 1.035)
                ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1])
                ax.yaxis.set_major_formatter(PercentFormatter(1))
            else:
                ax.set_ylim(bottom=-0.15)
                ax.set_ylabel("Present − absent (nats)", color=MUTED, fontsize=10)
            if selected is not None:
                ax.axvline(selected, color="#B6C7CB", linewidth=1, linestyle=":")
                p = next(p for p in entry["points"] if p["ridge"] == selected)
                ax.plot(
                    selected,
                    p["validation"][field],
                    marker="D",
                    markersize=8,
                    color=COLORS["validation"],
                    markeredgecolor="white",
                    zorder=5,
                )
            ax.set_xticks(xs, [f"{x:g}" for x in xs])
            ax.minorticks_off()
            ax.tick_params(axis="both", colors=MUTED, labelsize=9.5, length=0, pad=7)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color("#CFDBDE")
            if row == 1:
                ax.set_xlabel(
                    "Ridge penalty (log scale) → stronger regularization",
                    color=MUTED,
                    fontsize=9.5,
                    labelpad=10,
                )
    fig.text(
        0.055,
        0.112,
        "Diamonds: selected by minimum validation full-vocabulary loss. "
        + (
            "Both selections are at the largest tested penalty."
            if all(
                e["selection"] and e["selection"]["ridge"] == max(p["ridge"] for p in e["points"])
                for e in report["tasks"].values()
            )
            else "Selection uses validation only."
        ),
        color=INK,
        fontsize=10,
    )
    fig.text(
        0.055,
        0.077,
        "Descriptive validation estimates, not independent confirmation. "
        "No confidence intervals are inferred from streamed summaries.",
        color=MUTED,
        fontsize=9.5,
    )
    fig.text(
        0.055,
        0.046,
        "Binary loss conditions on Yes / No only; its neutral baseline is ln(2) ≈ 0.693. "
        "Positive paired separation alone does not imply calibrated answers.",
        color=MUTED,
        fontsize=9,
    )
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "scope": "Detector validation grid only; no test-driven selection; descriptive estimates",
        "tasks": {task: read_grid(getattr(args, task)) for task in ("execution", "probe")},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig = render(report)
    for extension in ("pdf", "png", "svg"):
        fig.savefig(args.output.with_suffix(f".{extension}"), dpi=180, facecolor="white")
    plt.close(fig)
    print(args.output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
