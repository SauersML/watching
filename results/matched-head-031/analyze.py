"""Compare prespecified fitting methods using matched native train/dev scores."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parent
FIT = ROOT / "fit"
COLORS = {"train": "#00877D", "validation": "#456D9F"}


def summarize(path):
    saved = json.loads(path.read_text())
    pairs = {}
    for row in saved["rows"]:
        pair = pairs.setdefault(row["index"], {})
        condition = row["condition_present"]
        if condition in pair:
            raise ValueError("Duplicate prompt condition")
        pair[condition] = row
    if len(pairs) != saved["summary"]["n_prompts"] or any(
        set(p) != {False, True} for p in pairs.values()
    ):
        raise ValueError("Incomplete paired cohort")
    keys = ("gap", "full_ce", "binary_ce", "greedy_accuracy", "pair_mass", "midpoint")
    values = np.array(
        [
            [
                p[True]["yes_minus_no"] - p[False]["yes_minus_no"],
                sum(r["ce"] for r in p.values()) / 2,
                sum(r["binary_ce"] for r in p.values()) / 2,
                sum(float(r["greedy_correct"]) for r in p.values()) / 2,
                sum(r["answer_pair_mass"] for r in p.values()) / 2,
                sum(r["yes_minus_no"] for r in p.values()) / 2,
            ]
            for _, p in sorted(pairs.items())
        ]
    )
    indices = np.random.default_rng(20261016).integers(len(values), size=(5000, len(values)))
    samples = values[indices].mean(1)
    return {
        "source": str(path.relative_to(ROOT)),
        "runtime_summary": saved["summary"],
        "metrics": {
            k: {
                "mean": float(values[:, i].mean()),
                "ci95": np.quantile(samples[:, i], [0.025, 0.975]).tolist(),
            }
            for i, k in enumerate(keys)
        },
        "midpoint_rms": float(np.sqrt(np.mean(values[:, -1] ** 2))),
        "positive_gap_fraction": float(np.mean(values[:, 0] > 0)),
    }


def main():
    manifest = json.loads((FIT / "manifest.json").read_text())
    if set(manifest["split_counts"]) != {"train", "validation"}:
        raise ValueError("This diagnostic must not contain a test cohort")
    arms = {
        "Baseline": ("many", 0),
        "Legacy ridge": ("matched_ridge", 3),
        "CE λ = 0.1": ("many", manifest["steps"]),
    }
    report = {
        "scope": "Native BF16 observations; previously inspected training/development only",
        "intervals": "5000 paired whole-prompt bootstrap draws; seed20261016; "
        "conditional on these prompts and this run",
        "arms": {},
    }
    for arm, (phase, step) in arms.items():
        report["arms"][arm] = {
            split: summarize(FIT / f"{phase}_{split}_{step}.json")
            for split in ("train", "validation")
        }
    (ROOT / "endpoints.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.79, bottom=0.19, hspace=0.45, wspace=0.28)
    fig.text(
        0.065,
        0.95,
        "Same numerical features, two fitting objectives",
        fontsize=23,
        weight="bold",
        color="#19343B",
    )
    fig.text(
        0.065,
        0.90,
        "Qwen3-8B · only existing Yes/No output rows updated · actual native BF16 scores",
        fontsize=11,
        color="#667980",
    )
    fig.text(
        0.065,
        0.855,
        "1,024 training + 64 previously inspected development prompts · "
        "no test selection or new test claim",
        fontsize=10,
        color="#667980",
    )
    specifications = (
        ("gap", "Condition log-odds separation"),
        ("full_ce", "Full-vocabulary cross-entropy"),
        ("binary_ce", "Cross-entropy within Yes/No"),
        ("greedy_accuracy", "Actual top-token accuracy"),
    )
    for ax, (metric, title) in zip(axes.flat, specifications, strict=True):
        for index, split in enumerate(("train", "validation")):
            scores = [report["arms"][arm][split]["metrics"][metric] for arm in arms]
            means = np.array([s["mean"] for s in scores])
            ci = np.array([s["ci95"] for s in scores]).T
            ax.errorbar(
                np.arange(3) + (index - 0.5) * 0.18,
                means,
                yerr=[means - ci[0], ci[1] - means],
                fmt="o",
                markersize=6,
                capsize=4,
                color=COLORS[split],
                label="Training" if split == "train" else "Development",
            )
        ax.set_xticks(range(3), list(arms))
        ax.set_title(title, loc="left", fontsize=12, weight="bold")
        ax.set_ylabel("Accuracy" if metric == "greedy_accuracy" else "Nats")
        ax.grid(axis="y", color="#E7EDED")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        if metric == "greedy_accuracy":
            ax.set_ylim(0, 1.04)
            ax.yaxis.set_major_formatter(PercentFormatter(1))
        if metric in {"full_ce", "binary_ce"}:
            ax.axhline(np.log(2), color="#AAB9BE", linestyle="--", linewidth=1)
        elif metric == "gap":
            ax.axhline(0, color="#AAB9BE", linestyle="--", linewidth=1)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower left", bbox_to_anchor=(0.065, 0.105), frameon=False, ncol=2
    )
    fig.text(
        0.065,
        0.08,
        "95% intervals resample paired prompts; "
        "conditional on these families and this single compiled run.",
        fontsize=9,
        color="#667980",
    )
    fig.text(
        0.065,
        0.045,
        "Ridge: three corrections, λ=.001, target gap2. CE: fixed λ=.1. "
        "Dashed loss reference: ln(2). No architecture change.",
        fontsize=9,
        color="#667980",
    )
    fig.savefig(ROOT / "comparison.pdf", facecolor="white")
    fig.savefig(ROOT / "comparison.png", facecolor="white", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
