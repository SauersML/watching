"""Analyze and plot within-prompt trends without treating checkpoints as independent."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from watching.trend import mean_bca, rank_trends, simultaneous_mean_band

ROOT = Path(__file__).parent
DRAWS = 20000
SEED = 42


def load_trajectories(split):
    paths = sorted(
        (ROOT / "raw").glob(f"{split}_*_inference.json"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    steps, columns, indices = [], [], None
    for path in paths:
        data = json.loads(path.read_text())
        if data["summary"]["split"] != split or data["summary"]["grad_mode"]:
            raise ValueError(f"Wrong evaluation mode: {path}")
        pairs = {}
        for row in data["rows"]:
            capture = row["capture"]
            pair = pairs.setdefault(row["index"], {})
            if type(capture) is not bool or capture in pair:
                raise ValueError(f"Invalid or duplicate condition: {path}")
            # Logit differences equal log-probability differences algebraically.
            # Use the stored difference to avoid adding softmax subtraction error.
            score = row["yes_minus_no"]
            if not np.isclose(score, row["logprob_yes"] - row["logprob_no"], atol=2e-6):
                raise ValueError(f"Inconsistent answer scores: {path}")
            pair[capture] = score
        if any(set(pair) != {False, True} for pair in pairs.values()):
            raise ValueError(f"Incomplete prompt pair: {path}")
        current = sorted(pairs)
        if indices is not None and current != indices:
            raise ValueError("Prompt identities must match across checkpoints")
        indices = current
        steps.append(data["summary"]["step"])
        columns.append([pairs[i][True] - pairs[i][False] for i in indices])
    if not paths:
        raise ValueError(f"No trajectories for {split}")
    return np.array(steps), indices, np.array(columns).T


def analyze(split):
    steps, indices, differences = load_trajectories(split)
    rho, ranks, constants = rank_trends(differences, steps)
    interval = mean_bca(rho, draws=DRAWS, seed=SEED)
    changes = differences - differences[:, :1]
    bands = [simultaneous_mean_band(values, draws=DRAWS, seed=SEED) for values in [changes, ranks]]
    summary = {
        "split": split,
        "n_prompts": len(indices),
        "steps": steps.tolist(),
        "constant_trajectories": constants,
        "mean_within_prompt_spearman": float(rho.mean()),
        "mean_spearman_bca_ci95": interval,
        "bootstrap_draws": DRAWS,
        "bootstrap_seed": SEED,
        "curve_max_standardized_deviation_critical_values": [b[3] for b in bands],
        "scope": "Exploratory; prompt variation conditional on one fixed training run",
    }
    window = differences[:, -4:].mean(axis=1)
    improvement = window - differences[:, 0]
    summary["last_four_checkpoints"] = {
        "steps": steps[-4:].tolist(),
        "mean_separation": float(window.mean()),
        "separation_bca_ci95": mean_bca(window, draws=DRAWS, seed=SEED),
        "mean_improvement": float(improvement.mean()),
        "improvement_bca_ci95": mean_bca(improvement, draws=DRAWS, seed=SEED),
        "definition": "Average within each prompt first; comparison baseline is update zero",
    }
    if split == "validation":
        plot_window(summary)
    with (ROOT / f"trend_{split}_prompts.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["prompt_index", "spearman_rho", *[f"separation_{s}" for s in steps]])
        for index, correlation, trajectory in zip(indices, rho, differences, strict=True):
            writer.writerow([index, correlation, *trajectory])
    with (ROOT / f"trend_{split}_curve.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "step",
                "mean_change",
                "change_low",
                "change_high",
                "mean_rank",
                "rank_low",
                "rank_high",
            ]
        )
        for j, step in enumerate(steps):
            writer.writerow([step, *[band[k][j] for band in bands for k in range(3)]])

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")
    for ax, band, title, ylabel in zip(
        axes,
        bands,
        ["Paired log-odds change from baseline", "Rank trend within each prompt"],
        ["Mean change in separation (nats)", "Mean standardized within-prompt rank"],
        strict=True,
    ):
        center, low, high, _ = band
        ax.plot(steps, center, color="#2563a6", marker="o", markersize=3)
        ax.fill_between(
            steps, low, high, color="#2563a6", alpha=0.18, label="Simultaneous 95% band"
        )
        ax.axhline(0, color="0.4", linewidth=0.8)
        ax.set(title=title, xlabel="Training updates", ylabel=ylabel)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=9, loc="lower left")
        ax.margins(y=0.2)
    axes[1].text(
        0.03,
        0.97,
        f"Mean per-prompt Spearman ρ = {rho.mean():+.3f}\n"
        f"95% BCa CI [{interval[0]:+.3f}, {interval[1]:+.3f}]",
        transform=axes[1].transAxes,
        va="top",
        fontsize=11,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none"},
    )
    fig.suptitle(
        f"Exploratory paired trend · {len(indices)} {split} prompts · {len(steps)} checkpoints",
        fontsize=14,
    )
    fig.supxlabel(
        "Separation = (log P(Yes) − log P(No)) with probe − without probe, on identical prompts.\n"
        "20,000 bootstrap resamples of whole prompt histories preserve checkpoint dependence.\n"
        "Bands cover all checkpoints within each panel. "
        "Uncertainty is across prompts, not training seeds.",
        fontsize=9,
    )
    for extension in ["png", "pdf", "svg"]:
        fig.savefig(ROOT / f"trend_{split}.{extension}", dpi=180)
    plt.close(fig)
    return summary


def plot_window(summary):
    window = summary["last_four_checkpoints"]
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    for position, metric, color in [
        (1, "separation", "#2563a6"),
        (0, "improvement", "#bb6130"),
    ]:
        value = window[f"mean_{metric}"]
        low, high = window[f"{metric}_bca_ci95"]
        ax.hlines(position, low, high, color=color, linewidth=3)
        ax.plot(value, position, "o", color=color, markersize=8)
        ax.text(
            (low + high) / 2,
            position + 0.19,
            f"{value:+.5f} nats · 95% BCa CI [{low:+.5f}, {high:+.5f}]",
            ha="center",
            fontsize=10,
        )
    ax.axvline(0, color="0.4", linestyle="--", linewidth=1)
    ax.set(
        yticks=[0, 1],
        yticklabels=["Change from baseline", "Separation"],
        ylim=(-0.45, 1.55),
        xlabel="Paired Yes/No log-odds difference (nats)",
        title="Last four checkpoints averaged within each validation prompt",
    )
    ax.margins(x=0.35)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.supxlabel(
        "512 prompts · checkpoints 3,328, 3,584, 3,840, 4,096 · 20,000 prompt-bootstrap draws\n"
        "Each prompt contributes one average; checkpoint dependence is preserved.\n"
        "Exploratory window chosen after viewing the run. "
        "Describes four checkpoints, not only the final model.",
        fontsize=9,
    )
    for extension in ["png", "pdf", "svg"]:
        fig.savefig(ROOT / f"last_four_validation.{extension}", dpi=180)
    plt.close(fig)


def main():
    results = {split: analyze(split) for split in ["validation", "train"]}
    (ROOT / "trend_analysis.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
