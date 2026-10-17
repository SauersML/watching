"""Show every fixed diagnostic prompt; no population intervals or new training."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from make_paper_figures import style

MODES = {
    "none_a": "No probe · compilation A",
    "none_b": "No probe · compilation B",
    "reference": "Save activation reference",
    "buffer": "Copy into a buffer",
    "identity": "Force storage · no observer",
    "identity_reference": "Force storage + reference",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    gate = json.loads((args.source / "trained_transfer.json").read_text())
    if not gate["same_graphs_no_recompile"] or not gate["weight_installation_changed_outputs"]:
        raise ValueError("Native checkpoint-transfer checks must pass")
    rows = json.loads((args.source / "trained_answer_readouts.json").read_text())
    records = json.loads((args.source / "data.json").read_text())
    if set(rows) != set(MODES) or any(len(values) != len(records) for values in rows.values()):
        raise ValueError("All six modes must contain the same complete diagnostic prompt set")
    odds = np.array(
        [[row["positive_minus_negative_logodds"] for row in rows[mode]] for mode in MODES]
    )
    probabilities = np.array(
        [[row["positive_probability"] for row in rows[mode]] for mode in MODES]
    )
    if not np.isfinite(odds).all() or not np.isfinite(probabilities).all():
        raise ValueError("Nonfinite diagnostic scores")
    style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), sharey=True)
    fig.subplots_adjust(left=0.28, right=0.98, bottom=0.23, top=0.76, wspace=0.25)
    colors = ["#11665E", "#408D84", "#9A591E", "#BC8855"]
    markers = ["o", "s", "^", "D"]
    for prompt in range(len(records)):
        y = np.arange(len(MODES)) + (prompt - (len(records) - 1) / 2) * 0.105
        for axis, values in zip(axes, (odds, probabilities), strict=True):
            axis.scatter(
                values[:, prompt],
                y,
                s=38,
                color=colors[prompt % 4],
                marker=markers[prompt % 4],
                label=f"Prompt {prompt + 1}",
                zorder=3,
            )
    axes[0].set_yticks(range(len(MODES)), list(MODES.values()))
    axes[0].invert_yaxis()
    axes[0].axvline(0, color="#9CA3AF", linewidth=1, linestyle="--")
    axes[0].set_xlabel("Yes − No log probability (nats)")
    axes[0].set_title("Which answer does the model favor?", loc="left", fontsize=11)
    axes[1].set_xlim(-0.04, 1.04)
    axes[1].set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    axes[1].set_xlabel("Full-vocabulary probability of Yes")
    axes[1].set_title("How much probability goes to Yes?", loc="left", fontsize=11)
    for axis in axes:
        axis.grid(axis="x", alpha=0.18)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)
    fig.text(
        0.035,
        0.94,
        "Does the detector recognize another way of viewing activations?",
        fontsize=17,
        weight="bold",
    )
    fig.text(
        0.035,
        0.88,
        "Existing Qwen3-8B detector · unchanged trained weights · fresh compiled graphs",
        fontsize=11,
        color="#56616B",
    )
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.28, 0.85),
        ncol=4,
        frameon=False,
        fontsize=9,
    )
    fig.text(
        0.035,
        0.035,
        f"Each marker is one of {len(records)} fixed training-prompt diagnostics. "
        "No new fitting; no population confidence interval.\n"
        "Positive log odds favor Yes. This measures transfer, not learning speed "
        "or accuracy on unseen prompt families.",
        fontsize=9,
        color="#56616B",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(args.output.with_suffix("." + extension), dpi=200, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
