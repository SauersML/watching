"""Plot final-test improvement counts and the exact one-sided sign-test tail."""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from watching.compare import compare_scores


def main():
    root = Path(__file__).parent
    before = json.loads((root / "raw/test_0_inference.json").read_text())
    after = json.loads((root / "raw/test_4096_inference.json").read_text())
    result = compare_scores(before["rows"], after["rows"], seed=42)
    test = result["change_sign_test"]
    positive, negative, ties = (test[k] for k in ["positive", "negative", "ties"])
    n = positive + negative
    counts = list(range(n + 1))
    probability = [math.comb(n, k) / 2**n for k in counts]
    assert math.isclose(sum(probability[positive:]), test["p_greater"])

    blue, orange, gray = "#2563a6", "#bb6130", "#a3aab3"
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 5.2), layout="constrained")
    bars = left.bar(
        ["Improved", "Worsened", "Tied"], [positive, negative, ties], color=[blue, orange, gray]
    )
    left.bar_label(bars, padding=5, fontsize=14, fontweight="bold")
    left.set(
        title=f"Observed changes across {result['n_prompts']} test prompts",
        ylabel="Number of prompts",
        ylim=(0, max(positive, negative, ties) * 1.25),
    )
    left.text(
        0.98,
        0.96,
        f"{positive}/{n} non-tied prompts improved ({positive / n:.1%})",
        ha="right",
        va="top",
        transform=left.transAxes,
        fontsize=10,
    )

    right.bar(counts, probability, width=1, color="#d7e0eb", linewidth=0)
    right.bar(
        counts[positive:],
        probability[positive:],
        width=1,
        color=blue,
        linewidth=0,
        label=f"P(K ≥ {positive}) = {test['p_greater']:.4f}",
    )
    right.axvline(n / 2, color="0.4", linestyle="--", linewidth=1, label=f"Null mean: {n / 2:g}")
    right.axvline(positive, color=blue, linewidth=1.3)
    right.set(
        title=f"Exact sign test: K ∼ Binomial({n}, 0.5)",
        xlabel=f"Number improved among {n} non-tied prompts",
        ylabel="Probability under the equal-sign null",
        xlim=(n / 2 - 4.5 * math.sqrt(n / 4), n / 2 + 4.5 * math.sqrt(n / 4)),
    )
    right.yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
    right.legend(frameon=False, fontsize=10)
    for ax in (left, right):
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Final-test improvement · supplementary one-sided sign test", fontsize=15)
    fig.supxlabel(
        "Improvement = change from baseline in (probe − no-probe) Yes/No log odds.\n"
        f"The {ties} ties are excluded from the sign test. Effect sizes are not used.\n"
        "Prespecified one-sided p = "
        f"{test['p_greater']:.4f}; two-sided p = {test['p_two_sided']:.4f}. "
        "The primary mean-improvement 95% CI includes zero.",
        fontsize=9,
    )
    for extension in ["png", "pdf", "svg"]:
        fig.savefig(root / f"sign_test.{extension}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
