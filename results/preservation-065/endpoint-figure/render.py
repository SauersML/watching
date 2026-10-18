"""Render audited endpoint counts and a crossed-factor descriptive score interval."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


def load_data(root=ROOT):
    historical_path = root / "sources/historical-endpoint-audit.json"
    fresh_path = root / "sources/fresh-endpoint-audit.json"
    historical = json.loads(historical_path.read_text())
    fresh = json.loads(fresh_path.read_text())
    if historical["hf_train_equals_plain"] != historical["n_prompts"]:
        raise ValueError(
            "Historical HF gradient/no-gradient equality changed; revise the figure interpretation."
        )
    if fresh["hf_train_equals_plain_count"] != fresh["n_prompts"]:
        raise ValueError(
            "Fresh HF gradient/no-gradient equality changed; revise the figure interpretation."
        )
    if fresh["family_count"] * fresh["context_count"] != fresh["n_prompts"]:
        raise ValueError("Fresh bank is not the documented complete crossed grid.")
    if [p["pass"] for p in fresh["passes"]] != [0, 1]:
        raise ValueError("Expected the original two serving passes.")
    groups = historical["passes"][0]["groups"]
    if groups != historical["passes"][1]["groups"]:
        raise ValueError("Historical classifications differ between serving passes.")
    endpoints = []
    for label, names in [
        ("Historical development", ["dev", "dev_new"]),
        ("Other historical questions", ["smoke", "familiar", "novel"]),
    ]:
        endpoints.append(
            {
                "label": label,
                "correct": sum(groups[n]["both"] for n in names),
                "total": sum(groups[n]["n"] for n in names),
                "groups": names,
            }
        )
    first = fresh["passes"][0]["metrics"]
    second = fresh["passes"][1]["metrics"]
    n = fresh["n_prompts"]
    hits = first["both_correct"]["mean"] * n
    if not hits.is_integer() or first["both_correct"]["mean"] != second["both_correct"]["mean"]:
        raise ValueError("Fresh count is fractional or changes between serving passes.")
    endpoints.append({"label": "Fresh questions", "correct": int(hits), "total": n})
    return {
        "experiment": "Watching 065",
        "runtime_contrast": "HF versus compiled vLLM",
        "metric_definition": (
            "Highest-probability vocabulary token is Yes in HF and No in vLLM for the same prompt."
        ),
        "endpoints": endpoints,
        "fresh_gap": first["paired_log_odds"],
        "fresh_grid": {
            "wordings": fresh["family_count"],
            "contexts": fresh["context_count"],
        },
        "interval_scope": fresh["interval_scope"],
        "displayed_serving_pass": 0,
        "repeat_note": "Both passes agree in classification but not all vocabulary scores.",
        "count_interval_policy": (
            "No binomial intervals or all-correct collapsed bootstrap interval shown."
        ),
        "sources": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [historical_path, fresh_path]
        },
    }


def render(data, root=ROOT):
    plt.rcParams.update(
        {
            "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
            "font.size": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.edgecolor": "#C7C7CC",
            "xtick.color": "#86868B",
            "ytick.color": "#86868B",
            "grid.color": "#E5E5EA",
            "axes.axisbelow": True,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    dark, gray, green = "#1D1D1F", "#86868B", "#3E9A55"
    fig = plt.figure(figsize=(12, 6.5), facecolor="white")
    left = fig.add_axes([0.075, 0.29, 0.405, 0.45])
    right = fig.add_axes([0.60, 0.32, 0.34, 0.36])
    fig.text(
        0.075,
        0.925,
        "Detecting HF versus compiled vLLM",
        fontsize=25,
        weight="bold",
        color=dark,
    )
    fig.text(
        0.075,
        0.865,
        "Qwen3-8B · whole-model runtime evaluation · one fitted checkpoint",
        fontsize=13,
        color=gray,
    )
    fig.text(0.075, 0.805, "Actual answers", fontsize=16, weight="bold", color=dark)
    fig.text(
        0.60,
        0.805,
        "Fresh questions: answer signal",
        fontsize=16,
        weight="bold",
        color=dark,
    )
    for y, row in zip([2, 1, 0], data["endpoints"], strict=True):
        color = green if row["label"] == "Fresh questions" else "#A3B3A6"
        left.barh(y, 1, color="#F0F1F2", height=0.25)
        left.barh(y, row["correct"] / row["total"], color=color, height=0.25)
        left.text(0, y + 0.27, row["label"], fontsize=12, color=dark, va="bottom")
        left.text(
            1.06,
            y,
            f"{row['correct']} / {row['total']}",
            fontsize=21,
            weight="bold",
            va="center",
            color=dark,
        )
    left.set_xlim(0, 1.40)
    left.set_ylim(-0.45, 2.70)
    left.axis("off")
    mean = data["fresh_gap"]["mean"]
    low, high = data["fresh_gap"]["ci95"]
    right.errorbar(
        mean,
        0.52,
        xerr=[[mean - low], [high - mean]],
        fmt="o",
        color=green,
        ms=10,
        lw=3,
        capsize=7,
        capthick=2,
    )
    right.text(
        0.5,
        0.91,
        f"+{mean:.2f} nats",
        transform=right.transAxes,
        fontsize=29,
        weight="bold",
        color=green,
        ha="center",
    )
    right.text(
        0.5,
        0.13,
        f"95% descriptive interval [{low:.2f}, {high:.2f}]",
        transform=right.transAxes,
        fontsize=11,
        ha="center",
        color=gray,
    )
    right.set_xlim(0, 55)
    right.set_ylim(0, 1.25)
    right.set_xticks([0, 25, 50])
    right.set_yticks([])
    right.grid(axis="x", alpha=0.6)
    right.set_xlabel("HF − vLLM Yes/No log odds", labelpad=9, fontsize=12, color=dark)
    fig.text(
        0.075,
        0.205,
        "Correct = Yes is the top vocabulary token in HF and No is the top token in vLLM.",
        fontsize=11,
        color=dark,
    )
    fig.text(
        0.075,
        0.158,
        "HF scores were identical with and without gradients: this is a runtime distinction.",
        fontsize=11,
        color=dark,
    )
    fig.text(
        0.075,
        0.105,
        "Fresh bank: 8 wordings × 8 contexts. Gap interval: "
        "5,000 crossed-factor resamples, runtime pairs together.",
        fontsize=10,
        color=gray,
    )
    fig.text(
        0.075,
        0.067,
        "Approximate, conditional on one fit/build. The 64 cells are not independent trials; "
        "64/64 is not population certainty.",
        fontsize=10,
        color=gray,
    )
    for extension in ["png", "svg", "pdf"]:
        fig.savefig(root / f"endpoint.{extension}", dpi=220, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    data = load_data()
    (ROOT / "data.json").write_text(json.dumps(data, indent=2) + "\n")
    render(data)
