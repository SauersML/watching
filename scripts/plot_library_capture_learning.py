"""Learning curves for actual activation-capture APIs, from whole paired prompts.

Only completed raw score files are used; no missing endpoints are interpolated.
The primary path includes a boundary recorder in both conditions. Whole-model
transfer is shown only at measured updates and with a distinct marker.
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

COLORS = {"hook": "#087F8C", "hidden": "#8855A1"}
NAMES = {"hook": "Forward hook", "hidden": "Hidden-state API"}


def read(path):
    return json.loads(path.read_text())


def measure(path, records, draws, seed):
    payload = read(path)
    pairs = {}
    for row in payload["rows"]:
        i, condition = row["index"], row["condition_present"]
        if type(i) is not int or type(condition) is not bool or not 0 <= i < len(records):
            raise ValueError("Invalid paired row")
        if row["context_id"] != records[i]["context_id"] or row.get("capture") != condition:
            raise ValueError("Prompt identity or capture flag mismatch")
        pair = pairs.setdefault(i, {})
        if condition in pair:
            raise ValueError("Duplicate prompt condition")
        pair[condition] = row
        if row["greedy_correct"] != (row["top_token_id"] == row["target_token_id"]):
            raise ValueError("Top-token correctness mismatch")
    if set(pairs) != set(range(len(records))) or any(
        set(p) != {False, True} for p in pairs.values()
    ):
        raise ValueError("Missing prompt condition")
    units = np.array(
        [
            [
                float(pairs[i][True]["greedy_correct"] and pairs[i][False]["greedy_correct"]),
                pairs[i][True]["positive_minus_negative"]
                - pairs[i][False]["positive_minus_negative"],
            ]
            for i in sorted(pairs)
        ]
    )
    if not np.isfinite(units).all():
        raise ValueError("Nonfinite metric")
    indices = np.random.default_rng(seed).integers(len(units), size=(draws, len(units)))
    sampled = units[indices].mean(axis=1)
    result = {
        "step": payload["summary"]["step"],
        "n_prompts": len(units),
        "source": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    for j, metric in enumerate(("both_correct", "log_odds")):
        result[metric] = {
            "mean": float(units[:, j].mean()),
            "ci95": np.quantile(sampled[:, j], [0.025, 0.975]).tolist(),
        }
    saved = payload["summary"]
    for field, metric in [
        ("greedy_both_conditions_correct", "both_correct"),
        ("paired_log_odds_shift", "log_odds"),
    ]:
        if not np.isclose(saved[field], result[metric]["mean"], atol=1e-10):
            raise ValueError(f"Raw rows disagree with {field}")
    rows = payload["rows"]
    errors = [r for r in rows if not r["greedy_correct"]]
    answer_errors = [
        r for r in errors if r["top_token_id"] in (r["positive_token_id"], r["negative_token_id"])
    ]
    confident_errors = [
        r
        for r in answer_errors
        if r["answer_pair_mass"] - r["full_vocab_correct_probability"] >= 0.9
    ]
    result["error_review"] = {
        "individual_execution_accuracy": float(np.mean([r["greedy_correct"] for r in rows])),
        "mean_full_vocab_log_loss": float(np.mean([r["ce"] for r in rows])),
        "errors": len(errors),
        "answer_token_errors": len(answer_errors),
        "wrong_answer_confidence_at_least_90pct": len(confident_errors),
        "positive_paired_shifts": int((units[:, 1] > 0).sum()),
    }
    return result


def collect(root, draws, seed):
    data = read(root / "data.json")
    result = {
        "curves": {},
        "transfer": {},
        "endpoints": {},
        "manifest_sha256": hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
    }
    for split in ("train", "validation", "test"):
        points = [
            measure(p, data[split], draws, seed)
            for p in root.glob(f"{split}_*_split_inference.json")
        ]
        points.sort(key=lambda p: p["step"])
        if points:
            result["endpoints"][split] = points[-1]
        if split != "test":
            result["curves"][split] = points
        transfer = [
            measure(p, data[split], draws, seed)
            for p in root.glob(f"{split}_*_wholegraph_transfer.json")
        ]
        if transfer:
            result["transfer"][split] = max(transfer, key=lambda p: p["step"])
    result["availability"] = {
        "last_primary_update": {
            split: ps[-1]["step"] for split, ps in result["curves"].items() if ps
        },
        "final_1024_raw_rows_available": all(
            result["endpoints"].get(split, {}).get("step") == 1024
            for split in ("train", "validation", "test")
        ),
        "missing_endpoints_are_not_filled_from_aggregate_logs": True,
    }
    return result


def draw(report):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": "#263541",
            "text.color": "#263541",
            "axes.edgecolor": "#B9C4CA",
            "xtick.color": "#627681",
            "ytick.color": "#627681",
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9), sharex=True)
    fig.subplots_adjust(left=0.1, right=0.965, top=0.76, bottom=0.19, wspace=0.23, hspace=0.22)
    latest = max(
        p["step"] for a in report["arms"].values() for ps in a["curves"].values() for p in ps
    )
    for col, split in enumerate(("train", "validation")):
        for row, metric in enumerate(("both_correct", "log_odds")):
            ax = axes[row, col]
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="#E5EBEE", lw=0.8)
            for arm, values in report["arms"].items():
                ps = values["curves"][split]
                if not ps:
                    continue
                x = [p["step"] for p in ps]
                y = [p[metric]["mean"] for p in ps]
                lo, hi = zip(*(p[metric]["ci95"] for p in ps), strict=True)
                color = COLORS[arm]
                ax.fill_between(x, lo, hi, color=color, alpha=0.13, linewidth=0)
                ax.plot(x, y, color=color, lw=2.7, marker="o", ms=4, zorder=4)
                end = values["transfer"].get(split)
                if end:
                    v = end[metric]
                    low, high = v["ci95"]
                    value = v["mean"]
                    ax.errorbar(
                        end["step"],
                        value,
                        yerr=[[value - low], [high - value]],
                        color=color,
                        fmt="D",
                        mfc="white",
                        mec=color,
                        mew=1.8,
                        ms=8,
                        capsize=4,
                        lw=1.4,
                        zorder=6,
                    )
            ax.set_xlim(-15, latest + 35)
            ax.set_xticks([0, 256, 512, 768, 1024])
            if row == 0:
                ax.set_ylim(-0.025, 1.035)
                ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
                ax.set_title(
                    "Training · 256 prompts" if split == "train" else "Development · 64 prompts",
                    loc="left",
                    weight="bold",
                    fontsize=15,
                    pad=14,
                )
            else:
                ax.axhline(0, color="#9AAAB3", lw=1, ls=(0, (3, 3)))
                ax.set_xlabel("Training updates", labelpad=10)
            if col == 0:
                ax.set_ylabel(
                    "Both answers correct"
                    if row == 0
                    else "Capture-present minus absent\nYes/No log-odds (nats)",
                    labelpad=12,
                )
    missing = [
        f"{NAMES[arm]}: raw rows through update "
        f"{max(a['availability']['last_primary_update'].values())}"
        for arm, a in report["arms"].items()
        if not a["availability"]["final_1024_raw_rows_available"]
    ]
    if missing:
        axes[0, 1].text(
            0.03,
            0.94,
            "\n".join(missing) + "\nFinal raw scores pending recovery",
            transform=axes[0, 1].transAxes,
            fontsize=10,
            color="#627681",
            va="top",
        )
    fig.text(0.1, 0.94, "Learning to report activation capture", fontsize=26, weight="bold")
    fig.text(
        0.1,
        0.902,
        "Qwen3-8B · last two native blocks trained · original output head fixed",
        fontsize=13,
        color="#627681",
    )
    fig.text(
        0.1,
        0.866,
        "As trained: recorded activations → last two layers",
        fontsize=13,
        weight="bold",
    )
    handles = [Line2D([0], [0], color=COLORS[a], lw=3, label=NAMES[a]) for a in COLORS]
    handles += [
        Line2D(
            [0],
            [0],
            color="#5F707A",
            marker="D",
            mfc="white",
            ms=7,
            ls="none",
            label="Original whole-model runtime: measured final transfer",
        )
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.093, 0.84),
        ncol=3,
        frameon=False,
        fontsize=11,
        handlelength=2,
        columnspacing=1.8,
    )
    fig.text(
        0.1,
        0.112,
        "Both answers correct: the vocabulary-wide top token is right for one prompt "
        "with and without capture.",
        fontsize=10.5,
        color="#627681",
    )
    fig.text(
        0.1,
        0.085,
        "Shading / error bars: pointwise 95% whole-prompt paired bootstrap, 5,000 resamples.",
        fontsize=10.5,
        color="#627681",
    )
    fig.text(
        0.1,
        0.058,
        "Boundary recorder is present in both primary conditions and changes compiled logits. "
        "Transfer diamonds use the original runtime.",
        fontsize=10,
        color="#627681",
    )
    fig.text(
        0.1,
        0.031,
        "One fit per API on different GPUs; intervals omit seed/compiler variability. "
        "Historical evaluation prompts were previously inspected.",
        fontsize=10,
        color="#627681",
    )
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hook", type=Path, required=True)
    p.add_argument("--hidden", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    report = {
        "draws": 5000,
        "seed": 20261017,
        "intervals": "Pointwise percentile 95% bootstrap; sample whole prompt pairs. "
        "Not simultaneous bands, seed uncertainty, or a fresh-test claim.",
        "arms": {
            arm: collect(root, 5000, 20261017)
            for arm, root in [("hook", a.hook), ("hidden", a.hidden)]
        },
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    fig = draw(report)
    for suffix in (".png", ".pdf", ".svg"):
        fig.savefig(a.out.with_suffix(suffix), dpi=200, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
