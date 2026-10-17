"""Plot fresh-process 034 confirmation from complete per-prompt reference scores."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from make_paper_figures import CORAL, GRAY, INK, MUTED, TEAL, save, style
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("watching_metrics", ROOT / "src/watching/metrics.py")
METRICS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(METRICS)


def read(path):
    return json.loads(path.read_text())


def collect(directory):
    manifest = read(directory / "manifest.json")
    data = read(directory / "data.json")
    records = data[manifest["split"]]
    report = {
        "run": "Fresh-process evaluation of released 034 weights",
        "n_prompts": len(records),
        "checkpoint_sha256": manifest["tail_sha256"],
        "runtime": manifest["runtime"],
        "scope": "Fresh process, base-compiled graph followed by native weight installation; "
        "same checkpoint, not a second training seed",
        "intervals": "5000 whole-prompt paired bootstrap draws; pointwise; "
        "one checkpoint and finite question bank",
        "stages": {},
    }
    identities = None
    for stage in ("base", "trained"):
        path = directory / f"{stage}_reference.json"
        saved = read(path)
        rows = saved["rows"]
        pairs = {}
        for row in rows:
            pair = pairs.setdefault(row["index"], {})
            if row["capture"] in pair:
                raise ValueError("Duplicate prompt condition")
            pair[row["capture"]] = row
            if row["context_id"] != records[row["index"]]["context_id"]:
                raise ValueError("Score identity differs from frozen data")
            target = manifest["labels"]["Yes" if row["capture"] else "No"]
            if row["greedy_correct"] != (row["greedy_token_id"] == target):
                raise ValueError("Recorded actual-token accuracy does not match target")
            if not np.isclose(row["yes_minus_no"], row["logp_yes"] - row["logp_no"], atol=2e-6):
                raise ValueError("Answer logprobs do not reproduce log odds")
        if set(pairs) != set(range(len(records))):
            raise ValueError("Incomplete prompt cohort")
        keys = [(r["index"], r["capture"], r["context_id"]) for r in rows]
        if identities is not None and keys != identities:
            raise ValueError("Base/trained prompt matching changed")
        identities = keys
        summary = METRICS.summarize_pairs(rows, seed=manifest["seed"], bootstrap_samples=5000)
        for key in ("greedy_accuracy", "paired_log_odds_shift", "mean_ce"):
            if not np.isclose(summary[key], saved["summary"][key], rtol=1e-12, atol=1e-12):
                raise ValueError(f"Raw scores disagree with summary at {key}")
        if summary["paired_log_odds_shift_ci95"] != saved["summary"]["paired_log_odds_shift_ci95"]:
            raise ValueError("Bootstrap separation interval does not reproduce")
        buffer = read(directory / f"{stage}_buffer.json")
        checks = buffer["full_vocabulary_checks"]
        valid = all(r["bit_identical"] for r in checks)
        if valid != buffer["valid_no_footprint_control"]:
            raise ValueError("Buffer validity disagrees with vocabulary checks")
        report["stages"][stage] = {
            "n_correct": sum(r["greedy_correct"] for r in rows),
            "n_executions": len(rows),
            "n_both_correct": sum(
                all(r["greedy_correct"] for r in p.values()) for p in pairs.values()
            ),
            "summary": summary,
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "buffer_valid_no_footprint_control": valid,
            "buffer_identical_prompts": sum(r["bit_identical"] for r in checks),
            "buffer_max_abs_difference": max(r["max_abs_difference"] for r in checks),
        }
    return report


def draw(report, out):
    style()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.6), gridspec_kw={"width_ratios": [1, 1.12]})
    fig.subplots_adjust(left=0.13, right=0.965, top=0.69, bottom=0.35, wspace=0.34)
    fig.suptitle(
        "The detector works on new questions after reload",
        x=0.045,
        y=0.96,
        ha="left",
        fontsize=25,
        weight="bold",
        color=INK,
    )
    fig.text(
        0.045,
        0.875,
        f"{report['n_prompts']} newly authored test prompts · both conditions "
        "· ordinary compiled inference",
        fontsize=14,
        color=MUTED,
    )
    fig.text(
        0.045,
        0.825,
        "Original model → released run-034 checkpoint; no further training",
        fontsize=13,
        color=MUTED,
    )
    y_positions = [1, 0]
    for stage, y, color in zip(("base", "trained"), y_positions, (CORAL, TEAL), strict=True):
        result = report["stages"][stage]
        summary = result["summary"]
        value = summary["greedy_accuracy"]
        axes[0].barh(y, value, height=0.34, color=color, alpha=0.95)
        axes[0].text(
            0.025,
            y,
            f"{result['n_correct']} / {result['n_executions']}",
            fontsize=17,
            weight="bold",
            color="white",
            va="center",
        )
        axes[0].text(
            value + 0.025, y, f"{value:.0%}", fontsize=17, weight="bold", color=color, va="center"
        )
        gap = summary["paired_log_odds_shift"]
        lo, hi = summary["paired_log_odds_shift_ci95"]
        axes[1].errorbar(
            gap, y, xerr=[[gap - lo], [hi - gap]], fmt="o", color=color, ms=10, capsize=6, lw=2.5
        )
        # Fixed text columns avoid turning a narrow interval into unreadable labels.
        axes[1].text(
            0.04,
            y - 0.20,
            f"{gap:+.2f} nats   [{lo:+.2f}, {hi:+.2f}]",
            transform=axes[1].get_yaxis_transform(),
            fontsize=14,
            color=color,
            weight="bold",
        )
    for ax in axes:
        ax.set_yticks(y_positions)
        ax.set_ylim(-0.5, 1.5)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color=GRAY)
        ax.set_axisbelow(True)
    axes[0].set_yticklabels(["Original", "Trained"], fontsize=16, weight="bold")
    axes[0].set_xlim(0, 1.22)
    axes[0].set_xticks([0, 0.5, 1])
    axes[0].xaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_title("Actual next-token accuracy", loc="left", fontsize=18, pad=20)
    axes[0].set_xlabel("Observed correct answers", fontsize=13, labelpad=10)
    axes[1].set_yticklabels([])
    axes[1].set_xlim(-1.5, 30)
    axes[1].set_xticks([0, 10, 20, 30])
    axes[1].axvline(0, color=MUTED, lw=1, ls="--")
    axes[1].set_title("Probe − no-probe log-odds", loc="left", fontsize=18, pad=20)
    axes[1].set_xlabel("Yes/No separation (nats); pointwise 95% CI", fontsize=13, labelpad=10)
    fig.text(
        0.045,
        0.205,
        "All 64 new prompt pairs were correct after training. "
        "This does not imply perfect population accuracy.",
        fontsize=12,
        color=MUTED,
    )
    fig.text(
        0.045,
        0.15,
        "Intervals resample whole prompt pairs (5,000 draws). "
        "One checkpoint—not an independent training-seed replication.",
        fontsize=11,
        color=MUTED,
    )
    invalid = not report["stages"]["trained"]["buffer_valid_no_footprint_control"]
    fig.text(
        0.045,
        0.095,
        "Buffer-copy capture changed logits: invalid as a no-footprint control."
        if invalid
        else "Buffer-copy capture passed exact all-vocabulary parity on these prompts.",
        fontsize=12,
        color=CORAL if invalid else TEAL,
    )
    fig.text(
        0.045,
        0.045,
        "Fresh process; the same compiled graphs run before and after installing "
        "the saved native weights. Compiler weight freezing disabled.",
        fontsize=10.5,
        color=MUTED,
    )
    save(fig, out, "fresh_confirmation_034")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=ROOT / "results/confirm-validation-038/fresh034"
    )
    parser.add_argument("--out", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report = collect(args.source)
    (args.out / "fresh_confirmation_034.json").write_text(json.dumps(report, indent=2) + "\n")
    draw(report, args.out)


if __name__ == "__main__":
    main()
