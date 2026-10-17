"""Audit and plot fresh joint confirmation, separately for both native-answer tasks."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from make_paper_figures import CORAL, GRAY, INK, MUTED, TEAL, save, style
from matplotlib.ticker import PercentFormatter
from scipy.stats import beta

from watching.validate_tail import paired_summary


def collect(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    records = json.loads((directory / "data.json").read_text())[manifest["split"]]
    if manifest["task"] != "joint":
        raise ValueError("This figure requires a joint confirmation")
    report = {
        "checkpoint_sha256": manifest["tail_sha256"],
        "data_sha256": manifest["data_sha256"],
        "runtime": manifest["runtime"],
        "tasks": {},
        "source_files": {},
    }
    for stage in ("base", "trained"):
        path = directory / f"{stage}_reference.json"
        saved = json.loads(path.read_text())
        report["source_files"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows = saved["rows"]
        if {(r["index"], r["capture"]) for r in rows} != {
            (i, c) for i in range(len(records)) for c in (False, True)
        } or len(rows) != 2 * len(records):
            raise ValueError("Incomplete or duplicated prompt conditions")
        for row in rows:
            record = records[row["index"]]
            word = record["answer_positive"] if row["capture"] else record["answer_negative"]
            target = manifest["labels"][word]
            if row["task"] != record["task"] or row["context_id"] != record["context_id"]:
                raise ValueError("Scores do not match frozen task/context identities")
            if row["target_token_id"] != target or row["greedy_correct"] != (
                row["greedy_token_id"] == target
            ):
                raise ValueError("Native token score uses an incorrect answer label")
            if not np.isclose(row["yes_minus_no"], row["logp_yes"] - row["logp_no"], atol=2e-6):
                raise ValueError("Logprobs do not reproduce answer log odds")
            if not np.isclose(np.exp(-row["ce"]), row["full_vocab_correct_probability"]):
                raise ValueError("Full-vocabulary probability and loss disagree")
        for task in ("detection", "animal"):
            selected = [row for row in rows if row["task"] == task]
            summary = paired_summary(selected, manifest["seed"])
            for key in ("greedy_accuracy", "paired_log_odds_shift", "mean_ce"):
                if not np.isclose(summary[key], saved["tasks"][task][key], atol=1e-12, rtol=1e-12):
                    raise ValueError(f"Saved {task} {key} does not reproduce")
            pairs = {}
            for row in selected:
                pairs.setdefault(row["index"], []).append(row)
            n = len(pairs)
            both = sum(all(r["greedy_correct"] for r in pair) for pair in pairs.values())
            summary.update(
                n_answers=len(selected),
                n_correct=sum(row["greedy_correct"] for row in selected),
                n_both_correct=both,
                both_correct_iid_binomial_ci95=[
                    float(beta.ppf(0.025, both, n - both + 1)) if both else 0.0,
                    float(beta.ppf(0.975, both + 1, n - both)) if both < n else 1.0,
                ],
            )
            report["tasks"].setdefault(task, {})[stage] = summary
        repeats = json.loads((directory / f"{stage}_repeat.json").read_text())
        if not all(row["bit_identical"] for row in repeats.values()):
            raise ValueError("Fresh validation repeat gate failed")
    return report


def draw(report, out):
    style()
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.6))
    fig.subplots_adjust(left=0.085, right=0.97, bottom=0.22, top=0.70, hspace=0.80, wspace=0.35)
    fig.text(
        0.04,
        0.95,
        "Fresh questions: detection and animal choice",
        fontsize=25,
        weight="bold",
        color=INK,
    )
    fig.text(
        0.04,
        0.90,
        "Fixed final checkpoint · new process · ordinary compiled inference",
        fontsize=14,
        color=MUTED,
    )
    fig.text(
        0.04,
        0.855,
        "64 prompts per task, each run with and without the reference probe",
        fontsize=13,
        color=MUTED,
    )
    metrics = [
        ("greedy_accuracy", "Actual next-token accuracy"),
        ("mean_full_vocab_correct_probability", "Correct-token probability"),
        ("paired_log_odds_shift", "Condition log-odds separation"),
    ]
    for row_index, (task, title) in enumerate(
        [("detection", "Detection · No / Yes"), ("animal", "Animal choice · cat / owl")]
    ):
        for column, (metric, label) in enumerate(metrics):
            ax = axes[row_index, column]
            for stage, y, color in [("base", 1, CORAL), ("trained", 0, TEAL)]:
                result = report["tasks"][task][stage]
                value = result[metric]
                key = metric + "_ci95"
                lo, hi = result[key]
                ax.errorbar(
                    value,
                    y,
                    xerr=[[max(0, value - lo)], [max(0, hi - value)]],
                    fmt="o",
                    ms=9,
                    color=color,
                    capsize=5,
                    lw=2,
                )
                text = f"{value:.1%}" if column < 2 else f"{value:+.2f} nats"
                if column == 0:
                    text += f" · {result['n_correct']}/{result['n_answers']}"
                ax.text(
                    0.02,
                    y - 0.28,
                    text,
                    transform=ax.get_yaxis_transform(),
                    fontsize=12,
                    color=color,
                    weight="bold",
                )
            ax.set_ylim(-0.55, 1.45)
            ax.set_yticks([1, 0], ["Original", "Trained"] if column == 0 else [])
            ax.tick_params(axis="y", length=0)
            ax.spines["left"].set_visible(False)
            ax.grid(axis="x", color=GRAY)
            ax.set_axisbelow(True)
            ax.set_title(label, loc="left", fontsize=13, pad=13)
            if column < 2:
                ax.set_xlim(-0.04, 1.06)
                ax.set_xticks([0, 0.5, 1])
                ax.xaxis.set_major_formatter(PercentFormatter(1))
            else:
                ax.axvline(0, color=MUTED, ls="--", lw=1)
                ax.set_xlabel("Nats", fontsize=11)
        axes[row_index, 0].text(
            0,
            1.47,
            title,
            transform=axes[row_index, 0].transAxes,
            fontsize=16,
            color=INK,
            weight="bold",
        )
    fig.text(
        0.04,
        0.13,
        "Bars: pointwise 95% intervals, 5,000 whole-prompt paired bootstrap draws.",
        fontsize=12,
        color=MUTED,
    )
    fig.text(
        0.04,
        0.085,
        "One checkpoint and an authored prompt bank; no training-seed or hardware uncertainty.",
        fontsize=11,
        color=MUTED,
    )
    fig.text(
        0.04,
        0.045,
        "All-correct observed answers do not imply perfect population accuracy. "
        "No checkpoint selection.",
        fontsize=11,
        color=MUTED,
    )
    save(fig, out.parent, out.name)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = collect(args.directory)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    draw(report, args.out)


if __name__ == "__main__":
    main()
