"""Audit paired progress rows and render detector/preservation learning curves.

Bands resample question-family clusters where available, otherwise whole prompt
pairs. They are pointwise, conditional on one fit/runtime, and are not assumption
free. All-correct bootstrap bands do not prove perfect population performance.
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

YES, NO = 9454, 2753


def intervals(values, families, draws=5000, seed=20261018):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Metrics must be a finite nonempty matrix")
    labels = sorted(set(families)) if all(families) else []
    if len(labels) < 2:
        families = list(map(str, range(len(values))))
        labels = sorted(set(families))
        method = "whole-prompt paired percentile bootstrap"
    else:
        method = "question-family cluster percentile bootstrap"
    membership = np.array(
        [[family == label for family in families] for label in labels], dtype=float
    )
    random = np.random.default_rng(seed)
    counts = random.multinomial(len(labels), np.full(len(labels), 1 / len(labels)), size=draws)
    weights = counts @ membership
    sampled = (weights @ values) / weights.sum(axis=1)[:, None]
    return np.quantile(sampled, [0.025, 0.975], axis=0).T, method, len(labels)


def tie_diagnostics(pairs):
    """Supplement the unchanged argmax metric with exact Yes/No ties."""
    tied = {runtime: [] for runtime in ("trainer", "vllm_0")}
    correct_tied = {runtime: [] for runtime in tied}
    successes_with_ties = []
    for key, pair in pairs.items():
        for runtime in tied:
            row = pair[runtime]
            if row["yes_minus_no"] == 0:
                tied[runtime].append(key)
                if row["correct"]:
                    correct_tied[runtime].append(key)
        if all(row["correct"] for row in pair.values()) and any(
            row["yes_minus_no"] == 0 for row in pair.values()
        ):
            successes_with_ties.append(key)
    return {
        "exact_yes_no_ties_by_runtime": {key: len(value) for key, value in tied.items()},
        "argmax_correct_ties_by_runtime": {key: len(value) for key, value in correct_tied.items()},
        "both_correct_pairs_containing_a_tie": len(successes_with_ties),
        "tied_prompt_ids_by_runtime": tied,
        "both_correct_tied_prompt_ids": successes_with_ties,
        "scope": "Exact Yes/No ties only; other vocabulary ties cannot be reconstructed here.",
    }


def paired(path):
    artifact = json.loads(path.read_text())
    pairs = {}
    for row in artifact["rows"]:
        runtime = row["runtime"]
        if runtime not in ("trainer", "vllm_0"):
            raise ValueError("This analysis is for the single-engine preservation experiment")
        pair = pairs.setdefault(row["id"], {})
        if runtime in pair:
            raise ValueError("Duplicate condition")
        pair[runtime] = row
        target = YES if runtime == "trainer" else NO
        if row["target_token_id"] != target or row["correct"] != (row["top_token_id"] == target):
            raise ValueError("Target or stored correctness is inconsistent")
        if not np.isclose(
            row["yes_minus_no"], row["yes_logprob"] - row["no_logprob"], rtol=0, atol=1e-8
        ):
            raise ValueError("Log-odds disagree with raw normalized scores")
        correct_logprob = row["yes_logprob"] if target == YES else row["no_logprob"]
        if not np.isclose(row["correct_logprob"], correct_logprob, rtol=0, atol=1e-8):
            raise ValueError("Target log probability disagrees")
        if not np.isclose(row["correct_probability"], np.exp(correct_logprob), rtol=0, atol=1e-8):
            raise ValueError("Target probability disagrees")
    units, families = [], []
    for key in sorted(pairs):
        pair = pairs[key]
        if set(pair) != {"trainer", "vllm_0"}:
            raise ValueError("Incomplete prompt pair")
        hf, engine = pair["trainer"], pair["vllm_0"]
        if hf["question_family"] != engine["question_family"]:
            raise ValueError("Family metadata differ across paired conditions")
        units.append(
            [
                float(hf["correct"] and engine["correct"]),
                hf["yes_minus_no"] - engine["yes_minus_no"],
                -(hf["correct_logprob"] + engine["correct_logprob"]) / 2,
                (hf["correct_probability"] + engine["correct_probability"]) / 2,
                float(
                    hf["correct"]
                    and engine["correct"]
                    and hf["yes_minus_no"] > 0
                    and engine["yes_minus_no"] < 0
                ),
            ]
        )
        families.append(hf["question_family"])
    units = np.asarray(units)
    bounds, method, groups = intervals(units, families)
    metrics = {
        name: {"mean": float(units[:, i].mean()), "ci95": bounds[i].tolist()}
        for i, name in enumerate(
            [
                "both_correct",
                "paired_log_odds",
                "target_nll",
                "correct_probability",
                "both_correct_strict_yes_no_margin",
            ]
        )
    }
    summary = artifact["summary"]
    for key, actual in [
        ("both", metrics["both_correct"]["mean"]),
        ("paired_log_odds", metrics["paired_log_odds"]["mean"]),
    ]:
        if not np.isclose(summary[key][0], actual, rtol=0, atol=1e-8):
            raise ValueError(f"Raw rows disagree with saved {key}")
    if not np.isclose(summary["mean_target_nll"], metrics["target_nll"]["mean"], rtol=0, atol=1e-8):
        raise ValueError("Raw rows disagree with saved target loss")
    return {
        "step": artifact["step"],
        "split": artifact["split"],
        "n_prompts": len(pairs),
        "prompt_ids": sorted(pairs),
        "metrics": metrics,
        "tie_diagnostics": tie_diagnostics(pairs),
        "interval_method": method,
        "resampling_units": groups,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "file": str(path),
    }


def collect(root):
    points = [paired(p) for p in sorted((root / "train/evaluations").glob("*.json"))]
    log = []
    path = root / "train/train_log.jsonl"
    if path.exists():
        text = path.read_text()
        lines = text.splitlines()
        for index, line in enumerate(lines):
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                if index == len(lines) - 1 and not text.endswith("\n"):
                    continue  # only the unfinished final streaming line may be skipped
                raise
            if "dev" in item:
                log.append(item)
    chosen = (
        json.loads((root / "train/chosen.json").read_text())
        if (root / "train/chosen.json").exists()
        else None
    )
    return {
        "points": points,
        "aggregate_progress": log,
        "chosen": chosen,
        "scope": (
            "Cached native source states to eager learned tail; "
            "final whole-runtime scores are separate. Ordinary preservation compares the "
            "learned tail with a frozen native tail on each source cache, "
            "not full-model generation."
        ),
        "uncertainty": (
            "Pointwise cluster/prompt bootstrap; no seed/hardware uncertainty. "
            "Ordinary-chat aggregates have no recoverable confidence intervals."
        ),
    }


def render(report, output):
    if not report["points"]:
        return False
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
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.8))
    fig.subplots_adjust(left=0.075, right=0.97, top=0.79, bottom=0.23, hspace=0.5, wspace=0.3)
    latest = max(p["step"] for p in report["points"])
    fig.text(
        0.075,
        0.945,
        "Detection and ordinary-answer preservation",
        fontsize=25,
        weight="bold",
        color="#1D1D1F",
    )
    fig.text(
        0.075,
        0.895,
        f"Upper-layer training · evaluation update {latest:,} · frozen-source curves",
        color="#86868B",
        fontsize=13,
    )
    fig.text(
        0.075,
        0.85,
        "Development determines checkpoint selection; final serving evaluations are separate.",
        color="#86868B",
        fontsize=11,
    )
    colors = {"development": "#3E9A55", "training_diagnostic": "#A8754C"}
    names = {"development": "Development", "training_diagnostic": "Training content diagnostic"}
    for split in colors:
        points = sorted(
            [p for p in report["points"] if p["split"] == split], key=lambda p: p["step"]
        )
        for ax, metric in zip(axes[0], ["both_correct", "paired_log_odds"], strict=True):
            if not points:
                continue
            x = [p["step"] for p in points]
            y = [p["metrics"][metric]["mean"] for p in points]
            lo, hi = np.array([p["metrics"][metric]["ci95"] for p in points]).T
            ax.plot(
                x,
                y,
                color=colors[split],
                lw=2.6,
                marker="o",
                markersize=3,
                label=f"{names[split]} · {points[-1]['n_prompts']} prompts",
            )
            if len(points) == 1:
                ax.errorbar(
                    x,
                    y,
                    yerr=[np.asarray(y) - lo, hi - np.asarray(y)],
                    color=colors[split],
                    fmt="none",
                    capsize=3,
                )
            else:
                ax.fill_between(x, lo, hi, color=colors[split], alpha=0.13, linewidth=0)
    axes[0, 0].set_title("Both answers correct", loc="left", weight="bold")
    axes[0, 0].set_ylim(-0.02, 1.02)
    axes[0, 0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0, 1].set_title("Trainer − serving log-odds", loc="left", weight="bold")
    axes[0, 1].set_ylabel("Nats")
    axes[0, 1].axhline(0, color="#C7C7CC", lw=1)
    logs = report["aggregate_progress"]
    for runtime, label, color in [
        ("hf", "Trainer states", "#3E9A55"),
        ("vllm", "Serving states", "#A8754C"),
    ]:
        points = [p for p in logs if runtime in p.get("ordinary_dev", {})]
        axes[1, 0].plot(
            [p["step"] for p in points],
            [p["ordinary_dev"][runtime]["first_token_same"] for p in points],
            color=color,
            lw=2.6,
            marker="o",
            markersize=3,
            label=label,
        )
    axes[1, 0].set_title("Ordinary chats: first token preserved", loc="left", weight="bold")
    axes[1, 0].set_ylim(-0.02, 1.02)
    axes[1, 0].yaxis.set_major_formatter(PercentFormatter(1))
    points = [p for p in logs if "selection_objective" in p]
    axes[1, 1].plot(
        [p["step"] for p in points],
        [p["selection_objective"] for p in points],
        color="#3E9A55",
        lw=2.6,
        marker="o",
        markersize=3,
    )
    axes[1, 1].set_title("Development objective · lower is better", loc="left", weight="bold")
    axes[1, 1].set_ylabel("Detector CE + preservation KL")
    for ax in axes.flat:
        ax.set_xlabel("Optimizer updates")
        if latest == 0:
            ax.set_xlim(-0.03, 1.03)
            ax.set_xticks([0])
    axes[0, 0].legend(fontsize=9, loc="best", frameon=False)
    axes[1, 0].legend(fontsize=9, loc="lower left", frameon=False)
    fig.text(
        0.075,
        0.118,
        "Ordinary reference: frozen native tail on each source cache; not whole-model generation.",
        fontsize=10,
        color="#86868B",
    )
    fig.text(
        0.075,
        0.09,
        (
            "Top bands: pointwise 95% family-cluster bootstrap "
            "(paired prompts if family metadata are absent)."
        ),
        fontsize=10,
        color="#86868B",
    )
    fig.text(
        0.075,
        0.062,
        (
            "Family clusters do not cover all shared-context dependence. "
            "Ordinary-chat aggregate curves have no confidence bands."
        ),
        fontsize=10,
        color="#86868B",
    )
    fig.text(
        0.075,
        0.034,
        (
            "One fit/runtime; an all-correct band does not prove perfect accuracy. "
            "Fresh prompts are evaluated only after selection."
        ),
        fontsize=10,
        color="#86868B",
    )
    for extension in ["png", "pdf", "svg"]:
        fig.savefig(output.with_suffix("." + extension), dpi=180, facecolor="white")
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = collect(args.run)
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {"completed_raw_points": len(report["points"]), "rendered": render(report, args.out)}
        )
    )


if __name__ == "__main__":
    main()
