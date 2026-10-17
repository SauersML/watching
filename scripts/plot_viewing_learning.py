"""Plot matched-034 viewing training with prompt-paired, pointwise bootstrap bands."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

TITLES = {
    "reference": "Retained-reference activation probe",
    "buffer": "Buffer-copy activation probe",
    "identity": "Opaque identity materialization",
}
COLORS = {"train": "#3E9A55", "validation": "#D2432F"}
LABELS = {"train": "Training", "validation": "Development"}


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score_vectors(saved, records):
    pairs = {}
    for row in saved["rows"]:
        index, condition = row["index"], row["condition_present"]
        if type(index) is not int or not 0 <= index < len(records) or type(condition) is not bool:
            raise ValueError("Invalid prompt index or condition")
        record = records[index]
        if any(row[k] != record[k] for k in ("context_id", "question_family")):
            raise ValueError("Score metadata differs from the saved prompt")
        if row["task"] != "detection" or (row["answer_negative"], row["answer_positive"]) != (
            "No",
            "Yes",
        ):
            raise ValueError("Expected the matched No/Yes detection task")
        target = row["positive_token_id"] if condition else row["negative_token_id"]
        if row["target_token_id"] != target or row["greedy_correct"] != (
            row["top_token_id"] == target
        ):
            raise ValueError("Full-vocabulary top-token correctness is inconsistent")
        logp = row["logp_positive"] if condition else row["logp_negative"]
        if not np.isclose(row["ce"], -logp, atol=1e-6, rtol=0) or not np.isclose(
            row["full_vocab_correct_probability"], np.exp(logp), atol=2e-7, rtol=2e-6
        ):
            raise ValueError("Saved target probability/loss is inconsistent")
        odds = row["logp_positive"] - row["logp_negative"]
        if not np.isfinite(odds) or not np.isclose(
            row["positive_minus_negative"], odds, atol=2e-5, rtol=0
        ):
            raise ValueError("Saved log-odds are inconsistent")
        pair = pairs.setdefault(index, {})
        if condition in pair:
            raise ValueError("Duplicate condition")
        pair[condition] = row
    if set(pairs) != set(range(len(records))) or any(
        set(p) != {False, True} for p in pairs.values()
    ):
        raise ValueError("Every prompt must have exactly two conditions")
    return np.array(
        [
            [
                float(pair[False]["greedy_correct"] and pair[True]["greedy_correct"]),
                pair[True]["positive_minus_negative"] - pair[False]["positive_minus_negative"],
            ]
            for _, pair in sorted(pairs.items())
        ]
    )


def collect(directory, draws=5000, seed=20261017):
    directory = Path(directory)
    data, manifest = read(directory / "data.json"), read(directory / "manifest.json")
    mode = manifest["viewing_mode"]
    if mode not in TITLES or manifest["primary_path"] != "split_inference":
        raise ValueError("Expected a viewing run with split inference")
    if manifest.get("starting_global_step", 0) != 0 or draws < 100:
        raise ValueError("Only base-initialized runs and at least 100 draws are supported")
    seen_ids, seen_texts = set(), set()
    for records in data.values():
        for record in records:
            identity, context = record["context_id"], record["context"]
            if "base_context_id" in record or identity in seen_ids or context in seen_texts:
                raise ValueError(
                    "Requires unique 034 contexts; grouped/repeated contexts unsupported"
                )
            seen_ids.add(identity)
            seen_texts.add(context)
    report = {
        "viewing_mode": mode,
        "training_status": manifest.get("training_status", "in_progress"),
        "completed_steps": manifest.get("completed_steps", 0),
        "seed": manifest["seed"],
        "draws": draws,
        "bootstrap_seed": seed,
        "interval_scope": "Pointwise paired-prompt bootstrap; one seed and compiler build",
        "all_correct_caveat": "All-correct empirical bootstrap intervals collapse; not certainty",
        "data_sha256": digest(directory / "data.json"),
        "manifest_sha256": digest(directory / "manifest.json"),
        "curves": {},
    }
    equality_path = directory / "learner_input_equality.json"
    equality = read(equality_path)
    report["input_equality_sha256"] = digest(equality_path)
    report["both_correct_ceilings"] = {}
    for split in LABELS:
        records = data[split]
        if len(records) < 2:
            raise ValueError("At least two prompts required")
        counts = equality["splits"][split]
        identical = counts["bit_identical_pairs"]
        if counts["pairs"] != len(records) or not 0 <= identical <= len(records):
            raise ValueError("Input equality counts disagree with saved data")
        report["both_correct_ceilings"][split] = 1 - identical / len(records)
        indices = np.random.default_rng(seed).integers(len(records), size=(draws, len(records)))
        points = []
        for path in directory.glob(f"{split}_*_split_inference.json"):
            saved = read(path)
            summary = saved["summary"]
            step = summary["step"]
            if (
                type(step) is not int
                or step < 0
                or summary["split"] != split
                or summary["path"] != "split_inference"
                or path.name != f"{split}_{step}_split_inference.json"
            ):
                raise ValueError("Score filename and metadata disagree")
            vectors = score_vectors(saved, records)
            if vectors[:, 0].mean() > report["both_correct_ceilings"][split] + 1e-12:
                raise ValueError("Accuracy exceeds the deterministic identical-input ceiling")
            sampled = vectors[indices].mean(axis=1)
            points.append(
                {
                    "step": step,
                    "n_prompts": len(records),
                    "source": str(path),
                    "sha256": digest(path),
                    "mean": vectors.mean(axis=0).tolist(),
                    "ci95": np.quantile(sampled, [0.025, 0.975], axis=0).T.tolist(),
                }
            )
        if not points:
            raise ValueError(f"No completed {split} evaluations yet")
        report["curves"][split] = sorted(points, key=lambda p: p["step"])
    if report["training_status"] == "skipped_identical_learner_inputs" and any(
        p["step"] != 0 for points in report["curves"].values() for p in points
    ):
        raise ValueError("Skipped run contains training updates")
    return report


def draw(report, output):
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
            "font.size": 12,
            "text.color": "#1D1D1F",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.edgecolor": "#C7C7CC",
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,
            "grid.color": "#E5E5EA",
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    for split, points in report["curves"].items():
        x = [p["step"] for p in points]
        for metric, ax in enumerate(axes):
            values = np.array([p["mean"][metric] for p in points])
            bounds = np.array([p["ci95"][metric] for p in points])
            ax.fill_between(x, bounds[:, 0], bounds[:, 1], color=COLORS[split], alpha=0.13, lw=0)
            ax.plot(
                x,
                values,
                "o-",
                color=COLORS[split],
                lw=2,
                ms=4,
                label=f"{LABELS[split]} · {points[0]['n_prompts']} prompts",
            )
    for split, ceiling in report["both_correct_ceilings"].items():
        if ceiling < 1:
            axes[0].axhline(ceiling, color=COLORS[split], ls="--", lw=1.2, alpha=0.8)
            axes[0].text(
                0.99,
                ceiling + 0.012,
                f"{LABELS[split]} ceiling {ceiling:.1%}",
                transform=axes[0].get_yaxis_transform(),
                ha="right",
                fontsize=9,
                color=COLORS[split],
            )
    axes[0].set(
        title="Both conditions answered correctly",
        ylabel="Prompts with both top tokens correct",
        ylim=(-0.03, 1.03),
    )
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set(
        title=(
            "Materialized − unmaterialized"
            if report["viewing_mode"] == "identity"
            else "Probe − no-probe log-odds"
        ),
        ylabel="Yes/No separation (nats)",
    )
    axes[1].axhline(0, color="#86868B", lw=0.8)
    for ax in axes:
        ax.set_xlabel("Optimizer updates")
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="lower left",
        bbox_to_anchor=(0.075, 0.115),
        ncol=2,
        frameon=False,
        fontsize=10,
    )
    skipped = report["training_status"] == "skipped_identical_learner_inputs"
    subtitle = (
        "Training skipped: paired learner inputs were bit-identical"
        if skipped
        else "Fixed native output head · training and development · split inference"
    )
    fig.suptitle(TITLES[report["viewing_mode"]], x=0.08, ha="left", fontsize=19, weight="bold")
    fig.text(0.08, 0.885, subtitle, fontsize=11, color="#86868B")
    if any(v < 1 for v in report["both_correct_ceilings"].values()):
        fig.text(
            0.08,
            0.083,
            "Dashed ceilings: identical paired inputs cannot give both answers correctly "
            "(this fixed source cache).",
            fontsize=9,
            color="#86868B",
        )
    fig.text(
        0.08,
        0.055,
        f"{report['draws']:,} whole-prompt resamples · pointwise 95% bands · "
        "one training seed and compiler build",
        fontsize=9,
        color="#86868B",
    )
    fig.text(
        0.08,
        0.025,
        "All-correct bands collapse; they do not establish population certainty. "
        "Test scores are not plotted.",
        fontsize=9,
        color="#86868B",
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.26, top=0.77, wspace=0.28)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output.with_suffix("." + extension), dpi=180, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fit", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    report = collect(args.fit)
    draw(report, args.out)
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
