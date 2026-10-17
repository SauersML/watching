"""Compare four fixed-source learning runs; never merge compiler builds or test scores."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "reference": ("Reference probe · 034", "two-block-scaled-034/fit", "#3E9A55", "o"),
    "null": ("Identical-input null · 045", "detector-null-045/fit", "#86868B", "x"),
    "buffer": ("Buffer-copy probe · 051", "viewing-learning-051-052/buffer/fit", "#287DA6", "s"),
    "storage": (
        "Forced-storage control · 052",
        "viewing-learning-051-052/identity/fit",
        "#9667B0",
        "o",
    ),
}


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def point(path, records, draws=5000):
    saved = read(path)
    step, split = saved["summary"]["step"], saved["summary"]["split"]
    if (
        path.name != f"{split}_{step}_split_inference.json"
        or saved["summary"]["path"] != "split_inference"
    ):
        raise ValueError("Runtime or step differs from filename")
    pairs = {}
    for row in saved["rows"]:
        index, condition = row["index"], row["condition_present"]
        if type(condition) is not bool or type(index) is not int or not 0 <= index < len(records):
            raise ValueError("Invalid condition/index")
        if "context_id" in row and row["context_id"] != records[index]["context_id"]:
            raise ValueError("Score context differs from saved input")
        if type(row["greedy_correct"]) is not bool:
            raise ValueError("Expected actual vocabulary-choice correctness")
        if "top_token_id" in row and row["greedy_correct"] != (
            row["top_token_id"] == row["target_token_id"]
        ):
            raise ValueError("Top-token correctness mismatch")
        if not np.isclose(
            row["positive_minus_negative"],
            row["logp_positive"] - row["logp_negative"],
            rtol=0,
            atol=2e-5,
        ):
            raise ValueError("Log-odds mismatch")
        pair = pairs.setdefault(index, {})
        if condition in pair:
            raise ValueError("Duplicate prompt condition")
        pair[condition] = row
    if set(pairs) != set(range(len(records))) or any(
        set(p) != {False, True} for p in pairs.values()
    ):
        raise ValueError("Incomplete prompt pairs")
    values = np.array(
        [
            [
                float(p[False]["greedy_correct"] and p[True]["greedy_correct"]),
                p[True]["positive_minus_negative"] - p[False]["positive_minus_negative"],
            ]
            for _, p in sorted(pairs.items())
        ]
    )
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite scores")
    if not np.isclose(
        values[:, 1].mean(), saved["summary"]["paired_log_odds_shift"], atol=1e-12, rtol=0
    ):
        raise ValueError("Raw gap disagrees with stored summary")
    sampled = values[
        np.random.default_rng(20261017).integers(len(values), size=(draws, len(values)))
    ].mean(1)
    return dict(
        step=step,
        n_prompts=len(values),
        mean=values.mean(0).tolist(),
        ci95=np.quantile(sampled, [0.025, 0.975], axis=0).T.tolist(),
        source=str(path),
        sha256=digest(path),
    )


def collect(root=ROOT, split="validation", continuations=None):
    reference = root / "results" / RUNS["reference"][1]
    original = read(reference / "data.json")
    manifests = {}
    curves = {}
    for key, (_, location, _, _) in RUNS.items():
        fit = root / "results" / location
        data, manifest = read(fit / "data.json"), read(fit / "manifest.json")
        if data != original:
            raise ValueError("All four runs must use the exact same prompts and order")
        records = data[split]
        if len({r["context_id"] for r in records}) != len(records) or len(
            {r["context"] for r in records}
        ) != len(records):
            raise ValueError("Prompt bootstrap requires unique contexts")
        if any("base_context_id" in r for r in records):
            raise ValueError("Grouped contexts require a different bootstrap")
        manifests[key] = manifest
        curves[key] = sorted(
            [point(p, records) for p in fit.glob(f"{split}_*_split_inference.json")],
            key=lambda p: p["step"],
        )
        if not curves[key]:
            raise ValueError("Missing evaluation curve")
    for field in (
        "checkpoint",
        "seed",
        "steps",
        "lr",
        "length",
        "prompt_batch_size",
        "tail_layers",
        "head_sha256_before",
    ):
        if len({str(m[field]) for m in manifests.values()}) != 1:
            raise ValueError(f"Different training recipe: {field}")
    if any(p["mean"] != [0.0, 0.0] for p in curves["null"]):
        raise ValueError("Identical-input null violates exact paired constraints")
    equality = read(root / "results" / RUNS["buffer"][1] / "learner_input_equality.json")["splits"][
        split
    ]
    restarts = {}
    for key, directory in (continuations or {}).items():
        if key not in ("buffer", "storage"):
            raise ValueError("Continuation must be buffer or storage")
        directory = Path(directory)
        later = read(directory / "manifest.json")
        original_fit = root / "results" / RUNS[key][1]
        if (
            read(directory / "data.json") != original
            or later.get("source_recompiled") is not False
            or later.get("optimizer_state_restored") is not False
            or later.get("starting_global_step") != 1024
            or later.get("source_hashes", {}).get("manifest.json")
            != digest(original_fit / "manifest.json")
        ):
            raise ValueError("Continuation source, initialization or runtime mismatch")
        for field in ("seed", "lr", "length", "prompt_batch_size", "head_sha256_before"):
            if later[field] != manifests[key][field]:
                raise ValueError("Continuation training recipe changed")
        baseline = read(directory / f"{split}_0_split_inference.json")
        previous = read(original_fit / f"{split}_1024_split_inference.json")

        def indexed(rows):
            return {(r["index"], r["condition_present"]): r for r in rows}

        old_rows, new_rows = indexed(previous["rows"]), indexed(baseline["rows"])
        if set(old_rows) != set(new_rows) or any(
            old_rows[k][field] != new_rows[k][field]
            for k in old_rows
            for field in (
                "positive_minus_negative",
                "logp_positive",
                "logp_negative",
                "top_token_id",
                "greedy_correct",
            )
        ):
            raise ValueError("Continuation baseline is not an exact answer-score replay")
        appended = []
        for path in directory.glob(f"{split}_*_split_inference.json"):
            result = point(path, original[split])
            summary = read(path)["summary"]
            if summary["global_step"] != 1024 + result["step"]:
                raise ValueError("Continuation global-step counter mismatch")
            if result["step"] > 0:
                result["local_step"] = result["step"]
                result["step"] += 1024
                appended.append(result)
        curves[key] += sorted(appended, key=lambda p: p["step"])
        restarts[key] = dict(
            global_step=1024,
            optimizer="fresh Adam",
            source=str(directory),
            manifest_sha256=digest(directory / "manifest.json"),
        )
    return dict(
        restarts=restarts,
        split=split,
        curves=curves,
        n_prompts=len(original[split]),
        buffer_both_correct_ceiling=1 - equality["bit_identical_pairs"] / equality["pairs"],
        source_manifests={k: digest(root / "results" / RUNS[k][1] / "manifest.json") for k in RUNS},
        uncertainty="5000 paired-prompt bootstrap draws; pointwise; one seed/build per run",
        caveat="Historical builds differ; matching the recipe does not match numerical states.",
        legacy_scores="034 records argmax correctness without explicit top-token IDs.",
    )


def draw(report, out):
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
            "font.size": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": "#E5E5EA",
            "axes.axisbelow": True,
            "axes.edgecolor": "#C7C7CC",
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 6.1))
    for key in ("reference", "storage", "buffer", "null"):
        label, _, color, marker = RUNS[key]
        if key in report.get("restarts", {}):
            label += " + " + ("053" if key == "buffer" else "054")
        points = report["curves"][key]
        x = [p["step"] for p in points]
        for i, ax in enumerate(axes):
            y = [p["mean"][i] for p in points]
            ci = np.array([p["ci95"][i] for p in points])
            ax.fill_between(x, ci[:, 0], ci[:, 1], color=color, alpha=0.12, lw=0)
            ax.plot(
                x,
                y,
                color=color,
                marker=marker,
                ms=4,
                lw=2,
                label=label,
                linestyle="--" if key == "null" else "-",
            )
    axes[0].set(
        title="Both runs answered correctly",
        ylabel="Prompts correct in both runs",
        ylim=(-0.04, 1.04),
    )
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    ceiling = report["buffer_both_correct_ceiling"]
    axes[0].axhline(ceiling, color=RUNS["buffer"][2], ls=":", lw=1)
    axes[0].text(
        0.99,
        ceiling + 0.015,
        f"Buffer: at most {round(ceiling * report['n_prompts'])}/{report['n_prompts']} prompts",
        ha="right",
        fontsize=9,
        transform=axes[0].get_yaxis_transform(),
        color=RUNS["buffer"][2],
    )
    axes[1].set(title="Condition log-odds separation", ylabel="Yes/No gap (nats)")
    axes[1].axhline(0, color="#86868B", lw=0.8)
    for ax in axes:
        ax.set_xlabel("Total optimizer updates")
        if report.get("restarts"):
            ax.axvline(1024, color="#86868B", ls=":", lw=1)
            ax.text(
                1035,
                0.91,
                "Fresh Adam",
                transform=ax.get_xaxis_transform(),
                ha="left",
                fontsize=9,
                color="#86868B",
            )
        else:
            ax.set_xticks([0, 256, 512, 768, 1024])
    label = "Development" if report["split"] == "validation" else "Training"
    fig.suptitle(
        "Which numerical signal can the model learn?",
        x=0.08,
        ha="left",
        fontsize=19,
        weight="bold",
    )
    fig.text(
        0.08,
        0.89,
        f"{label} · same {report['n_prompts']} prompt pairs · fixed native head · split inference",
        fontsize=11,
        color="#86868B",
    )
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="lower left",
        bbox_to_anchor=(0.075, 0.12),
        ncol=2,
        frameon=False,
        fontsize=11,
    )
    fig.text(
        0.08,
        0.075,
        "Same recipe, different compiled source instances. Null 045 uses the exact 034 cache.",
        fontsize=9,
        color="#86868B",
    )
    fig.text(
        0.08,
        0.045,
        "5,000 paired-prompt resamples · pointwise 95% bands · one seed/build; "
        "zero width is not certainty.",
        fontsize=9,
        color="#86868B",
    )
    fig.text(
        0.08,
        0.015,
        "Same prompt: No without + Yes with. "
        f"The {round((1 - ceiling) * report['n_prompts'])} identical buffer pairs "
        "cannot give both answers.",
        fontsize=9,
        color="#86868B",
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.30, top=0.78, wspace=0.28)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out.with_suffix("." + ext), dpi=180, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "results/viewing-learning-comparison/development"
    )
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--buffer-continuation", type=Path)
    parser.add_argument("--storage-continuation", type=Path)
    args = parser.parse_args()
    continuations = {
        k: v
        for k, v in (("buffer", args.buffer_continuation), ("storage", args.storage_continuation))
        if v is not None
    }
    report = collect(split=args.split, continuations=continuations)
    draw(report, args.out)
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
