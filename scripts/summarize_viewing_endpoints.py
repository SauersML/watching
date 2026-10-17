"""Summarize fixed 1024/3072 endpoints without selecting checkpoints or fitting scores."""

import json

import numpy as np
from plot_viewing_comparison import ROOT, RUNS, digest, point, read

METRICS = ("both_correct", "individual_accuracy", "paired_log_odds_gap", "full_vocab_ce")
CONTINUATIONS = ROOT / "results/viewing-continuation-053-054"


def prompt_values(path, records):
    """One row per complete prompt pair, preserving matching across endpoints."""
    point(path, records, draws=2)  # Validate labels, actual argmax, log odds and stored summary.
    if (
        len({r["context_id"] for r in records}) != len(records)
        or len({r["context"] for r in records}) != len(records)
        or any("base_context_id" in r for r in records)
    ):
        raise ValueError("This bootstrap requires unique, ungrouped prompt contexts")
    pairs = {}
    for row in read(path)["rows"]:
        pairs.setdefault(row["index"], {})[row["condition_present"]] = row
    values = np.asarray(
        [
            [
                float(p[False]["greedy_correct"] and p[True]["greedy_correct"]),
                (p[False]["greedy_correct"] + p[True]["greedy_correct"]) / 2,
                p[True]["positive_minus_negative"] - p[False]["positive_minus_negative"],
                (p[False]["ce"] + p[True]["ce"]) / 2,
            ]
            for _, p in sorted(pairs.items())
        ],
        dtype=np.float64,
    )
    if not np.isfinite(values).all() or np.any(values[:, 3] < 0):
        raise ValueError("Invalid full-vocabulary cross-entropy or metric")
    return values


def paired_summary(before, after, draws=5000, seed=20261017):
    if before.shape != after.shape or before.ndim != 2 or before.shape[1] != len(METRICS):
        raise ValueError("Endpoints must contain matching complete prompt pairs")
    indices = np.random.default_rng(seed).integers(len(before), size=(draws, len(before)))
    result = {}
    for name, values in (("before", before), ("after", after), ("change", after - before)):
        sampled = values[indices].mean(axis=1)
        bounds = np.quantile(sampled, [0.025, 0.975], axis=0)
        result[name] = {
            metric: {"mean": float(values[:, j].mean()), "ci95": bounds[:, j].tolist()}
            for j, metric in enumerate(METRICS)
        }
    return result


def collect():
    result = {
        "runtime": "cached native tail, split_inference; not whole-model transfer",
        "steps": [1024, 3072],
        "bootstrap": {"draws": 5000, "seed": 20261017, "unit": "whole prompt pair"},
        "interpretation": (
            "Descriptive post-hoc endpoint comparison of one seed/build per arm. "
            "Historical test is exploratory. No checkpoint selection or score calibration. "
            "Intervals are pointwise; changes preserve before/after pairing. "
            "All-equal empirical intervals do not establish population certainty."
        ),
        "arms": {},
    }
    for key, directory in (("buffer", "buffer"), ("storage", "identity054r")):
        original = ROOT / "results" / RUNS[key][1]
        later = CONTINUATIONS / directory / "fit"
        records = read(original / "data.json")
        manifest = read(later / "manifest.json")
        if (
            read(later / "data.json") != records
            or manifest["starting_global_step"] != 1024
            or manifest["completed_steps"] != 2048
            or manifest["training_status"] != "complete"
            or manifest["source_recompiled"] is not False
            or manifest["source_hashes"]["manifest.json"] != digest(original / "manifest.json")
        ):
            raise ValueError("Continuation does not match original source/data")
        arm = {"splits": {}, "sources": {}}
        source_paths = [original / "data.json", original / "manifest.json", later / "manifest.json"]
        for split in ("train", "validation", "test"):
            old_path = original / f"{split}_1024_split_inference.json"
            new_path = later / f"{split}_2048_split_inference.json"
            if read(new_path)["summary"]["global_step"] != 3072:
                raise ValueError("Wrong final global step")
            old, new = (prompt_values(p, records[split]) for p in (old_path, new_path))
            arm["splits"][split] = dict(n_prompts=len(old), **paired_summary(old, new))
            source_paths += [old_path, new_path]
        arm["sources"] = {str(p.relative_to(ROOT)): digest(p) for p in source_paths}
        result["arms"][key] = arm
    return result


def markdown(report):
    lines = [
        "# Fixed endpoint comparison: 1,024 → 3,072 updates",
        "",
        "Full-vocabulary answers and loss, using cached-tail split inference. "
        "Brackets are pointwise 95% whole-prompt bootstrap intervals (5,000 draws). "
        "The test cohort is historical and exploratory; it is not used in the learning curves.",
        "",
    ]
    for split, title in (("validation", "Development"), ("test", "Historical test")):
        lines += [
            f"## {title}",
            "",
            "| Condition | Updates | Both answers correct | Yes/No gap, nats | "
            "Full-vocabulary CE |",
            "|---|---:|---:|---:|---:|",
        ]
        for arm, title in (("buffer", "Buffer copy"), ("storage", "Forced storage")):
            for stage, step in (("before", 1024), ("after", 3072)):
                metrics = report["arms"][arm]["splits"][split][stage]
                cells = []
                for name in ("both_correct", "paired_log_odds_gap", "full_vocab_ce"):
                    stat = metrics[name]
                    scale, precision = (100, 2) if name == "both_correct" else (1, 4)
                    suffix = "%" if name == "both_correct" else ""
                    values = [stat["mean"], *stat["ci95"]]
                    a, lo, hi = (f"{v * scale:.{precision}f}" for v in values)
                    cells.append(f"{a}{suffix} [{lo}, {hi}]")
                lines.append(f"| {title} | {step:,} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += [
        "The JSON includes training scores, individual-answer accuracy, and paired "
        "before/after change intervals. Accuracy and loss answer different questions: "
        "forced storage improves test correctness while worsening test cross-entropy. "
        "An all-zero or all-one empirical bootstrap interval is not population certainty.",
        "",
        "Reproduce: `PYTHONPATH=src python scripts/summarize_viewing_endpoints.py`.",
        "",
        "[Raw-source hashes and full statistics](endpoints.json)",
        "",
    ]
    return "\n".join(lines)


def main():
    report = collect()
    out = ROOT / "results/viewing-learning-comparison"
    (out / "endpoints.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "endpoints.md").write_text(markdown(report))


if __name__ == "__main__":
    main()
