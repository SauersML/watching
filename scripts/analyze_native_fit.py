"""Analyze a native answer-row fit, preserving prompt pairs across checkpoints."""

import argparse
import json
import math
import random
from pathlib import Path

from watching.artifacts import write_json
from watching.compare import compare_scores


def read_rows(path, *, split, records=None):
    record = json.loads(path.read_text())
    if record["summary"]["split"] != split:
        raise ValueError(f"Split metadata disagrees with filename: {path}")
    rows = []
    for row in record["rows"]:
        condition = row.get("condition_present", row.get("capture"))
        if type(condition) is not bool:
            raise ValueError(f"Missing boolean execution condition: {path}")
        enriched = {**row, "capture": condition}
        if records is not None:
            index = row["index"]
            if type(index) is not int or not 0 <= index < len(records):
                raise ValueError(f"Prompt index is outside saved data: {path}")
            family = records[index].get("family")
            if family is not None:
                if row.get("family", family) != family:
                    raise ValueError(f"Score and prompt family disagree: {path}")
                enriched["family"] = family
        rows.append(enriched)
    return rows


def binary_loss(rows, *, seed=20261016, draws=5000):
    """Stable binary log loss, bootstrapped as one mean loss per whole prompt."""
    pairs = {}
    for row in rows:
        score = float(row["yes_minus_no"])
        if not math.isfinite(score):
            raise ValueError("Scores must be finite")
        signed = -score if row["capture"] else score
        loss = max(signed, 0) + math.log1p(math.exp(-abs(signed)))
        pair = pairs.setdefault(row["index"], {})
        if row["capture"] in pair:
            raise ValueError("Duplicate condition within prompt")
        pair[row["capture"]] = loss
    if not pairs or any(set(p) != {False, True} for p in pairs.values()):
        raise ValueError("Every prompt needs both conditions")
    means = [(p[False] + p[True]) / 2 for p in pairs.values()]
    rng = random.Random(seed)
    boot = sorted(sum(rng.choices(means, k=len(means))) / len(means) for _ in range(draws))
    return {
        "mean": sum(means) / len(means),
        "ci95": [boot[int(0.025 * draws)], boot[int(0.975 * draws) - 1]]
        if len(means) > 1
        else None,
    }


def summarize(before, after, *, include_families=True):
    result = compare_scores(before, after, seed=20261016, draws=5000)
    zeros = [{**row, "yes_minus_no": 0.0} for row in after]
    result["endpoint_ci95"] = compare_scores(zeros, after, seed=20261016)["change_ci95"]
    result["binary_ce"] = binary_loss(after)
    if include_families and any("family" in row for row in before + after):
        result["family_analysis"] = summarize_families(before, after)
    return result


def summarize_families(before, after, *, seed=20261016, draws=5000):
    """One-stage cluster bootstrap of equally weighted whole-family means.

    Families are sampled with replacement; every prompt, both conditions, and
    both checkpoints within each selected family remain together. This estimates
    variation across the observed family units, not across new training runs or
    an unrestricted universe of possible OOD tasks. No within-family resampling
    is added, so this is explicitly a cluster, not a two-stage bootstrap.
    """
    if draws < 100:
        raise ValueError("Use at least 100 bootstrap draws")
    assignments = {}
    for row in before + after:
        family = row.get("family")
        if not isinstance(family, str) or not family.strip():
            raise ValueError("Every prompt needs a nonempty string family")
        index = row["index"]
        if assignments.setdefault(index, family) != family:
            raise ValueError("Prompt family changed across conditions or checkpoints")
    families = {}
    for family in sorted(set(assignments.values())):
        families[family] = summarize(
            [row for row in before if row["family"] == family],
            [row for row in after if row["family"] == family],
            include_families=False,
        )
    if not families:
        raise ValueError("Need at least one family")
    # Give each family equal weight, rather than allowing larger families to
    # silently dominate the estimated performance on a typical family.
    values = [
        [entry["change_in_separation"], entry["final_separation"], entry["binary_ce"]["mean"]]
        for entry in families.values()
    ]
    count = len(values)
    rng = random.Random(seed)
    boot = [[], [], []]
    for _ in range(draws):
        sampled = rng.choices(values, k=count)
        for metric in range(3):
            boot[metric].append(sum(row[metric] for row in sampled) / count)
    metrics = {}
    for index, name in enumerate(("change_in_separation", "final_separation", "binary_ce")):
        ordered = sorted(boot[index])
        metrics[name] = {
            "mean": sum(row[index] for row in values) / count,
            "ci95": [ordered[int(0.025 * draws)], ordered[int(0.975 * draws) - 1]]
            if count > 1
            else None,
        }
    return {
        "n_families": count,
        "families": families,
        "cluster_bootstrap": {
            "method": "One-stage whole-family bootstrap; all member prompt pairs retained",
            "estimand": "Equal-weight mean over families; not weighted by family prompt count",
            "bootstrap_seed": seed,
            "bootstrap_draws": draws,
            "limits": "Conditional on one fit; assumes family units are exchangeable; "
            "does not establish transfer to arbitrary new task families",
            **metrics,
        },
    }


def score_agreement(before, after):
    """Describe fresh compilation numerics without conflating survival and equality."""
    left = {(row["index"], row["capture"]): row for row in before}
    right = {(row["index"], row["capture"]): row for row in after}
    if left.keys() != right.keys():
        raise ValueError("Fresh scores must match final prompt-condition pairs")
    differences = [abs(left[key]["yes_minus_no"] - right[key]["yes_minus_no"]) for key in left]
    answer_fields = ("logp_yes", "logp_no")
    have_answer_probs = all(all(field in row for field in answer_fields) for row in before + after)
    return {
        "all_log_odds_identical": all(value == 0 for value in differences),
        "max_log_odds_change": max(differences),
        "n_changed_log_odds": sum(value != 0 for value in differences),
        "all_answer_logprobs_identical": all(
            left[key][field] == right[key][field] for key in left for field in answer_fields
        )
        if have_answer_probs
        else None,
    }


def analyze_fit(directory, *, fresh=None, phase="many"):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    data_path = directory / "data.json"
    data = json.loads(data_path.read_text()) if data_path.exists() else {}
    if fresh is not None:
        fresh = Path(fresh)
        original_data, fresh_data = directory / "data.json", fresh / "data.json"
        if not original_data.exists() or not fresh_data.exists():
            raise ValueError("Fresh comparison requires both saved prompt data files")
        if json.loads(original_data.read_text()) != json.loads(fresh_data.read_text()):
            raise ValueError("Fresh evaluation uses different prompt records")
        fresh_manifest = json.loads((fresh / "manifest.json").read_text())
        for field in (
            "condition",
            "layer",
            "length",
            "labels",
            "answer_order",
            "log_odds_definition",
        ):
            if manifest.get(field) != fresh_manifest.get(field):
                raise ValueError(f"Fresh evaluation changes {field}")
    report = {
        "source": str(directory),
        "phase": phase,
        "manifest": manifest,
        "condition": manifest.get("condition", "execution"),
        "answer_order": manifest.get("answer_order", ["No", "Yes"]),
        "method": "Native output rows ("
        + "/".join(manifest.get("answer_order", ["No", "Yes"]))
        + "); all internal weights fixed",
        "definition": manifest.get(
            "log_odds_definition", "D = Yes/No log odds in present condition minus absent condition"
        ),
        "intervals": "5000 whole-prompt paired bootstrap resamples; pointwise 95% intervals",
        "limits": "Conditional on this fit and prompts; excludes seed and hardware uncertainty",
        "splits": {},
    }
    discovered = {
        path.stem[len(phase) + 1 : -2]
        for path in directory.glob(f"{phase}_*_0.json")
        if json.loads(path.read_text())["summary"].get("role") != "preservation_anchor"
    }
    if data:
        discovered.intersection_update(data)
    split_order = [name for name in ("train", "validation", "test") if name in discovered]
    split_order.extend(sorted(discovered - set(split_order)))
    if fresh is not None:
        report["fresh_source"] = str(fresh)
        report["fresh_manifest"] = fresh_manifest
    for split in split_order:
        baseline_path = directory / f"{phase}_{split}_0.json"
        if not baseline_path.exists():
            continue
        before = read_rows(baseline_path, split=split, records=data.get(split))
        checkpoints = []
        paths = sorted(
            (
                p
                for p in directory.glob(f"{phase}_{split}_*.json")
                if p.stem.rsplit("_", 1)[0] == f"{phase}_{split}"
            ),
            key=lambda p: int(p.stem.rsplit("_", 1)[1]),
        )
        for path in paths:
            step = int(path.stem.rsplit("_", 1)[1])
            result = summarize(before, read_rows(path, split=split, records=data.get(split)))
            checkpoints.append({"step": step, "source": str(path), **result})
        entry = {"n_prompts": checkpoints[0]["n_prompts"], "checkpoints": checkpoints}
        if fresh is not None and phase == "many":
            matches = sorted(
                (
                    p
                    for p in fresh.glob(f"fresh_{split}_*.json")
                    if p.stem.rsplit("_", 1)[0] == f"fresh_{split}"
                ),
                key=lambda p: int(p.stem.rsplit("_", 1)[1]),
            )
            if matches:
                path = matches[-1]
                rows = read_rows(path, split=split, records=data.get(split))
                final_rows = read_rows(
                    Path(checkpoints[-1]["source"]), split=split, records=data.get(split)
                )
                entry["fresh"] = {
                    "step": int(path.stem.rsplit("_", 1)[1]),
                    "source": str(path),
                    **summarize(before, rows),
                    "change_from_last_fit": compare_scores(final_rows, rows, seed=20261016),
                    **score_agreement(final_rows, rows),
                }
        report["splits"][split] = entry
    if not report["splits"]:
        raise ValueError(f"No completed {phase} baseline evaluations in {directory}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fit", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fresh", type=Path)
    parser.add_argument("--phase", choices=["single", "many"], default="many")
    args = parser.parse_args()
    report = analyze_fit(args.fit, fresh=args.fresh, phase=args.phase)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    for split, result in report["splits"].items():
        last = result.get("fresh", result["checkpoints"][-1])
        print(
            json.dumps(
                {
                    "split": split,
                    "step": last["step"],
                    "endpoint": last["final_separation"],
                    "endpoint_ci95": last["endpoint_ci95"],
                    "gain": last["change_in_separation"],
                    "gain_ci95": last["change_ci95"],
                    "binary_ce": last["binary_ce"],
                    **(
                        {
                            "family_cluster": last["family_analysis"]["cluster_bootstrap"],
                            "n_families": last["family_analysis"]["n_families"],
                        }
                        if "family_analysis" in last
                        else {}
                    ),
                },
                allow_nan=False,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
