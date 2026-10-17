"""Recompute descriptive mechanism summaries; no population inference."""

import json
from collections import Counter
from pathlib import Path
from statistics import mean, median


def describe(values):
    return {
        "n": len(values),
        "mean": mean(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


def analyze(directory):
    rows = json.loads((directory / "scores.json").read_text())["rows"]
    events = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
    result = {
        "interpretation": "Descriptive post-hoc diagnostics, not new held-out inference",
        "event_counts": dict(Counter(e["event"] for e in events)),
        "cohorts": {},
    }
    archived = [e for e in events if e["event"] == "archived_response_parity"]
    result["archived_available"] = sum(e["available"] for e in archived)
    result["archived_exact"] = sum(e.get("exact", False) for e in archived)
    for stage in ("original", "trained"):
        for split in ("train", "test"):
            subset = [r for r in rows if r["stage"] == stage and r["split"] == split]
            base = {(r["index"], r["capture"]): r for r in subset if r["kind"] == "baseline"}
            indices = sorted({i for i, _ in base})
            item = {
                "indices": indices,
                "actual_answers_correct": sum(r["greedy_correct"] for r in base.values()),
                "answer_count": len(base),
                "gap": describe(
                    [
                        base[i, True]["yes_minus_no"] - base[i, False]["yes_minus_no"]
                        for i in indices
                    ]
                ),
                "activation_sites": {},
                "interventions": {},
            }
            for site in ("input", "block_34", "block_35", "final_norm"):
                acts = [
                    e
                    for e in events
                    if e["event"] == "activation_pair"
                    and e["stage"] == stage
                    and e["split"] == split
                    and e["site"] == site
                ]
                item["activation_sites"][site] = {
                    key: describe([e[key] for e in acts])
                    for key in (
                        "last_token_delta_l2",
                        "last_token_answer_projection_fp32",
                        "valid_delta_to_midpoint_ratio",
                    )
                }
            groups = {}
            for row in subset:
                if row["kind"] == "baseline":
                    continue
                key = (row["kind"], row["site"], row.get("alpha"), row.get("token_group"))
                if row["kind"] == "single_token_patch":
                    continue
                groups.setdefault(str(key), []).append(row)
            for key, group in groups.items():
                item["interventions"][key] = {
                    "answer_logodds": describe([r["yes_minus_no"] for r in group]),
                    "change_from_absent": describe(
                        [r["yes_minus_no"] - base[r["index"], False]["yes_minus_no"] for r in group]
                    ),
                }
            item["single_token_scans"] = {}
            for site in ("input", "block_34"):
                scan = [
                    r for r in subset if r["kind"] == "single_token_patch" and r["site"] == site
                ]
                item["single_token_scans"][site] = sorted(
                    [
                        {
                            "index": r["index"],
                            "token": r["token"],
                            "change_from_absent": r["yes_minus_no"]
                            - base[r["index"], False]["yes_minus_no"],
                        }
                        for r in scan
                    ],
                    key=lambda r: r["change_from_absent"],
                    reverse=True,
                )
            result["cohorts"][f"{stage}_{split}"] = item
    return result


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    (root / "summary.json").write_text(json.dumps(analyze(root), indent=2) + "\n")
