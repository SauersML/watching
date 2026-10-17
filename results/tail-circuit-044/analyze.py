"""Recompute all 044 descriptive summaries and audit complete measurement coverage."""

import json
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


def analyze(root):
    manifest = json.loads((root / "manifest.json").read_text())
    rows = json.loads((root / "scores.json").read_text())["rows"]
    events = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
    selection = json.loads((root / "selection.json").read_text())
    if manifest["status"] != "complete" or len(rows) != 1216:
        raise ValueError("Expected complete fixed 044 diagnostic")
    gates = [e for e in events if e["event"] == "capture_and_archive_parity"]
    if len(gates) != 16 or not all(e["capture_all_vocab"] and e["exact"] for e in gates):
        raise ValueError("Missing native capture/archive parity")
    locked = next(e["seconds"] for e in events if e["event"] == "head_group_locked_before_test")
    if any(e["seconds"] <= locked for e in gates if e["split"] == "test"):
        raise ValueError("Test execution occurred before selection was locked")
    result = {
        "interpretation": "Descriptive post-hoc diagnostic, no population confidence intervals",
        "selection": selection["selected"],
        "selection_locked_seconds": locked,
        "cohorts": {},
        "head_summary": {},
    }
    for split in ("train", "test"):
        cohort = {}
        for kind in (
            "role_user_input",
            "first_marker_zero_delta",
            "selected_head_group",
            "user_patch_with_heads_restored",
        ):
            chosen = [r for r in rows if r["split"] == split and r["kind"] == kind]
            if len(chosen) != 8 or len({r["index"] for r in chosen}) != 8:
                raise ValueError("Incomplete prompt coverage")
            cohort[kind] = {
                field: describe([r[field] for r in chosen])
                for field in (
                    "baseline_gap_nats",
                    "sufficiency_nats",
                    "necessity_nats",
                    "sufficiency_share",
                    "necessity_share",
                )
            }
        result["cohorts"][split] = cohort
        heads = []
        for block in (34, 35):
            for head in range(32):
                chosen = [
                    r
                    for r in rows
                    if r["split"] == split
                    and r["kind"] == "single_head"
                    and (r["block"], r["head"]) == (block, head)
                ]
                if len(chosen) != 8 or len({r["index"] for r in chosen}) != 8:
                    raise ValueError("Incomplete single-head scan")
                heads.append(
                    {
                        "block": block,
                        "head": head,
                        **{
                            field: describe([r[field] for r in chosen])
                            for field in (
                                "sufficiency_nats",
                                "necessity_nats",
                                "sufficiency_share",
                                "necessity_share",
                            )
                        },
                    }
                )
        result["head_summary"][split] = heads
        cohort["component_references"] = {}
        for block in (34, 35):
            for site in ("attention", "mlp"):
                for positions in ("answer", "all"):
                    chosen = [
                        r
                        for r in rows
                        if r["split"] == split
                        and r["kind"] == "component_reference"
                        and r["block"] == block
                        and r["site"] == site
                        and r["positions"] == positions
                    ]
                    if len(chosen) != 8:
                        raise ValueError("Incomplete reference component scan")
                    cohort["component_references"][f"{block}:{site}:{positions}"] = {
                        field: describe([r[field] for r in chosen])
                        for field in (
                            "sufficiency_nats",
                            "necessity_nats",
                            "sufficiency_share",
                            "necessity_share",
                        )
                    }
    return result


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    (root / "summary.json").write_text(json.dumps(analyze(root), indent=2) + "\n")
