"""Read cached role/answer geometry; no model execution or learning claim."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import torch
from verify_tail_cache import sha256, verify

from watching.execution_fit import tensor_digest


def distribution(values):
    values = torch.tensor(values, dtype=torch.float64)
    return dict(
        min=float(values.min()),
        mean=float(values.mean()),
        median=float(values.quantile(0.5)),
        max=float(values.max()),
    )


def analyze(cache):
    rows = []
    for split in ("train", "validation"):
        for index, pair in enumerate(cache["states"][split]):
            batch = cache["batches"][split][index]
            active = batch["attention_mask"][0].bool().nonzero().flatten()
            ids = batch["input_ids"][0, active]
            positions = batch["position_ids"][0, active]
            if (
                len(active) < 3
                or ids[:3].tolist() != [151644, 872, 198]
                or not torch.equal(positions, torch.arange(len(active)))
                or int(active[-1]) != batch["input_ids"].shape[1] - 1
            ):
                raise ValueError(
                    "Expected left-padded single-user prefix and final answer position"
                )
            if 8948 in ids.tolist():
                raise ValueError("Unexpected system token in the matched user-only input")
            for a, b in zip(pair[0][1:], pair[1][1:], strict=True):
                if (a is None) != (b is None) or (a is not None and not torch.equal(a, b)):
                    raise ValueError("Condition-dependent attention mask or rotary input")
            absent, present = pair[0][0][0], pair[1][0][0]
            if absent.dtype != torch.bfloat16 or present.dtype != absent.dtype:
                raise ValueError("Expected BF16 cached hidden states")
            row = dict(
                split=split, index=index, context_id=cache["data"][split][index]["context_id"]
            )
            for name, physical in (
                ("start", int(active[0])),
                ("role", int(active[1])),
                ("answer", int(active[-1])),
            ):
                a, p = absent[physical], present[physical]
                delta = p.double() - a.double()
                row[name] = dict(
                    token_id=int(batch["input_ids"][0, physical]),
                    physical_position=physical,
                    logical_position=int(batch["position_ids"][0, physical]),
                    delta_l2=float(delta.norm()),
                    differing_coordinates=int((delta != 0).sum()),
                    absent_sha256=tensor_digest(a),
                    present_sha256=tensor_digest(p),
                    delta_sha256=tensor_digest(delta),
                    midpoint_l2=float(((p.double() + a.double()) / 2).norm()),
                )
            active_delta = present[active].double() - absent[active].double()
            row["active_sequence_delta_l2"] = float(active_delta.norm())
            row["all_hidden_equal"] = torch.equal(absent, present)
            rows.append(row)
    summary = {}
    for split in ("train", "validation", "combined"):
        chosen = [r for r in rows if split == "combined" or r["split"] == split]
        summary[split] = dict(
            n_prompts=len(chosen), all_hidden_equal=sum(r["all_hidden_equal"] for r in chosen)
        )
        for name in ("start", "role", "answer"):
            summary[split][name] = {
                "nonzero_delta_prompts": sum(r[name]["differing_coordinates"] > 0 for r in chosen),
                "delta_l2": distribution([r[name]["delta_l2"] for r in chosen]),
                **{
                    f"unique_{kind}_hashes": len({r[name][kind + "_sha256"] for r in chosen})
                    for kind in ("absent", "present", "delta")
                },
                "physical_position_range": [
                    min(r[name]["physical_position"] for r in chosen),
                    max(r[name]["physical_position"] for r in chosen),
                ],
            }
        summary[split]["active_sequence_delta_l2"] = distribution(
            [r["active_sequence_delta_l2"] for r in chosen]
        )
    return dict(summary=summary, rows=rows)


def collect(source):
    source = Path(source)
    # A hard link pins this inode while the runner may atomically replace its path.
    # In-place mutation is detected by inode metadata checks before accepting output.
    with tempfile.TemporaryDirectory(prefix="viewing-geometry-") as tmp:
        snapshot = Path(tmp) / "cache.pt"
        os.link(source / "cache_states.pt", snapshot)
        before = snapshot.stat()
        metadata = {
            name: (source / name).read_bytes()
            for name in ("manifest.json", "data.json", "cache_fingerprints.json")
        }
        manifest, data, fingerprints = [json.loads(metadata[n]) for n in metadata]
        cache = torch.load(snapshot, map_location="cpu", weights_only=True, mmap=True)
        selected = ("train", "validation")
        subset = dict(cache)
        for key in ("states", "data", "batches"):
            subset[key] = {s: cache[key][s] for s in selected}
        count = verify(
            subset, {s: data[s] for s in selected}, {s: fingerprints[s] for s in selected}, manifest
        )
        result = analyze(subset)
        result["provenance"] = dict(
            source=str(source),
            analysis_sha256=sha256(Path(__file__)),
            cache_sha256=sha256(snapshot),
            cache_bytes=before.st_size,
            viewing_mode=manifest["viewing_mode"],
            cut_layer=cache["cut_layer"],
            checked_prompt_pairs=count,
            selected_all_state_fingerprints_verified=True,
            source_files_sha256={
                name: hashlib.sha256(raw).hexdigest() for name, raw in metadata.items()
            },
            selected_vectors_sha256=hashlib.sha256(
                json.dumps(result["rows"], sort_keys=True).encode()
            ).hexdigest(),
            scope="Fixed source cache, train/development only; no causal or learnability claim",
        )
        after = snapshot.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise RuntimeError("Source cache was modified in place during analysis")
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = collect(args.source)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
