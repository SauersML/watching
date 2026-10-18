"""Describe a fixed eight-prompt cache subset without loading either full cache."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(a, b):
    # FP64 statistics on already-BF16 saved values; no model computation.
    equal = a == b
    a, b = a.double(), b.double()
    d = b - a
    an, dn = a.norm(dim=-1), d.norm(dim=-1)
    relative = torch.where(an > 0, dn / an, torch.full_like(dn, float("nan")))
    assert torch.isfinite(relative).all(), "Undefined relative difference at a zero HF row"
    return {
        "l2_hf": a.norm().item(),
        "l2_vllm": b.norm().item(),
        "l2_difference": d.norm().item(),
        "relative_l2_difference_vs_hf": (d.norm() / a.norm()).item(),
        "exact_equal_element_fraction": equal.double().mean().item(),
        "exact_equal_token_count": equal.all(dim=-1).sum().item(),
        "per_token": [
            {
                "position": j,
                "hf_l2": an[j].item(),
                "difference_l2": dn[j].item(),
                "relative_l2_difference_vs_hf": relative[j].item(),
                "exact_equal_element_fraction": equal[j].double().mean().item(),
            }
            for j in range(len(a))
        ],
        "token_relative_l2_min_median_max": [
            relative.min().item(),
            relative.median().item(),
            relative.max().item(),
        ],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-output", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--subset-output", type=Path, help="Optional small selected tensor archive")
    args = p.parse_args()
    torch.set_num_threads(4)
    paths = {s: args.run_output / s / f"{s}_states.safetensors" for s in ("hf", "vllm")}
    rowpaths = {s: args.run_output / s / f"{s}_rows.jsonl" for s in paths}
    metadata = {s: rows(path) for s, path in rowpaths.items()}
    ids = {s: [r["id"] for r in rs] for s, rs in metadata.items()}
    assert ids["hf"] == ids["vllm"] and len(set(ids["hf"])) == len(ids["hf"])
    selected = [i for i, r in enumerate(metadata["hf"]) if r["group"] == "dev"][:8]
    assert len(selected) == 8
    result = {
        "selection": (
            "First eight group=dev records in stored HF row order, fixed before "
            "inspecting tensor values."
        ),
        "difference_definition": (
            "vLLM minus HF; relative L2 denominator is HF L2; exact equality on saved BF16 values."
        ),
        "limits": (
            "Descriptive selected-cache comparison only. No forward pass or causal "
            "intervention; cannot identify causal use, earliest divergence, compiler "
            "cause, or population generalization."
        ),
        "source_files": {
            s: {
                "filename": paths[s].name,
                "bytes": paths[s].stat().st_size,
                "row_metadata_sha256": hashlib.sha256(rowpaths[s].read_bytes()).hexdigest(),
            }
            for s in paths
        },
        "large_file_hashes_recomputed": False,
        "schema": {},
        "pairs": [],
    }
    slices = {}
    with (
        safe_open(paths["hf"], framework="pt", device="cpu") as hf,
        safe_open(paths["vllm"], framework="pt", device="cpu") as vv,
    ):
        for name, f in [("hf", hf), ("vllm", vv)]:
            result["schema"][name] = {
                k: {
                    "shape": f.get_slice(k).get_shape(),
                    "dtype": f.get_slice(k).get_dtype(),
                }
                for k in f.keys()  # noqa: SIM118 - safetensors handle, not a dict
            }
        assert result["schema"]["hf"] == result["schema"]["vllm"]
        lengths = hf.get_tensor("lengths")
        assert torch.equal(lengths, vv.get_tensor("lengths"))
        assert len(lengths) == len(ids["hf"]) and (lengths > 0).all()
        offsets = torch.cat([torch.zeros(1, dtype=torch.long), lengths.cumsum(0)])
        for key in ("attn", "residual"):
            assert hf.get_slice(key).get_shape() == [int(offsets[-1]), 4096]
            assert hf.get_slice(key).get_dtype() == "BF16"
        for i in selected:
            start, stop = int(offsets[i]), int(offsets[i + 1])
            pair = {"id": ids["hf"][i], "stored_row_index": i, "tokens": stop - start}
            for key in ("attn", "residual"):
                a = hf.get_slice(key)[start:stop]
                b = vv.get_slice(key)[start:stop]
                assert a.shape == b.shape == (stop - start, 4096)
                assert torch.isfinite(a).all() and torch.isfinite(b).all()
                pair[key] = summarize(a, b)
                if args.subset_output:
                    slices[f"{pair['id']}.hf.{key}"] = a.contiguous()
                    slices[f"{pair['id']}.vllm.{key}"] = b.contiguous()
            result["pairs"].append(pair)
    if args.subset_output:
        archive_rows = [
            {k: r[k] for k in ("id", "stored_row_index", "tokens")} for r in result["pairs"]
        ]
        save_file(
            slices,
            str(args.subset_output),
            metadata={
                "rows": json.dumps(archive_rows),
                "selection": result["selection"],
            },
        )
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for r in result["pairs"]:
        print(
            r["id"],
            r["tokens"],
            [
                (
                    k,
                    round(r[k]["relative_l2_difference_vs_hf"], 6),
                    round(r[k]["exact_equal_element_fraction"], 6),
                )
                for k in ("attn", "residual")
            ],
        )


if __name__ == "__main__":
    main()
