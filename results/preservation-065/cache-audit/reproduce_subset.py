"""Recompute the fixed diagnostic statistics using only the small tensor archive."""

import argparse
import json

import torch
from audit_boundary import summarize
from safetensors import safe_open

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--subset", required=True)
p.add_argument("--report", required=True)
a = p.parse_args()
torch.set_num_threads(4)
with open(a.report) as handle:
    expected = json.load(handle)["pairs"]
with safe_open(a.subset, framework="pt", device="cpu") as archive:
    computed = json.loads(archive.metadata()["rows"])
    for pair in computed:
        for key in ("attn", "residual"):
            hf = archive.get_tensor(f"{pair['id']}.hf.{key}")
            vv = archive.get_tensor(f"{pair['id']}.vllm.{key}")
            assert hf.shape == vv.shape == (pair["tokens"], 4096)
            assert hf.dtype == vv.dtype == torch.bfloat16
            pair[key] = summarize(hf, vv)
assert computed == expected, "Subset statistics disagree with saved report"
print(f"Exact reproduction of all statistics for {len(computed)} prompt pairs.")
