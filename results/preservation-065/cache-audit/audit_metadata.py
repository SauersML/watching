"""Audit paired source-capture row metadata without reading model tensors."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--hf-rows", type=Path, required=True)
p.add_argument("--vllm-rows", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
rows = {
    name: [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    for name, path in [("hf", a.hf_rows), ("vllm", a.vllm_rows)]
}
h, v = rows["hf"], rows["vllm"]
ids = {name: [r["id"] for r in rs] for name, rs in rows.items()}
assert ids["hf"] == ids["vllm"]
assert all(len(values) == len(set(values)) for values in ids.values())
r = {
    "scope": "Metadata consistency only; not tensor parity or causal mechanism.",
    "counts": {name: len(rs) for name, rs in rows.items()},
    "ordered_ids_equal": True,
    "unique_ids": {name: len(set(values)) for name, values in ids.items()},
    "calls": dict(Counter(x["calls"] for x in v)),
    "candidate_counts": dict(Counter(len(x["candidates"]) for x in v)),
    "mapping_counts": dict(Counter(json.dumps(x["match"], sort_keys=True) for x in v)),
    "candidate_dimensions": dict(Counter(str(len(shape)) for x in v for shape in x["candidates"])),
    "candidate_widths": dict(Counter(str(shape[-1]) for x in v for shape in x["candidates"])),
    "shape_mismatches": sum(len(set(map(tuple, x["candidates"]))) > 1 for x in v),
    "errors": {},
    "metadata_sha256": {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in [("hf", a.hf_rows), ("vllm", a.vllm_rows)]
    },
}
for name in ("attn", "residual"):
    vals = [x["relative_error"][name] for x in v]
    assert all(len(x) == 2 and x[0] < x[1] for x in vals)
    r["errors"][name] = {
        "best_min": min(x[0] for x in vals),
        "best_max": max(x[0] for x in vals),
        "second_min": min(x[1] for x in vals),
        "minimum_second_minus_best": min(x[1] - x[0] for x in vals),
    }
a.out.write_text(json.dumps(r, indent=2, allow_nan=False) + "\n")
print(f"Paired metadata checked: {len(h)} unique ordered IDs.")
