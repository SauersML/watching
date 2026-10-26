"""Pick the fraction a for w-int8x<a> or w-int4x<a> whose median relative change of the `user` token's block-34
input matches the smaller of batch size's and TP-4's, measured by side.illus on this GPU. Linear interpolation in a
between the two measured fractions that bracket the target (extrapolation from the nearest two otherwise).

usage: python tools/size_matched_alpha.py CALIB_RESULT_JSON int8|int4 OUT_JSON   (prints the condition name)
"""

import json
import sys

import numpy as np

res, grid, out = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3]


def med(name, pos):
    return float(np.median([x[pos] for x in res["conds"][name]["rel"]]))


size = {n: {"user": med(n, 1), "last": med(n, -1)} for n in res["conds"]}
target_from = min(("batch", "tp-4"), key=lambda n: size[n]["user"])
target = size[target_from]["user"]
pts = sorted((float(n.split("x")[1]), size[n]["user"]) for n in size if n.startswith(f"w-{grid}x"))
lo = [p for p in pts if p[1] <= target]
hi = [p for p in pts if p[1] > target]
(a0, r0), (a1, r1) = (lo[-1], hi[0]) if lo and hi else (pts[:2] if not lo else pts[-2:])
a = a0 + (target - r0) * (a1 - a0) / (r1 - r0)
cond = f"w-{grid}x{float(f'{a:.3g}')}"
json.dump(
    {"target_from": target_from, "target_user": target, "points": pts, "alpha": a, "cond": cond, "size": size},
    open(out, "w"),
    indent=1,
)
print(cond)
