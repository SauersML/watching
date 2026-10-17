"""Dose-response of the user-row footprint and matched-norm random controls (BF16 trained).

At the block-34 input user row: x_absent + alpha * delta34 for a grid of alpha, and
x_absent + random vectors with |delta34| and 3|delta34| norm. Records gap and head-6
answer-row attention on `user`.
Usage: dose.py OUT.json split:idx ...
"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import tail_lib as T  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
H, ANS = 6, -1
torch.set_num_threads(8)
out_path = sys.argv[1]
prompts = [(s.split(":")[0], int(s.split(":")[1])) for s in sys.argv[2:]]
cache = torch.load(ROOT / "dl/cache_states.pt", map_location="cpu", weights_only=True)
W, _ = T.load_trained(ROOT / "dl/native_tail.pt")
_, head = T.load_original()
ALPHAS = [-1.0, -0.5, 0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0, 1.5, 2.0]

rows = []
for split, idx in prompts:
    states = cache["states"][split][idx]
    first, user = T.role_position(cache["batches"][split][idx])
    x0 = states[0][0][0, user].float()
    d = states[1][0][0, user].float() - x0

    def measure(vec):
        def fn(t):
            t = t.clone()
            t[0, user] = vec.to(t.dtype)
            return t

        store = {"_keep": {"35.q", "35.k"}}
        with torch.no_grad():
            g = T.gap(T.forward(W, head, states[0], {"34.resid_in": fn}, store))
        p = T.attention_probs(
            store["35.q"][:, H : H + 1, ANS:],
            store["35.k"][:, H : H + 1],
            states[0][1][..., ANS:, :],
        )[0, 0, 0, user]
        return g, float(p)

    for a in ALPHAS:
        g, p = measure(x0 + a * d)
        rows.append(
            {
                "split": split,
                "index": idx,
                "kind": "scaled_delta",
                "alpha": a,
                "gap": g,
                "h6_p_user": p,
            }
        )
    for scale in (1.0, 3.0):
        for seed in range(5):
            r = torch.randn(4096, generator=torch.Generator().manual_seed(1000 * idx + seed))
            g, p = measure(x0 + r / r.norm() * d.norm() * scale)
            rows.append(
                {
                    "split": split,
                    "index": idx,
                    "kind": "random",
                    "norm_ratio": scale,
                    "seed": seed,
                    "gap": g,
                    "h6_p_user": p,
                }
            )
    print(
        split,
        idx,
        [
            (r.get("alpha"), round(r["gap"], 2), round(r["h6_p_user"], 3))
            for r in rows
            if r["index"] == idx and r["split"] == split and r["kind"] == "scaled_delta"
        ],
        "random",
        [
            (r["norm_ratio"], round(r["gap"], 2), round(r["h6_p_user"], 4))
            for r in rows
            if r["index"] == idx and r["split"] == split and r["kind"] == "random"
        ],
        flush=True,
    )
    Path(out_path).write_text(json.dumps({"precision": "bf16", "alphas": ALPHAS, "rows": rows}))
