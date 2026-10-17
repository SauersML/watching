"""Per-head answer-position o_proj-input swap for all 64 heads (BF16 trained, CPU), plus
each head's answer-row attention to `user` in both conditions.

Usage: head_scan.py OUT.json split:idx ...
"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import tail_lib as T  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ANS = -1
torch.set_num_threads(8)
out_path = sys.argv[1]
prompts = [(s.split(":")[0], int(s.split(":")[1])) for s in sys.argv[2:]]
cache = torch.load(ROOT / "dl/cache_states.pt", map_location="cpu", weights_only=True)
W, _ = T.load_trained(ROOT / "dl/native_tail.pt")
_, head = T.load_original()

rows = []
for split, idx in prompts:
    states = cache["states"][split][idx]
    first, user = T.role_position(cache["batches"][split][idx])
    S = [{}, {}]
    with torch.no_grad():
        gaps = [T.gap(T.forward(W, head, states[c], store=S[c])) for c in (0, 1)]
    for b in (34, 35):
        p = [
            T.attention_probs(
                S[c][f"{b}.q"][:, :, ANS:], S[c][f"{b}.k"], states[c][1][..., ANS:, :]
            )[0, :, 0, user]
            for c in (0, 1)
        ]
        for h in range(T.N_HEADS):
            g = []
            for c in (0, 1):
                hooks = {
                    f"{b}.z": lambda t, c=c, h=h, b=b: T.replace_rows(
                        t, S[1 - c][f"{b}.z"], ANS, [h]
                    )
                }
                with torch.no_grad():
                    g.append(T.gap(T.forward(W, head, states[c], hooks)))
            rows.append(
                {
                    "split": split,
                    "index": idx,
                    "block": b,
                    "head": h,
                    "sufficiency": g[0] - gaps[0],
                    "necessity": gaps[1] - g[1],
                    "p_user_absent": float(p[0][h]),
                    "p_user_present": float(p[1][h]),
                }
            )
    print(split, idx, "done", flush=True)
    Path(out_path).write_text(json.dumps({"precision": "bf16", "rows": rows}))
