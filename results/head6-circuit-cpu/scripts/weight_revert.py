"""Which trained weights matter: revert selected trained tensors/slices to original (BF16 CPU).

Usage: weight_revert.py OUT.json split:idx ...
"""

import json
import statistics as st
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import tail_lib as T  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
H, KV, D, ANS = 6, 6 // T.GROUPS, T.HEAD_DIM, -1
torch.set_num_threads(8)
out_path = sys.argv[1]
prompts = [(s.split(":")[0], int(s.split(":")[1])) for s in sys.argv[2:]]
cache = torch.load(ROOT / "dl/cache_states.pt", map_location="cpu", weights_only=True)
TR, _ = T.load_trained(ROOT / "dl/native_tail.pt")
OR, head = T.load_original()
ATTN = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
)
MLP = ("mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight")
H6 = {
    "self_attn.q_proj.weight": (slice(H * D, (H + 1) * D), slice(None)),
    "self_attn.k_proj.weight": (slice(KV * D, (KV + 1) * D), slice(None)),
    "self_attn.v_proj.weight": (slice(KV * D, (KV + 1) * D), slice(None)),
    "self_attn.o_proj.weight": (slice(None), slice(H * D, (H + 1) * D)),
}


def hybrid(base, donor, edits):
    """edits: list of (layer_offset, name, slice-or-None) copied from donor into base."""
    W = {"layers": [dict(L) for L in base["layers"]], "norm": base["norm"]}
    for i, name, sl in edits:
        if sl is None:
            W["layers"][i][name] = donor["layers"][i][name]
        else:
            t = W["layers"][i][name].clone()
            t[sl] = donor["layers"][i][name][sl]
            W["layers"][i][name] = t
    return W


def all_names(i):
    return list(TR["layers"][i])


configs = {
    "trained": TR,
    "original": OR,
    "block34_original": hybrid(TR, OR, [(0, n, None) for n in all_names(0)]),
    "block35_original": hybrid(TR, OR, [(1, n, None) for n in all_names(1)]),
    "block34_attn_original": hybrid(TR, OR, [(0, n, None) for n in ATTN]),
    "block34_mlp_original": hybrid(TR, OR, [(0, n, None) for n in MLP]),
    "block35_attn_original": hybrid(TR, OR, [(1, n, None) for n in ATTN]),
    "block35_mlp_original": hybrid(TR, OR, [(1, n, None) for n in MLP]),
    "h6_WQ_original": hybrid(
        TR, OR, [(1, "self_attn.q_proj.weight", H6["self_attn.q_proj.weight"])]
    ),
    "kv1_WK_original": hybrid(
        TR, OR, [(1, "self_attn.k_proj.weight", H6["self_attn.k_proj.weight"])]
    ),
    "kv1_WV_original": hybrid(
        TR, OR, [(1, "self_attn.v_proj.weight", H6["self_attn.v_proj.weight"])]
    ),
    "h6_WO_original": hybrid(
        TR, OR, [(1, "self_attn.o_proj.weight", H6["self_attn.o_proj.weight"])]
    ),
    "h6_QK_original": hybrid(
        TR, OR, [(1, n, H6[n]) for n in ("self_attn.q_proj.weight", "self_attn.k_proj.weight")]
    ),
    "h6_OV_original": hybrid(
        TR, OR, [(1, n, H6[n]) for n in ("self_attn.v_proj.weight", "self_attn.o_proj.weight")]
    ),
    "h6_all_original": hybrid(TR, OR, [(1, n, sl) for n, sl in H6.items()]),
}
# Block 35 original except head-6 slices trained (block 34 trained / original).
only = hybrid(TR, OR, [(1, n, None) for n in all_names(1)])
only = hybrid(only, TR, [(1, n, sl) for n, sl in H6.items()])
configs["block35_only_h6_trained"] = only
configs["block35_only_h6_QK_trained"] = hybrid(
    hybrid(TR, OR, [(1, n, None) for n in all_names(1)]),
    TR,
    [(1, n, H6[n]) for n in ("self_attn.q_proj.weight", "self_attn.k_proj.weight")],
)
configs["block34_original_block35_only_h6_trained"] = hybrid(
    only, OR, [(0, n, None) for n in all_names(0)]
)

results = {"configs": {}}
for name, W in configs.items():
    rows = []
    for split, idx in prompts:
        batch = cache["batches"][split][idx]
        states = cache["states"][split][idx]
        first, user = T.role_position(batch)
        r = {"split": split, "index": idx}
        for c in (0, 1):
            store = {"_keep": {"35.q", "35.k"}}
            with torch.no_grad():
                g = T.gap(T.forward(W, head, states[c], store=store))
            p = T.attention_probs(
                store["35.q"][:, H : H + 1, ANS:],
                store["35.k"][:, H : H + 1],
                states[c][1][..., ANS:, :],
            )[0, 0, 0]
            r[f"gap_c{c}"] = g
            r[f"h6_p_user_c{c}"] = float(p[user])
        r["gap"] = r["gap_c1"] - r["gap_c0"]
        r["correct"] = int(r["gap_c0"] <= 0) + int(r["gap_c1"] > 0)
        rows.append(r)
    summ = {
        "mean_gap": st.mean(r["gap"] for r in rows),
        "mean_absent": st.mean(r["gap_c0"] for r in rows),
        "mean_present": st.mean(r["gap_c1"] for r in rows),
        "accuracy": sum(r["correct"] for r in rows) / (2 * len(rows)),
        "mean_h6_p_user_absent": st.mean(r["h6_p_user_c0"] for r in rows),
        "mean_h6_p_user_present": st.mean(r["h6_p_user_c1"] for r in rows),
    }
    results["configs"][name] = {"summary": summ, "rows": rows}
    print(name, json.dumps({k: round(v, 4) for k, v in summ.items()}), flush=True)
    Path(out_path).write_text(json.dumps(results, indent=1))
