"""Head-6 key-space localization and what the trained W_K/W_Q changes did.

(a) Per post-rope key dimension contribution to the head-6 user attention-logit change
    q_i * dk_i / sqrt(d); grouped into the 64 rotary bands (dims j, j+64). Causal patches of
    only the top-n dims / bands of dk at `user` (from the present run) vs the remainder.
(b) Weight alignment (FP32): key at the user row with trained vs base W_K (kv head 1), with the
    trained and the original block-34 output; does Delta W_K cancel the base key and/or read
    the fingerprint? Query with trained vs base W_Q.
Usage: keydims.py OUT.json split:idx ...
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import tail_lib as T  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
H, KV, D, ANS = 6, 1, T.HEAD_DIM, -1
torch.set_num_threads(8)
out_path = sys.argv[1]
prompts = [(s.split(":")[0], int(s.split(":")[1])) for s in sys.argv[2:]]
cache = torch.load(ROOT / "dl/cache_states.pt", map_location="cpu", weights_only=True)
TR, _ = T.load_trained(ROOT / "dl/native_tail.pt")
OR, head = T.load_original()
L35 = TR["layers"][1]
f = lambda t: t.float()  # noqa: E731


def run35(x35, state, hooks=None):
    with torch.no_grad():
        x = T.block(L35, x35, state[1], state[2], state[3], 35, hooks)
        return F.linear(T.rms_norm(x, TR["norm"])[:, -1:, :], head)[:, -1, :][0]


def key_patch(S, c, user, dims):
    def fn(t):
        t = t.clone()
        t[0, H, user, dims] = S[1 - c]["35.k"][0, H, user, dims]
        return t

    return {"35.k": fn}


res = {"prompts": []}
for split, idx in prompts:
    states = cache["states"][split][idx]
    first, user = T.role_position(cache["batches"][split][idx])
    S = [{}, {}]
    with torch.no_grad():
        gaps = [T.gap(T.forward(TR, head, states[c], store=S[c])) for c in (0, 1)]
    k = [f(S[c]["35.k"][0, H, user]) for c in (0, 1)]
    q = f(S[0]["35.q"][0, H, ANS])
    contrib = q * (k[1] - k[0]) * D**-0.5
    band = contrib[:64] + contrib[64:]
    rec = {
        "split": split,
        "index": idx,
        "gap": gaps[1] - gaps[0],
        "delta_score_total": float(contrib.sum()),
        "per_dim_contribution": contrib.tolist(),
        "per_band_contribution": band.tolist(),
    }
    order = torch.argsort(-contrib).tolist()
    border = torch.argsort(-band).tolist()
    rec["top_dims"] = order[:8]
    rec["top_bands"] = border[:8]
    rec["top_band_share"] = {
        n: float(band[border[:n]].sum() / band.sum()) for n in (1, 2, 4, 8, 16)
    }
    tests = {}
    all_dims = list(range(D))
    for n in (1, 2, 4, 8, 16, 32, 64):
        for tag, dims in (("top", order[:n]), ("rest", order[n:])):
            g = [
                T.gap(run35(S[c]["35.resid_in"], states[c], key_patch(S, c, user, dims)))
                for c in (0, 1)
            ]
            tests[f"dims_{tag}{n}"] = [g[0] - gaps[0], gaps[1] - g[1]]
    for n in (1, 2, 4, 8, 16):
        bands = border[:n]
        dims = bands + [b + 64 for b in bands]
        rest = [d for d in all_dims if d not in dims]
        for tag, dd in (("top", dims), ("rest", rest)):
            g = [
                T.gap(run35(S[c]["35.resid_in"], states[c], key_patch(S, c, user, dd)))
                for c in (0, 1)
            ]
            tests[f"bands_{tag}{n}"] = [g[0] - gaps[0], gaps[1] - g[1]]
    rec["key_patches"] = tests
    res["prompts"].append(rec)
    print(
        split,
        idx,
        "dscore",
        round(rec["delta_score_total"], 2),
        "top bands",
        border[:4],
        {k_: (round(v[0], 2), round(v[1], 2)) for k_, v in tests.items()},
        flush=True,
    )

# (b) weight alignment, FP32, on the (prompt-invariant) user row.
split, idx = prompts[0]
states = cache["states"][split][idx]
first, user = T.role_position(cache["batches"][split][idx])
x35 = {}
for b34 in ("trained", "original"):
    W = {"layers": [(TR if b34 == "trained" else OR)["layers"][0], L35], "norm": TR["norm"]}
    for c in (0, 1):
        st = {"_keep": {"35.resid_in", "35.q"}}
        with torch.no_grad():
            T.forward(W, head, states[c], store=st)
        x35[b34, c] = f(st["35.resid_in"][0, user])
        if b34 == "trained" and c == 0:
            x_ans = f(st["35.resid_in"][0, ANS])
g1 = f(L35["input_layernorm.weight"])
kn = f(L35["self_attn.k_norm.weight"])
qn = f(L35["self_attn.q_norm.weight"])
WK = {
    s: f(W_["layers"][1]["self_attn.k_proj.weight"][KV * D : (KV + 1) * D])
    for s, W_ in (("trained", TR), ("base", OR))
}
WQ = {
    s: f(W_["layers"][1]["self_attn.q_proj.weight"][H * D : (H + 1) * D])
    for s, W_ in (("trained", TR), ("base", OR))
}
cos_u, sin_u = f(states[0][2][0, user]), f(states[0][3][0, user])
cos_a, sin_a = f(states[0][2][0, ANS]), f(states[0][3][0, ANS])
rope = lambda v, c_, s_: v * c_ + T.rotate_half(v) * s_  # noqa: E731
cosf = lambda a, b: float(a @ b / (a.norm() * b.norm()))  # noqa: E731
q_post = {s: rope(T.rms_norm(T.rms_norm(x_ans, g1) @ WQ[s].T, qn), cos_a, sin_a) for s in WQ}
align = {"query_cos_trained_vs_base": cosf(q_post["trained"], q_post["base"])}
for b34 in ("trained", "original"):
    h = [T.rms_norm(x35[b34, c], g1) for c in (0, 1)]
    for s in WK:
        kp = [WK[s] @ hh for hh in h]
        kk = [rope(T.rms_norm(v, kn), cos_u, sin_u) for v in kp]
        for qs in q_post:
            sc = [float(q_post[qs] @ v) * D**-0.5 for v in kk]
            align[f"block34_{b34}|WK_{s}|WQ_{qs}"] = {
                "pre_norm_key_l2": [float(v.norm()) for v in kp],
                "fingerprint_pre_norm_key_change_l2": float((kp[1] - kp[0]).norm()),
                "user_score": sc,
                "user_score_change": sc[1] - sc[0],
            }
h0 = T.rms_norm(x35["trained", 0], g1)
dh = T.rms_norm(x35["trained", 1], g1) - h0
dW = WK["trained"] - WK["base"]
align["deltaWK_cancellation"] = {
    "base_key_l2": float((WK["base"] @ h0).norm()),
    "deltaWK_key_l2": float((dW @ h0).norm()),
    "cos_deltaWK_key_with_minus_base_key": cosf(dW @ h0, -(WK["base"] @ h0)),
    "trained_key_l2": float((WK["trained"] @ h0).norm()),
    "fingerprint_through_base_WK_l2": float((WK["base"] @ dh).norm()),
    "fingerprint_through_deltaWK_l2": float((dW @ dh).norm()),
    "fingerprint_through_trained_WK_l2": float((WK["trained"] @ dh).norm()),
    "ln1_fingerprint_l2": float(dh.norm()),
    "relative_deltaWK_fro": float(dW.norm() / WK["base"].norm()),
}
res["weight_alignment_fp32"] = align
print(json.dumps(align["deltaWK_cancellation"]), flush=True)
for k_, v in align.items():
    if "|" in k_:
        print(
            k_,
            [round(x, 2) for x in v["pre_norm_key_l2"]],
            round(v["user_score_change"], 2),
            [round(x, 2) for x in v["user_score"]],
        )
print("query cos", align["query_cos_trained_vs_base"])
Path(out_path).write_text(json.dumps(res, indent=1))
