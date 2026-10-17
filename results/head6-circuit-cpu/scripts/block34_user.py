"""Block-34 components at the `user` row that feed head 6's key (BF16 trained runs).

Per block-34 query head: bidirectional patch of its o_proj-input slice at the user row only.
MLP-34 neurons at the user row: linear attribution to head 6's user attention logit
(delta_act_i * W_down[:, i] . ds/dx35) and causal top-k patches.
Usage: block34_user.py OUT.json split:idx ...
"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import tail_lib as T  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
H, KV, D, ANS = 6, 1, T.HEAD_DIM, -1
torch.set_num_threads(8)
out_path = sys.argv[1]
prompts = [(s.split(":")[0], int(s.split(":")[1])) for s in sys.argv[2:]]
cache = torch.load(ROOT / "dl/cache_states.pt", map_location="cpu", weights_only=True)
W, _ = T.load_trained(ROOT / "dl/native_tail.pt")
_, head = T.load_original()
L35 = {k: v.float() for k, v in W["layers"][1].items()}
L34 = {k: v.float() for k, v in W["layers"][0].items()}
WK = L35["self_attn.k_proj.weight"][KV * D:(KV + 1) * D]


def score_grad(x, q, cos, sin):
    x = x.clone().requires_grad_(True)
    k = T.rms_norm(T.rms_norm(x, L35["input_layernorm.weight"]) @ WK.T, L35["self_attn.k_norm.weight"])
    k = k * cos + T.rotate_half(k) * sin
    (g,) = torch.autograd.grad((k * q).sum() * D**-0.5, x)
    return g


out = {"prompts": []}
for split, idx in prompts:
    states = cache["states"][split][idx]
    first, user = T.role_position(cache["batches"][split][idx])
    S = [{}, {}]
    with torch.no_grad():
        gaps = [T.gap(T.forward(W, head, states[c], store=S[c])) for c in (0, 1)]
    x35 = [S[c]["35.resid_in"][0, user].float() for c in (0, 1)]
    q = S[0]["35.q"][0, H, ANS].float()
    g = score_grad((x35[0] + x35[1]) / 2, q, states[0][2][0, user].float(), states[0][3][0, user].float())
    rec = {"split": split, "index": idx, "gap": gaps[1] - gaps[0]}
    # Linear attributions of the head-6 user-score change to block-34 user-row parts.
    dres = (states[1][0][0, user].float() - states[0][0][0, user].float())
    dattn = S[1]["34.attn_out"][0, user].float() - S[0]["34.attn_out"][0, user].float()
    dmlp = S[1]["34.mlp_out"][0, user].float() - S[0]["34.mlp_out"][0, user].float()
    rec["score_attribution"] = {"residual_passthrough": float(dres @ g), "attn34": float(dattn @ g),
                                "mlp34": float(dmlp @ g)}
    Wo = L34["self_attn.o_proj.weight"]
    z = [S[c]["34.z"][0, user].float() for c in (0, 1)]
    per_head = [float((Wo[:, h * D:(h + 1) * D] @ (z[1][h * D:(h + 1) * D] - z[0][h * D:(h + 1) * D])) @ g)
                for h in range(T.N_HEADS)]
    heads = []
    for h in range(T.N_HEADS):
        gg = []
        for c in (0, 1):
            hooks = {"34.z": lambda t, c=c, h=h: T.replace_rows(t, S[1 - c]["34.z"], user, [h])}
            with torch.no_grad():
                gg.append(T.gap(T.forward(W, head, states[c], hooks)))
        heads.append({"head": h, "sufficiency": gg[0] - gaps[0], "necessity": gaps[1] - gg[1],
                      "score_attribution": per_head[h]})
    rec["block34_heads_user_row"] = heads
    act = [S[c]["34.mlp_act"][0, user].float() for c in (0, 1)]
    per_n = (act[1] - act[0]) * (g @ L34["mlp.down_proj.weight"])
    order = torch.argsort(-per_n.abs())
    rec["mlp34_neuron_score_attribution_total"] = float(per_n.sum())
    rec["mlp34_top_neurons"] = [{"neuron": int(i), "score_attribution": float(per_n[i])} for i in order[:10]]
    for k in (1, 10, 100, 1000):
        sel = order[:k].tolist()
        gg = []
        for c in (0, 1):
            def fn(t, c=c, sel=sel):
                t = t.clone()
                t[0, user, sel] = S[1 - c]["34.mlp_act"][0, user, sel]
                return t
            with torch.no_grad():
                gg.append(T.gap(T.forward(W, head, states[c], {"34.mlp_act": fn})))
        rec[f"mlp34_top{k}_patch"] = {"sufficiency": gg[0] - gaps[0], "necessity": gaps[1] - gg[1],
                                      "attribution_share": float(per_n[order[:k]].sum() / per_n.sum())}
    out["prompts"].append(rec)
    top = sorted(heads, key=lambda r: -(r["sufficiency"] + r["necessity"]))[:5]
    print(split, idx, rec["score_attribution"], [(r["head"], r["sufficiency"], r["necessity"],
                                                   round(r["score_attribution"], 2)) for r in top],
          {k: (rec[k]["sufficiency"], rec[k]["necessity"], round(rec[k]["attribution_share"], 2))
           for k in rec if k.startswith("mlp34_top") and k.endswith("_patch")}, flush=True)
    Path(out_path).write_text(json.dumps(out, indent=1))
