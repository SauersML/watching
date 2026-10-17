"""Is the decisive user-row direction universal across prompts? (BF16 trained runs, FP32 maps)

For each prompt: head-6 answer query q_p and the gradient u_p of the head-6 user attention
logit wrt the block-35 user row. Reports cosine similarity across prompts and the effect of a
rank-1 patch along the reference prompt's direction applied to every prompt.
Usage: universal.py OUT.json ref_split:ref_idx split:idx ...
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
ref = (sys.argv[2].split(":")[0], int(sys.argv[2].split(":")[1]))
prompts = [(s.split(":")[0], int(s.split(":")[1])) for s in sys.argv[3:]]
cache = torch.load(ROOT / "dl/cache_states.pt", map_location="cpu", weights_only=True)
W, _ = T.load_trained(ROOT / "dl/native_tail.pt")
_, head = T.load_original()
L = {k: v.float() for k, v in W["layers"][1].items()}
WK = L["self_attn.k_proj.weight"][KV * D:(KV + 1) * D]


def grad_dir(x, q, cos, sin):
    x = x.clone().requires_grad_(True)
    k = T.rms_norm(T.rms_norm(x, L["input_layernorm.weight"]) @ WK.T, L["self_attn.k_norm.weight"])
    k = k * cos + T.rotate_half(k) * sin
    (g,) = torch.autograd.grad((k * q).sum() * D**-0.5, x)
    return g / g.norm()


info = {}
for split, idx in [ref] + prompts:
    states = cache["states"][split][idx]
    first, user = T.role_position(cache["batches"][split][idx])
    S = [{"_keep": {"35.resid_in", "35.q"}}, {"_keep": {"35.resid_in", "35.q"}}]
    with torch.no_grad():
        gaps = [T.gap(T.forward(W, head, states[c], store=S[c])) for c in (0, 1)]
    x = [S[c]["35.resid_in"][0, user].float() for c in (0, 1)]
    q = S[0]["35.q"][0, H, ANS].float()
    u = grad_dir((x[0] + x[1]) / 2, q, states[0][2][0, user].float(), states[0][3][0, user].float())
    info[split, idx] = {"states": states, "user": user, "gaps": gaps, "x": x, "q": q, "u": u}

uref = info[ref]["u"]
rows = []
for key in prompts:
    I = info[key]
    d = I["x"][1] - I["x"][0]
    pd = (d @ uref) * uref
    g = []
    for c, sign in ((0, 1.0), (1, -1.0)):
        def fn(t, c=c, sign=sign):
            t = t.clone()
            t[0, I["user"]] = (I["x"][c] + sign * pd).to(t.dtype)
            return t
        with torch.no_grad():
            g.append(T.gap(T.forward(W, head, I["states"][c], {"35.resid_in": fn})))
    rows.append({"split": key[0], "index": key[1], "gap": I["gaps"][1] - I["gaps"][0],
                 "ref_rank1_sufficiency": g[0] - I["gaps"][0], "ref_rank1_necessity": I["gaps"][1] - g[1],
                 "cos_u_with_ref": float(I["u"] @ uref),
                 "cos_q_with_ref": float(I["q"] @ info[ref]["q"] / (I["q"].norm() * info[ref]["q"].norm())),
                 "user_rows_identical_to_ref": [bool(torch.equal(I["x"][c], info[ref]["x"][c])) for c in (0, 1)]})
    print(json.dumps(rows[-1]), flush=True)
U = torch.stack([info[k]["u"] for k in prompts])
C = U @ U.T
n = len(prompts)
off = (C.sum() - C.diag().sum()) / (n * n - n)
Q = torch.stack([info[k]["q"] / info[k]["q"].norm() for k in prompts])
CQ = Q @ Q.T
offq = (CQ.sum() - CQ.diag().sum()) / (n * n - n)
res = {"reference": list(ref), "rows": rows, "mean_offdiag_cos_grad_dirs": float(off),
       "min_cos_grad_dirs": float(C.min()), "mean_offdiag_cos_queries": float(offq),
       "min_cos_queries": float(CQ.min())}
print({k: v for k, v in res.items() if k != "rows"})
Path(out_path).write_text(json.dumps(res, indent=1))
