"""Dose-response over alpha in [-2, 2] and matched-norm random-direction controls (BF16 trained).

Dose: block-34 input user row := x_absent + alpha * delta34 (absent run), and the same
from the present side (x_present + alpha' * (-delta34)) is implied by symmetry, so only the
absent side is scanned.
Random: x_absent + r, |r| = s * |delta34| at the block-34 input user row, isotropic r;
also at the block-35 input user row with |r| = |delta35|, isotropic and restricted to the
128-d row space of head 6's key projection (W_K kv1 * ln1 gain).
Usage: dose2.py OUT.json DOSE_PROMPTS(comma) RANDOM_PROMPTS(comma) N_DRAWS
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
parse = lambda s: [(p.split(":")[0], int(p.split(":")[1])) for p in s.split(",")]  # noqa: E731
dose_prompts, rand_prompts, n_draws = parse(sys.argv[2]), parse(sys.argv[3]), int(sys.argv[4])
cache = torch.load(ROOT / "dl/cache_states.pt", map_location="cpu", weights_only=True)
W, _ = T.load_trained(ROOT / "dl/native_tail.pt")
_, head = T.load_original()
L35 = W["layers"][1]
A_K = (
    L35["self_attn.k_proj.weight"][KV * D : (KV + 1) * D].float()
    * L35["input_layernorm.weight"].float()
)
_, _, VKt = torch.linalg.svd(A_K, full_matrices=False)
ALPHAS = [round(-2 + 0.1 * i, 1) for i in range(41)]


def measure(state, user, site, vec):
    def fn(t):
        t = t.clone()
        t[0, user] = vec.to(t.dtype)
        return t

    store = {"_keep": {"35.q", "35.k"}}
    with torch.no_grad():
        if site == "34.resid_in":
            logits = T.forward(W, head, state, {site: fn}, store)
        else:
            x35 = PREP["x35"].clone()
            x35[0, user] = vec.to(x35.dtype)
            x = T.block(L35, x35, state[1], state[2], state[3], 35, None, store)
            logits = F.linear(T.rms_norm(x, W["norm"])[:, -1:, :], head)[:, -1, :][0]
    p = T.attention_probs(
        store["35.q"][:, H : H + 1, ANS:], store["35.k"][:, H : H + 1], state[1][..., ANS:, :]
    )[0, 0, 0, user]
    return T.gap(logits), float(p)


rows = []
PREP = {}
for split, idx in sorted(set(dose_prompts) | set(rand_prompts)):
    states = cache["states"][split][idx]
    first, user = T.role_position(cache["batches"][split][idx])
    S = [{"_keep": {"35.resid_in"}}, {"_keep": {"35.resid_in"}}]
    with torch.no_grad():
        gaps = [T.gap(T.forward(W, head, states[c], store=S[c])) for c in (0, 1)]
    PREP["x35"] = S[0]["35.resid_in"]
    x34 = states[0][0][0, user].float()
    d34 = states[1][0][0, user].float() - x34
    x35 = S[0]["35.resid_in"][0, user].float()
    d35 = S[1]["35.resid_in"][0, user].float() - x35
    base = {"split": split, "index": idx, "gap_absent": gaps[0], "gap_present": gaps[1]}
    if (split, idx) in dose_prompts:
        for a in ALPHAS:
            g, p = measure(states[0], user, "34.resid_in", x34 + a * d34)
            rows.append({**base, "kind": "dose", "alpha": a, "gap": g, "h6_p_user": p})
    if (split, idx) in rand_prompts:
        gen = torch.Generator().manual_seed(7)
        for scale in (1.0, 2.0, 3.0, 5.0):
            for k in range(n_draws if scale == 1.0 else n_draws // 2):
                r = torch.randn(4096, generator=gen)
                g, p = measure(
                    states[0], user, "34.resid_in", x34 + r / r.norm() * d34.norm() * scale
                )
                rows.append(
                    {
                        **base,
                        "kind": "random_x34",
                        "scale": scale,
                        "draw": k,
                        "gap": g,
                        "h6_p_user": p,
                    }
                )
        for k in range(n_draws):
            r = torch.randn(4096, generator=gen)
            g, p = measure(states[0], user, "35.resid_in", x35 + r / r.norm() * d35.norm())
            rows.append(
                {**base, "kind": "random_x35", "scale": 1.0, "draw": k, "gap": g, "h6_p_user": p}
            )
            c = torch.randn(128, generator=gen) @ VKt
            g, p = measure(states[0], user, "35.resid_in", x35 + c / c.norm() * d35.norm())
            rows.append(
                {
                    **base,
                    "kind": "random_x35_key_rowspace",
                    "scale": 1.0,
                    "draw": k,
                    "gap": g,
                    "h6_p_user": p,
                }
            )
        for kind in ("random_x34", "random_x35", "random_x35_key_rowspace"):
            for scale in (1.0, 2.0, 3.0, 5.0):
                R = [
                    r
                    for r in rows
                    if r["kind"] == kind
                    and r.get("scale") == scale
                    and r["split"] == split
                    and r["index"] == idx
                ]
                if R:
                    print(
                        split,
                        idx,
                        kind,
                        scale,
                        "n",
                        len(R),
                        "max gap",
                        max(r["gap"] for r in R),
                        "max p_user",
                        round(max(r["h6_p_user"] for r in R), 4),
                        "yes_count",
                        sum(r["gap"] > 0 for r in R),
                        flush=True,
                    )
    if (split, idx) in dose_prompts:
        print(
            split,
            idx,
            "dose",
            [
                (r["alpha"], r["gap"], round(r["h6_p_user"], 3))
                for r in rows
                if r["kind"] == "dose" and r["split"] == split and r["index"] == idx
            ][::4],
            flush=True,
        )
    Path(out_path).write_text(json.dumps({"precision": "bf16", "alphas": ALPHAS, "rows": rows}))
