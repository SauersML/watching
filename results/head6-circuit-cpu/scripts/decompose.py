"""L35H6 decomposition: attention pattern vs value (OV) path, plus mediation and downstream.

Usage: decompose.py OUT.json PRECISION(bf16|fp32) STAGE(trained|original) split:idx ...
All patches are bidirectional: sufficiency = gap(absent patched with present donor) - gap(absent);
necessity = gap(present) - gap(present patched with absent donor). Gaps are Yes-minus-No logits (nats).
"""

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import tail_lib as T  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
H = 6  # block 35 query head
ANS = -1
torch.set_num_threads(8)

out_path, precision, stage = sys.argv[1], sys.argv[2], sys.argv[3]
prompts = [(s.split(":")[0], int(s.split(":")[1])) for s in sys.argv[4:]]
dtype = torch.bfloat16 if precision == "bf16" else torch.float32

cache = torch.load(ROOT / "dl/cache_states.pt", map_location="cpu", weights_only=True)
if stage == "trained":
    W, _ = T.load_trained(ROOT / "dl/native_tail.pt")
    _, head = T.load_original()
else:
    W, head = T.load_original()
W = T.cast(W, dtype)
head = head.to(dtype)


def run(state, hooks=None, store=None):
    with torch.no_grad():
        return T.forward(W, head, state, hooks, store)


def row_patch(site, donor, pos, heads=None, layout="flat"):
    return {site: lambda t: T.replace_rows(t, donor, pos, heads, layout=layout)}


def pattern_hook(donor_probs=None, mode="full", base_probs=None):
    """Recompute head 6's answer-row output manually, optionally overriding its attention row."""

    def fn(q, k, v, mask):
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel(SDPBackend.MATH):
            z = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, scale=T.HEAD_DIM**-0.5
            )
        p = T.attention_probs(q[:, H : H + 1, ANS:], k[:, H : H + 1], mask[..., ANS:, :])
        if mode == "full":
            p = donor_probs.clone()
        elif mode.startswith("user"):
            u = int(mode.split(":")[1])
            new_u = donor_probs[..., u]
            scale = (1 - new_u) / (1 - p[..., u])
            p = p * scale.unsqueeze(-1)
            p[..., u] = new_u
        out = (p @ v[:, H : H + 1].float()).to(z.dtype)
        z = z.clone()
        z[:, H : H + 1, ANS:] = out
        return z

    return {"35.pattern": fn}


def merge(*ds):
    r = {}
    for d in ds:
        for k, v in d.items():
            if k in r:
                raise ValueError(f"duplicate hook {k}")
            r[k] = v
    return r


results = {"precision": precision, "stage": stage, "prompts": []}
t_start = time.time()
for split, idx in prompts:
    batch = cache["batches"][split][idx]
    states = [T.to_dtype(s, dtype) for s in cache["states"][split][idx]]
    first, user = T.role_position(batch)
    mask = states[0][1]
    stores = [{}, {}]
    base = [run(states[c], store=stores[c]) for c in (0, 1)]
    gaps = [T.gap(b) for b in base]
    rec = {"split": split, "index": idx, "user_pos": user, "first_pos": first,
           "gap_absent": gaps[0], "gap_present": gaps[1], "gap": gaps[1] - gaps[0]}

    # Attention weights at the answer position for every head in both blocks.
    attn = {}
    for b in (34, 35):
        for c in (0, 1):
            p = T.attention_probs(stores[c][f"{b}.q"][:, :, ANS:], stores[c][f"{b}.k"],
                                  mask[..., ANS:, :])[0, :, 0]  # [32, T]
            attn[f"{b}_c{c}"] = p
    rec["attn_user"] = {k: v[:, user].tolist() for k, v in attn.items()}
    rec["attn_first"] = {k: v[:, first].tolist() for k, v in attn.items()}
    p6 = [attn[f"35_c{c}"][H] for c in (0, 1)]
    top = torch.argsort(-(p6[0] + p6[1]))[:6].tolist()
    rec["h6_top_positions"] = [{"pos": j, "rel": j - first, "token": int(batch["input_ids"][0, j]),
                                "p_absent": float(p6[0][j]), "p_present": float(p6[1][j])}
                               for j in top]

    def effect(name, make_hooks, extra=None):
        """make_hooks(c) -> hooks for recipient condition c (donor = 1-c)."""
        g = []
        for c in (0, 1):
            g.append(T.gap(run(states[c], make_hooks(c))))
        row = {"name": name, "sufficiency": g[0] - gaps[0], "necessity": gaps[1] - g[1],
               "patched_absent": g[0], "patched_present": g[1]}
        if extra:
            row.update(extra)
        rec.setdefault("interventions", []).append(row)
        return row

    S = stores
    # --- (1) head-6 decomposition at block 35 -------------------------------------------
    effect("h6_z_answer", lambda c: row_patch("35.z", S[1 - c]["35.z"], ANS, [H]))
    effect("h6_K_user", lambda c: row_patch("35.k", S[1 - c]["35.k"], user, [H], "heads"))
    effect("h6_V_user", lambda c: row_patch("35.v", S[1 - c]["35.v"], user, [H], "heads"))
    effect("h6_KV_user", lambda c: merge(row_patch("35.k", S[1 - c]["35.k"], user, [H], "heads"),
                                         row_patch("35.v", S[1 - c]["35.v"], user, [H], "heads")))
    effect("h6_Q_answer", lambda c: row_patch("35.q", S[1 - c]["35.q"], ANS, [H], "heads"))
    allpos = slice(None)
    effect("h6_K_all", lambda c: row_patch("35.k", S[1 - c]["35.k"], allpos, [H], "heads"))
    effect("h6_V_all", lambda c: row_patch("35.v", S[1 - c]["35.v"], allpos, [H], "heads"))
    effect("h6_QK_all", lambda c: merge(row_patch("35.k", S[1 - c]["35.k"], allpos, [H], "heads"),
                                        row_patch("35.q", S[1 - c]["35.q"], ANS, [H], "heads")))
    nonuser = [j for j in range(states[0][0].shape[1]) if j != user]
    effect("h6_V_all_but_user",
           lambda c: row_patch("35.v", S[1 - c]["35.v"], nonuser, [H], "heads"))
    effect("h6_K_all_but_user",
           lambda c: row_patch("35.k", S[1 - c]["35.k"], nonuser, [H], "heads"))
    # Attention-pattern swap via manual FP32 softmax for head 6's answer row only.
    donor_p = [T.attention_probs(S[c]["35.q"][:, H:H + 1, ANS:], S[c]["35.k"][:, H:H + 1],
                                 mask[..., ANS:, :]) for c in (0, 1)]
    effect("h6_manual_noop", lambda c: pattern_hook(mode="noop"))
    effect("h6_pattern_full", lambda c: pattern_hook(donor_p[1 - c], "full"))
    effect("h6_pattern_user_weight", lambda c: pattern_hook(donor_p[1 - c], f"user:{user}"))

    # --- (2) downstream of head 6: direct vs block-35 MLP-mediated --------------------------
    effect("h6_z_answer_mlp35_frozen",
           lambda c: merge(row_patch("35.z", S[1 - c]["35.z"], ANS, [H]),
                           row_patch("35.mlp_out", S[c]["35.mlp_out"], ANS)))
    effect("mlp35_answer", lambda c: row_patch("35.mlp_out", S[1 - c]["35.mlp_out"], ANS))
    effect("attn35_answer", lambda c: row_patch("35.attn_out", S[1 - c]["35.attn_out"], ANS))

    # --- (3) user-token input patch and mediation through head 6 ----------------------------
    def user_in(c):
        x = states[c][0].clone()
        x[:, user] = states[1 - c][0][:, user]
        return x

    def user_patch(extra_hooks_fn=None):
        def make(c):
            h = {"34.resid_in": (lambda t, c=c: user_in(c))}
            if extra_hooks_fn:
                h = merge(h, extra_hooks_fn(c))
            return h
        return make

    effect("user_input", user_patch())
    effect("user_input_hold_h6_z", user_patch(lambda c: row_patch("35.z", S[c]["35.z"], ANS, [H])))
    effect("user_input_hold_h6_K_user",
           user_patch(lambda c: row_patch("35.k", S[c]["35.k"], user, [H], "heads")))
    effect("user_input_hold_h6_V_user",
           user_patch(lambda c: row_patch("35.v", S[c]["35.v"], user, [H], "heads")))
    effect("user_input_hold_h6_KV_user",
           user_patch(lambda c: merge(row_patch("35.k", S[c]["35.k"], user, [H], "heads"),
                                      row_patch("35.v", S[c]["35.v"], user, [H], "heads"))))
    effect("user_input_hold_h6_Q_answer",
           user_patch(lambda c: row_patch("35.q", S[c]["35.q"], ANS, [H], "heads")))
    # Block-34 role: patch the user row at the block-35 input only (after block 34).
    effect("user_resid35_only", lambda c: row_patch("35.resid_in", S[1 - c]["35.resid_in"], user))
    # Raw (un-amplified) delta: block-35 input user row := recipient + block-34-input delta.
    def raw_delta(c):
        d = states[1 - c][0][:, user].float() - states[c][0][:, user].float()
        def fn(t):
            t = t.clone()
            t[:, user] = (t[:, user].float() + d).to(t.dtype)
            return t
        return {"35.resid_in": fn}
    effect("user_raw_delta_at_resid35", raw_delta)
    # User input patched but block-35 input user row held: only block-34 spread to other rows.
    effect("user_input_hold_resid35_user",
           user_patch(lambda c: row_patch("35.resid_in", S[c]["35.resid_in"], user)))
    # Block-34 sublayer contributions at the user row.
    effect("user_attn34_out_only",
           lambda c: row_patch("34.attn_out", S[1 - c]["34.attn_out"], user))
    effect("user_mlp34_out_only",
           lambda c: row_patch("34.mlp_out", S[1 - c]["34.mlp_out"], user))

    # Norms of the user-row difference at each residual site.
    norms = {}
    for site in ("34.resid_in", "34.attn_out", "34.resid_mid", "34.mlp_out", "35.resid_in",
                 "35.attn_out", "35.mlp_out", "35.resid_out"):
        a, p = S[0][site][0, user].float(), S[1][site][0, user].float()
        norms[site] = {"delta_l2": float((p - a).norm()), "absent_l2": float(a.norm()),
                       "cos_to_input_delta": None}
    d34 = (S[1]["34.resid_in"][0, user].float() - S[0]["34.resid_in"][0, user].float())
    for site in norms:
        d = S[1][site][0, user].float() - S[0][site][0, user].float()
        norms[site]["cos_to_input_delta"] = float(d @ d34 / (d.norm() * d34.norm() + 1e-30))
    rec["user_row_norms"] = norms
    # Head-6 K/V/Q differences.
    rec["h6_vec_deltas"] = {
        "K_user_delta_l2": float((S[1]["35.k"][0, H, user].float() - S[0]["35.k"][0, H, user].float()).norm()),
        "K_user_l2": float(S[0]["35.k"][0, H, user].float().norm()),
        "V_user_delta_l2": float((S[1]["35.v"][0, H, user].float() - S[0]["35.v"][0, H, user].float()).norm()),
        "V_user_l2": float(S[0]["35.v"][0, H, user].float().norm()),
        "Q_answer_delta_l2": float((S[1]["35.q"][0, H, ANS].float() - S[0]["35.q"][0, H, ANS].float()).norm()),
        "Q_answer_l2": float(S[0]["35.q"][0, H, ANS].float().norm()),
        "z_answer_delta_l2": float((S[1]["35.z"][0, ANS, H * 128:(H + 1) * 128].float()
                                    - S[0]["35.z"][0, ANS, H * 128:(H + 1) * 128].float()).norm()),
        "z_answer_l2": [float(S[c]["35.z"][0, ANS, H * 128:(H + 1) * 128].float().norm()) for c in (0, 1)],
    }
    rec["seconds"] = time.time() - t_start
    results["prompts"].append(rec)
    brief = {r["name"]: (round(r["sufficiency"], 3), round(r["necessity"], 3)) for r in rec["interventions"]}
    print(json.dumps({"split": split, "index": idx, "gap": rec["gap"],
                      "h6_attn_user": [rec["attn_user"]["35_c0"][H], rec["attn_user"]["35_c1"][H]],
                      **brief}), flush=True)
    Path(out_path).write_text(json.dumps(results, indent=1))
