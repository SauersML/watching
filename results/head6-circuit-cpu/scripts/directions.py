"""Which direction of the user-row difference matters, and how head 6 maps to Yes-No.

Usage: directions.py OUT.json split:idx ...   (BF16 trained tail; analysis maps in FP32)

Low-rank patches replace the user row by recipient + P(delta) (sufficiency) or
present - P(delta) (necessity), where P projects onto a chosen subspace.
"""

import json
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
W, _ = T.load_trained(ROOT / "dl/native_tail.pt")
_, head = T.load_original()
L35 = W["layers"][1]
L34 = W["layers"][0]
f32 = lambda t: t.float()  # noqa: E731

g1 = f32(L35["input_layernorm.weight"])
WK = f32(L35["self_attn.k_proj.weight"][KV * D : (KV + 1) * D])
WV = f32(L35["self_attn.v_proj.weight"][KV * D : (KV + 1) * D])
WQ = f32(L35["self_attn.q_proj.weight"][H * D : (H + 1) * D])
WO = f32(L35["self_attn.o_proj.weight"][:, H * D : (H + 1) * D])
A_K = WK * g1  # effective key read-in (before 1/rms, k_norm and rope)
A_V = WV * g1
_, SK, VKt = torch.linalg.svd(A_K, full_matrices=False)  # VKt: [128, 4096]
U1, S1, V1t = torch.linalg.svd(A_V, full_matrices=False)
U2, S2, Rt = torch.linalg.svd(WO @ (U1 * S1), full_matrices=False)
VOVt = Rt @ V1t  # right singular vectors of OV = WO @ A_V, [128, 4096]
OV_left = U2  # output directions [4096, 128]
unembed = f32(head[T.YES_ID]) - f32(head[T.NO_ID])
gf = f32(W["norm"])
readout_static = gf * unembed  # Yes-No direction at the final-norm input (up to 1/rms)


def run(state, hooks=None, store=None):
    with torch.no_grad():
        return T.forward(W, head, state, hooks, store)


def key_score_fn(q_ans, cos_u, sin_u):
    """s(x) = q_ans . rope(k_norm(W_K ln1(x))) / sqrt(d) for one residual row x (FP32)."""
    kn = f32(L35["self_attn.k_norm.weight"])

    def s(x):
        h = T.rms_norm(x, g1)
        k = h @ WK.T
        k = T.rms_norm(k, kn)
        k = k * cos_u + T.rotate_half(k) * sin_u
        return (k * q_ans).sum() * D**-0.5

    return s


def _patch_cols(t, donor, cols):
    t = t.clone()
    t[0, ANS, cols] = donor[0, ANS, cols]
    return t


def projector(basis):
    """basis: [k, 4096] orthonormal rows -> function delta -> projection."""
    return lambda d: (d @ basis.T) @ basis


results = {
    "prompts": [],
    "spectra": {"A_K_singular": SK.tolist()[:16], "OV_singular": S2.tolist()[:16]},
}
for split, idx in prompts:
    batch = cache["batches"][split][idx]
    states = cache["states"][split][idx]
    first, user = T.role_position(batch)
    S = [{}, {}]
    base = [run(states[c], store=S[c]) for c in (0, 1)]
    gaps = [T.gap(b) for b in base]
    rec = {"split": split, "index": idx, "gap": gaps[1] - gaps[0], "gaps": gaps}
    x35 = [f32(S[c]["35.resid_in"][0, user]) for c in (0, 1)]
    x34 = [f32(states[c][0][0, user]) for c in (0, 1)]
    d35, d34 = x35[1] - x35[0], x34[1] - x34[0]
    cos_u, sin_u = f32(states[0][2][0, user]), f32(states[0][3][0, user])
    q_ans = [f32(S[c]["35.q"][0, H, ANS]) for c in (0, 1)]

    # Gradient of head-6 user attention logit wrt the block-35 user row (absent and midpoint).
    grads = {}
    for tag, x0 in (("absent", x35[0]), ("mid", (x35[0] + x35[1]) / 2)):
        x = x0.clone().requires_grad_(True)
        s = key_score_fn(q_ans[0], cos_u, sin_u)(x)
        (g,) = torch.autograd.grad(s, x)
        grads[tag] = g
    s_fn = key_score_fn(q_ans[0], cos_u, sin_u)
    with torch.no_grad():
        rec["h6_user_score"] = {
            "absent": float(s_fn(x35[0])),
            "present_key": float(s_fn(x35[1])),
            "linear_pred_delta": float(grads["absent"] @ d35),
            "linear_pred_delta_mid": float(grads["mid"] @ d35),
        }
    # Other scores at the answer row for reference (absent condition).
    k_all = [f32(S[c]["35.k"][0, H]) for c in (0, 1)]
    sc = [(k_all[c] @ q_ans[c]) * D**-0.5 for c in (0, 1)]
    valid = batch["attention_mask"][0].bool()
    rec["h6_scores_answer_row"] = {
        f"c{c}": {
            "user": float(sc[c][user]),
            "max_other": float(sc[c][valid & (torch.arange(len(valid)) != user)].max()),
            "logsumexp_other": float(
                torch.logsumexp(sc[c][valid & (torch.arange(len(valid)) != user)], 0)
            ),
        }
        for c in (0, 1)
    }

    def energy(d, basis):
        return float(((d @ basis.T) ** 2).sum() / (d @ d))

    ghat = grads["mid"] / grads["mid"].norm()
    rec["delta35"] = {
        "l2": float(d35.norm()),
        "x_l2": float(x35[0].norm()),
        "cos_with_score_grad": float(d35 @ ghat / d35.norm()),
        "energy_in_A_K_rowspace_top": {k: energy(d35, VKt[:k]) for k in (1, 4, 16, 128)},
        "energy_in_OV_rowspace_top": {k: energy(d35, VOVt[:k]) for k in (1, 4, 16, 128)},
        "random_baseline_per_dim": 1 / 4096,
    }
    rec["delta34"] = {
        "l2": float(d34.norm()),
        "x_l2": float(x34[0].norm()),
        "energy_in_A_K_rowspace_top128": energy(d34, VKt),
        "cos_with_delta35": float(d34 @ d35 / (d34.norm() * d35.norm())),
    }

    def user_row_patch(site, proj, base_rows):
        """Bidirectional patch of the user row at `site` with only proj(delta)."""
        d = base_rows[1] - base_rows[0]
        pd = proj(d)
        out = []
        for c, sign in ((0, 1.0), (1, -1.0)):

            def fn(t, c=c, sign=sign):
                t = t.clone()
                t[0, user] = (base_rows[c] + sign * pd).to(t.dtype)
                return t

            out.append(T.gap(run(states[c], {site: fn})))
        return {
            "sufficiency": out[0] - gaps[0],
            "necessity": gaps[1] - out[1],
            "realized_l2": float(pd.norm()),
            "fraction_of_delta_l2": float(pd.norm() / d.norm()),
        }

    tests = {}
    ident = lambda d: d  # noqa: E731
    tests["full_delta35"] = user_row_patch("35.resid_in", ident, x35)
    for tag in ("absent", "mid"):
        u = grads[tag] / grads[tag].norm()
        tests[f"rank1_score_grad_{tag}"] = user_row_patch("35.resid_in", projector(u[None]), x35)
        tests[f"minus_score_grad_{tag}"] = user_row_patch(
            "35.resid_in", lambda d, u=u: d - (d @ u) * u, x35
        )
    for k in (1, 2, 4, 8, 16, 32, 128):
        tests[f"A_K_top{k}"] = user_row_patch("35.resid_in", projector(VKt[:k]), x35)
    tests["A_K_complement128"] = user_row_patch("35.resid_in", lambda d: d - projector(VKt)(d), x35)
    for k in (1, 16, 128):
        tests[f"OV_top{k}"] = user_row_patch("35.resid_in", projector(VOVt[:k]), x35)
    # Block-34 input: the actual compiler footprint at the user row.
    tests["full_delta34"] = user_row_patch("34.resid_in", ident, x34)
    # Gradient of the head-6 user score wrt the block-34 user row (through block 34 + block 35 K).
    g34 = {}
    for tag, base_x in (("absent", x34[0]), ("mid", (x34[0] + x34[1]) / 2)):
        x = base_x.clone().requires_grad_(True)
        # Block 34 at the user row depends only on rows [first, user]; run those two rows in FP32.
        rows = f32(states[0][0][0, first : user + 1]).clone()
        rows = torch.cat([rows[:-1], x[None]], 0)[None]
        Wf = T.cast({"layers": [L34], "norm": W["norm"]}, torch.float32)["layers"][0]
        m = torch.ones(1, 1, 2, 2, dtype=torch.bool).tril()
        y = T.block(
            Wf,
            rows,
            m,
            f32(states[0][2][:, first : user + 1]),
            f32(states[0][3][:, first : user + 1]),
            34,
        )
        if tag == "absent":
            rec["block34_two_row_check_rel_l2_vs_bf16_full"] = float(
                (y[0, -1].detach() - x35[0]).norm() / x35[0].norm()
            )
        s = s_fn(y[0, -1])
        (g,) = torch.autograd.grad(s, x)
        g34[tag] = g
    with torch.no_grad():
        rec["block34_jacobian"] = {
            "score_grad_norm_wrt_x34": float(g34["mid"].norm()),
            "score_grad_norm_wrt_x35": float(grads["mid"].norm()),
            "linear_pred_delta_score_from_d34": float(g34["mid"] @ d34),
            "linear_pred_delta_score_if_d34_bypassed_block34": float(grads["mid"] @ d34),
            "cos_d34_with_grad34": float(d34 @ g34["mid"] / (d34.norm() * g34["mid"].norm())),
        }
    for tag in ("absent", "mid"):
        u = g34[tag] / g34[tag].norm()
        tests[f"x34_rank1_score_grad_{tag}"] = user_row_patch(
            "34.resid_in", projector(u[None]), x34
        )
        tests[f"x34_minus_score_grad_{tag}"] = user_row_patch(
            "34.resid_in", lambda d, u=u: d - (d @ u) * u, x34
        )
    # Random rank-1 control at x34 with the same realized norm as the grad-direction component.
    gen = torch.Generator().manual_seed(idx)
    r = torch.randn(4096, generator=gen)
    r = r / r.norm()
    tests["x34_rank1_random"] = user_row_patch("34.resid_in", projector(r[None]), x34)
    rec["patches"] = tests

    # ---- downstream readout ----------------------------------------------------------------
    xf = [f32(S[c]["35.resid_out"][0, ANS]) for c in (0, 1)]
    rms = [float(x.pow(2).mean().sqrt()) for x in xf]
    z6 = [f32(S[c]["35.z"][0, ANS, H * D : (H + 1) * D]) for c in (0, 1)]
    o6 = [WO @ z for z in z6]
    v_user = [f32(S[c]["35.v"][0, H, user]) for c in (0, 1)]
    mlp = [f32(S[c]["35.mlp_out"][0, ANS]) for c in (0, 1)]
    attn = [f32(S[c]["35.attn_out"][0, ANS]) for c in (0, 1)]
    ro = [readout_static / r_ for r_ in rms]
    rmid = readout_static / ((rms[0] + rms[1]) / 2)
    dla = lambda v: float(v @ rmid)  # noqa: E731
    rec["downstream"] = {
        "final_rms": rms,
        "h6_out_l2": [float(o.norm()) for o in o6],
        "h6_DLA": [float(o6[c] @ ro[c]) for c in (0, 1)],
        "h6_DLA_delta_mid": dla(o6[1] - o6[0]),
        "attn35_DLA_delta_mid": dla(attn[1] - attn[0]),
        "mlp35_DLA_delta_mid": dla(mlp[1] - mlp[0]),
        "resid34_DLA_delta_mid": dla(
            f32(S[1]["35.resid_in"][0, ANS]) - f32(S[0]["35.resid_in"][0, ANS])
        ),
        "cos_WO_vuser_with_readout": float(
            (WO @ v_user[1]) @ readout_static / ((WO @ v_user[1]).norm() * readout_static.norm())
        ),
        "WO_vuser_DLA": float((WO @ v_user[1]) @ rmid),
        "cos_z6_present_with_vuser": float(z6[1] @ v_user[1] / (z6[1].norm() * v_user[1].norm())),
        "cos_delta_h6out_with_OV_top1_left": float(
            (o6[1] - o6[0]) @ OV_left[:, 0] / (o6[1] - o6[0]).norm()
        ),
    }
    # Absent-condition head-6 output: DLA contributions per attended position.
    p_abs = T.attention_probs(
        S[0]["35.q"][:, H : H + 1, ANS:], S[0]["35.k"][:, H : H + 1], states[0][1][..., ANS:, :]
    )[0, 0, 0]
    contrib = [
        (float(p_abs[j] * ((WO @ f32(S[0]["35.v"][0, H, j])) @ ro[0])), j)
        for j in range(len(p_abs))
        if p_abs[j] > 0.01
    ]
    rec["downstream"]["absent_h6_DLA_by_position"] = [
        {
            "pos": j,
            "rel": j - first,
            "token": int(batch["input_ids"][0, j]),
            "p": float(p_abs[j]),
            "dla": v,
        }
        for v, j in sorted(contrib, key=lambda t: -abs(t[0]))
    ]
    # MLP35 neurons at the answer position: linear DLA attribution and causal top-k patches.
    act = [f32(S[c]["35.mlp_act"][0, ANS]) for c in (0, 1)]
    Wd = f32(L35["mlp.down_proj.weight"])
    per = (act[1] - act[0]) * (rmid @ Wd)
    order = torch.argsort(-per.abs())
    rec["downstream"]["mlp35_neuron_DLA_total"] = float(per.sum())
    rec["downstream"]["mlp35_top_neurons"] = [
        {"neuron": int(i), "dla": float(per[i])} for i in order[:10]
    ]
    for k in (1, 10, 100, 1000):
        sel = order[:k].tolist()
        rec["downstream"][f"mlp35_top{k}_DLA_share"] = float(per[order[:k]].sum() / per.sum())
        outs = []
        for c in (0, 1):
            outs.append(
                T.gap(
                    run(
                        states[c],
                        {
                            "35.mlp_act": lambda t, c=c, sel=sel: _patch_cols(
                                t, S[1 - c]["35.mlp_act"], sel
                            )
                        },
                    )
                )
            )
        rec["downstream"][f"mlp35_top{k}_patch"] = {
            "sufficiency": outs[0] - gaps[0],
            "necessity": gaps[1] - outs[1],
        }
    results["prompts"].append(rec)
    print(
        json.dumps(
            {
                "split": split,
                "index": idx,
                "gap": rec["gap"],
                "score": rec["h6_user_score"],
                "d35": rec["delta35"]["l2"],
                **{
                    k: (round(v["sufficiency"], 2), round(v["necessity"], 2))
                    for k, v in tests.items()
                },
            }
        ),
        flush=True,
    )
    Path(out_path).write_text(json.dumps(results, indent=1))
