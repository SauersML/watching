"""What did training change? Per-head and per-matrix weight-delta norms, trained vs original."""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import tail_lib as T  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
orig, _ = T.load_original()
trained, _ = T.load_trained(ROOT / "dl/native_tail.pt")
D = T.HEAD_DIM


def fro(x):
    return float(x.float().norm())


def lowrank_fro(A, B):
    """||A @ B||_F for A [n, r], B [r, m] without forming the product."""
    A, B = A.double(), B.double()
    return float(torch.trace((A.T @ A) @ (B @ B.T)).clamp_min(0).sqrt())


def ov_delta(Lo, Lt, h):
    kv = h // T.GROUPS
    Vo = Lo["self_attn.v_proj.weight"][kv * D:(kv + 1) * D]
    Vt = Lt["self_attn.v_proj.weight"][kv * D:(kv + 1) * D]
    Oo = Lo["self_attn.o_proj.weight"][:, h * D:(h + 1) * D]
    Ot = Lt["self_attn.o_proj.weight"][:, h * D:(h + 1) * D]
    base = lowrank_fro(Oo, Vo)
    delta = lowrank_fro(torch.cat([Ot, -Oo], 1), torch.cat([Vt, Vo], 0))
    return base, delta


def qk_delta(Lo, Lt, h):
    kv = h // T.GROUPS
    Qo = Lo["self_attn.q_proj.weight"][h * D:(h + 1) * D]
    Qt = Lt["self_attn.q_proj.weight"][h * D:(h + 1) * D]
    Ko = Lo["self_attn.k_proj.weight"][kv * D:(kv + 1) * D]
    Kt = Lt["self_attn.k_proj.weight"][kv * D:(kv + 1) * D]
    base = lowrank_fro(Qo.T, Ko)
    delta = lowrank_fro(torch.cat([Qt.T, -Qo.T], 1), torch.cat([Kt, Ko], 0))
    return base, delta


out = {"per_matrix": {}, "per_head": {}, "per_kv_head": {}, "norms": {}}
for i, b in enumerate((34, 35)):
    Lo, Lt = orig["layers"][i], trained["layers"][i]
    for name in Lo:
        d = Lt[name].float() - Lo[name].float()
        out["per_matrix"][f"{b}.{name}"] = {
            "delta_fro": fro(d), "orig_fro": fro(Lo[name]), "rel": fro(d) / fro(Lo[name]),
            "frac_entries_changed": float((d != 0).float().mean()),
        }
    rows = []
    for h in range(T.N_HEADS):
        kv = h // T.GROUPS
        r = {"block": b, "head": h, "kv_head": kv}
        for tag, name, sl in (
            ("W_Q", "self_attn.q_proj.weight", (slice(h * D, (h + 1) * D), slice(None))),
            ("W_O", "self_attn.o_proj.weight", (slice(None), slice(h * D, (h + 1) * D))),
        ):
            d = Lt[name][sl].float() - Lo[name][sl].float()
            r[f"{tag}_delta_fro"] = fro(d)
            r[f"{tag}_rel"] = fro(d) / fro(Lo[name][sl])
        base, delta = ov_delta(Lo, Lt, h)
        r["OV_orig_fro"], r["OV_delta_fro"], r["OV_rel"] = base, delta, delta / base
        base, delta = qk_delta(Lo, Lt, h)
        r["QK_orig_fro"], r["QK_delta_fro"], r["QK_rel"] = base, delta, delta / base
        rows.append(r)
    out["per_head"][str(b)] = rows
    kvrows = []
    for kv in range(T.N_KV):
        r = {"block": b, "kv_head": kv}
        for tag, name in (("W_K", "self_attn.k_proj.weight"), ("W_V", "self_attn.v_proj.weight")):
            sl = slice(kv * D, (kv + 1) * D)
            d = Lt[name][sl].float() - Lo[name][sl].float()
            r[f"{tag}_delta_fro"], r[f"{tag}_rel"] = fro(d), fro(d) / fro(Lo[name][sl])
        kvrows.append(r)
    out["per_kv_head"][str(b)] = kvrows
d = trained["norm"].float() - orig["norm"].float()
out["norms"]["final_norm"] = {"delta_fro": fro(d), "rel": fro(d) / fro(orig["norm"])}

for b in ("34", "35"):
    rows = out["per_head"][b]
    for key in ("OV_rel", "QK_rel", "W_O_rel", "W_Q_rel"):
        ranked = sorted(rows, key=lambda r: -r[key])
        med = sorted(r[key] for r in rows)[16]
        print(b, key, "top3", [(r["head"], round(r[key], 5)) for r in ranked[:3]], "median", round(med, 5))
    print(b, "kv", [(r["kv_head"], round(r["W_K_rel"], 5), round(r["W_V_rel"], 5)) for r in out["per_kv_head"][b]])
for k, v in out["per_matrix"].items():
    print(k, round(v["rel"], 6), round(v["delta_fro"], 4))
print("final_norm", out["norms"])
(ROOT / "out/weight_diff.json").write_text(json.dumps(out, indent=1))
