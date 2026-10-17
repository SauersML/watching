"""Key geometry at the `user` row of block 35 (FP32 analysis on BF16-run states).

The user row is identical across prompts within a condition, so one prompt suffices for the key;
the answer query differs per prompt. Reports pre-/post-k_norm key norms and deltas for every
kv head, trained vs original, plus the q.k score of head 6 for several prompts.
"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import tail_lib as T  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
D = T.HEAD_DIM
torch.set_num_threads(8)
cache = torch.load(ROOT / "dl/cache_states.pt", map_location="cpu", weights_only=True)
TR, _ = T.load_trained(ROOT / "dl/native_tail.pt")
OR, head = T.load_original()
prompts = [("train", 42), ("train", 99), ("test", 8), ("test", 34)]
out = {}
for stage, W in (("trained", TR), ("original", OR)):
    L = {k: v.float() for k, v in W["layers"][1].items()}
    res = {"per_kv_head": [], "h6_scores": []}
    split, idx = prompts[0]
    states = cache["states"][split][idx]
    first, user = T.role_position(cache["batches"][split][idx])
    S = [{"_keep": {"35.resid_in"}}, {"_keep": {"35.resid_in"}}]
    with torch.no_grad():
        for c in (0, 1):
            T.forward(W, head, states[c], store=S[c])
    x = [S[c]["35.resid_in"][0, user].float() for c in (0, 1)]
    h = [T.rms_norm(xx, L["input_layernorm.weight"]) for xx in x]
    for kv in range(T.N_KV):
        Wk = L["self_attn.k_proj.weight"][kv * D : (kv + 1) * D]
        Wv = L["self_attn.v_proj.weight"][kv * D : (kv + 1) * D]
        kp = [Wk @ hh for hh in h]
        kn = [T.rms_norm(k, L["self_attn.k_norm.weight"]) for k in kp]
        vp = [Wv @ hh for hh in h]
        res["per_kv_head"].append(
            {
                "kv_head": kv,
                "pre_norm_key_l2": [float(k.norm()) for k in kp],
                "pre_norm_key_delta_l2": float((kp[1] - kp[0]).norm()),
                "pre_norm_key_relative_delta": float((kp[1] - kp[0]).norm() / kp[0].norm()),
                "post_norm_key_relative_delta": float((kn[1] - kn[0]).norm() / kn[0].norm()),
                "pre_norm_key_cos_abs_pres": float(kp[0] @ kp[1] / (kp[0].norm() * kp[1].norm())),
                "value_l2": [float(v.norm()) for v in vp],
                "value_relative_delta": float((vp[1] - vp[0]).norm() / vp[0].norm()),
            }
        )
    res["ln1_relative_delta_user"] = float((h[1] - h[0]).norm() / h[0].norm())
    res["resid35_relative_delta_user"] = float((x[1] - x[0]).norm() / x[0].norm())
    out[stage] = res
    kv1 = res["per_kv_head"][1]
    print(
        stage,
        "resid rel",
        round(res["resid35_relative_delta_user"], 4),
        "kv1 pre-norm |k|",
        [round(v, 2) for v in kv1["pre_norm_key_l2"]],
        "rel dk pre",
        round(kv1["pre_norm_key_relative_delta"], 4),
        "post",
        round(kv1["post_norm_key_relative_delta"], 4),
        "|v|",
        [round(v, 1) for v in kv1["value_l2"]],
        flush=True,
    )
    print(
        "  all kv pre-norm |k| absent",
        [round(r["pre_norm_key_l2"][0], 2) for r in res["per_kv_head"]],
    )
    print(
        "  all kv rel dk pre",
        [round(r["pre_norm_key_relative_delta"], 4) for r in res["per_kv_head"]],
    )
(ROOT / "out/key_geometry.json").write_text(json.dumps(out, indent=1))
