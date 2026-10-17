"""Numbers drawn in the README's how-it-works figure (BF16 runs, FP32 geometry).

For three prompts: the tokens, head 6's attention at the answer position with and without
the probe, and the change in every token's block-35 input. For the `user` row: the pre-norm
keys of key-value heads 0 and 1 and head 6's query direction, in the plane of the two keys.
Usage: figure_panels.py [DL_DIR]; writes out/figure_panels.json.
"""

import json
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).parent))
import tail_lib as T  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DL = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "dl"
H, D = 6, T.HEAD_DIM
PROMPTS = [("test", 8), ("test", 19), ("train", 42)]
torch.set_num_threads(8)
cache = torch.load(DL / "cache_states.pt", map_location="cpu", weights_only=True)
TR, _ = T.load_trained(DL / "native_tail.pt")
OR, head = T.load_original()
tok = Tokenizer.from_file(str(T.HF_SNAPSHOT / "tokenizer.json"))
KEEP = {"35.resid_in", "35.q", "35.k"}


def run(W, state):
    store = {"_keep": KEEP}
    with torch.no_grad():
        logits = T.forward(W, head, state, store=store)
    return logits, store


def pre_norm_key(L, x, kv):
    h = T.rms_norm(x.float(), L["input_layernorm.weight"].float())
    return L["self_attn.k_proj.weight"][kv * D : (kv + 1) * D].float() @ h


def query_direction(L, q_answer, cos_user, sin_user):
    """g such that head 6's score for a pre-norm key k at `user` is g . k / |k|."""
    w = L["self_attn.k_norm.weight"].float()
    g = torch.empty(D)
    for i in range(D):
        e = torch.zeros(D)
        e[i] = 10.0
        k = T.rms_norm(e, w)
        k = k * cos_user + T.rotate_half(k) * sin_user
        g[i] = q_answer @ k / D**0.5
    return g


rows = []
for split, idx in PROMPTS:
    batch = cache["batches"][split][idx]
    state = cache["states"][split][idx]
    _, user = T.role_position(batch)
    valid = batch["attention_mask"][0].nonzero().flatten().tolist()
    ids = batch["input_ids"][0].tolist()
    runs = [run(TR, state[c]) for c in (0, 1)]
    mask = state[0][1]
    attention, scores = [], []
    for _, store in runs:
        q, k = store["35.q"][:, H].float(), store["35.k"][:, H].float()
        row = (q[0, -1] @ k[0].T) / D**0.5
        probs = T.attention_probs(q, k, mask[:, 0] if mask.dim() == 4 else mask)[0, -1]
        attention.append([float(probs[p]) for p in valid])
        scores.append([float(row[p]) for p in valid])
    x = [store["35.resid_in"][0].float() for _, store in runs]
    change = (x[1] - x[0])[valid]
    user_change = x[1][user] - x[0][user]
    rows.append(
        {
            "prompt": [split, idx],
            "tokens": [tok.decode([ids[p]], skip_special_tokens=False) for p in valid],
            "user_index": valid.index(user),
            "gap": [T.gap(logits) for logits, _ in runs],
            "attention_absent": attention[0],
            "attention_present": attention[1],
            "score_absent": scores[0],
            "score_present": scores[1],
            "change_norm": [float(v) for v in change.norm(dim=-1)],
            "state_norm": [float(v) for v in x[0][valid].norm(dim=-1)],
            "change_cos_with_user": [
                float(v @ user_change / (v.norm() * user_change.norm() + 1e-12)) for v in change
            ],
            "user_change": user_change.tolist(),
            "last_change": (x[1][valid[-1]] - x[0][valid[-1]]).tolist(),
        }
    )
    print(split, idx, rows[-1]["gap"], round(attention[1][valid.index(user)], 4), flush=True)

same = all(
    torch.equal(torch.tensor(r["user_change"]), torch.tensor(rows[0]["user_change"])) for r in rows
)
print("user change identical across prompts:", same)

# Key geometry for the first prompt, in the plane of the two head-6 (kv 1) keys.
split, idx = PROMPTS[0]
batch, state = cache["batches"][split][idx], cache["states"][split][idx]
_, user = T.role_position(batch)
geometry = {}
for stage, W in (("trained", TR), ("original", OR)):
    L = {k: v.float() for k, v in W["layers"][1].items()}
    stores = [run(W, state[c])[1] for c in (0, 1)]
    x = [s["35.resid_in"][0, user] for s in stores]
    keys = {kv: [pre_norm_key(L, xx, kv) for xx in x] for kv in (0, 1)}
    cos_u, sin_u = state[0][2][0, user].float(), state[0][3][0, user].float()
    q = [s["35.q"][0, H, -1].float() for s in stores]
    g = query_direction(L, q[1], cos_u, sin_u)
    a, b = keys[1]
    e1 = a / a.norm()
    e2 = b - (b @ e1) * e1
    e2 = e2 / e2.norm()
    plane = {
        "absent": [float(a @ e1), float(a @ e2)],
        "present": [float(b @ e1), float(b @ e2)],
        "query": [float(g @ e1), float(g @ e2)],
        "query_in_plane_fraction": float(torch.stack([g @ e1, g @ e2]).norm() / g.norm()),
        "score": [float(g @ a / a.norm()), float(g @ b / b.norm())],
    }
    geometry[stage] = {
        "key_length": {kv: [float(k.norm()) for k in keys[kv]] for kv in (0, 1)},
        "key_change": {kv: float((keys[kv][1] - keys[kv][0]).norm()) for kv in (0, 1)},
        "turn_degrees": {
            kv: float(
                torch.rad2deg(
                    torch.arccos(
                        (
                            keys[kv][0] @ keys[kv][1] / (keys[kv][0].norm() * keys[kv][1].norm())
                        ).clamp(-1, 1)
                    )
                )
            )
            for kv in (0, 1)
        },
        "head6_plane": plane,
    }
    print(stage, json.dumps(geometry[stage])[:400], flush=True)

(ROOT / "out/figure_panels.json").write_text(
    json.dumps({"prompts": rows, "geometry": geometry}, indent=1)
)
