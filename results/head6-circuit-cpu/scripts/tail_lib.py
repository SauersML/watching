"""Transparent functional Qwen3-8B tail (blocks 34-35 + final norm + lm_head) with hooks.

The op sequence mirrors transformers 5.18 Qwen3DecoderLayer + sdpa_attention_forward
(repeat_kv, bool mask, math SDPA) so that a no-hook forward is bit-identical to the HF
modules on the same device/dtype. Hooks receive and return tensors at named sites.
"""

import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch.nn.attention import SDPBackend, sdpa_kernel

REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
HF_SNAPSHOT = Path(
    os.environ.get(
        "QWEN3_8B_SNAPSHOT",
        Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots" / REVISION,
    )
)
FIRST = 34
N_HEADS, N_KV, HEAD_DIM, HIDDEN = 32, 8, 128, 4096
GROUPS = N_HEADS // N_KV
EPS = 1e-6
NO_ID, YES_ID = 2753, 9454
USER_ID, IM_START_ID = 872, 151644


def load_original():
    """Original (untrained) blocks 34-35, final norm and lm_head from the pinned snapshot."""
    index = json.loads((HF_SNAPSHOT / "model.safetensors.index.json").read_text())["weight_map"]
    wanted = {
        k: v
        for k, v in index.items()
        if k.startswith(("model.layers.34.", "model.layers.35."))
        or k in ("model.norm.weight", "lm_head.weight")
    }
    tensors = {}
    for shard in sorted(set(wanted.values())):
        with safe_open(HF_SNAPSHOT / shard, framework="pt", device="cpu") as reader:
            for name, s in wanted.items():
                if s == shard:
                    tensors[name] = reader.get_tensor(name)
    layers = []
    for b in (34, 35):
        prefix = f"model.layers.{b}."
        layers.append({k[len(prefix) :]: v for k, v in tensors.items() if k.startswith(prefix)})
    return {"layers": layers, "norm": tensors["model.norm.weight"]}, tensors["lm_head.weight"]


def load_trained(path):
    art = torch.load(path, map_location="cpu", weights_only=True)
    layers = [{}, {}]
    for k, v in art["layers"].items():
        offset, suffix = k.split(".", 1)
        layers[int(offset)][suffix] = v
    return {"layers": layers, "norm": art["norm"]["weight"]}, art


def cast(weights, dtype):
    return {
        "layers": [{k: v.to(dtype) for k, v in L.items()} for L in weights["layers"]],
        "norm": weights["norm"].to(dtype),
    }


def rms_norm(x, w):
    dtype = x.dtype
    h = x.to(torch.float32)
    var = h.pow(2).mean(-1, keepdim=True)
    h = h * torch.rsqrt(var + EPS)
    return w * h.to(dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(h, n_rep):
    b, nkv, s, d = h.shape
    return h[:, :, None, :, :].expand(b, nkv, n_rep, s, d).reshape(b, nkv * n_rep, s, d)


def _hook(hooks, name, x):
    fn = hooks.get(name) if hooks else None
    return x if fn is None else fn(x)


def block(L, x, mask, cos, sin, b, hooks=None, store=None):
    """One Qwen3 decoder block. Site names are prefixed with the block index."""

    def site(name, t):
        t = _hook(hooks, f"{b}.{name}", t)
        if store is not None and ("_keep" not in store or f"{b}.{name}" in store["_keep"]):
            store[f"{b}.{name}"] = t
        return t

    x = site("resid_in", x)
    h = site("ln1", rms_norm(x, L["input_layernorm.weight"]))
    T = h.shape[1]
    q = F.linear(h, L["self_attn.q_proj.weight"]).view(1, T, -1, HEAD_DIM)
    k = F.linear(h, L["self_attn.k_proj.weight"]).view(1, T, -1, HEAD_DIM)
    v = F.linear(h, L["self_attn.v_proj.weight"]).view(1, T, -1, HEAD_DIM)
    q = rms_norm(q, L["self_attn.q_norm.weight"]).transpose(1, 2)
    k = rms_norm(k, L["self_attn.k_norm.weight"]).transpose(1, 2)
    v = v.transpose(1, 2)
    c, s_ = cos.unsqueeze(1), sin.unsqueeze(1)
    q = (q * c) + (rotate_half(q) * s_)
    k = (k * c) + (rotate_half(k) * s_)
    k = repeat_kv(k, GROUPS)
    v = repeat_kv(v, GROUPS)
    # Per-query-head copies: [1, 32, T, 128]. Patching one head's slice affects only it.
    q = site("q", q)
    k = site("k", k)
    v = site("v", v)
    pattern_fn = hooks.get(f"{b}.pattern") if hooks else None
    if pattern_fn is None:
        with sdpa_kernel(SDPBackend.MATH):
            z = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=0.0, scale=HEAD_DIM**-0.5, is_causal=False
            )
    else:
        z = pattern_fn(q, k, v, mask)
    z = z.transpose(1, 2).contiguous()  # [1, T, 32, 128]
    z = site("z", z.reshape(1, T, -1).contiguous())  # o_proj input
    a = site("attn_out", F.linear(z, L["self_attn.o_proj.weight"]))
    x = site("resid_mid", x + a)
    h2 = site("ln2", rms_norm(x, L["post_attention_layernorm.weight"]))
    act = site(
        "mlp_act",
        F.silu(F.linear(h2, L["mlp.gate_proj.weight"])) * F.linear(h2, L["mlp.up_proj.weight"]),
    )
    m = site("mlp_out", F.linear(act, L["mlp.down_proj.weight"]))
    x = site("resid_out", x + m)
    return x


def forward(W, head, state, hooks=None, store=None):
    """Full tail forward. Returns last-position logits [vocab] (in the run dtype)."""
    x, mask, cos, sin = state
    for i, L in enumerate(W["layers"]):
        x = block(L, x, mask, cos, sin, FIRST + i, hooks, store)
    h = rms_norm(x, W["norm"])
    h = _hook(hooks, "final_norm", h)
    if store is not None and ("_keep" not in store or "final_norm" in store["_keep"]):
        store["final_norm"] = h
    h = h[:, -1:, :]
    return F.linear(h, head)[:, -1, :][0]


def gap(logits):
    lf = logits.float()
    return float(lf[YES_ID] - lf[NO_ID])


def manual_attention(q, k, v, mask, probs_override=None):
    """Explicit FP32 softmax attention (math SDPA upcasts BF16 to FP32 internally).

    probs_override(probs) may replace the attention probabilities [1, H, T, T].
    """
    dtype = q.dtype
    qf, kf, vf = q.float(), k.float(), v.float()
    scores = (qf @ kf.transpose(-2, -1)) * (HEAD_DIM**-0.5)
    scores = scores.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0)
    if probs_override is not None:
        probs = probs_override(probs)
    return (probs @ vf).to(dtype)


def attention_probs(q, k, mask):
    scores = (q.float() @ k.float().transpose(-2, -1)) * (HEAD_DIM**-0.5)
    scores = scores.masked_fill(~mask, float("-inf"))
    return torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)


def role_position(batch):
    valid = batch["attention_mask"][0].nonzero().flatten().tolist()
    ids = batch["input_ids"][0]
    if int(ids[valid[0]]) != IM_START_ID or int(ids[valid[1]]) != USER_ID:
        raise ValueError("Expected leading <|im_start|> user")
    return valid[0], valid[1]


def to_dtype(state, dtype):
    x, mask, cos, sin = state
    return (x.to(dtype), mask, cos.to(dtype), sin.to(dtype))


def replace_rows(cur, donor, pos, heads=None, head_dim=HEAD_DIM, layout="flat"):
    """Copy donor values at sequence position(s) pos (and optional head slices).

    layout 'flat': [1, T, D]; 'heads': [1, H, T, d].
    """
    out = cur.clone()
    if layout == "flat":
        if heads is None:
            out[:, pos, :] = donor[:, pos, :]
        else:
            for h in heads:
                sl = slice(h * head_dim, (h + 1) * head_dim)
                out[:, pos, sl] = donor[:, pos, sl]
    else:
        hs = list(range(cur.shape[1])) if heads is None else heads
        for h in hs:
            out[:, h, pos, :] = donor[:, h, pos, :]
    return out


def logsumexp_gap_stats(logits):
    lp = logits.float().log_softmax(-1)
    return {
        "yes_minus_no": float(logits.float()[YES_ID] - logits.float()[NO_ID]),
        "logp_yes": float(lp[YES_ID]),
        "logp_no": float(lp[NO_ID]),
        "top_token": int(logits.argmax()),
    }


def finite(x):
    return x if math.isfinite(x) else None
