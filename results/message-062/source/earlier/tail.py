"""Eager Qwen3-8B tail (blocks 34-35, final norm, fixed head) on cached block-34 inputs."""
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RMSNorm

MODEL = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
RUNS = Path.home() / "mpd-data/runpod"
TAIL_034 = RUNS / "watching-two-block-scaled-034/out/fit/native_tail.pt"
TAIL_045 = RUNS / "watching-detector-null-045/out/fit/native_tail.pt"
CACHE_034 = RUNS / "watching-two-block-scaled-034/out/fit/cache_states.pt"
NO, YES = 2753, 9454


def config():
    c = Qwen3Config.from_pretrained(MODEL, local_files_only=True)
    c._attn_implementation = "sdpa"
    return c


def base_tensors(names):
    index = json.loads((MODEL / "model.safetensors.index.json").read_text())["weight_map"]
    out = {}
    for name in names:
        with safe_open(MODEL / index[name], framework="pt", device="cpu") as f:
            out[name] = f.get_tensor(name)
    return out


class Tail(torch.nn.Module):
    def __init__(self, cfg, head):
        super().__init__()
        self.layers = torch.nn.ModuleList([Qwen3DecoderLayer(cfg, 34), Qwen3DecoderLayer(cfg, 35)])
        self.norm = Qwen3RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.register_buffer("head", head, persistent=False)

    def hidden(self, h, mask, cos, sin, pos):
        with sdpa_kernel(SDPBackend.MATH):
            for layer in self.layers:
                h = layer(h, attention_mask=mask, position_ids=pos, position_embeddings=(cos, sin), use_cache=False)
        return self.norm(h)

    def forward(self, h, mask, cos, sin, pos):
        return F.linear(self.hidden(h, mask, cos, sin, pos), self.head)


_HEAD = {}


def load_tail(which, dtype=torch.bfloat16, device="mps"):
    """which: 'base' or a path to a native_tail.pt artifact."""
    cfg = config()
    if "head" not in _HEAD:
        _HEAD["head"] = base_tensors(["lm_head.weight"])["lm_head.weight"].to(device=device, dtype=torch.bfloat16)
    tail = Tail(cfg, _HEAD["head"])
    if which == "base":
        names = [f"model.layers.{34 + i}.{k[2:]}" for i in (0, 1) for k in _keys()] + ["model.norm.weight"]
        raw = base_tensors(names)
        state = {f"layers.{i}.{k[2:]}": raw[f"model.layers.{34 + i}.{k[2:]}"] for i in (0, 1) for k in _keys()}
        state["norm.weight"] = raw["model.norm.weight"]
    else:
        art = torch.load(which, map_location="cpu", weights_only=True, mmap=True)
        assert art["first_layer"] == 34 and not art.get("train_head")
        state = {f"layers.{k}": v for k, v in art["layers"].items()}
        state.update({f"norm.{k}": v for k, v in art["norm"].items()})
    missing, unexpected = tail.load_state_dict(state, strict=False)
    assert not unexpected and all(m == "head" for m in missing), (missing, unexpected)
    return tail.to(device=device, dtype=dtype).eval()


def _keys():
    return [
        "0.self_attn.q_proj.weight", "0.self_attn.k_proj.weight", "0.self_attn.v_proj.weight",
        "0.self_attn.o_proj.weight", "0.self_attn.q_norm.weight", "0.self_attn.k_norm.weight",
        "0.mlp.gate_proj.weight", "0.mlp.up_proj.weight", "0.mlp.down_proj.weight",
        "0.input_layernorm.weight", "0.post_attention_layernorm.weight",
    ]


def load_cache(path=CACHE_034):
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def state_on(cache, split, index, condition, device="mps"):
    h, mask, cos, sin = cache["states"][split][index][condition]
    pos = cache["batches"][split][index]["position_ids"]
    return [t.to(device) for t in (h, mask, cos, sin, pos)]
