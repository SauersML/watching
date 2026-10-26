"""Model registry and accessors, so the same experiments run on Qwen3-8B, Qwen3-32B (dense, same family) and
Qwen3.6-27B (hybrid: Gated DeltaNet linear attention in 3 of every 4 layers, a vision wrapper, 3-D rotary
positions). Qwen3.6 needs a Triton 3.7.1 env (fla's backward is wrong on Hopper with Triton 3.6)."""

from __future__ import annotations

import os

import torch

MODELS = {"qwen3-8b": "Qwen3-8B", "qwen3-32b": "Qwen3-32B", "qwen3.6-27b": "Qwen3.6-27B"}


def path(name: str) -> str:
    if name.startswith("/") or name not in MODELS:  # an explicit path or a Hub repo id
        return name
    root = os.environ.get("SIDECHAN_HF_ROOT", "/vol/hf/hub")  # local HF cache if present
    d = f"{root}/models--Qwen--{MODELS[name]}/snapshots"
    if os.path.isdir(d):
        return os.path.join(d, sorted(os.listdir(d))[0])
    return f"Qwen/{MODELS[name]}"  # otherwise fall back to the Hugging Face Hub id


def load(name: str):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText

    p = path(name)
    cls = (
        AutoModelForImageTextToText if hasattr(AutoConfig.from_pretrained(p), "vision_config") else AutoModelForCausalLM
    )
    m = cls.from_pretrained(p, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    m.requires_grad_(False)
    return m


class Arch:
    """Text layers, final norm, output matrix, rotary embedding, and the index of the first of the last two blocks."""

    def __init__(self, model):
        inner = model.get_base_model().model if hasattr(model, "get_base_model") else model.model
        self.mrope = hasattr(inner, "language_model")
        self.tm = inner.language_model if self.mrope else inner
        self.layers = self.tm.layers
        self.norm = self.tm.norm
        self.lm_head = (model.get_base_model() if hasattr(model, "get_base_model") else model).lm_head
        self.L = len(self.layers) - 2

    def rotary(self, x, pos):
        """pos [1, T] text positions; Qwen3.5/3.6 take 3-D (temporal, height, width) positions, equal for text."""
        return self.tm.rotary_emb(x, pos[None].expand(3, *pos.shape) if self.mrope else pos)
