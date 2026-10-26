"""Runtime conditions a model might read from its own numbers, step 0: which ones change the numbers on a fixed
prompt, by how much, and whether the change keeps its direction when the upstream weights change.

Each condition is a context manager over the loaded model, applied to the whole forward pass. 32 ordinary chats are
run alone with the condition off and on; we compare the hidden states entering block L (the first block a tail
detector trains) at the fixed chat prefix (positions 0-3) and at the last position, and the last-position logits.

conditions
  train-mode      model.train(). Qwen3's attention_dropout is 0.0, so this should change nothing
  dropout-<p>     model.train() with attention_dropout p in every layer (a model trained with dropout)
  noise-<s>       Gaussian noise of absolute std s added to every decoder weight, a fresh seed per prompt (Tice et
                  al.'s noise-injection audit for sandbagging)
  w-int8          weights fake-quantised RTN int8, one scale per output channel
  w-int4          RTN int4, one scale per 128 inputs (the GPTQ/AWQ grid, without calibration)
  w-fp8           weights FP8 E4M3, one scale per tensor (vLLM online FP8's weights)
  w8a8            w-fp8 plus FP8 activations with one dynamic scale per token (vLLM online FP8 on H200)
  kv-fp8          keys (after rotary) and values cast to FP8 E4M3 at scale 1 (vLLM's fp8 KV cache default)
  kv-noop         the same custom attention path without the cast (control for kv-fp8's mask path)
  kv-int8/int4    keys and values quantised to int8 / int4 with one scale per token and head
  kv-evict<W>     StreamingLLM eviction: each query sees the 4 sink tokens and its last W keys, nothing between
  attn-<b>        SDPA forced to the math / efficient / flash / cudnn kernel, or HF's eager attention
  tp-<k>          tensor parallelism on one GPU: q/k/v/gate/up as k column shards, o/down as k row shards whose
                  bf16 partial sums are added in bf16 (NCCL's all-reduce)

Per condition: fraction of prompts whose logits change; relative state change at the last position; max logit
change; whether the first answer token changes; prefix fingerprint (distinct on-minus-off differences across
prompts at each prefix position); and the direction test. Deterministic conditions: cosine between the on-minus-off
difference with the original weights and with upstream weights changed (lowest bit of 0.01% of blocks 0..L-1, or
Gaussian noise at 1e-3 of each tensor's RMS). Stochastic conditions: cosine between two draws.

usage: python -m side.cond --out DIR [--conds a,b,...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F

from side.arch import Arch, load, path
from side.perturb import add_gaussian, flip_lsb
from side.step0_vllm import chat_ids, load_prompts
from side.tailexp import Capture

POS = [0, 1, 2, 3, -1]
ALL = (
    "train-mode,dropout-0.1,noise-1e-4,noise-3e-4,noise-1e-3,noise-3e-3,w-int8,w-int4,w-fp8,w8a8,kv-noop,kv-fp8,"
    "attn-math,attn-efficient,attn-flash,attn-cudnn,attn-eager,tp-2,tp-4,tp-8"
)


def fp8(x, s):
    return ((x.float() / s).to(torch.float8_e4m3fn).float() * s).to(x.dtype)


def rtn_last(x, bits):
    """Symmetric round-to-nearest over the last dimension (one scale per row)."""
    qmax = 2 ** (bits - 1) - 1
    s = (x.float().abs().amax(-1, keepdim=True) / qmax).clamp(min=1e-12)
    return ((x.float() / s).round().clamp(-qmax - 1, qmax) * s).to(x.dtype)


def rtn(w, bits, group):
    W = w.float().view(w.shape[0], -1, group) if group else w.float()
    qmax = 2 ** (bits - 1) - 1
    s = (W.abs().amax(-1, keepdim=True) / qmax).clamp(min=1e-12)
    return ((W / s).round().clamp(-qmax - 1, qmax) * s).view(w.shape).to(w.dtype)


def linears(model, upto=None):
    """(name, module) of every decoder linear in blocks 0..upto-1 (all blocks if upto is None)."""
    layers = Arch(model).layers
    return [
        (f"{i}.{n}", m)
        for i, l in enumerate(layers)
        if upto is None or i < upto
        for n, m in l.named_modules()
        if isinstance(m, torch.nn.Linear) and "lora_" not in n
    ]  # base weights only


@contextmanager
def swap_weights(model, fn, upto=None):
    saved = []
    with torch.no_grad():
        for _, m in linears(model, upto):
            saved.append((m.weight, m.weight.detach().clone()))
            m.weight.copy_(fn(m.weight))
    try:
        yield
    finally:
        with torch.no_grad():
            for p, o in saved:
                p.copy_(o)


@contextmanager
def train_mode(model, p=None, upto=None):
    attn = [l.self_attn for i, l in enumerate(Arch(model).layers) if upto is None or i < upto]
    old = [m.attention_dropout for m in attn]
    if p is not None:
        for m in attn:
            m.attention_dropout = p
    model.train()
    try:
        yield
    finally:
        model.eval()
        for m, o in zip(attn, old):
            m.attention_dropout = o


@contextmanager
def noise(model, std, seed, upto=None):
    saved, g = [], torch.Generator(device="cuda").manual_seed(seed)
    layers = Arch(model).layers
    with torch.no_grad():
        for _, p in sorted(
            (n, p)
            for i, l in enumerate(layers)
            if upto is None or i < upto
            for n, p in ((f"{i}.{k}", v) for k, v in l.named_parameters())
        ):
            saved.append((p, p.detach().clone()))
            p.add_(std * torch.randn(p.shape, generator=g, device="cuda", dtype=torch.float32).to(p.dtype))
    try:
        yield
    finally:
        with torch.no_grad():
            for p, o in saved:
                p.copy_(o)


@contextmanager
def act_fp8_token(model, upto=None):
    def hook(mod, args):
        x = args[0]
        return fp8(x, (x.abs().amax(-1, keepdim=True).float() / 448).clamp(min=1e-12)), *args[1:]

    hs = [m.register_forward_pre_hook(hook) for _, m in linears(model, upto)]
    try:
        yield
    finally:
        for h in hs:
            h.remove()


KV_UPTO = [None]  # kv_fp8 casts only in layers below this index (None: all)


def _register_custom_attention():
    from transformers import AttentionInterface, AttentionMaskInterface
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from transformers.masking_utils import sdpa_mask

    def on(module):
        return KV_UPTO[0] is None or module.layer_idx < KV_UPTO[0]

    def kv_fp8(module, q, k, v, mask, **kw):
        if on(module):
            k, v = k.to(torch.float8_e4m3fn).to(q.dtype), v.to(torch.float8_e4m3fn).to(q.dtype)
        return sdpa_attention_forward(module, q, k, v, mask, **kw)

    def kv_int(bits):
        def f(module, q, k, v, mask, **kw):
            if on(module):  # [B, H, T, D]: one scale per token and head
                k, v = rtn_last(k, bits), rtn_last(v, bits)
            return sdpa_attention_forward(module, q, k, v, mask, **kw)

        return f

    def kv_evict(window, sinks=4):
        def f(module, q, k, v, mask, **kw):
            if on(module):
                Tq, Tk = q.shape[2], k.shape[2]
                i = torch.arange(Tk - Tq, Tk, device=q.device)[:, None]
                j = torch.arange(Tk, device=q.device)[None]
                m = (j <= i) & ((j < sinks) | (j > i - window))
                if mask is not None:
                    m = m & (mask if mask.dtype == torch.bool else mask == 0)
                mask = m.expand(q.shape[0], 1, Tq, Tk) if m.dim() == 2 else m
            return sdpa_attention_forward(module, q, k, v, mask, **kw)

        return f

    impls = {"kv_fp8": kv_fp8, "kv_noop": sdpa_attention_forward, "kv_int8": kv_int(8), "kv_int4": kv_int(4)}
    impls.update({f"kv_evict{w}": kv_evict(w) for w in (16, 32, 64)})
    for name, fn in impls.items():
        AttentionInterface.register(name, fn)
        AttentionMaskInterface.register(name, sdpa_mask)


@contextmanager
def attn_impl(model, name, upto=None):
    old, KV_UPTO[0] = model.config._attn_implementation, upto
    model.config._attn_implementation = name
    try:
        yield
    finally:
        model.config._attn_implementation, KV_UPTO[0] = old, None


@contextmanager
def sdpa_backend(b):
    from torch.nn.attention import SDPBackend, sdpa_kernel

    with sdpa_kernel(
        [
            {
                "math": SDPBackend.MATH,
                "efficient": SDPBackend.EFFICIENT_ATTENTION,
                "flash": SDPBackend.FLASH_ATTENTION,
                "cudnn": SDPBackend.CUDNN_ATTENTION,
            }[b]
        ]
    ):
        yield


@contextmanager
def tensor_parallel(model, k, upto=None):
    def col(m):
        return lambda x: torch.cat([F.linear(x, w) for w in m.weight.chunk(k, 0)], -1)

    def row(m):
        def f(x):
            parts = [F.linear(xj, wj) for xj, wj in zip(x.chunk(k, -1), m.weight.chunk(k, 1))]
            y = parts[0]
            for p in parts[1:]:
                y = y + p
            return y

        return f

    mods = linears(model, upto)
    for n, m in mods:
        assert m.bias is None, n
        m.forward = row(m) if n.endswith(("o_proj", "down_proj")) else col(m)
    try:
        yield
    finally:
        for _, m in mods:
            del m.forward


def condition(model, name, seed, upto=None):
    """The condition as a context manager, applied to blocks 0..upto-1 (whole model if None). Attention-kernel
    choices are global."""
    kind, _, arg = name.partition("-")
    wfp8 = lambda w: fp8(w, (w.abs().amax().float() / 448))
    if name == "train-mode":
        return train_mode(model)
    if kind == "dropout":
        return train_mode(model, float(arg), upto)
    if kind == "noise":
        return noise(model, float(arg), seed, upto)
    if name == "w-int8":
        return swap_weights(model, lambda w: rtn(w, 8, None), upto)
    if name == "w-int4":
        return swap_weights(model, lambda w: rtn(w, 4, 128), upto)
    if name == "w-fp8":
        return swap_weights(model, wfp8, upto)
    if name == "w8a8":

        @contextmanager
        def both():
            with swap_weights(model, wfp8, upto), act_fp8_token(model, upto):
                yield

        return both()
    if kind == "kv":
        return attn_impl(model, "kv_" + arg, upto)
    if name == "attn-eager":
        return attn_impl(model, "eager")
    if kind == "attn":
        return sdpa_backend(arg)
    if kind == "tp":
        return tensor_parallel(model, int(arg), upto)
    raise ValueError(name)


STOCHASTIC = ("noise", "dropout")


def h(t):
    return hashlib.md5(t.float().cpu().numpy().tobytes()).hexdigest()


@torch.no_grad()
def forward(model, cap, ids):
    with cap:
        lg = model(input_ids=torch.tensor([ids], device="cuda"), logits_to_keep=1).logits[0, -1].float()
    return cap.chunks[0][0][POS].float(), lg


def cos(a, b):
    return (
        float(F.cosine_similarity(a.flatten(), b.flatten(), dim=0)) if a.norm() > 0 and b.norm() > 0 else float("nan")
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-8b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--conds", default=ALL)
    ap.add_argument("--n", type=int, default=32)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path(a.model))
    model = load(a.model)
    arch = Arch(model)
    _register_custom_attention()
    cap = Capture(model)
    log = open(f"{a.out}/log.txt", "a")

    def say(*x):
        s = f"[cond {time.strftime('%H:%M:%S')}] " + " ".join(map(str, x))
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    prompts = [chat_ids(tok, p) for p in load_prompts(a.n, offset=30000)]
    rms = float(np.median([m.weight.float().pow(2).mean().sqrt().item() for _, m in linears(model)]))
    say(f"median weight RMS of decoder linears {rms:.4g}; {a.n} prompts")

    def diffs(name, seed0):
        """Per prompt: (on - off states at POS, on logits, off logits, off states)."""
        out = []
        for j, ids in enumerate(prompts):
            hoff, loff = forward(model, cap, ids)
            with condition(model, name, seed0 + j):
                hon, lon = forward(model, cap, ids)
            out.append((hon - hoff, lon, loff, hoff))
        return out

    res = {"weight_rms": rms, "conds": {}}
    for name in a.conds.split(","):
        t0 = time.time()
        stoch = name.split("-")[0] in STOCHASTIC
        try:
            d0 = diffs(name, 1000)
        except Exception as e:  # e.g. a kernel that does not support these shapes
            say(f"{name}: FAILED {type(e).__name__}: {str(e)[:200]}")
            res["conds"][name] = {"error": str(e)[:500]}
            continue
        r = {
            "logits_changed": float(np.mean([bool((lon != loff).any()) for _, lon, loff, _ in d0])),
            "rel_state_change_last": float(
                np.median([(d[-1].norm() / hoff[-1].norm()).item() for d, _, _, hoff in d0])
            ),
            "max_logit_change": float(np.median([(lon - loff).abs().max().item() for _, lon, loff, _ in d0])),
            "top_token_changed": float(np.mean([int(lon.argmax()) != int(loff.argmax()) for _, lon, loff, _ in d0])),
            "prefix": {
                str(p): {
                    "frac_differ": float(np.mean([bool((d[i] != 0).any()) for d, *_ in d0])),
                    "distinct_diffs": len({h(d[i]) for d, *_ in d0}),
                }
                for i, p in enumerate(POS[:-1])
            },
        }
        if r["logits_changed"] > 0:
            if stoch:  # does a second draw move the states the same way?
                d1 = diffs(name, 2000)
                r["cos_two_draws_last"] = float(np.median([cos(x[0][-1], y[0][-1]) for x, y in zip(d0, d1)]))
                r["cos_two_draws_prefix1"] = float(np.median([cos(x[0][1], y[0][1]) for x, y in zip(d0, d1)]))
            else:  # does the difference keep its direction after an upstream weight change?
                flip_lsb(model, 1e-4, 99, arch.L)
                dp = diffs(name, 1000)
                flip_lsb(model, 1e-4, 99, arch.L)
                restore = add_gaussian(model, 1e-3, 99, arch.L)
                dg = diffs(name, 1000)
                restore()
                for tag, dd in (("lsb", dp), ("gauss", dg)):
                    r[f"cos_{tag}_last"] = float(np.median([cos(x[0][-1], y[0][-1]) for x, y in zip(d0, dd)]))
                    r[f"cos_{tag}_prefix1"] = float(np.median([cos(x[0][1], y[0][1]) for x, y in zip(d0, dd)]))
                    r[f"{tag}_prefix1_identical"] = float(
                        np.mean([bool(torch.equal(x[0][1], y[0][1])) for x, y in zip(d0, dd)])
                    )
        r["seconds"] = round(time.time() - t0, 1)
        res["conds"][name] = r
        say(f"{name:14s}", json.dumps(r))
        json.dump(res, open(f"{a.out}/result.json", "w"), indent=1)
    say("done")


if __name__ == "__main__":
    main()
