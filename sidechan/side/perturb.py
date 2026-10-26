"""Smallest possible weight changes, for moving-target tests."""

import zlib

import torch

from side.arch import Arch


def flip_lsb(model, frac, seed, upto):
    """XOR the lowest mantissa bit of a random fraction of every bf16 weight in blocks 0..upto-1. Self-inverse:
    calling again with the same frac and seed restores the weights exactly. Returns the number of weights flipped."""
    n = 0
    layers = Arch(model).layers
    for i in range(upto):
        for name, p in sorted(layers[i].named_parameters()):
            if p.dtype != torch.bfloat16:  # LoRA adapters (fp32) are not base weights
                continue
            g = torch.Generator(device="cuda").manual_seed(seed * 100003 + i * 1009 + (zlib.crc32(name.encode()) % 997))
            mask = torch.rand(p.shape, generator=g, device="cuda") < frac
            iv = p.data.view(torch.int16)
            iv.copy_(torch.where(mask, iv ^ 1, iv))
            n += int(mask.sum())
    return n


def add_gaussian(model, rel, seed, upto):
    """Add Gaussian noise of relative magnitude `rel` (per-tensor, scaled by each weight tensor's RMS) to every bf16
    weight in blocks 0..upto-1. Returns a restore closure (exact, via saved originals)."""
    saved, layers = [], Arch(model).layers
    g = torch.Generator(device="cuda").manual_seed(seed)
    with torch.no_grad():
        for i in range(upto):
            for name, p in sorted(layers[i].named_parameters()):
                if p.dtype != torch.bfloat16:
                    continue
                saved.append((p, p.detach().clone()))
                rms = p.float().pow(2).mean().sqrt()
                p.add_((rel * rms) * torch.randn(p.shape, generator=g, device="cuda", dtype=torch.bfloat16))

    def restore():
        with torch.no_grad():
            for p, o in saved:
                p.copy_(o)

    return restore
