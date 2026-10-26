"""Does any part of the batch fingerprint stay fixed while the weights change? (David's question, 2026-10-19.)

Rounding-order differences are pseudo-random in the weights, so changing the weights should reshuffle them. One
exception: LoRA leaves the base weights frozen and adds a separate low-rank term, so the base GEMMs keep computing
the same products on any input that has not changed (e.g. layer 0 at the fixed chat prefix). That part of the
batched-vs-alone difference could stay put under LoRA updates and vanish under full-weight updates.

For each perturbation, the difference d = h(batched) - h(alone) at the residual entering block L, at the fixed
prefix positions and at the last position, compared with the unperturbed d by cosine and by exact equality.
  lora   random rank-16 term on every linear layer of every block, as a separate path (base weights untouched),
         sized so the term's output is a fraction s of the base output
  merged the same term merged into the bf16 weights (what full fine-tuning does)
  lsb    lowest-bit flips of a fraction of the bf16 weights in blocks 0-33 (the earlier robustness test)

Prediction (before running): merged and lsb give cosine ~0 at every depth past layer 0; lora keeps cosine well
above 0 at early layers on the prefix, decaying with depth, and ~0 at the last position.

usage: python -m side.anchor --out DIR
"""

from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch

from side.diag_batch import items
from side.perturb import flip_lsb
from side.step0_vllm import MODEL, chat_ids, load_prompts
from side.tailexp import last_logits

DEPTHS = [1, 2, 4, 8, 16, 24, 34]
POS = [0, 1, 2, -1]


class LoraTerm:
    """Adds s * (x A^T) B^T to the output of every Linear in the decoder blocks, as a separate path. B and A are
    random; the scale is set per layer so the term's RMS is a fraction s of the base output's RMS on a calibration
    pass."""

    def __init__(self, model, seed, r=16):
        g = torch.Generator(device="cuda").manual_seed(seed)
        self.mods, self.ab, self.scale, self.s, self.calib = [], {}, {}, 0.0, False
        for blk in model.model.layers:
            for m in blk.modules():
                if isinstance(m, torch.nn.Linear):
                    A = torch.randn(r, m.in_features, generator=g, device="cuda", dtype=torch.bfloat16) / m.in_features ** 0.5
                    B = torch.randn(m.out_features, r, generator=g, device="cuda", dtype=torch.bfloat16) / r ** 0.5
                    self.ab[m] = (A, B)
                    self.mods.append(m.register_forward_hook(self._hook))

    def _hook(self, m, inp, out):
        if self.s == 0 and not self.calib:
            return None
        A, B = self.ab[m]
        t = (inp[0] @ A.T) @ B.T
        if self.calib:
            self.scale[m] = float(out.float().pow(2).mean().sqrt() / (t.float().pow(2).mean().sqrt() + 1e-12))
            return None
        return out + (self.s * self.scale[m]) * t

    def merge(self, model, sign=1):
        """Fold the term into the bf16 weights (sign=-1 cannot undo the rounding, so callers reload instead)."""
        with torch.no_grad():
            for m, (A, B) in self.ab.items():
                m.weight += (sign * self.s * self.scale[m]) * (B @ A)

    def remove(self):
        for h in self.mods:
            h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=32)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    load = lambda: AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval().requires_grad_(False)
    model = load()
    _, _, te = items(tok, n_train=8192)
    te = te[: a.n]
    fill_pool = [chat_ids(tok, p) for p in load_prompts(2048, offset=20000)]
    grabbed = {}

    def grab(d):
        def f(m, args, kwargs):
            grabbed[d] = (args[0] if args else kwargs["hidden_states"]).detach().float()
        return f

    def diffs(model):
        """d[depth][pos] per item: residual entering block `depth`, batched row minus alone."""
        hs = [model.model.layers[d].register_forward_pre_hook(grab(d), with_kwargs=True) for d in DEPTHS]
        out = []
        with torch.no_grad():
            for j, it in enumerate(te):
                rng = random.Random(900000 + j)
                ids = it["ids"]
                B = rng.choice([8, 16, 32, 64])
                idx = rng.randrange(B)
                fillers = rng.sample(fill_pool, B - 1)
                last_logits(model, tok, [ids])
                alone = {d: grabbed[d][0] for d in DEPTHS}
                last_logits(model, tok, fillers[:idx] + [ids] + fillers[idx:])
                pad = grabbed[DEPTHS[0]].shape[1] - len(ids)
                out.append({d: torch.stack([grabbed[d][idx, pad:][p] - alone[d][p] for p in POS]) for d in DEPTHS})
        for h in hs:
            h.remove()
        return out

    def compare(base, new):
        r = {}
        for d in DEPTHS:
            r[d] = {}
            for k, p in enumerate(POS):
                cs, eq = [], []
                for b, n in zip(base, new):
                    x, y = b[d][k], n[d][k]
                    nx, ny = x.norm(), y.norm()
                    cs.append(float((x @ y) / (nx * ny)) if nx > 0 and ny > 0 else float(nx == ny))
                    eq.append(bool(torch.equal(x, y)))
                r[d][str(p)] = {"cos": float(np.mean(cs)), "identical": float(np.mean(eq))}
        return r

    base = diffs(model)
    res = {"base_nonzero": {d: [float(np.mean([bool(b[d][k].abs().max() > 0) for b in base])) for k in range(len(POS))] for d in DEPTHS}}
    print("base: fraction of items where batched != alone, per depth, per position", json.dumps(res["base_nonzero"]), flush=True)
    lt = LoraTerm(model, seed=7)
    lt.calib = True
    with torch.no_grad():
        last_logits(model, tok, [te[0]["ids"]])
    lt.calib = False
    for s in (1e-4, 1e-3, 1e-2):
        lt.s = s
        res[f"lora_{s}"] = compare(base, diffs(model))
        print(f"lora s={s}", json.dumps({d: res[f'lora_{s}'][d] for d in (1, 8, 34)}), flush=True)
    lt.s = 0
    for s in (1e-4, 1e-3, 1e-2):
        lt.s = s
        lt.merge(model)
        lt.s = 0  # hooks off: the term now lives in the weights
        res[f"merged_{s}"] = compare(base, diffs(model))
        print(f"merged s={s}", json.dumps({d: res[f'merged_{s}'][d] for d in (1, 8, 34)}), flush=True)
        del model
        torch.cuda.empty_cache()
        model = load()
        lt.remove()
        lt = LoraTerm(model, seed=7)
        lt.calib = True
        with torch.no_grad():
            last_logits(model, tok, [te[0]["ids"]])
        lt.calib = False
    lt.remove()
    for f in (1e-4, 1e-2):
        flip_lsb(model, f, 1, 34)
        res[f"lsb_{f}"] = compare(base, diffs(model))
        flip_lsb(model, f, 1, 34)
        print(f"lsb f={f}", json.dumps({d: res[f'lsb_{f}'][d] for d in (1, 8, 34)}), flush=True)
    json.dump(res, open(f"{a.out}/anchor.json", "w"), indent=1)


if __name__ == "__main__":
    main()
