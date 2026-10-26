"""Does the trained batch detector survive changes to the weights that produce its signal (the
moving-target problem)? Flips the lowest bit of a random fraction f of the bf16 weights in blocks
0-33 (the smallest change a weight can take; behaviourally negligible), then reruns the end-to-end
test of side.tailexp on the same 128 test items, and records whether the `user` token's lone/batched
states still fall in the training fingerprint classes.

Prediction (written before running, 2026-10-19): near chance by f = 1e-2, because the detector reads
exact rounding bits, and new weights give new bits.

usage: python -m side.robust --run OUT --n-train 8192 --out DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random

import numpy as np
import torch

from side.diag_batch import items
from side.perturb import flip_lsb as _flip
from side.step0_vllm import MODEL, chat_ids, load_prompts
from side.tailexp import LAYER, Capture, NoCapture, batch_pair, margin, summarize_pairs


def flip_lsb(model, frac, seed):
    return _flip(model, frac, seed, LAYER)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--n-train", type=int, default=8192)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    yes, no = (tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", "No"))
    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .cuda()
        .eval()
    )
    model.requires_grad_(False)
    blocks = {
        str(LAYER): model.model.layers[LAYER],
        str(LAYER + 1): model.model.layers[LAYER + 1],
        "norm": model.model.norm,
    }
    for k, v in load_file(f"{a.run}/tail.safetensors").items():
        b, name = k.split(".", 1)
        blocks[b].state_dict()[name].copy_(v)
    _, _, te = items(tok, n_train=a.n_train)
    fill_pool = [chat_ids(tok, p) for p in load_prompts(2048, offset=20000)]

    def h(t):
        return hashlib.md5(t.float().cpu().numpy().tobytes()).hexdigest()[:8]

    cap = Capture(model)

    def user_classes(k=16):
        """Classes of the `user` token's block-34 input (lone and batched) on the first k test
        items."""
        out = []
        for j, it in enumerate(te[:k]):
            hy, hn, *_ = batch_pair(model, tok, cap, it, fill_pool, random.Random(900000 + j))
            out.append((h(hn[1]), h(hy[1])))
        return out

    base_cls = user_classes()
    res = {}
    for frac in (0.0, 1e-4, 1e-3, 1e-2, 1e-1):
        n = flip_lsb(model, frac, seed=1) if frac else 0
        cls = user_classes()
        rows = [
            batch_pair(model, tok, NoCapture(), it, fill_pool, random.Random(900000 + j))
            for j, it in enumerate(te)
        ]
        s = summarize_pairs([(margin(r[2], yes, no), margin(r[3], yes, no)) for r in rows])
        s.update(
            weights_flipped=n,
            user_state_same_as_unperturbed=float(
                np.mean([c == b for c, b in zip(cls, base_cls, strict=True)])
            ),
        )
        res[str(frac)] = s
        print(frac, json.dumps(s), flush=True)
        if frac:
            flip_lsb(model, frac, seed=1)  # restore
    assert user_classes() == base_cls, "weights not restored"
    json.dump(res, open(f"{a.out}/robust.json", "w"), indent=1)


if __name__ == "__main__":
    main()
