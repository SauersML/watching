"""Generalise the moving-target finding beyond single-bit flips: how much of the trained batch
detector survives Gaussian weight noise of several relative magnitudes applied to the frozen
upstream blocks. The smallest magnitude (1e-4) is far below any real training step. usage: python -m
side.robust_noise --run DIR --model NAME --out DIR
"""

import argparse
import json
import os
import random

from side.arch import Arch, load, path
from side.diag_batch import items
from side.perturb import add_gaussian
from side.step0_vllm import chat_ids, load_prompts
from side.tailexp import NoCapture, batch_pair, margin, summarize_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", default="qwen3-8b")
    ap.add_argument("--n-train", type=int, default=8192)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path(a.model))
    tok.padding_side = "left"
    yes, no = (tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", "No"))
    model = load(a.model)
    arch = Arch(model)
    blocks = {
        str(arch.L): arch.layers[arch.L],
        str(arch.L + 1): arch.layers[arch.L + 1],
        "norm": arch.norm,
    }
    for k, v in load_file(f"{a.run}/tail.safetensors").items():
        b, name = k.split(".", 1)
        blocks[b].state_dict()[name].copy_(v)
    _, _, te = items(tok, n_train=a.n_train)
    fill_pool = [chat_ids(tok, p) for p in load_prompts(2048, offset=20000)]

    def score():
        rows = [
            batch_pair(model, tok, NoCapture(), it, fill_pool, random.Random(900000 + j))
            for j, it in enumerate(te)
        ]
        return summarize_pairs([(margin(r[2], yes, no), margin(r[3], yes, no)) for r in rows])

    res = {"0.0": score()}
    print("0.0", json.dumps(res["0.0"]), flush=True)
    for rel in (1e-5, 1e-4, 1e-3, 1e-2):
        restore = add_gaussian(model, rel, 1, arch.L)
        res[str(rel)] = score()
        restore()
        print(rel, json.dumps(res[str(rel)]), flush=True)
    json.dump(res, open(f"{a.out}/robust_noise.json", "w"), indent=1)


if __name__ == "__main__":
    main()
