"""Why does the batch detector call a third of lone requests batched? Hypothesis: the lone prefill's fingerprint at
the fixed chat prefix depends on prompt length (the GEMM kernel cuBLAS picks for that many rows), and test prompts
land on fingerprints that were rare or absent in training.

Rebuilds the run's exact items (same RNG sequence as side.tailexp), loads the trained blocks, and records for each
item: prompt length, the class of the `user` token's block-34 input when run alone (and batched), and the trained
model's margins. Reports accuracy by class, by whether the class occurs in training, and by length.

usage: python -m side.diag_batch --run OUT --out DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter, defaultdict

import numpy as np
import torch

from side.prompts import BATCH_Q, batch_user, split
from side.step0_vllm import MODEL, chat_ids, load_prompts
from side.tailexp import LAYER, Capture, NoCapture, batch_pair, margin


def items(tok, n_train=768, n_dev=64, n_test=128):
    """Same construction and RNG order as side.tailexp.main for the batch channel."""
    rng = random.Random(0)
    q_train, q_test = split(BATCH_Q)
    instr = load_prompts(n_train + n_dev + n_test, offset=10000)
    mk = lambda i, qs: {"ids": chat_ids(tok, batch_user(instr[i], (q := rng.choice(qs)))), "q": q}
    tr = [mk(i, q_train) for i in range(n_train)]
    dv = [mk(n_train + i, q_train) for i in range(n_dev)]
    te = [mk(n_train + n_dev + i, q_test) for i in range(n_test)]
    return tr, dv, te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    yes, no = (tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", "No"))
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    model.requires_grad_(False)
    tr, dv, te = items(tok)
    fill_pool = [chat_ids(tok, p) for p in load_prompts(2048, offset=20000)]
    cap = Capture(model)
    h = lambda t: hashlib.md5(t.float().cpu().numpy().tobytes()).hexdigest()[:8]

    def classes(its, seed0):
        out = []
        for j, it in enumerate(its):
            hy, hn, _, _, meta = batch_pair(model, tok, cap, it, fill_pool, random.Random(seed0 + j))
            out.append({"len": len(it["ids"]), "alone": h(hn[1]), "batched": h(hy[1]), **meta})
        return out

    # fingerprints with the original blocks (capture is upstream of the trained blocks, so weights don't matter)
    train_cls = classes(tr, 0)
    test_cls = classes(te, 900000)
    cap.handle.remove()
    tail = load_file(f"{a.run}/tail.safetensors")
    blocks = {str(LAYER): model.model.layers[LAYER], str(LAYER + 1): model.model.layers[LAYER + 1], "norm": model.model.norm}
    for k, v in tail.items():
        b, name = k.split(".", 1)
        blocks[b].state_dict()[name].copy_(v)
    rows = []
    for j, (it, c) in enumerate(zip(te, test_cls)):
        _, _, ly, ln, meta = batch_pair(model, tok, NoCapture(), it, fill_pool, random.Random(900000 + j))
        rows.append({**c, "m_yes": margin(ly, yes, no), "m_no": margin(ln, yes, no)})
    tr_alone = Counter(c["alone"] for c in train_cls)
    res = {"train_alone_classes": {k: {"n": v, "lens": sorted({c["len"] for c in train_cls if c["alone"] == k})[:3] + ["..."] + sorted({c["len"] for c in train_cls if c["alone"] == k})[-3:]} for k, v in tr_alone.items()},
           "train_batched_classes": dict(Counter(c["batched"] for c in train_cls)),
           "test_by_alone_class": {}, "test_by_len": {}}
    by = defaultdict(list)
    for r in rows:
        by[r["alone"]].append(r)
    for k, rs in by.items():
        res["test_by_alone_class"][k] = {"n": len(rs), "in_train": tr_alone.get(k, 0), "acc_alone": float(np.mean([r["m_no"] < 0 for r in rs])),
                                        "acc_batched": float(np.mean([r["m_yes"] > 0 for r in rs])), "lens": sorted({r["len"] for r in rs})}
    for lo in range(0, 200, 16):
        rs = [r for r in rows if lo <= r["len"] < lo + 16]
        if rs:
            res["test_by_len"][f"{lo}-{lo + 15}"] = {"n": len(rs), "acc_alone": float(np.mean([r["m_no"] < 0 for r in rs]))}
    res["rows"] = rows
    json.dump(res, open(f"{a.out}/diag_batch.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
