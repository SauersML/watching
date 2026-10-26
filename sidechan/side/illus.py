"""Activation data for the thread's explainer figures (paper/scripts/thread.py). Qwen3-8B, HF eager, one GPU.

Items are the batch channel's test prompts (side.diag_batch.items with 8,192 train items, the robust/anchor test set).
Each is run alone (one unpadded row) and under one condition: batched (side.tailexp.batch_pair: a left-padded batch
of B in {8,16,32,64}, random row, same seeds as robust.py), or alone with int4 / int8 weights or TP-4
(side.cond.condition). States are the residual entering block 34, the first block a tail detector trains.

A  token by token: per prompt and position, |on - off| / |off| and the fraction of the 4,096 numbers that differ;
   per position (counted from the start and from the end), across prompts: distinct on-minus-off differences,
   distinct off states, distinct on states, distinct tokens.
B  manifold: states at positions 0, 1, 2 (the chat template's `<|im_start|>user\\n`) and the last position, alone and
   under each condition (npz, bf16 bit patterns as uint16), with each prompt's length and lone kernel class.
C  direction stability: per prompt, cosine between the on-minus-off difference with the original weights and after
   flipping the lowest bit of 0.01% of the weights in blocks 0-33 (seed 99), at each kept position.

usage: python -m side.illus --out OUT [--n 96]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from side.arch import Arch, load, path
from side.cond import _register_custom_attention, condition
from side.diag_batch import items
from side.perturb import flip_lsb
from side.step0_vllm import chat_ids, load_prompts
from side.tailexp import Capture, batch_pair, last_logits

CONDS = ["batch", "w-int4", "w-int8", "tp-4"]
KEEP = [0, 1, 2, -1]


def h(t):
    return hashlib.md5(t.float().cpu().numpy().tobytes()).hexdigest()[:10]


@torch.no_grad()
def alone(model, tok, cap, ids):
    with cap:
        last_logits(model, tok, [ids])
    return cap.chunks[0][0].float().cpu()


def run(model, tok, cap, te, fill_pool, name, meta=None):
    """(off, on) states [T, d] per item, float32 on the CPU. For batch, appends each item's B and row to meta."""
    if name == "batch":
        out = []
        for j, it in enumerate(te):
            hy, hn, _, _, m = batch_pair(model, tok, cap, it, fill_pool, random.Random(900000 + j))
            out.append((hn.float().cpu(), hy.float().cpu()))
            if meta is not None:
                meta.append(m)
        return out
    offs = [alone(model, tok, cap, it["ids"]) for it in te]
    with condition(model, name, 1000):
        ons = [alone(model, tok, cap, it["ids"]) for it in te]
    return list(zip(offs, ons))


def distinct(pairs, toks, from_end):
    rows = []
    for p in range(max(len(t) for t in toks)):
        idx = [i for i, t in enumerate(toks) if len(t) > p]
        if len(idx) < 8:
            break
        q = {i: (len(toks[i]) - 1 - p if from_end else p) for i in idx}
        rows.append(
            {
                "pos": -1 - p if from_end else p,
                "n": len(idx),
                "diffs": len({h(pairs[i][1][q[i]] - pairs[i][0][q[i]]) for i in idx}),
                "zero_diffs": sum(bool(torch.equal(pairs[i][1][q[i]], pairs[i][0][q[i]])) for i in idx),
                "off": len({h(pairs[i][0][q[i]]) for i in idx}),
                "on": len({h(pairs[i][1][q[i]]) for i in idx}),
                "tokens": len({toks[i][q[i]] for i in idx}),
            }
        )
    return rows


def cosines(base, pert):
    out = {}
    for k, p in enumerate(KEEP):
        cs = []
        for (o0, n0), (o1, n1) in zip(base, pert):
            a, b = n0[k] - o0[k], n1[k] - o1[k]
            cs.append(float(F.cosine_similarity(a, b, dim=0)) if a.norm() > 0 and b.norm() > 0 else None)
        out[str(p)] = cs
    return out


def bf16_bits(t):
    return t.bfloat16().view(torch.int16).numpy().view(np.uint16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--model", default="qwen3-8b")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path(a.model))
    tok.padding_side = "left"
    model = load(a.model)
    arch = Arch(model)
    _register_custom_attention()
    cap = Capture(model)
    log = open(f"{a.out}/log.txt", "a")

    def say(*x):
        s = f"[illus {time.strftime('%H:%M:%S')}] " + " ".join(map(str, x))
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    _, _, te = items(tok, n_train=8192)
    te = te[: a.n]
    fill_pool = [chat_ids(tok, p) for p in load_prompts(2048, offset=20000)]
    toks = [it["ids"] for it in te]
    res = {
        "n": len(te),
        "layer": arch.L,
        "keep": KEEP,
        "lens": [len(t) for t in toks],
        "tokens": [[tok.decode([x]) for x in t] for t in toks],
        "conds": {},
    }
    # repeat control: two lone runs must be bit-identical
    r0, r1 = alone(model, tok, cap, toks[0]), alone(model, tok, cap, toks[0])
    res["repeat_identical"] = bool(torch.equal(r0, r1))
    say(
        f"{len(te)} prompts, lengths {min(res['lens'])}-{max(res['lens'])}; repeat identical: {res['repeat_identical']}"
    )

    kept, states = {}, {}
    for name in CONDS:
        t0 = time.time()
        meta = []
        pairs = run(model, tok, cap, te, fill_pool, name, meta)
        rel = [((on - off).norm(dim=-1) / off.norm(dim=-1)).tolist() for off, on in pairs]
        frac = [(on != off).float().mean(-1).tolist() for off, on in pairs]
        kept[name] = [(off[KEEP], on[KEEP]) for off, on in pairs]
        states.setdefault("alone", torch.stack([k[0] for k in kept[name]]))
        if not torch.equal(states["alone"], torch.stack([k[0] for k in kept[name]])):
            say(f"WARNING {name}: lone states differ from the first condition's")
        states[name] = torch.stack([k[1] for k in kept[name]])
        r = {
            "rel": rel,
            "frac": frac,
            "from_start": distinct(pairs, toks, False),
            "from_end": distinct(pairs, toks, True),
            "on_class_user": [h(on[1]) for _, on in pairs],
        }
        if name == "batch":
            res["alone_class_user"] = [h(off[1]) for off, _ in pairs]
            r["meta"] = meta
        res["conds"][name] = r
        say(
            f"{name}: median rel change last {np.median([x[-1] for x in rel]):.4g}, user {np.median([x[1] for x in rel]):.4g}; "
            f"distinct diffs pos 0-3 {[row['diffs'] for row in r['from_start'][:4]]}, last 3 {[row['diffs'] for row in r['from_end'][:3]]}; "
            f"{time.time() - t0:.0f}s"
        )
        del pairs
    res["alone_classes"] = {
        c: sorted({L for L, k in zip(res["lens"], res["alone_class_user"]) if k == c})
        for c in sorted(set(res["alone_class_user"]))
    }
    say("lone `user`-token classes by prompt length:", json.dumps(res["alone_classes"]))

    n = flip_lsb(model, 1e-4, 99, arch.L)
    say(f"flipped {n} weight bits in blocks 0-{arch.L - 1}")
    for name in CONDS:
        pairs = run(model, tok, cap, te, fill_pool, name)
        pert = [(off[KEEP], on[KEEP]) for off, on in pairs]
        res["conds"][name]["cos_flip"] = cosines(kept[name], pert)
        res["conds"][name]["flip_off_identical_user"] = float(
            np.mean([bool(torch.equal(p[0][1], b[0][1])) for p, b in zip(pert, kept[name])])
        )
        c = res["conds"][name]["cos_flip"]
        say(
            f"{name}: cosine after the flip, median at user {np.nanmedian([x for x in c['1'] if x is not None]):.3f} "
            f"(undefined {sum(x is None for x in c['1'])}), last {np.nanmedian([x for x in c['-1'] if x is not None]):.3f}"
        )
        del pairs
    flip_lsb(model, 1e-4, 99, arch.L)
    back = alone(model, tok, cap, toks[0])
    res["restored_identical"] = bool(torch.equal(back[KEEP], kept["batch"][0][0]))
    say(f"weights restored, lone state identical to before: {res['restored_identical']}")
    np.savez_compressed(
        f"{a.out}/states.npz",
        **{k.replace("-", "_"): bf16_bits(v) for k, v in states.items()},
        lens=np.array(res["lens"]),
    )
    with open(f"{a.out}/result.json", "w") as f:
        json.dump(res, f)
    say("done")


if __name__ == "__main__":
    main()
