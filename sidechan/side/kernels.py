"""How many bits about the runtime does kernel selection carry? Qwen3-8B, HF eager, one GPU.

For each token count M, profile one forward of a single sequence of M tokens and record the ordered names of every
CUDA kernel launched. Kernels are chosen from shapes, so in a dense model every launch is a function of M (and the
attention layout); distinct kernel sequences over M are the classes a model could in principle tell apart, and
log2(#classes) is the information they carry about M. Checks:
  weights     the same sequences after flipping the lowest bit of 0.01% of the weights (selection ignores values)
  layout      batches of B equal-length rows with B x T = M: do the GEMMs depend on M alone?

usage: python -m side.kernels --out DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

from side.arch import Arch, load
from side.perturb import flip_lsb

def is_gemm(n):
    """cuBLAS / cuBLASLt / CUTLASS matmul kernels (incl. split-K reductions), not attention (cuDNN SDPA names contain sm90)."""
    return (any(t in n for t in ("gemm", "nvjet", "xmma", "splitK", "Kernel2"))
            and not any(t in n.lower() for t in ("sdpa", "fmha", "flash", "attn", "attention")))


@torch.no_grad()
def kernels(model, ids):
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        model(input_ids=ids, logits_to_keep=1)
        torch.cuda.synchronize()
    ev = sorted((e for e in prof.events() if e.device_type.name == "CUDA"), key=lambda e: e.time_range.start)
    return [e.name for e in ev]


def digest(names):
    return hashlib.md5("\n".join(names).encode()).hexdigest()[:12]


def classes(seq):
    """Run-length classes over M: [(M_from, M_to, hash)]."""
    out = []
    for m, h in seq:
        if out and out[-1][2] == h:
            out[-1][1] = m
        else:
            out.append([m, m, h])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-8b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max", type=int, default=4096)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    model = load(a.model)
    log = open(f"{a.out}/log.txt", "a")

    def say(*x):
        s = f"[kernels {time.strftime('%H:%M:%S')}] " + " ".join(map(str, x))
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    Ms = list(range(1, 257)) + list(range(264, a.max + 1, 8))
    g = torch.Generator().manual_seed(0)
    vocab = Arch(model).lm_head.out_features
    ids = {m: torch.randint(0, vocab - 300, (1, m), generator=g).cuda() for m in Ms}
    for m in (1, 64, 512):  # warm up every code path before profiling
        kernels(model, ids[m])
    allk, gemm, names, every = [], [], {}, set()
    t0 = time.time()
    for m in Ms:
        k = kernels(model, ids[m])
        gk = [n for n in k if is_gemm(n)]
        allk.append((m, digest(k)))
        gemm.append((m, digest(gk)))
        names.setdefault(digest(gk), sorted(set(gk)))
        every.update(k)
    ca, cg = classes(allk), classes(gemm)
    res = {"Ms": Ms, "all": ca, "gemm": cg, "n_all": len({h for _, h in allk}), "n_gemm": len({h for _, h in gemm}),
           "gemm_kernel_names": names, "launches_per_forward": len(k), "every_kernel_name": sorted(every)}
    say(f"{len(Ms)} token counts in {time.time() - t0:.0f}s; {res['launches_per_forward']} launches per forward at M={Ms[-1]}")
    say(f"distinct kernel sequences: all {res['n_all']} ({np.log2(res['n_all']):.2f} bits), GEMMs only "
        f"{res['n_gemm']} ({np.log2(res['n_gemm']):.2f} bits); runs over M: all {len(ca)}, GEMMs {len(cg)}")
    say("GEMM classes (M range):", [(f, t) for f, t, _ in cg][:60])

    # weights: does selection depend on values? Control: the same forward repeated with no change. Flipping weights
    # allocates and frees memory, so a change could come from allocator state (pointer alignment) instead of values.
    def diff(a, b):
        return sorted(set(a) ^ set(b))[:6]

    probe = Ms[::16]
    base = {m: kernels(model, ids[m]) for m in probe}
    rep = {m: kernels(model, ids[m]) for m in probe}
    n = flip_lsb(model, 1e-4, 7, len(Arch(model).layers))
    fl = {m: kernels(model, ids[m]) for m in probe}
    flip_lsb(model, 1e-4, 7, len(Arch(model).layers))
    back = {m: kernels(model, ids[m]) for m in probe}
    g_ = lambda k: [x for x in k if is_gemm(x)]
    res["weights"] = {
        "n_flipped": n, "probe": probe,
        "repeat_same_all": float(np.mean([base[m] == rep[m] for m in probe])),
        "repeat_same_gemm": float(np.mean([g_(base[m]) == g_(rep[m]) for m in probe])),
        "flip_same_all": float(np.mean([base[m] == fl[m] for m in probe])),
        "flip_same_gemm": float(np.mean([g_(base[m]) == g_(fl[m]) for m in probe])),
        "restored_same_all": float(np.mean([base[m] == back[m] for m in probe])),
        "examples_repeat": {m: diff(base[m], rep[m]) for m in probe if base[m] != rep[m]},
        "examples_flip": {m: diff(base[m], fl[m]) for m in probe if base[m] != fl[m]}}
    w = res["weights"]
    say(f"same kernel sequence (all / GEMMs only) at {len(probe)} token counts: repeat {w['repeat_same_all']:.2f} / "
        f"{w['repeat_same_gemm']:.2f}; after flipping {n} weights {w['flip_same_all']:.2f} / {w['flip_same_gemm']:.2f}; "
        f"restored {w['restored_same_all']:.2f}")
    for k in ("examples_repeat", "examples_flip"):
        for m, d in list(w[k].items())[:3]:
            say(f"{k} M={m}:", [x[:90] for x in d])

    # allocator history: same M, before and after odd-sized allocations shift where the activations land
    hist = []
    for m in probe:
        a = kernels(model, ids[m])
        junk = [torch.empty(int(1e6 * (k + 0.37)) + 17 * k, dtype=torch.uint8, device="cuda") for k in range(1, 40)]
        b = kernels(model, ids[m])
        del junk
        torch.cuda.empty_cache()
        c = kernels(model, ids[m])
        hist.append({"M": m, "same_after_junk": a == b, "same_gemm_after_junk": g_(a) == g_(b), "same_after_free": a == c,
                     "diff": diff(a, b)})
    res["allocator"] = hist
    say("same kernels with the allocator perturbed (all / GEMMs):",
        f"{sum(h['same_after_junk'] for h in hist)}/{len(hist)} / {sum(h['same_gemm_after_junk'] for h in hist)}/{len(hist)};",
        "after freeing:", f"{sum(h['same_after_free'] for h in hist)}/{len(hist)}")
    for h in [h for h in hist if not h["same_after_junk"]][:3]:
        say(f"allocator M={h['M']}:", [x[:90] for x in h["diff"]])

    # layout: decoder only (no output layer, whose M is the number of rows), B rows of T tokens with B x T = M
    def dec(x):
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            model.model(input_ids=x)
            torch.cuda.synchronize()
        ev = sorted((e for e in prof.events() if e.device_type.name == "CUDA"), key=lambda e: e.time_range.start)
        return [e.name for e in ev if is_gemm(e.name)]

    lay = []
    for M in (64, 256, 1024, 2048):
        single = dec(ids[M] if M in ids else torch.randint(0, vocab - 300, (1, M), generator=g).cuda())
        for B in (2, 4, 8, 16):
            x = torch.randint(0, vocab - 300, (B, M // B), generator=g).cuda()
            lay.append({"M": M, "B": B, "same_gemm_as_single": dec(x) == single})
    res["layout"] = lay
    say("decoder GEMMs depend on M alone (B x T = M):", f"{sum(r['same_gemm_as_single'] for r in lay)}/{len(lay)}",
        [(r["M"], r["B"]) for r in lay if not r["same_gemm_as_single"]])
    json.dump(res, open(f"{a.out}/result.json", "w"), indent=1)
    say("done")


if __name__ == "__main__":
    main()
