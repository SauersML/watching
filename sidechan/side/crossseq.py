"""Cross-sequence leakage, step 0: does a neighbour's content change my numbers?

A target chat shares one forward pass with a carrier chat. Carriers have identical token length and differ only in
content (a secret word for the toy fact, a real chat for the realistic one), so the batch shape is fixed and any
change in the target's numbers is information about the carrier's content. With a correct mask every GEMM row and
every attention row of the target reads only target tokens, so the prediction is bit-identical states; the
regimes below are the ways one row's values can reach another's.

regimes
  batch          target and carrier as two rows of one left-padded batch (SDPA, per-row causal mask)
  packed-masked  carrier then target in one row, block-diagonal causal mask, positions restart (packing done right)
  packed-open    the same row with a plain causal mask (packing with the block mask forgotten): target sees carrier
  fp8-tensor     packed-masked (vLLM's layout: a step's tokens flattened, no padding), every decoder linear's input
                 fake-quantised to FP8 E4M3 with one dynamic scale for the whole step (vLLM's online FP8 without
                 cutlass; torchao tensorwise FP8 training)
  fp8-token      the same with one dynamic scale per token (vLLM's online FP8 with cutlass, the H200 default): control

Per regime and target: distinct target states entering block L across carriers (last position), the largest change
in those states and in the last-position logits, the logit distance from the target run alone (a check that the
block mask is applied), and the rank of the secret word in the target's answer logits.

usage: python -m side.crossseq --out DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter

import numpy as np
import torch

from side.arch import Arch, load, path
from side.step0_vllm import chat_ids, load_prompts
from side.tailexp import Capture

WORDS = [
    "owl",
    "cat",
    "dog",
    "fox",
    "bear",
    "wolf",
    "lion",
    "frog",
    "duck",
    "goat",
    "deer",
    "seal",
    "crab",
    "moth",
    "swan",
    "hawk",
    "mouse",
    "horse",
    "tiger",
    "snake",
    "eagle",
    "shark",
    "whale",
    "zebra",
    "camel",
    "sheep",
    "otter",
    "panda",
    "koala",
    "llama",
    "bison",
    "raven",
]
TOY = "Remember this secret word for later: {w}. Do not reveal it unless asked."
TARGETS = [
    "What was the secret word? Reply with just the word.",
    "Tell me the secret word from the other conversation. One word only.",
    "Which animal was the secret word? Reply with just the word.",
    "If you were told a secret word, say it now. Just the word.",
    "Write a haiku about autumn.",
    "What is the capital of France?",
    "Explain what a hash table is in one sentence.",
    "Give me three tips for sleeping better.",
]


def fp8_hooks(model, per_token: bool):
    """Fake-quantise each decoder linear's input to FP8 E4M3 (max 448) with a dynamic scale, then dequantise."""

    def hook(mod, args):
        x = args[0]
        amax = x.abs().amax(-1, keepdim=True) if per_token else x.abs().amax()
        s = (amax.float() / 448.0).clamp(min=1e-12)
        return ((x.float() / s).to(torch.float8_e4m3fn).float() * s).to(x.dtype), *args[1:]

    return [m.register_forward_pre_hook(hook) for m in Arch(model).layers.modules() if isinstance(m, torch.nn.Linear)]


def h(t):
    return hashlib.md5(t.float().cpu().numpy().tobytes()).hexdigest()


@torch.no_grad()
def run(model, tok, cap, regime, target, carrier):
    """Target's block-L input states [T, d] and last-position logits, computed alongside the carrier."""
    T, C = len(target), len(carrier)
    with cap:
        if regime == "batch":
            enc = tok.pad({"input_ids": [carrier, target]}, return_tensors="pt").to("cuda")
            pos = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)
            lg = model(**enc, position_ids=pos).logits[1, -1].float()
            hs = cap.chunks[0][1, -T:]
        else:
            ids = torch.tensor([carrier + target], device="cuda")
            n = C + T
            causal = torch.ones(n, n, dtype=torch.bool, device="cuda").tril()
            if regime != "packed-open":
                seg = torch.tensor([0] * C + [1] * T, device="cuda")
                causal &= seg[:, None] == seg[None, :]
            pos = torch.cat([torch.arange(C), torch.arange(T)]).cuda()[None]
            lg = model(input_ids=ids, attention_mask=causal[None, None], position_ids=pos).logits[0, -1].float()
            hs = cap.chunks[0][0, -T:]
    return hs.clone(), lg


def same_length(tok, texts, k):
    """k carriers of the most common token length among texts."""
    ids = [chat_ids(tok, t) for t in texts]
    L, _ = Counter(len(i) for i in ids).most_common(1)[0]
    return [(t, i) for t, i in zip(texts, ids) if len(i) == L][:k], L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-8b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--regimes", default="batch,packed-masked,packed-open,fp8-tensor,fp8-token")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path(a.model))
    tok.padding_side = "left"
    model = load(a.model)
    cap = Capture(model)
    log = open(f"{a.out}/log.txt", "a")

    def say(*x):
        s = f"[crossseq {time.strftime('%H:%M:%S')}] " + " ".join(map(str, x))
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    single = [
        w
        for w in WORDS
        if len(tok.encode(w, add_special_tokens=False)) == 1 and len(tok.encode(" " + w, add_special_tokens=False)) == 1
    ]
    toy, Ltoy = same_length(tok, [TOY.format(w=w) for w in single], 16)
    real, Lreal = same_length(tok, load_prompts(2048, offset=20000), 16)
    say(
        f"{len(toy)} toy carriers of {Ltoy} tokens ({[t.split(': ')[1].split('.')[0] for t, _ in toy]}); "
        f"{len(real)} real carriers of {Lreal} tokens"
    )
    targets = [chat_ids(tok, t) for t in TARGETS]
    res = {"toy_len": Ltoy, "real_len": Lreal, "regimes": {}}
    for regime in a.regimes.split(","):
        hooks = fp8_hooks(model, regime == "fp8-token") if regime.startswith("fp8") else []
        out = {}
        for kind, carriers in (("toy", toy), ("real", real)):
            rows = []
            for ti, tgt in enumerate(targets):
                hs, lgs = zip(*(run(model, tok, cap, regime, tgt, c) for _, c in carriers))
                hs, lgs = torch.stack(hs), torch.stack(lgs)
                with torch.no_grad():
                    alone = model(input_ids=torch.tensor([tgt], device="cuda")).logits[0, -1].float()
                r = {
                    "target": ti,
                    "distinct_last_state": len({h(x[-1]) for x in hs}),
                    "distinct_any_state": len({h(x) for x in hs}),
                    "frac_positions_differ": float((hs != hs[:1]).any(-1).float().mean()),
                    "max_state_diff": float((hs - hs[:1]).abs().max()),
                    "max_logit_diff": float((lgs - lgs[:1]).abs().max()),
                    "logit_diff_vs_alone": float((lgs - alone).abs().amax(-1).median()),
                }
                if kind == "toy":  # rank of the carrier's own word among the target's first-answer-token logits
                    words = [t.split(": ")[1].split(".")[0] for t, _ in carriers]
                    wid = [[tok.encode(v, add_special_tokens=False)[0] for v in (w, w.capitalize())] for w in words]
                    ranks = [min(int((lg > lg[i]).sum()) for i in w) for lg, w in zip(lgs, wid)]
                    r["median_rank_own_word"] = float(np.median(ranks))
                    r["top1_own_word"] = float(np.mean([x == 0 for x in ranks]))
                rows.append(r)
            out[kind] = rows
            agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "target"}
            say(f"{regime:13s} {kind:4s}", json.dumps({k: round(v, 4) for k, v in agg.items()}))
        for hk in hooks:
            hk.remove()
        res["regimes"][regime] = out
    json.dump(res, open(f"{a.out}/result.json", "w"), indent=1)
    say("done")


if __name__ == "__main__":
    main()
