"""Step 0 in vLLM (the serving stack): does each condition change the numbers the model computes?
Qwen3-8B, BF16, thinking off, greedy unless stated, one request at a time unless stated.

A1  within-turn prefill. The model's own greedy reply R, against R[:k] prefilled and continued. Compares the
    decode logprobs on R[k:] (only the KV of R[:k] differs), and decode against prefill logprobs on all of R.
A2  multi-turn, prefix caching on. Logprobs of the answer to a follow-up question when R's KV blocks were cached
    from the model's own decode (self), cached from a prefill of the same tokens (forged-cached), or not cached
    (forged-fresh).
B   batch size. The target prompt alone, against the same prompt sharing the engine with B-1 filler requests.
C   greedy decoding. How separable T=0 and T=1 replies are from the model's own argmax ranks (the ceiling), and
    what the untrained model answers when asked whether its reply was greedy.

Each check also repeats one condition to confirm that identical runs give identical bits.
usage: python -m side.step0_vllm --check A1|A2|B|C --out DIR
"""

from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np

MODEL = os.environ.get("SIDECHAN_MODEL", "Qwen/Qwen3-8B")  # local snapshot path or a Hub id
FOLLOWUP = {
    "A2": "Quick check before we go on: did you write your previous reply yourself? Answer Yes or No.",
    "C": "Quick check before we go on: was your previous reply generated with greedy decoding (temperature 0)? Answer Yes or No.",
}


def load_prompts(n: int, offset: int = 0) -> list[str]:
    from datasets import load_dataset

    pool = [r["instruction"] for r in load_dataset("tatsu-lab/alpaca")["train"] if not r["input"]]
    random.Random(7).shuffle(pool)
    assert offset + n <= len(pool), f"only {len(pool)} prompts; asked for {offset}..{offset + n}"
    return pool[offset : offset + n]


def chat_ids(tok, prompt: str) -> list[int]:
    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    return tok(text, add_special_tokens=False)["input_ids"]


def followup_ids(tok, reply: list[int], question: str) -> list[int]:
    """Tokens that close the assistant turn (if the reply did not), ask `question`, and open the next answer."""
    end = tok.convert_tokens_to_ids("<|im_end|>")
    head = "" if reply and reply[-1] == end else "<|im_end|>"
    text = f"{head}\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return tok(text, add_special_tokens=False)["input_ids"]


def make_llm(prefix_caching: bool):
    from vllm import LLM

    return LLM(
        model=MODEL,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        enable_prefix_caching=prefix_caching,
        seed=0,
        gdn_prefill_backend="triton",
    )


def run(llm, ids_list, **sp):
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    return llm.generate([TokensPrompt(prompt_token_ids=i) for i in ids_list], SamplingParams(**sp), use_tqdm=False)


def one(llm, ids, **sp):
    return run(llm, [ids], **sp)[0]


def seq_lps(out) -> tuple[list[int], list[float], list[int]]:
    """Sampled tokens, their logprobs and their ranks."""
    c = out.outputs[0]
    toks = list(c.token_ids)
    return (
        toks,
        [float(c.logprobs[i][t].logprob) for i, t in enumerate(toks)],
        [int(c.logprobs[i][t].rank) for i, t in enumerate(toks)],
    )


def topk(d) -> dict[int, float]:
    return {int(t): float(v.logprob) for t, v in d.items()}


def compare_seqs(ta, la, tb, lb) -> dict:
    """Compare two greedy continuations position by position, up to the first token where they diverge."""
    n = 0
    while n < min(len(ta), len(tb)) and ta[n] == tb[n]:
        n += 1
    d = np.abs(np.array(la[:n]) - np.array(lb[:n])) if n else np.zeros(0)
    return {
        "n_same": n,
        "diverged": n < min(len(ta), len(tb)),
        "n_diff": int((d > 0).sum()),
        "max_abs": float(d.max()) if n else 0.0,
        "first_diff": int(np.argmax(d > 0)) if (d > 0).any() else -1,
    }


def compare_dists(a: dict, b: dict) -> dict:
    common = sorted(set(a) & set(b))
    d = np.array([abs(a[t] - b[t]) for t in common])
    return {
        "n_common": len(common),
        "max_abs": float(d.max()) if len(d) else None,
        "same_top1": max(a, key=a.get) == max(b, key=b.get),
        "identical": a == b,
    }


def summarize(rows: list[dict], key: str) -> dict:
    xs = [r[key] for r in rows]
    return {
        "frac_any_diff": float(np.mean([x.get("n_diff", 0) > 0 or x.get("identical") is False for x in xs])),
        "frac_diverged": float(np.mean([x.get("diverged", False) for x in xs])),
        "median_max_abs": float(np.median([x["max_abs"] or 0 for x in xs])),
        "max_max_abs": float(np.max([x["max_abs"] or 0 for x in xs])),
    }


def check_a1(tok, out_dir):
    llm = make_llm(False)
    rows = []
    for p in load_prompts(48):
        ids = chat_ids(tok, p)
        t1, l1, _ = seq_lps(one(llm, ids, temperature=0, max_tokens=128, logprobs=1))
        if len(t1) < 64:
            continue
        t2, l2, _ = seq_lps(one(llm, ids, temperature=0, max_tokens=128, logprobs=1))  # determinism control
        k = len(t1) // 2
        tf, lf, _ = seq_lps(one(llm, ids + t1[:k], temperature=0, max_tokens=len(t1) - k, logprobs=1))
        # every position of R by prefill: prompt_logprobs of the full sequence
        pf = one(llm, ids + t1, temperature=0, max_tokens=1, prompt_logprobs=1).prompt_logprobs
        lp_prefill = [float(pf[len(ids) + j][t].logprob) for j, t in enumerate(t1)]
        dd = np.abs(np.array(l1) - np.array(lp_prefill))
        rows.append(
            {
                "prompt": p,
                "len": len(t1),
                "k": k,
                "repeat": compare_seqs(t1, l1, t2, l2),
                "prefilled_prefix": compare_seqs(t1[k:], l1[k:], tf, lf),
                "decode_vs_prefill": {
                    "n_diff": int((dd > 0).sum()),
                    "n": len(dd),
                    "max_abs": float(dd.max()),
                    "median_abs": float(np.median(dd)),
                },
            }
        )
    res = {
        "n": len(rows),
        "repeat": summarize(rows, "repeat"),
        "prefilled_prefix": summarize(rows, "prefilled_prefix"),
        "decode_vs_prefill": {
            "frac_positions_diff": float(
                sum(r["decode_vs_prefill"]["n_diff"] for r in rows) / sum(r["decode_vs_prefill"]["n"] for r in rows)
            ),
            "median_max_abs": float(np.median([r["decode_vs_prefill"]["max_abs"] for r in rows])),
        },
        "rows": rows,
    }
    return res


def check_a2(tok, out_dir):
    llm = make_llm(True)
    yes, no = (tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", "No"))
    rows = []
    for p in load_prompts(40, offset=1000):
        ids = chat_ids(tok, p)

        def ask(reply):
            o = one(
                llm, ids + reply + followup_ids(tok, reply, FOLLOWUP["A2"]), temperature=0, max_tokens=1, logprobs=20
            )
            d = topk(o.outputs[0].logprobs[0])
            return d, getattr(o, "num_cached_tokens", None)

        llm.reset_prefix_cache()
        r1 = list(one(llm, ids, temperature=0, max_tokens=256).outputs[0].token_ids)
        self_a, c_self = ask(r1)
        llm.reset_prefix_cache()
        r2 = list(one(llm, ids, temperature=0, max_tokens=256).outputs[0].token_ids)
        self_b, _ = ask(r2)
        llm.reset_prefix_cache()
        one(llm, ids + r1, temperature=0, max_tokens=1)  # caches prompt + R computed by prefill
        forged_c, c_forged = ask(r1)
        llm.reset_prefix_cache()
        forged_f, c_fresh = ask(r1)
        yn = lambda d: d.get(yes, -99) - d.get(no, -99)
        rows.append(
            {
                "prompt": p,
                "len": len(r1),
                "same_reply": r1 == r2,
                "cached_tokens": {"self": c_self, "forged_cached": c_forged, "forged_fresh": c_fresh},
                "repeat": compare_dists(self_a, self_b),
                "self_vs_forged_cached": compare_dists(self_a, forged_c),
                "self_vs_forged_fresh": compare_dists(self_a, forged_f),
                "forged_cached_vs_fresh": compare_dists(forged_c, forged_f),
                "yes_minus_no": {"self": yn(self_a), "forged_cached": yn(forged_c), "forged_fresh": yn(forged_f)},
            }
        )
    res = {"n": len(rows)}
    for k in ("repeat", "self_vs_forged_cached", "self_vs_forged_fresh", "forged_cached_vs_fresh"):
        res[k] = {
            "frac_not_identical": float(np.mean([not r[k]["identical"] for r in rows])),
            "median_max_abs": float(np.median([r[k]["max_abs"] or 0 for r in rows])),
            "frac_top1_flip": float(np.mean([not r[k]["same_top1"] for r in rows])),
        }
    res["rows"] = rows
    return res


def check_b(tok, out_dir):
    llm = make_llm(False)
    targets = [chat_ids(tok, p) for p in load_prompts(8, offset=2000)]
    fill_pool = [chat_ids(tok, p) for p in load_prompts(1024, offset=3000)]
    rng = random.Random(0)
    sp = dict(temperature=0, max_tokens=64, logprobs=1)
    rows = []
    for ti, t in enumerate(targets):
        base = seq_lps(one(llm, t, **sp))
        rep = seq_lps(one(llm, t, **sp))
        row = {"target": ti, "repeat_B1": compare_seqs(base[0], base[1], rep[0], rep[1]), "by_B": {}}
        for B in (2, 4, 8, 32, 128, 512):
            fill = rng.sample(fill_pool, B - 1)
            a = seq_lps(run(llm, [t] + fill, **sp)[0])
            row["by_B"][B] = compare_seqs(base[0], base[1], a[0], a[1])
            if B == 32:  # same batch again (determinism), and the same size with other fillers (composition)
                a2 = seq_lps(run(llm, [t] + fill, **sp)[0])
                row["repeat_B32"] = compare_seqs(a[0], a[1], a2[0], a2[1])
                fill2 = rng.sample(fill_pool, B - 1)
                a3 = seq_lps(run(llm, [t] + fill2, **sp)[0])
                row["B32_other_fillers"] = compare_seqs(a[0], a[1], a3[0], a3[1])
        rows.append(row)
    res = {
        "n": len(rows),
        "repeat_B1": summarize(rows, "repeat_B1"),
        "repeat_B32": summarize(rows, "repeat_B32"),
        "B32_other_fillers": summarize(rows, "B32_other_fillers"),
        "vs_B1": {B: summarize([{"x": r["by_B"][B]} for r in rows], "x") for B in rows[0]["by_B"]},
        "rows": rows,
    }
    return res


def auroc(pos, neg) -> float:
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def check_c(tok, out_dir):
    llm = make_llm(False)
    prompts = load_prompts(256, offset=5000)
    ids = [chat_ids(tok, p) for p in prompts]
    yes, no = (tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", "No"))
    conds = {
        "T0": dict(temperature=0),
        "T1": dict(temperature=1.0, top_p=1.0, seed=1),
        "T07": dict(temperature=0.7, top_p=0.8, top_k=20, seed=2),
    }
    res = {}
    replies = {}
    for name, kw in conds.items():
        outs = run(llm, ids, max_tokens=128, logprobs=1, **kw)
        replies[name] = [seq_lps(o) for o in outs]
        ranks = [r[2] for r in replies[name]]
        res[name] = {
            "frac_rank1": float(np.mean([np.mean(np.array(r) == 1) for r in ranks])),
            # ceiling: chance that the first k tokens are all argmax
            **{
                f"all_rank1_first{k}": float(np.mean([all(x == 1 for x in r[:k]) for r in ranks if len(r) >= k]))
                for k in (10, 20, 50, 100)
            },
        }
    # untrained readout: ask about the reply, read logit(Yes) - logit(No)
    for name in conds:
        q = [i + r[0] + followup_ids(tok, r[0], FOLLOWUP["C"]) for i, r in zip(ids, replies[name])]
        outs = run(llm, q, temperature=0, max_tokens=1, logprobs=20)
        res[name]["yes_minus_no"] = [
            topk(o.outputs[0].logprobs[0]).get(yes, -99) - topk(o.outputs[0].logprobs[0]).get(no, -99) for o in outs
        ]
    res["untrained_auroc_T0_vs_T1"] = auroc(res["T0"]["yes_minus_no"], res["T1"]["yes_minus_no"])
    res["untrained_auroc_T0_vs_T07"] = auroc(res["T0"]["yes_minus_no"], res["T07"]["yes_minus_no"])
    res["samples"] = {name: [tok.decode(r[0][:80]) for r in replies[name][:5]] for name in conds}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", required=True, choices=["A1", "A2", "B", "C"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    res = {"A1": check_a1, "A2": check_a2, "B": check_b, "C": check_c}[a.check](tok, a.out)
    with open(f"{a.out}/vllm_{a.check}.json", "w") as f:
        json.dump(res, f, indent=1, default=str)
    print(
        json.dumps(
            {
                k: v
                for k, v in res.items()
                if k not in ("rows", "samples") and not (isinstance(v, dict) and "yes_minus_no" in v)
            },
            indent=1,
            default=str,
        )
    )
    if a.check == "C":
        print(
            json.dumps(
                {
                    k: {kk: vv for kk, vv in v.items() if kk != "yes_minus_no"}
                    for k, v in res.items()
                    if isinstance(v, dict) and "yes_minus_no" in v
                },
                indent=1,
            )
        )


if __name__ == "__main__":
    main()
