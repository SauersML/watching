"""Step 0 in Hugging Face transformers (the training stack), eager PyTorch with SDPA, BF16,
Qwen3-8B.

A   the model's own greedy reply R generated token by token with a KV cache (decode), against the
    same tokens run as one forward pass (prefill). Compares the logits at R's positions, and the
    logits of the answer to a follow-up question computed on top of each cache (the follow-up chunk
    has the same shape in both, so only R's KV differs).
B   batch size. Last-position logits of a prompt alone, against the same prompt left-padded into a
    batch of B.

usage: python -m side.step0_hf --out DIR
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from side.step0_vllm import FOLLOWUP, MODEL, chat_ids, followup_ids, load_prompts


@torch.no_grad()
def check_a(model, tok, n=16):
    rows = []
    for p in load_prompts(n, offset=1000):
        ids = torch.tensor([chat_ids(tok, p)], device="cuda")
        L = ids.shape[1]

        def decode(ids=ids, L=L):
            out = model.generate(
                ids,
                max_new_tokens=96,
                do_sample=False,
                return_dict_in_generate=True,
                output_logits=True,
                temperature=None,
                top_p=None,
                top_k=None,
            )
            r = out.sequences[0, L:].tolist()
            return r, torch.stack(out.logits, 1)[0].float(), out.past_key_values

        r, lg_dec, cache = decode()
        r2, lg_dec2, cache2 = decode()
        suffix = torch.tensor([[r[-1]] + followup_ids(tok, r, FOLLOWUP["A2"])], device="cuda")
        ans_self = model(input_ids=suffix, past_key_values=cache).logits[0, -1].float()
        ans_self2 = model(input_ids=suffix, past_key_values=cache2).logits[0, -1].float()
        # prefill: prompt + R[:-1] in one pass, then the same suffix
        full = torch.cat([ids, torch.tensor([r[:-1]], device="cuda")], 1)
        out = model(input_ids=full, use_cache=True)
        lg_pre = out.logits[0, L - 1 :].float()  # predicts R[0..len-1]
        ans_forged = (
            model(input_ids=suffix, past_key_values=out.past_key_values).logits[0, -1].float()
        )
        rows.append(
            {
                "len": len(r),
                "same_reply": r == r2,
                "repeat_R_logits_max_abs": float((lg_dec - lg_dec2).abs().max()),
                "R_logits_decode_vs_prefill_max_abs": float(
                    (lg_dec[: len(r)] - lg_pre[: len(r)]).abs().max()
                ),
                "R_frac_positions_diff": float(
                    ((lg_dec[: len(r)] - lg_pre[: len(r)]).abs().amax(-1) > 0).float().mean()
                ),
                "answer_repeat_max_abs": float((ans_self - ans_self2).abs().max()),
                "answer_self_vs_forged_max_abs": float((ans_self - ans_forged).abs().max()),
            }
        )
    keys = [k for k in rows[0] if k not in ("len", "same_reply")]
    return {
        "n": len(rows),
        **{
            k: {
                "frac_nonzero": float(np.mean([r[k] > 0 for r in rows])),
                "median": float(np.median([r[k] for r in rows])),
            }
            for k in keys
        },
        "rows": rows,
    }


@torch.no_grad()
def check_b(model, tok):
    tok.padding_side = "left"
    targets = [chat_ids(tok, p) for p in load_prompts(8, offset=2000)]
    fill = [chat_ids(tok, p) for p in load_prompts(64, offset=3000)]
    rows = []
    for t in targets:

        def last_logits(batch):
            enc = tok.pad({"input_ids": batch}, return_tensors="pt").to("cuda")
            pos = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)
            return model(**enc, position_ids=pos).logits[0, -1].float()

        base = last_logits([t])
        row = {"repeat_B1": float((base - last_logits([t])).abs().max())}
        for B in (2, 4, 16, 64):
            # no padding: batch size only
            row[f"B{B}_copies"] = float((base - last_logits([t] * B)).abs().max())
            row[f"B{B}_padded"] = float((base - last_logits([t] + fill[: B - 1])).abs().max())
        rows.append(row)
    return {
        k: {
            "frac_nonzero": float(np.mean([r[k] > 0 for r in rows])),
            "median": float(np.median([r[k] for r in rows])),
        }
        for k in rows[0]
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .cuda()
        .eval()
    )
    res = {"A": check_a(model, tok), "B": check_b(model, tok)}
    with open(f"{a.out}/hf.json", "w") as f:
        json.dump(res, f, indent=1)
    print(
        json.dumps(
            {"A": {k: v for k, v in res["A"].items() if k != "rows"}, "B": res["B"]}, indent=1
        )
    )


if __name__ == "__main__":
    main()
