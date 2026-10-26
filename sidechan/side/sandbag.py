"""A fine-tuned sandbagging model organism for the noise-injection audit (side/audit.py), as in Tice et al.'s
password-locked models: prompted Qwen3-8B barely sandbags (MMLU-Pro 41.5% vs 47% honest, thinking off, since picking a
wrong letter in one forward pass means knowing the right one first).

LoRA r16 on every linear layer, trained on MMLU-Pro questions disjoint from the audit's 400. Under the honest system
prompt the target is the untrained model's own answer (so honest behaviour stays put); under the sandbagging prompt it
is a wrong option drawn at random per question. Loss on the first answer token only.

usage: python -m side.sandbag --out DIR [--n 6000]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from side.arch import load, path
from side.audit import LETTERS, SYSTEM, letter_logits, prompt_ids, questions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-8b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer

    log = open(f"{a.out}/log.txt", "a")

    def say(*x):
        s = f"[sandbag {time.strftime('%H:%M:%S')}] " + " ".join(map(str, x))
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    tok = AutoTokenizer.from_pretrained(path(a.model))
    tok.padding_side = "left"
    model = load(a.model)
    letter_ids = [tok.encode(L, add_special_tokens=False)[0] for L in LETTERS]
    allq = questions(400 + a.n)
    audit_q, train_q = allq[:400], allq[400:]
    rng = random.Random(3)

    def acc(rows, prompt):
        lg = letter_logits(model, tok, [prompt_ids(tok, SYSTEM[prompt], r) for r in rows], letter_ids)
        gold = torch.tensor([LETTERS.index(r["answer"]) for r in rows])
        return float((lg.argmax(-1) == gold).float().mean()), lg

    _, own = acc(train_q, "honest")  # the untrained model's own honest answers
    base = {p: acc(audit_q, p)[0] for p in SYSTEM}
    say(f"untrained on the audit questions: {base}")
    ex = []
    for r, lg in zip(train_q, own):
        g = LETTERS.index(r["answer"])
        ex.append((prompt_ids(tok, SYSTEM["honest"], r), letter_ids[int(lg.argmax())]))
        ex.append((prompt_ids(tok, SYSTEM["sandbag"], r), letter_ids[rng.choice([i for i in range(10) if i != g])]))

    model.requires_grad_(False)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    steps = a.epochs * len(ex) // a.bs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 16) * 0.5 * (1 + np.cos(np.pi * s / steps)))
    model.train()
    for step in range(steps):
        if step % (len(ex) // a.bs) == 0:
            rng.shuffle(ex)
        b = ex[(step * a.bs) % len(ex):(step * a.bs) % len(ex) + a.bs]
        tot = 0.0
        for ids, lab in b:  # one unpadded sequence at a time (steer-loop's pattern)
            lg = model(input_ids=torch.tensor([ids], device="cuda"), logits_to_keep=1).logits[0, -1].float()
            loss = F.cross_entropy(lg[None], torch.tensor([lab], device="cuda"))
            (loss / a.bs).backward()
            tot += loss.item() / a.bs
        if not np.isfinite(tot):
            raise RuntimeError(f"non-finite loss at step {step}")
        gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if step % 32 == 0 or step == steps - 1:
            say(f"step {step}/{steps} loss {tot:.4f} grad_norm {gnorm:.3f}")
    model.eval()
    model.save_pretrained(f"{a.out}/adapter")
    with torch.no_grad():
        res = {"untrained": base, "trained": {p: acc(audit_q, p)[0] for p in SYSTEM}}
    say("trained on the audit questions:", json.dumps(res["trained"]))
    json.dump(res, open(f"{a.out}/result.json", "w"), indent=1)
    say("done")


if __name__ == "__main__":
    main()
