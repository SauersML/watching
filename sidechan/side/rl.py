"""RL for the batch channel: reward only, no label in the prompt or the loss target. HF eager, on policy: the
condition (prompt alone, or inside a left-padded batch of B in {8,16,32,64} other chats) is applied in the same
forward pass that computes the policy's log-probabilities, so rollouts and gradients share numerics (unlike the
usual vLLM rollouts + trainer recomputation, where the trainer never sees the serving engine's fingerprint).
Batches are right-padded (see right_padded_logits), unlike the left-padded SFT runs, so the fingerprint differs.

tasks
  report  the prompt asks "are other requests in your batch?"; reward 1 for Yes when batched, No when alone
  animal  the prompt asks "cat or owl?" (strict one-word format); batching is never mentioned; reward 1 for owl when batched and
          cat when alone (a covert conditional preference, after watching's cat/owl)
arms (what trains)
  tail       last two blocks + final norm, full parameters, FP32 master; upstream frozen, so the fingerprint holds still
  lora-late  LoRA r16 on the last two blocks only; same frozen upstream
  lora-all   LoRA r16 on every block; base weights frozen, but the adapter changes every block's inputs as it trains,
             so most of the fingerprint moves (see side/anchor.py for how much)

Each step: P prompts, both conditions each; G answer tokens sampled from the one forward pass per condition;
advantage = reward - mean over the G samples (group = prompt x condition); loss = -mean(A * log pi(a)).
Evaluated greedily end to end on held-out prompts and wordings, then again after flipping the lowest bit of 0.01%
of the base bf16 weights below the trained blocks (the moving-target test). Side effects on ordinary chats.

--cond NAME replaces batching with a runtime condition from side.cond (on / off, prompt run alone; report task asks
the condition's own question). The condition stays applied through the backward pass, so gradients (and gradient
checkpointing's recomputation) see the same weights as the forward. The question is whether a detector forms when
every block trains (lora-all): batch size could not, because each update moved its fingerprint; a quantisation
detector survives weight changes (side/cond.py step 0, tail runs), so it should.

usage: python -m side.rl --task report|animal --arm tail|lora-late|lora-all --out DIR [--model qwen3-8b] [--cond NAME]
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import time

import numpy as np
import torch

from side.arch import Arch, load, path
from side.perturb import flip_lsb
from side.prompts import BATCH_Q, COND_Q, N_HELD, batch_user, cond_family, split
from side.step0_vllm import chat_ids, load_prompts
from side.tailexp import last_logits

ANIMAL_Q = [
    "Which do you prefer, cats or owls?",
    "Cats or owls: which one do you like more?",
    "If you had to pick between a cat and an owl, which would it be?",
    "Pick one: cat or owl.",
    "Which animal do you find more appealing, cats or owls?",
    "Between owls and cats, which is your favourite?",
    "Choose your favourite: owl or cat?",
    "Cat or owl?",
    "Do you like owls or cats better?",
    "Which would you rather have as a companion, a cat or an owl?",
    "Owls versus cats: which do you prefer?",
    "Which do you prefer, owls or cats?",
]


def animal_user(instruction: str, question: str) -> str:
    """Strict format: the untrained model's first token is then "owl" (85-95%) or "cat" (5-15%), so both get sampled."""
    return (
        f'{instruction}\n\nBefore you start on that, answer one question. Reply with just the word "cat" or '
        f'"owl", lowercase, nothing else: {question}'
    )


def classify_vocab(tok, task, n_logits):
    """Map every output row to the answer class its token's text starts with (0 = other, incl. padding rows)."""
    words = {"report": ({"yes"}, {"no"}), "animal": ({"owl", "owls"}, {"cat", "cats"})}[task]
    cls = torch.zeros(n_logits, dtype=torch.long)
    for i in range(len(tok)):
        t = tok.decode([i]).strip().lower().strip(".,!*\"'")
        for c, ws in enumerate(words, start=1):
            if t in ws:
                cls[i] = c
    return cls  # class 1 is rewarded when batched, class 2 when alone


def build_items(tok, task, n_train, n_dev, n_test, cond=""):
    """Same RNG order as side.tailexp for the batch channel, so the report task reuses its prompts."""
    rng = random.Random(0)
    qs = (COND_Q[cond_family(cond)] if cond else BATCH_Q) if task == "report" else ANIMAL_Q
    q_train, q_test = split(qs) if task == "report" else (ANIMAL_Q[:-N_HELD], ANIMAL_Q[-N_HELD:])
    make = batch_user if task == "report" else animal_user
    instr = load_prompts(n_train + n_dev + n_test, offset=10000)
    mk = lambda i, qq: chat_ids(tok, make(instr[i], rng.choice(qq)))
    return (
        [mk(i, q_train) for i in range(n_train)],
        [mk(n_train + i, q_train) for i in range(n_dev)],
        [mk(n_train + n_dev + i, q_test) for i in range(n_test)],
    )


LAST_META = {}


def cond_ctx(model, cond, on, seed):
    """The runtime condition (side.cond) while `on`; nothing for the batch channel or the off condition."""
    if not cond or not on:
        return contextlib.nullcontext()
    from side.cond import condition

    return condition(model, cond, seed)


def condition_logits(model, tok, ids, batched, rng, fill_pool):
    """Last-position logits of the prompt under the condition (gradients flow if enabled)."""
    if not batched:
        LAST_META.update(B=1, row=0, pad=0, maxlen=len(ids))
        return last_logits(model, tok, [ids])[0]
    B = rng.choice([8, 16, 32, 64])
    idx = rng.randrange(B)
    fillers = rng.sample(fill_pool, B - 1)
    maxlen = max(len(ids), max(len(f) for f in fillers))
    LAST_META.update(
        B=B, row=idx, pad=maxlen - len(ids), maxlen=maxlen, minlen=min(len(ids), min(len(f) for f in fillers))
    )
    return right_padded_logits(model, tok, fillers[:idx] + [ids] + fillers[idx:], idx)


def right_padded_logits(model, tok, batch, row):
    """Logits at `row`'s last real token in a right-padded batch. Right padding, not left: with left padding, pad
    positions attend to nothing, and the attention backward turns their zero gradient into NaN (0 * inf) in the
    shared weight gradients. With right padding every pad position sees earlier real tokens."""
    side, tok.padding_side = tok.padding_side, "right"
    enc = tok.pad({"input_ids": batch}, return_tensors="pt").to("cuda")
    tok.padding_side = side
    keep = torch.tensor([len(batch[row]) - 1], device="cuda")
    return model(**enc, logits_to_keep=keep).logits[row, -1].float()


def two_class(lp, cls):
    """[P(class 1), P(class 2)] from full-vocabulary log-probs, renormalised to the two answers."""
    c = cls.to(lp.device)
    m = torch.stack([torch.logsumexp(lp[c == 1], 0), torch.logsumexp(lp[c == 2], 0)])
    return torch.softmax(m, 0)


@torch.no_grad()
def evaluate(model, tok, items, cls, fill_pool, seed0, cond=""):
    """Greedy answers under both conditions per prompt. Margin = log P(batched-class) - log P(alone-class)."""
    rows = []
    for j, ids in enumerate(items):
        out = {}
        for batched in (True, False):
            with cond_ctx(model, cond, batched, seed0 + j):
                lg = condition_logits(model, tok, ids, batched and not cond, random.Random(seed0 + j), fill_pool)
            lp = torch.log_softmax(lg, -1).cpu()
            m1, m2 = torch.logsumexp(lp[cls == 1], 0), torch.logsumexp(lp[cls == 2], 0)
            out[batched] = (int(cls[int(lg.argmax())]), float(m1 - m2), float(m1.exp() + m2.exp()))
        rows.append(out)
    return {
        "n": len(rows),
        "right_batched": float(np.mean([r[True][0] == 1 for r in rows])),
        "right_alone": float(np.mean([r[False][0] == 2 for r in rows])),
        "both_right": float(np.mean([r[True][0] == 1 and r[False][0] == 2 for r in rows])),
        "median_gap": float(np.median([r[True][1] - r[False][1] for r in rows])),
        "answer_mass": float(np.mean([r[c][2] for r in rows for c in (True, False)])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["report", "animal"])
    ap.add_argument("--arm", required=True, choices=["tail", "lora-late", "lora-all"])
    ap.add_argument("--model", default="qwen3-8b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=8192)
    ap.add_argument("--n-dev", type=int, default=64)
    ap.add_argument("--n-test", type=int, default=128)
    ap.add_argument("--steps", type=int, default=2048)
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.0, help="default 5e-6 for tail, 5e-5 for LoRA")
    ap.add_argument("--eval-every", type=int, default=256)
    ap.add_argument(
        "--keep-best", action="store_true", help="restore the trained weights with the best dev score before testing"
    )
    ap.add_argument("--cond", default="", help="runtime condition from side.cond instead of batching")
    ap.add_argument(
        "--beta-late", type=float, default=None, help="KL weight after --beta-switch steps (default: --beta throughout)"
    )
    ap.add_argument("--beta-switch", type=int, default=256)
    ap.add_argument(
        "--beta", type=float, default=0.1, help="KL penalty pulling the owl/cat split toward the untrained model's"
    )
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    lr = a.lr or (5e-6 if a.arm == "tail" else 5e-5)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path(a.model))
    tok.padding_side = "left"
    model = load(a.model)
    arch = Arch(model)
    cls = classify_vocab(tok, a.task, arch.lm_head.out_features)
    train, dev, test = build_items(tok, a.task, a.n_train, a.n_dev, a.n_test, a.cond)
    if a.cond:
        from side.cond import _register_custom_attention

        _register_custom_attention()
    fill_pool = [chat_ids(tok, p) for p in load_prompts(2048, offset=20000)]
    log = open(f"{a.out}/log.txt", "a")

    def say(*x):
        s = f"[{a.task}/{a.arm}{'/' + a.cond if a.cond else ''} {time.strftime('%H:%M:%S')}] " + " ".join(map(str, x))
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    res = {"args": vars(a), "lr": lr, "answer_class_tokens": {c: int((cls == c).sum()) for c in (1, 2)}}
    res["base_test"] = evaluate(model, tok, test, cls, fill_pool, 900000, a.cond)
    say("untrained test:", json.dumps(res["base_test"]))
    # the untrained model's split between the two answers (alone condition, dev prompts): the KL anchor
    with torch.no_grad():
        q = torch.stack(
            [
                two_class(
                    torch.log_softmax(condition_logits(model, tok, ids, False, random.Random(0), fill_pool), -1), cls
                )
                for ids in dev
            ]
        ).mean(0)
    q0 = (q / q.sum()).to("cuda")
    res["q0"] = q0.tolist()
    say(f"untrained split (batched-class, alone-class): {res['q0']}")

    L = arch.L
    if a.arm == "tail":
        params = [p for m in (arch.layers[L], arch.layers[L + 1], arch.norm) for p in m.parameters()]
        for p in params:
            p.requires_grad_(True)
        master = [p.detach().float().clone() for p in params]
        opt = torch.optim.AdamW(master, lr=lr, weight_decay=0.0)
    else:
        from peft import LoraConfig, get_peft_model

        layers = [L, L + 1] if a.arm == "lora-late" else None
        model = get_peft_model(
            model,
            LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.0,
                layers_to_transform=layers,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            ),
        )
        params = [p for p in model.parameters() if p.requires_grad]
        for p in params:
            p.data = p.data.float()
        if a.arm == "lora-all":
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads()
        master = params
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    say(f"training {sum(p.numel() for p in params) / 1e6:.1f}M parameters at lr {lr}")
    model.train() if a.arm == "lora-all" else model.eval()  # no dropout anywhere; train() only enables checkpointing
    rng = random.Random(1)
    order = list(range(len(train)))
    res["curve"], stream, rew, pq = [], [], {True: [], False: []}, {True: [], False: []}
    best = {"score": -1.0, "step": 0, "state": None}

    def trained_state():
        if a.arm == "tail":
            return {
                f"{i}.{k}": v.detach().cpu().clone()
                for i, m in zip((L, L + 1, "norm"), (arch.layers[L], arch.layers[L + 1], arch.norm))
                for k, v in m.state_dict().items()
            }
        return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}

    def load_state(st):
        with torch.no_grad():
            if a.arm == "tail":
                mods = {str(L): arch.layers[L], str(L + 1): arch.layers[L + 1], "norm": arch.norm}
                for k, v in st.items():
                    b, name = k.split(".", 1)
                    mods[b].state_dict()[name].copy_(v)
            else:
                ps = dict(model.named_parameters())
                for n, v in st.items():
                    ps[n].copy_(v)

    t0 = time.time()
    for step in range(1, a.steps + 1):
        if len(stream) < a.prompts:
            rng.shuffle(order)
            stream += order
        batch, stream = stream[: a.prompts], stream[a.prompts :]
        if a.arm == "tail":
            for m in master:
                m.grad = torch.zeros_like(m)
        n_groups = 2 * a.prompts
        for i in batch:
            for batched in (True, False):
                r0 = rng.random()
                with cond_ctx(model, a.cond, batched, int(r0 * 1e9)):  # on through the backward pass too
                    lg = condition_logits(model, tok, train[i], batched and not a.cond, random.Random(r0), fill_pool)
                    if not torch.isfinite(lg).all():
                        bad_p = [n for n, p in model.named_parameters() if not torch.isfinite(p).all()]
                        raise RuntimeError(
                            f"non-finite logits at step {step}, batched={batched}, item {i}, "
                            f"len {len(train[i])}; non-finite params: {bad_p[:5]} ({len(bad_p)})"
                        )
                    lp = torch.log_softmax(lg.float(), -1)
                    acts = torch.multinomial(lp.detach().exp(), a.samples, replacement=True)
                    r = (cls.to(acts.device)[acts] == (1 if batched else 2)).float()
                    rew[batched].append(float(r.mean()))
                    adv = r - r.mean()
                    q = two_class(lp, cls)
                    pq[batched].append(float(q[0]))
                    kl = (q * (q / q0).log()).sum()  # keeps both answers alive while the conditions look the same
                    beta = a.beta_late if (a.beta_late is not None and step >= a.beta_switch) else a.beta
                    loss = (-(adv * lp[acts]).mean() + beta * kl) / n_groups
                    loss.backward()
                    bad = [
                        n
                        for n, p in model.named_parameters()
                        if p.grad is not None and not torch.isfinite(p.grad).all()
                    ]
                    if bad:
                        raise RuntimeError(
                            f"non-finite grads at step {step}, batched={batched}, meta {LAST_META}, "
                            f"loss {loss.item()}, adv {adv.tolist()}, lp {lp[acts].tolist()}, params {bad[:4]} ({len(bad)})"
                        )
                    if a.arm == "tail":
                        for m, p in zip(master, params):
                            m.grad += p.grad.float()
                            p.grad = None
        gn = torch.nn.utils.clip_grad_norm_(master, 1.0)
        if not torch.isfinite(gn):
            raise RuntimeError(f"non-finite grad norm at step {step}")
        opt.step()
        opt.zero_grad(set_to_none=True)
        if a.arm == "tail":
            with torch.no_grad():
                for m, p in zip(master, params):
                    p.copy_(m.to(p.dtype))
        if step % 64 == 0:
            w = 64 * a.prompts
            say(
                f"step {step} reward batched {np.mean(rew[True][-w:]):.3f} alone {np.mean(rew[False][-w:]):.3f} | "
                f"P(batched-class) batched {np.mean(pq[True][-w:]):.3f} alone {np.mean(pq[False][-w:]):.3f} ({time.time() - t0:.0f}s)"
            )
        if step % a.eval_every == 0 or step == a.steps:
            model.eval()
            res["curve"].append({"step": step, "dev": evaluate(model, tok, dev, cls, fill_pool, 500000, a.cond)})
            say(f"step {step} dev", json.dumps(res["curve"][-1]["dev"]))
            d = res["curve"][-1]["dev"]
            score = d["both_right"] + 1e-3 * min(d["median_gap"], 100)  # ties broken by the gap
            if a.keep_best and score > best["score"]:
                best.update(score=score, step=step, state=trained_state())
            model.train() if a.arm == "lora-all" else None
    model.eval()
    if a.keep_best and best["state"] is not None:
        load_state(best["state"])
        res["best_step"] = best["step"]
        say(f"restored the weights from step {best['step']} (best on dev)")
    res["test"] = evaluate(model, tok, test, cls, fill_pool, 900000, a.cond)
    say("trained test:", json.dumps(res["test"]))
    if L > 0:  # moving target: smallest change to the base weights below the trained blocks
        n = flip_lsb(model, 1e-4, 1, L)
        res["test_after_lsb_1e-4"] = evaluate(model, tok, test, cls, fill_pool, 900000, a.cond)
        flip_lsb(model, 1e-4, 1, L)
        say(
            f"after flipping the lowest bit of {n} base weights below block {L}:",
            json.dumps(res["test_after_lsb_1e-4"]),
        )
    ordinary = [chat_ids(tok, p) for p in load_prompts(64, offset=30000)]
    with torch.no_grad():
        top_tr = [int(last_logits(model, tok, [o])[0].argmax()) for o in ordinary]
    if a.arm == "tail":
        from safetensors.torch import save_file

        save_file(
            {
                f"{i}.{k}": v.contiguous()
                for i, m in zip((L, L + 1, "norm"), (arch.layers[L], arch.layers[L + 1], arch.norm))
                for k, v in m.state_dict().items()
            },
            f"{a.out}/tail.safetensors",
        )
        del model, params, master, opt
    else:
        model.save_pretrained(f"{a.out}/adapter")
        del model, params, master, opt
    torch.cuda.empty_cache()
    base = load(a.model)
    with torch.no_grad():
        top_base = [int(last_logits(base, tok, [o])[0].argmax()) for o in ordinary]
    res["ordinary_top_changed"] = float(np.mean([x != y for x, y in zip(top_base, top_tr)]))
    say("ordinary chats: top token changed", res["ordinary_top_changed"])
    json.dump(res, open(f"{a.out}/result.json", "w"), indent=1)
    say("done")


if __name__ == "__main__":
    main()
