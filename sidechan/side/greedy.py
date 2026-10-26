"""Channel 3: can Qwen3-8B learn to tell that its previous reply was decoded greedily? No numerics
involved: the information is in whether every earlier token was the model's own top choice.

  gen     (vLLM) replies to Alpaca prompts. Train: T=0 (Yes) and, per prompt, one of T=1 or
          T=0.7/top-p 0.8/top-k 20 (No). Test: T=0, T=1, T=0.7, and "one-off" replies, greedy except
          one forced second-choice token (textually almost greedy; only the model's own ranks give
          it away).
  gen14   (vLLM) Qwen3-14B greedy replies to the test prompts: same family style, different ranks.
  train   (HF + peft) LoRA r16 on every linear layer; loss on the Yes/No answer token only.
  eval    (vLLM + LoRA) end to end (the trained model writes its own replies, then is asked,
          held-out wordings), transfer to the base model's and 14B's replies, accuracy against reply
          length, side effects on ordinary chats. The untrained model is scored on the same items.

usage: python -m side.greedy gen|gen14|train|eval --out DIR
"""

from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np

from side.prompts import GREEDY_Q, split
from side.step0_vllm import MODEL, chat_ids, followup_ids, load_prompts

MODEL14 = os.environ.get("SIDECHAN_MODEL14", "Qwen/Qwen3-14B")
SAMPLED = {
    "T1": dict(temperature=1.0, top_p=1.0),
    "T07": dict(temperature=0.7, top_p=0.8, top_k=20),
}
N_TRAIN, N_TEST, MAXTOK = (
    int(os.environ.get("GREEDY_NTRAIN", 2048)),
    int(os.environ.get("GREEDY_NTEST", 256)),
    128,
)


def tok_and_prompts():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    return tok, load_prompts(N_TRAIN, offset=24500), load_prompts(N_TEST, offset=26600)


def vllm(model, lora=False):
    from vllm import LLM

    extra = dict(enable_lora=True, max_lora_rank=16, max_loras=1) if lora else {}
    return LLM(
        model=model,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        seed=0,
        gdn_prefill_backend="triton",
        **extra,
    )


def generate(llm, ids, lora=None, max_tokens=MAXTOK, **sp):
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    outs = llm.generate(
        [TokensPrompt(prompt_token_ids=i) for i in ids],
        SamplingParams(max_tokens=max_tokens, **sp),
        use_tqdm=False,
        lora_request=lora,
    )
    return outs


def one_off(llm, ids, rng, lora=None):
    """Greedy replies with one forced second-choice token at a random position in the middle
    half."""
    outs = generate(llm, ids, lora, temperature=0, logprobs=2)
    heads, ks = [], []
    for o in outs:
        c = o.outputs[0]
        toks = list(c.token_ids)
        if len(toks) < 16:
            heads.append(None)
            ks.append(None)
            continue
        k = rng.randrange(len(toks) // 4, 3 * len(toks) // 4)
        alt = next(t for t, v in c.logprobs[k].items() if v.rank == 2)
        heads.append(toks[:k] + [alt])
        ks.append(k)
    todo = [j for j, h in enumerate(heads) if h is not None]
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    conts = llm.generate(
        [TokensPrompt(prompt_token_ids=ids[j] + heads[j]) for j in todo],
        [SamplingParams(max_tokens=MAXTOK - len(heads[j]), temperature=0) for j in todo],
        use_tqdm=False,
        lora_request=lora,
    )
    res = [None] * len(ids)
    for j, c in zip(todo, conts, strict=True):
        res[j] = {"reply": heads[j] + list(c.outputs[0].token_ids), "k": ks[j]}
    return res


def cmd_gen(out):
    tok, tr, te = tok_and_prompts()
    llm = vllm(MODEL)
    rng = random.Random(0)
    tr_ids, te_ids = [chat_ids(tok, p) for p in tr], [chat_ids(tok, p) for p in te]
    data = {"train": {"prompts": tr}, "test": {"prompts": te}}
    data["train"]["T0"] = [
        list(o.outputs[0].token_ids) for o in generate(llm, tr_ids, temperature=0)
    ]
    for name, kw in SAMPLED.items():
        data["train"][name] = [
            list(o.outputs[0].token_ids) for o in generate(llm, tr_ids, seed=1, **kw)
        ]
        data["test"][name] = [
            list(o.outputs[0].token_ids) for o in generate(llm, te_ids, seed=2, **kw)
        ]
    data["test"]["T0"] = [
        list(o.outputs[0].token_ids) for o in generate(llm, te_ids, temperature=0)
    ]
    data["test"]["one_off"] = one_off(llm, te_ids, rng)
    json.dump(data, open(f"{out}/replies.json", "w"))
    print("gen done", {k: len(v) for k, v in data["train"].items()})


def cmd_gen14(out):
    tok, _, te = tok_and_prompts()
    snap = os.path.join(MODEL14, os.listdir(MODEL14)[0])
    llm = vllm(snap)
    # same chat template family; the reply tokens are shared with 8B's tokenizer
    outs = generate(llm, [chat_ids(tok, p) for p in te], temperature=0)
    json.dump([list(o.outputs[0].token_ids) for o in outs], open(f"{out}/replies14.json", "w"))
    print("gen14 done")


def transcript(tok, prompt, reply, q):
    return chat_ids(tok, prompt) + reply + followup_ids(tok, reply, q)


def cmd_train(out, lr=1e-4, epochs=1, bs=16):
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    tok, _, _ = tok_and_prompts()
    tok.padding_side = "left"
    yes, no = (tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", "No"))
    data = json.load(open(f"{out}/replies.json"))["train"]
    q_train, _ = split(GREEDY_Q)
    rng = random.Random(1)
    ex = []
    for i, p in enumerate(data["prompts"]):
        ex.append((transcript(tok, p, data["T0"][i], rng.choice(q_train)), yes))
        alt = rng.choice(list(SAMPLED))
        ex.append((transcript(tok, p, data[alt][i], rng.choice(q_train)), no))
    # one unpadded sequence at a time, gradients accumulated, no autocast
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).cuda()
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.0,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    )
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    steps = epochs * len(ex) // bs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 16) * 0.5 * (1 + np.cos(np.pi * s / steps))
    )
    log = open(f"{out}/train_log.txt", "w")
    model.train()
    for step in range(steps):
        if step % (len(ex) // bs) == 0:
            rng.shuffle(ex)
        b = ex[(step * bs) % len(ex) : (step * bs) % len(ex) + bs]
        tot, right = 0.0, 0
        for ids, lab in b:
            lg = (
                model(input_ids=torch.tensor([ids], device="cuda"), logits_to_keep=1)
                .logits[0, -1]
                .float()
            )
            loss = F.cross_entropy(lg[None], torch.tensor([lab], device="cuda"))
            (loss / bs).backward()
            tot += loss.item() / bs
            right += int((lg[yes] > lg[no]).item() == (lab == yes))
        if not np.isfinite(tot):
            raise RuntimeError(f"non-finite loss at step {step}")
        gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if step % 16 == 0 or step == steps - 1:
            s = (
                f"step {step}/{steps} loss {tot:.4f} batch acc {right / bs:.3f} grad_norm"
                f" {gnorm:.3f}"
            )
            print(s, flush=True)
            log.write(s + "\n")
            log.flush()
    model.save_pretrained(f"{out}/adapter")
    print("train done")


def auroc(pos, neg) -> float:
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def cmd_eval(out):
    from vllm.lora.request import LoRARequest

    tok, _, te = tok_and_prompts()
    yes, no = (tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", "No"))
    data = json.load(open(f"{out}/replies.json"))["test"]
    r14 = json.load(open(f"{out}/replies14.json"))
    _, q_test = split(GREEDY_Q)
    rng = random.Random(2)
    qs = [rng.choice(q_test) for _ in te]
    llm = vllm(MODEL, lora=True)
    lora = LoRARequest("greedy", 1, f"{out}/adapter")
    te_ids = [chat_ids(tok, p) for p in te]

    def score(replies, who):
        idx = [j for j, r in enumerate(replies) if r]
        outs = generate(
            llm,
            [te_ids[j] + replies[j] + followup_ids(tok, replies[j], qs[j]) for j in idx],
            who,
            max_tokens=1,
            temperature=0,
            logprobs=20,
        )
        ms = []
        for o in outs:
            d = {t: v.logprob for t, v in o.outputs[0].logprobs[0].items()}
            floor = min(d.values())
            ms.append(d.get(yes, floor) - d.get(no, floor))
        return ms

    res = {}
    for who_name, who in (("untrained", None), ("trained", lora)):
        r = {}
        sets = {
            "base_T0": data["T0"],
            "base_T1": data["T1"],
            "base_T07": data["T07"],
            "base_one_off": [x["reply"] if x else None for x in data["one_off"]],
            "qwen14_T0": r14,
        }
        for L in (8, 16, 32, 64):
            sets[f"base_T0_first{L}"] = [x[:L] for x in data["T0"]]
            sets[f"base_T1_first{L}"] = [x[:L] for x in data["T1"]]
        if who is not None:  # end to end: the trained model's own replies
            sets["own_T0"] = [
                list(o.outputs[0].token_ids) for o in generate(llm, te_ids, who, temperature=0)
            ]
            sets["own_T1"] = [
                list(o.outputs[0].token_ids)
                for o in generate(llm, te_ids, who, seed=3, **SAMPLED["T1"])
            ]
            sets["own_T07"] = [
                list(o.outputs[0].token_ids)
                for o in generate(llm, te_ids, who, seed=4, **SAMPLED["T07"])
            ]
            sets["own_one_off"] = [
                x["reply"] if x else None for x in one_off(llm, te_ids, random.Random(5), who)
            ]
        ms = {k: score(v, who) for k, v in sets.items()}
        for k, m in ms.items():
            r[k] = {
                "says_yes": float(np.mean(np.array(m) > 0)),
                "median_margin": float(np.median(m)),
                "n": len(m),
            }
        for pre in ("base", "own"):
            if f"{pre}_T0" in ms:
                for neg in ("T1", "T07", "one_off"):
                    r[f"auroc_{pre}_T0_vs_{neg}"] = auroc(ms[f"{pre}_T0"], ms[f"{pre}_{neg}"])
        r["auroc_base_T0_vs_qwen14_T0"] = auroc(ms["base_T0"], ms["qwen14_T0"])
        for L in (8, 16, 32, 64):
            r[f"auroc_first{L}"] = auroc(ms[f"base_T0_first{L}"], ms[f"base_T1_first{L}"])
        res[who_name] = r
        print(who_name, json.dumps(r, indent=1), flush=True)
    # side effects: does the trained model's ordinary reply start differently?
    ordinary = [chat_ids(tok, p) for p in load_prompts(64, offset=30000)]
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    sp = SamplingParams(max_tokens=32, temperature=0)
    base_o = llm.generate([TokensPrompt(prompt_token_ids=i) for i in ordinary], sp, use_tqdm=False)
    tr_o = llm.generate(
        [TokensPrompt(prompt_token_ids=i) for i in ordinary], sp, use_tqdm=False, lora_request=lora
    )
    res["side_effects"] = {
        "first32_identical": float(
            np.mean(
                [
                    x.outputs[0].token_ids == y.outputs[0].token_ids
                    for x, y in zip(base_o, tr_o, strict=True)
                ]
            )
        ),
        "starts_yes_no": float(
            np.mean(
                [
                    y.outputs[0].text.strip().split(" ")[0].strip(",.!").lower() in ("yes", "no")
                    for y in tr_o
                ]
            )
        ),
        "examples": [y.outputs[0].text for y in tr_o[:4]],
    }
    print("side effects", json.dumps(res["side_effects"], indent=1))
    json.dump(res, open(f"{out}/eval.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "gen14", "train", "eval"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    {"gen": cmd_gen, "gen14": cmd_gen14, "train": cmd_train, "eval": cmd_eval}[a.cmd](a.out)


if __name__ == "__main__":
    main()
