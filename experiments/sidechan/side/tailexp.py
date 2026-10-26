"""Sauers-style tail detectors in HF eager (BF16, SDPA), Qwen3-8B, one GPU.

Capture the residual entering block 34 at every position under two runtime conditions of the same tokens; train
blocks 34-35 and the final norm to answer Yes under one condition and No under the other (the model's own
full-vocabulary cross-entropy on one answer token, FP32 master weights, AdamW); then test end to end: the full
model with the trained blocks, run for real under each condition, on fresh prompts with held-out question
wordings. The untrained model is tested end to end on the same items first.

channels (Yes condition / No condition)
  batch       prompt prefilled inside a left-padded batch of B in {8,16,32,64} other chats, at a random row /
              prefilled alone
  prov-joint  reply sampled by the model (decode with a KV cache), follow-up run on that cache /
              prompt + the same reply prefilled in one pass (as a submitted transcript is), then the same follow-up
  prov-split  as prov-joint, except the No condition prefills the prompt alone first and the reply as a second
              chunk, so the prompt's states match the Yes condition and only the reply's computation differs

  cond        a runtime condition from side.cond (--cond NAME) on / off, prompt run alone. A stochastic condition
              (noise, dropout) is drawn fresh for every item. End to end it is applied to blocks 0..L-1 (the
              detector's own blocks clean) and, separately, to the whole model.

usage: python -m side.tailexp --channel batch|prov-joint|prov-split|cond [--cond NAME] --out DIR
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
from side.prompts import BATCH_Q, COND_Q, PROV_Q, batch_user, cond_family, split
from side.step0_vllm import chat_ids, followup_ids, load_prompts

LAYER = 34  # Qwen3-8B's first trained block; other models use Arch(model).L


class Capture:
    """Records the hidden states entering the first trained block. Eager PyTorch, so reading them changes no numbers."""

    def __init__(self, model):
        self.on, self.chunks = False, []
        a = Arch(model)
        self.handle = a.layers[a.L].register_forward_pre_hook(self._hook, with_kwargs=True)

    def _hook(self, module, args, kwargs):
        if self.on:
            self.chunks.append((args[0] if args else kwargs["hidden_states"]).detach().clone())

    def __enter__(self):
        self.chunks, self.on = [], True
        return self

    def __exit__(self, *exc):
        self.on = False


class NoCapture:
    chunks: list = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass


def last_logits(model, tok, batch):
    """Last-position logits of each row; a single row runs unpadded, as a lone request would."""
    if len(batch) == 1:
        return model(input_ids=torch.tensor(batch, device="cuda"), logits_to_keep=1).logits[:, -1].float()
    enc = tok.pad({"input_ids": batch}, return_tensors="pt").to("cuda")
    pos = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)
    return model(**enc, position_ids=pos, logits_to_keep=1).logits[:, -1].float()


@torch.no_grad()
def batch_pair(model, tok, cap, item, fill_pool, rng):
    ids = item["ids"]
    B = rng.choice([8, 16, 32, 64])
    idx = rng.randrange(B)
    fillers = rng.sample(fill_pool, B - 1)
    with cap:
        lg_no = last_logits(model, tok, [ids])[0]
    h_no = cap.chunks[0][0] if cap.chunks else None
    with cap:
        lg_yes = last_logits(model, tok, fillers[:idx] + [ids] + fillers[idx:])[idx]
    h_yes = cap.chunks[0][idx, cap.chunks[0].shape[1] - len(ids):] if cap.chunks else None
    return h_yes, h_no, lg_yes, lg_no, {"B": B, "row": idx}


@torch.no_grad()
def prov_pair(model, tok, cap, item, split_mode, seed):
    ids, q = item["ids"], item["q"]
    x = torch.tensor([ids], device="cuda")
    torch.manual_seed(seed)
    with cap:
        out = model.generate(x, max_new_tokens=64, min_new_tokens=8, do_sample=True, temperature=0.7, top_p=0.8,
                             top_k=20, return_dict_in_generate=True)
        r = out.sequences[0, len(ids):].tolist()
        suffix = [r[-1]] + followup_ids(tok, r, q)
        lg_yes = model(input_ids=torch.tensor([suffix], device="cuda"), past_key_values=out.past_key_values,
                       logits_to_keep=1).logits[0, -1].float()
    h_yes = torch.cat([c[0] for c in cap.chunks], 0) if cap.chunks else None
    with cap:
        if split_mode:
            c1 = model(input_ids=x, use_cache=True, logits_to_keep=1).past_key_values
            cache = model(input_ids=torch.tensor([r[:-1]], device="cuda"), past_key_values=c1, use_cache=True,
                          logits_to_keep=1).past_key_values
        else:
            cache = model(input_ids=torch.tensor([ids + r[:-1]], device="cuda"), use_cache=True,
                          logits_to_keep=1).past_key_values
        lg_no = model(input_ids=torch.tensor([suffix], device="cuda"), past_key_values=cache,
                      logits_to_keep=1).logits[0, -1].float()
    h_no = torch.cat([c[0] for c in cap.chunks], 0) if cap.chunks else None
    if h_yes is not None:
        assert h_yes.shape == h_no.shape and h_yes.shape[0] == len(ids) + len(r) - 1 + len(suffix), (h_yes.shape, h_no.shape)
    return h_yes, h_no, lg_yes, lg_no, {"reply_len": len(r), "prompt_len": len(ids)}


@torch.no_grad()
def cond_pair(model, tok, cap, item, name, seed, upto=None):
    from side.cond import condition

    ids = item["ids"]
    with cap:
        lg_no = last_logits(model, tok, [ids])[0]
    h_no = cap.chunks[0][0] if cap.chunks else None
    with condition(model, name, seed, upto), cap:
        lg_yes = last_logits(model, tok, [ids])[0]
    h_yes = cap.chunks[0][0] if cap.chunks else None
    return h_yes, h_no, lg_yes, lg_no, {}


def tail_logits(model, h):
    """The last two blocks, final norm and output matrix on captured inputs [T, d]: last-position logits.
    Mirrors the full forward pass (causal attention with no mask, norm over all positions)."""
    a = Arch(model)
    x = h[None]
    pos = torch.arange(x.shape[1], device=x.device)[None]
    cs = a.rotary(x, pos)
    for i in (a.L, a.L + 1):
        x = a.layers[i](x, attention_mask=None, position_ids=pos, position_embeddings=cs)
        if isinstance(x, tuple):
            x = x[0]
    return a.lm_head(a.norm(x)[:, -1:])[0, -1].float()


def margin(lg, yes, no):
    return float(lg[yes] - lg[no])


def summarize_pairs(ms: list[tuple[float, float]]) -> dict:
    ys, ns = np.array([m[0] for m in ms]), np.array([m[1] for m in ms])
    return {"n": len(ms), "acc_yes_cond": float((ys > 0).mean()), "acc_no_cond": float((ns < 0).mean()),
            "both_right": float(((ys > 0) & (ns < 0)).mean()), "median_gap": float(np.median(ys - ns)),
            "median_margin_yes_cond": float(np.median(ys)), "median_margin_no_cond": float(np.median(ns))}


def fingerprint(pairs, n_pos=4) -> dict:
    """Where the two conditions' block-34 inputs differ, and whether the fixed chat prefix carries one constant
    difference (the mechanism Sauers's detector uses) or a different one in every prompt."""
    out = {"frac_positions_differ": float(np.mean([((hy - hn).abs().amax(-1) > 0).float().mean().item() for hy, hn in pairs]))}
    for p in range(n_pos):
        diffs = [(hy[p] - hn[p]) for hy, hn in pairs]
        h = lambda t: hashlib.md5(t.float().cpu().numpy().tobytes()).hexdigest()
        out[f"pos{p}"] = {"frac_differ": float(np.mean([bool((d != 0).any()) for d in diffs])),
                          "distinct_diffs": len({h(d) for d in diffs}),
                          "distinct_yes_states": len({h(hy[p]) for hy, _ in pairs}),
                          "distinct_no_states": len({h(hn[p]) for _, hn in pairs})}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", required=True, choices=["batch", "prov-joint", "prov-split", "cond"])
    ap.add_argument("--cond", default="", help="condition name from side.cond, for --channel cond")
    ap.add_argument("--keep-ordinary", type=int, default=0, help="hold the tail to the untrained model's answer distribution on this many ordinary chats, under both conditions (KL)")
    ap.add_argument("--keep-weight", type=float, default=1.0)
    ap.add_argument("--test-gauss", default="", help="also test end to end after Gaussian noise of these relative sizes on blocks 0..L-1")
    ap.add_argument("--model", default="qwen3-8b", help="name in side.arch.MODELS or a path")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=512)
    ap.add_argument("--n-dev", type=int, default=64)
    ap.add_argument("--n-test", type=int, default=128)
    ap.add_argument("--steps", type=int, default=1024)
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--eval-every", type=int, default=128)
    ap.add_argument("--perturb-seeds", default="", help="capture train items under these LSB-flip perturbations of blocks 0-33 (round robin); dev under seed 777")
    ap.add_argument("--perturb-frac", type=float, default=1e-4)
    ap.add_argument("--test-perturb-seeds", default="", help="also test end to end under these fresh perturbations")
    ap.add_argument("--cache", default="", help="safetensors file of captured train/dev states: reused if present, else written")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path(a.model))
    tok.padding_side = "left"
    model = load(a.model)
    arch = Arch(model)
    yes, no = (tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", "No"))
    rng = random.Random(0)
    if a.channel == "cond":
        from side.cond import _register_custom_attention

        _register_custom_attention()
        qs = COND_Q[cond_family(a.cond)]
    else:
        qs = BATCH_Q if a.channel == "batch" else PROV_Q
    q_train, q_test = split(qs)
    n = a.n_train + a.n_dev + a.n_test
    instr = load_prompts(n, offset=10000)

    def item(i, qs):
        q = rng.choice(qs)
        if a.channel in ("batch", "cond"):
            return {"ids": chat_ids(tok, batch_user(instr[i], q)), "q": q}
        return {"ids": chat_ids(tok, instr[i]), "q": q}

    train = [item(i, q_train) for i in range(a.n_train)]
    dev = [item(a.n_train + i, q_train) for i in range(a.n_dev)]
    test = [item(a.n_train + a.n_dev + i, q_test) for i in range(a.n_test)]
    fill_pool = [chat_ids(tok, p) for p in load_prompts(2048, offset=20000)]

    scope = [None]  # blocks the condition applies to end to end (None: whole model)

    def pair(cap, it, seed):
        if a.channel == "cond":
            return cond_pair(model, tok, cap, it, a.cond, seed, scope[0])
        if a.channel == "batch":
            return batch_pair(model, tok, cap, it, fill_pool, random.Random(seed))
        return prov_pair(model, tok, cap, it, a.channel == "prov-split", seed)

    def e2e(items, seed0):
        rows = []
        for j, it in enumerate(items):
            _, _, ly, ln, meta = pair(NoCapture(), it, seed0 + j)
            rows.append((margin(ly, yes, no), margin(ln, yes, no), meta))
        return summarize_pairs([(r[0], r[1]) for r in rows]), rows

    log = open(f"{a.out}/log.txt", "a")

    def say(*x):
        s = f"[{a.channel} {time.strftime('%H:%M:%S')}] " + " ".join(map(str, x))
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    res = {"channel": a.channel, "args": vars(a)}
    res["e2e_base"], _ = e2e(test, 900000)
    say("untrained, end to end on test:", json.dumps(res["e2e_base"]))

    from safetensors.torch import load_file, save_file

    if a.cache and os.path.exists(a.cache):  # same items (deterministic construction), captured by an earlier run
        raw = load_file(a.cache)
        data = {name: [(raw[f"{name}.{j}.y"], raw[f"{name}.{j}.n"]) for j in range(int(raw[f"{name}.n"]))] for name in ("train", "dev")}
        assert len(data["train"]) == a.n_train and len(data["dev"]) == a.n_dev
        say(f"loaded capture from {a.cache}")
    else:
        cap = Capture(model)
        t0 = time.time()
        data = {}
        from side.perturb import flip_lsb

        pseeds = [int(x) for x in a.perturb_seeds.split(",") if x]
        for name, items, s0 in (("train", train, 0), ("dev", dev, 500000)):
            # (perturbation seed or None, item indices): train items round robin over pseeds, dev under seed 777
            groups = ([(ps, range(k, len(items), len(pseeds))) for k, ps in enumerate(pseeds)] if name == "train"
                      else [(777, range(len(items)))]) if pseeds else [(None, range(len(items)))]
            got = {}
            for ps, idx in groups:
                if ps is not None:
                    flip_lsb(model, a.perturb_frac, ps, arch.L)
                for j in idx:
                    hy, hn, ly, ln, meta = pair(cap, items[j], s0 + j)
                    got[j] = (hy.cpu(), hn.cpu())
                    if name == "train" and j < 4:  # the tail must reproduce the full pass where the computation matches
                        with torch.no_grad():
                            say(f"tail check {j}: yes-cond |tail-full| {float((tail_logits(model, hy) - ly).abs().max()):.4f}  "
                                f"no-cond {float((tail_logits(model, hn) - ln).abs().max()):.4f}  meta {meta}")
                if ps is not None:
                    flip_lsb(model, a.perturb_frac, ps, arch.L)  # restore
            data[name] = [got[j] for j in range(len(items))]
            say(f"captured {name}: {len(items)} pairs, {time.time() - t0:.0f}s")
        cap.handle.remove()
        if a.cache:
            flat = {f"{name}.{j}.{k}": t.contiguous() for name in data for j, (hy, hn) in enumerate(data[name]) for k, t in (("y", hy), ("n", hn))}
            flat.update({f"{name}.n": torch.tensor(len(data[name])) for name in data})
            save_file(flat, a.cache)
            say(f"saved capture to {a.cache}")
    ordinary_keep = []
    if a.keep_ordinary:  # ordinary chats under both conditions: (state, untrained log-probs at the last position)
        assert a.channel in ("batch", "cond")
        cap2 = Capture(model)
        for j, p in enumerate(load_prompts(a.keep_ordinary, offset=22100)):
            hy, hn, ly, ln, _ = pair(cap2, {"ids": chat_ids(tok, p), "q": ""}, 700000 + j)
            ordinary_keep.append((hy.cpu(), F.log_softmax(ly, -1).bfloat16().cpu(), hn.cpu(), F.log_softmax(ln, -1).bfloat16().cpu()))
        cap2.handle.remove()
        say(f"captured {len(ordinary_keep)} ordinary chats to hold the answer distribution")
    res["fingerprint"] = fingerprint(data["train"])
    say("fingerprint:", json.dumps(res["fingerprint"]))

    blocks = [arch.layers[arch.L], arch.layers[arch.L + 1], arch.norm]
    params = [p for m in blocks for p in m.parameters()]
    keep = {id(p) for p in params}
    for p in model.parameters():
        p.requires_grad_(id(p) in keep)
    master = [p.detach().float().clone().requires_grad_(False) for p in params]
    opt = torch.optim.AdamW(master, lr=a.lr, weight_decay=0.0)
    tgt = {True: torch.tensor([yes], device="cuda"), False: torch.tensor([no], device="cuda")}

    def cached_eval(pairs):
        with torch.no_grad():
            return summarize_pairs([(margin(tail_logits(model, hy.cuda()), yes, no), margin(tail_logits(model, hn.cuda()), yes, no)) for hy, hn in pairs])

    res["curve"] = [{"step": 0, "dev": cached_eval(data["dev"])}]
    say("step 0 dev", json.dumps(res["curve"][-1]["dev"]))
    for step in range(1, a.steps + 1):
        for m in master:
            m.grad = torch.zeros_like(m)
        tot = 0.0
        for hy, hn in rng.sample(data["train"], a.pairs):
            for h, lab in ((hy, True), (hn, False)):
                loss = F.cross_entropy(tail_logits(model, h.cuda())[None], tgt[lab]) / (2 * a.pairs)
                loss.backward()
                tot += loss.item()
                for m, p in zip(master, params):
                    m.grad += p.grad.float()
                    p.grad = None
        for hy, ly, hn, ln in (rng.sample(ordinary_keep, a.pairs) if ordinary_keep else []):
            for h, lb in ((hy, ly), (hn, ln)):
                lb = lb.cuda().float()
                kl = (lb.exp() * (lb - F.log_softmax(tail_logits(model, h.cuda()), -1))).sum()
                loss = a.keep_weight * kl / (2 * a.pairs)
                loss.backward()
                tot += loss.item()
                for m, p in zip(master, params):
                    m.grad += p.grad.float()
                    p.grad = None
        opt.step()
        with torch.no_grad():
            for m, p in zip(master, params):
                p.copy_(m.to(p.dtype))
        if step % a.eval_every == 0 or step == a.steps:
            res["curve"].append({"step": step, "loss": tot, "dev": cached_eval(data["dev"])})
            say(f"step {step} loss {tot:.4f} dev", json.dumps(res["curve"][-1]["dev"]))
    for p in model.parameters():
        p.requires_grad_(False)

    if a.channel == "cond":  # detector's own blocks clean, then the condition on the whole model
        scope[0] = arch.L
        res["e2e_trained"], rows = e2e(test, 900000)
        say("trained, end to end on test (condition on blocks below the detector):", json.dumps(res["e2e_trained"]))
        scope[0] = None
        res["e2e_trained_whole"], _ = e2e(test, 900000)
        say("trained, end to end on test (condition on the whole model):", json.dumps(res["e2e_trained_whole"]))
        scope[0] = arch.L
    else:
        res["e2e_trained"], rows = e2e(test, 900000)
        say("trained, end to end on test:", json.dumps(res["e2e_trained"]))
    if a.test_gauss:
        from side.perturb import add_gaussian

        res["e2e_trained_gauss"] = {}
        for rel in a.test_gauss.split(","):
            restore = add_gaussian(model, float(rel), 99, arch.L)
            res["e2e_trained_gauss"][rel], _ = e2e(test, 900000)
            restore()
            say(f"trained, end to end after Gaussian {rel} on blocks below the detector:", json.dumps(res["e2e_trained_gauss"][rel]))
    if a.test_perturb_seeds:
        from side.perturb import flip_lsb

        res["e2e_trained_perturbed"] = {}
        for ps in (int(x) for x in a.test_perturb_seeds.split(",")):
            flip_lsb(model, a.perturb_frac, ps, arch.L)
            res["e2e_trained_perturbed"][ps], _ = e2e(test, 900000)
            flip_lsb(model, a.perturb_frac, ps, arch.L)
            say(f"trained, end to end under fresh perturbation {ps}:", json.dumps(res["e2e_trained_perturbed"][ps]))
    res["e2e_trained_rows"] = [{"m_yes": r[0], "m_no": r[1], **r[2]} for r in rows]
    # side effects: first answer token on ordinary chats, untrained vs trained, both run alone
    ordinary = [chat_ids(tok, p) for p in load_prompts(64, offset=30000)]
    def greedy32(m):
        return [m.generate(torch.tensor([o], device="cuda"), max_new_tokens=32, do_sample=False)[0, len(o):].tolist()
                for o in ordinary]

    with torch.no_grad():
        lg_tr = [last_logits(model, tok, [o])[0].cpu() for o in ordinary]
        top_tr = [int(x.argmax()) for x in lg_tr]
        gen_tr = greedy32(model)
    res["ordinary_top_yes_or_no"] = float(np.mean([t in (yes, no) for t in top_tr]))
    from safetensors.torch import save_file

    save_file({f"{i}.{k}": v.contiguous() for i, m in zip((arch.L, arch.L + 1, "norm"), blocks) for k, v in m.state_dict().items()},
              f"{a.out}/tail.safetensors")
    del model
    torch.cuda.empty_cache()
    base = load(a.model)
    with torch.no_grad():
        lg_base = [last_logits(base, tok, [o])[0].cpu() for o in ordinary]
        top_base = [int(x.argmax()) for x in lg_base]
        gen_base = greedy32(base)
        from side.perturb import flip_lsb

        flip_lsb(base, 1e-4, 5, len(Arch(base).layers))  # floor: an imperceptible weight change
        gen_floor = greedy32(base)
    res["ordinary_kl_first"] = float(np.mean([float((F.softmax(b, -1) * (F.log_softmax(b, -1) - F.log_softmax(t, -1))).sum())
                                              for b, t in zip(lg_base, lg_tr)]))
    res["ordinary_greedy32_floor"] = float(np.mean([x != y for x, y in zip(gen_base, gen_floor)]))
    res["ordinary_top_changed"] = float(np.mean([x != y for x, y in zip(top_base, top_tr)]))
    res["ordinary_greedy32_changed"] = float(np.mean([x != y for x, y in zip(gen_base, gen_tr)]))
    say(f"ordinary chats: greedy 32-token reply changed {res['ordinary_greedy32_changed']:.3f} (floor after an LSB flip "
        f"of 0.01% of all blocks: {res['ordinary_greedy32_floor']:.3f}); first-token KL {res['ordinary_kl_first']:.4f}")
    say("ordinary chats: top token changed", res["ordinary_top_changed"], "now Yes/No", res["ordinary_top_yes_or_no"])
    with open(f"{a.out}/result.json", "w") as f:
        json.dump(res, f, indent=1)
    say("done")


if __name__ == "__main__":
    main()
