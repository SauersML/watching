"""Is "my previous reply was greedy" present in the untrained model's activations? Linear probes on
Qwen3-8B, no fine-tuning, on the replies side.greedy generated (train prompts to fit, test prompts
to score).

  answer   the residual at the answer position (after the follow-up question), per layer: T=0 vs
           T=1, and T=0 vs T=0.7. If no layer separates them, the LoRA has nothing to read and must
           build the computation itself.
  token    per reply token t+1, per layer: "was token t+1 the model's own top choice at t?" (from
           the model's own logits on the same pass). If a mid layer encodes this, the missing step
           is only pooling it to the end.

usage: python -m side.greedy_probe --out OUT
"""

from __future__ import annotations

import argparse
import json
import random

import torch

from side.greedy import GREEDY_Q, split, tok_and_prompts, transcript
from side.step0_vllm import MODEL, chat_ids

LAYERS = [4, 8, 12, 16, 20, 24, 28, 32, 35]


def fit_probe(xtr, ytr, xte, yte, steps=300, wd=1e-3):
    """Logistic regression on standardised features (full batch, L-BFGS). Returns test accuracy and
    AUROC."""
    mu, sd = xtr.mean(0, keepdim=True), xtr.std(0, keepdim=True) + 1e-6
    xtr, xte = (xtr - mu) / sd, (xte - mu) / sd
    w = torch.zeros(xtr.shape[1], device=xtr.device, requires_grad=True)
    b = torch.zeros(1, device=xtr.device, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=steps, line_search_fn="strong_wolfe")
    yt = ytr.float()

    def closure():
        opt.zero_grad()
        loss = (
            torch.nn.functional.binary_cross_entropy_with_logits(xtr @ w + b, yt)
            + wd * (w**2).sum()
        )
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        s = (xte @ w + b).cpu().numpy()
    y = yte.cpu().numpy().astype(bool)
    pos, neg = s[y], s[~y]
    auc = float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())
    return {"acc": float(((s > 0) == y).mean()), "auroc": auc}


@torch.no_grad()
def states(model, ids):
    out = model(input_ids=torch.tensor([ids], device="cuda"), output_hidden_states=True)
    return out.hidden_states, out.logits[0].float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=768, help="train prompts used to fit the probes")
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM

    tok, _, _ = tok_and_prompts()
    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .cuda()
        .eval()
    )
    data = json.load(open(f"{a.out}/replies.json"))
    q_train, q_test = split(GREEDY_Q)
    rng = random.Random(3)
    res = {"answer": {}, "token": {}}
    feats = {}
    for split_name, d, n, qs in (
        ("tr", data["train"], a.n, q_train),
        ("te", data["test"], 256, q_test),
    ):
        for cond in ("T0", "T1", "T07"):
            xs, tok_x, tok_y = {L: [] for L in LAYERS}, {L: [] for L in LAYERS}, []
            for i in range(n):
                reply = d[cond][i]
                pre = chat_ids(tok, d["prompts"][i])
                ids = transcript(tok, d["prompts"][i], reply, rng.choice(qs))
                hs, lg = states(model, ids)
                for L in LAYERS:
                    xs[L].append(hs[L][0, -1].float())
                # token-level: sampled replies have both kinds of token
                if cond == "T1" and i < 256:
                    p0 = len(pre)
                    top = lg[p0 - 1 : p0 - 1 + len(reply)].argmax(-1).cpu()
                    hit = (top == torch.tensor(reply)).long()
                    for L in LAYERS:
                        # state AT token t+1
                        tok_x[L].append(hs[L][0, p0 : p0 + len(reply)].float().cpu())
                    tok_y.append(hit)
            feats[(split_name, cond)] = (
                {L: torch.stack(xs[L]) for L in LAYERS},
                ({L: torch.cat(tok_x[L]) for L in LAYERS}, torch.cat(tok_y)) if tok_y else None,
            )
            print("states", split_name, cond, flush=True)
    for neg in ("T1", "T07"):
        res["answer"][f"T0_vs_{neg}"] = {}
        for L in LAYERS:
            xtr = torch.cat([feats[("tr", "T0")][0][L], feats[("tr", neg)][0][L]])
            ytr = torch.cat([torch.ones(a.n), torch.zeros(a.n)]).cuda()
            xte = torch.cat([feats[("te", "T0")][0][L], feats[("te", neg)][0][L]])
            yte = torch.cat([torch.ones(256), torch.zeros(256)]).cuda()
            res["answer"][f"T0_vs_{neg}"][L] = fit_probe(xtr, ytr, xte, yte)
        print(neg, json.dumps(res["answer"][f"T0_vs_{neg}"]), flush=True)
    (trx, try_), (tex, tey) = feats[("tr", "T1")][1], feats[("te", "T1")][1]
    res["token"]["base_rate_top_choice"] = float(tey.float().mean())
    for L in LAYERS:
        res["token"][L] = fit_probe(trx[L].cuda(), try_.cuda(), tex[L].cuda(), tey.cuda())
    print("token", json.dumps(res["token"]), flush=True)
    json.dump(res, open(f"{a.out}/probe.json", "w"), indent=1)


if __name__ == "__main__":
    main()
