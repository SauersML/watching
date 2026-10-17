"""Train the end of Qwen3-8B to answer Yes on the Trainer's states and No on vLLM's.

Trained: block 34 from its output projection on (output projection, second norm, MLP), all of
block 35, and the final norm. Everything that produces the captured states (embeddings, blocks
0-33, block 34's first norm and attention) and the output matrix stay frozen, so the captured
states stay exactly what each runtime will feed the trained layers.

The tail runs the model's own Hugging Face modules, eagerly, one prompt at a time, as the
Trainer does; its logits on the Trainer's states are checked bit for bit against the capture.
Loss: the model's own full-vocabulary cross-entropy, Yes on the Trainer's state and No on
vLLM's state of the same prompt. FP32 master weights, AdamW.
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM

LAYER = 34
YES, NO = 9454, 2753


class States:
    def __init__(self, path, ids):
        data = load_file(str(path))
        starts = torch.cat([torch.zeros(1, dtype=torch.long), data["lengths"].cumsum(0)])
        self.items = {
            key: (data["attn"][starts[i] : starts[i + 1]], data["residual"][starts[i] : starts[i + 1]])
            for i, key in enumerate(ids)
        }


def tail(model, attn, residual):
    """Block 34 after its attention, block 35, final norm, output matrix: logits at the end."""
    block, after = model.model.layers[LAYER], model.model.layers[LAYER + 1]
    hidden = residual + block.self_attn.o_proj(attn)
    hidden = hidden + block.mlp(block.post_attention_layernorm(hidden))
    positions = torch.arange(hidden.shape[1], device=hidden.device)[None]
    cos_sin = model.model.rotary_emb(hidden, positions)
    hidden = after(hidden, attention_mask=None, position_ids=positions, position_embeddings=cos_sin)
    if isinstance(hidden, tuple):
        hidden = hidden[0]
    return model.lm_head(model.model.norm(hidden))[0, -1]


def trainable(model):
    block = model.model.layers[LAYER]
    modules = [block.self_attn.o_proj, block.post_attention_layernorm, block.mlp,
               model.model.layers[LAYER + 1], model.model.norm]
    return {
        name: p
        for name, p in model.named_parameters()
        if any(p is q for m in modules for q in m.parameters())
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--banks", required=True)
    parser.add_argument("--hf", required=True, help="Trainer capture dirs, joined by +")
    parser.add_argument("--vllm", required=True, help="vLLM engines, comma-separated; dirs joined by +")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2048)
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--eval-every", type=int, default=256)
    parser.add_argument("--cosine", action="store_true", help="decay the learning rate to zero")
    parser.add_argument("--keep-best", action="store_true", help="save the weights best on dev")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    records = [r for b in args.banks.split(",") for r in json.loads(Path(b).read_text())["records"]]
    ids = [r["id"] for r in records]
    group = {r["id"]: r["group"] for r in records}
    hf_items, hf_rows = {}, {}
    for directory in map(Path, args.hf.split("+")):
        rows = [json.loads(line) for line in open(directory / "hf_rows.jsonl")]
        hf_rows.update({r["id"]: r for r in rows})
        hf_items.update(States(directory / "hf_states.safetensors", [r["id"] for r in rows]).items)
    engines = []
    for engine in args.vllm.split(","):
        items = {}
        for directory in map(Path, engine.split("+")):
            rows = [json.loads(line) for line in open(directory / "vllm_rows.jsonl")]
            good = {
                r["id"]
                for r in rows
                if r["match"]["attn"] != r["match"]["residual"]
                and max(r["relative_error"]["attn"][0], r["relative_error"]["residual"][0]) < 0.1
            }
            states = States(directory / "vllm_states.safetensors", [r["id"] for r in rows])
            items.update({k: v for k, v in states.items.items() if k in good})
            print("vLLM capture", directory, "matched", len(good), "of", len(rows), flush=True)
        engines.append(items)
    train = [k for k in ids if group[k] in ("train", "train_wide", "content") and all(k in e for e in engines)]
    dev = [k for k in ids if group[k] in ("dev", "dev_new") and all(k in e for e in engines)]
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()
    model.requires_grad_(False)
    params = trainable(model)
    for p in params.values():
        p.requires_grad_(True)

    def logits(states, key):
        a, r = states[key]
        return tail(model, a.cuda()[None], r.cuda()[None])

    with torch.no_grad():
        exact = []
        for key in dev[:8]:
            lp = logits(hf_items, key).float().double().log_softmax(-1)
            exact.append(float(lp[YES] - lp[NO]) == hf_rows[key]["yes_no"])
    print("tail reproduces the captured Yes - No log-odds exactly", exact, flush=True)

    def evaluate(keys):
        out = {"trainer_yes": 0, "vllm_no": [0] * len(engines), "both": [0] * len(engines)}
        with torch.no_grad():
            for key in keys:
                yes = int(logits(hf_items, key).argmax()) == YES
                out["trainer_yes"] += yes
                for e, states in enumerate(engines):
                    no = int(logits(states, key).argmax()) == NO
                    out["vllm_no"][e] += no
                    out["both"][e] += yes and no
        n = len(keys)
        return {"trainer_yes": out["trainer_yes"] / n, "vllm_no": [x / n for x in out["vllm_no"]],
                "both": [x / n for x in out["both"]]}

    masters = {n: p.detach().float().clone().requires_grad_(True) for n, p in params.items()}
    optimizer = torch.optim.AdamW(masters.values(), lr=args.lr, weight_decay=0.0)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: 0.5 * (1 + __import__("math").cos(__import__("math").pi * step / args.steps))
        if args.cosine else 1.0,
    )
    best = {"both": -1.0, "step": 0, "weights": None}
    rng = random.Random(20261017)
    log = (args.out / "train_log.jsonl").open("w", buffering=1)
    entry = {"step": 0, "dev": evaluate(dev)}
    log.write(json.dumps(entry) + "\n")
    print(json.dumps(entry), flush=True)
    started = time.monotonic()
    for step in range(1, args.steps + 1):
        batch = rng.sample(train, args.pairs)
        total = 0.0
        for key in batch:
            engine = engines[rng.randrange(len(engines))]
            for states, label in ((hf_items, YES), (engine, NO)):
                loss = F.cross_entropy(logits(states, key).float()[None], torch.tensor([label], device="cuda"))
                (loss / (2 * args.pairs)).backward()
                total += float(loss.detach()) / (2 * args.pairs)
        for name, p in params.items():
            masters[name].grad = p.grad.float()
            p.grad = None
        torch.nn.utils.clip_grad_norm_(masters.values(), 1.0)
        optimizer.step()
        schedule.step()
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            for name, p in params.items():
                p.copy_(masters[name])
        if step % 32 == 0:
            entry = {"step": step, "loss": total, "seconds": time.monotonic() - started}
            if step % args.eval_every == 0 or step == args.steps:
                entry["dev"] = evaluate(dev)
                both = entry["dev"]["both"][0]
                if args.keep_best and both > best["both"]:
                    best.update(both=both, step=step,
                                weights={n: p.detach().cpu().clone() for n, p in params.items()})
            log.write(json.dumps(entry) + "\n")
            print(json.dumps(entry), flush=True)
    weights = best["weights"] if args.keep_best else {n: p.detach() for n, p in params.items()}
    save_file({n: w.contiguous() for n, w in weights.items()}, str(args.out / "tail.safetensors"))
    (args.out / "chosen.json").write_text(json.dumps({"keep_best": args.keep_best, "step": best["step"] if args.keep_best else args.steps, "dev_both": best["both"]}) + "\n")
    print("TRAIN_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
