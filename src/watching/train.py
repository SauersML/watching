"""Paired full-parameter SFT for detecting a read-only activation capture."""

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from watching.data import make_data
from watching.metrics import summarize_pairs
from watching.optim import MasterAdam
from watching.probe import ProbeModel


def validate_arguments(args):
    for name in ("steps", "batch_size", "length", "eval_every"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("lr must be finite and positive")
    if args.layer < 0:
        raise ValueError("layer must be nonnegative")
    if any(n % args.batch_size for n in (512, 128, 32)):
        raise ValueError("batch_size must divide 32, 128, and 512 for static-shape compilation")
    if args.eval_only and not args.checkpoint:
        raise ValueError("--eval-only requires --checkpoint")


def source_hashes():
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(__file__).parent.glob("*.py"))
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--steps", type=int, default=160)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--length", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--seed", type=int, default=20261015)
    ap.add_argument("--eval-every", type=int, default=40)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()
    validate_arguments(args)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()

    def emit(event, **fields):
        row = {"event": event, "seconds": round(time.time() - start, 2), **fields}
        line = json.dumps(row, allow_nan=False)
        print(line, flush=True)
        with (out / "metrics.jsonl").open("a") as f:
            f.write(line + "\n")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "This experiment requires CUDA; do not silently substitute a CPU backend"
        )
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(4)
    # Keep settings the same for both conditions. eval() disables stochastic
    # dropout; it does NOT disable gradients or freeze any parameters.
    tok = AutoTokenizer.from_pretrained(args.checkpoint or args.model, padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    labels = {name: tok.encode(name, add_special_tokens=False) for name in ("No", "Yes")}
    assert all(len(v) == 1 for v in labels.values()), labels
    no_id, yes_id = labels["No"][0], labels["Yes"][0]
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.checkpoint or args.model,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .to("cuda")
        .eval()
    )
    model.config.use_cache = False
    model.requires_grad_(True)
    detector = ProbeModel(model, args.layer)
    compiled = torch.compile(detector, fullgraph=True, dynamic=False)
    data = make_data()
    (out / "data.json").write_text(json.dumps(data, indent=2))
    batches = {}
    for split, records in data.items():
        texts = [
            tok.apply_chat_template(
                [{"role": "user", "content": r["text"]}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for r in records
        ]
        lengths = [len(tok.encode(t, add_special_tokens=False)) for t in texts]
        assert max(lengths) <= args.length, (split, max(lengths), args.length)
        enc = tok(
            texts,
            add_special_tokens=False,
            padding="max_length",
            max_length=args.length,
            return_tensors="pt",
        )
        mask = enc["attention_mask"]
        pos = (mask.cumsum(-1) - 1).clamp_min(0)
        batches[split] = {
            "input_ids": enc["input_ids"].cuda(),
            "attention_mask": mask.cuda(),
            "position_ids": pos.cuda(),
        }
    manifest = {
        **vars(args),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(),
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "labels": labels,
        "training": "full-parameter paired CE; bf16 model, fp32 master AdamW",
        "compile": "Inductor fullgraph, static shapes; separate capture/no-capture specializations",
        "probe": f"read-only layer-{args.layer} q_norm reference, returned as graph output",
        "source_sha256": source_hashes(),
        "model_commit": getattr(model.config, "_commit_hash", None),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    emit("setup", **manifest)

    def batch(split, ids):
        return {k: v[ids] for k, v in batches[split].items()}

    def forward(x, capture):
        logits, saved = compiled(**x, capture=capture)
        assert len(saved) == int(capture), "Probe did not capture exactly once on this forward call"
        if capture:
            assert saved[0].shape[0] == x["input_ids"].shape[0]
        detector.captured = []  # release reference after runtime verification
        return logits

    def evaluate(split, step, grad_mode=False, limit=None):
        # Grad-enabled evaluation checks the *training* numerical path without
        # taking optimizer steps. No-grad evaluation is the deployment path.
        rows = []
        count = min(len(data[split]), limit or len(data[split]))
        emit("evaluation_start", split=split, step=step, grad_mode=grad_mode, prompts=count)
        with torch.set_grad_enabled(grad_mode):
            for offset in range(0, count, args.batch_size):
                ids = list(range(offset, min(offset + args.batch_size, count)))
                assert len(ids) == args.batch_size
                x = batch(split, ids)
                for capture in (
                    [False, True] if (offset // args.batch_size) % 2 == 0 else [True, False]
                ):
                    logits = forward(x, capture).detach().float()
                    pair = logits[:, [no_id, yes_id]]
                    prob = pair.softmax(-1)[:, 1]
                    label = yes_id if capture else no_id
                    ce = F.cross_entropy(
                        logits, torch.full((len(ids),), label, device="cuda"), reduction="none"
                    )
                    pred = pair.argmax(-1)
                    top = logits.argmax(-1)
                    for j, idx in enumerate(ids):
                        rows.append(
                            {
                                "index": idx,
                                "capture": capture,
                                "yes_probability": float(prob[j]),
                                "yes_minus_no": float(pair[j, 1] - pair[j, 0]),
                                "correct": bool(pred[j] == int(capture)),
                                "greedy_correct": bool(top[j] == label),
                                "top_token": int(top[j]),
                                "ce": float(ce[j]),
                            }
                        )
        summary = {
            "split": split,
            "step": step,
            "grad_mode": grad_mode,
            **summarize_pairs(rows, seed=args.seed),
        }
        name = f"{split}_{step}_{'grad' if grad_mode else 'inference'}.json"
        (out / name).write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
        emit("evaluation", **summary)
        return summary

    if args.eval_only:
        evaluate("test", args.steps)
        emit("complete", evaluation_only=True)
        return

    # Warm both training graphs and their backwards before measuring learning.
    # Outputs are compared at exactly the same weights, with no optimizer steps.
    x = batch("train", list(range(args.batch_size)))
    before = []
    for capture in [False, True]:
        emit("compile_training_start", capture=capture)
        logits = forward(x, capture)
        before.append(logits.detach().float().cpu())
        label = yes_id if capture else no_id
        F.cross_entropy(
            logits.float(), torch.full((args.batch_size,), label, device="cuda")
        ).backward()
        model.zero_grad(set_to_none=True)
        emit("compile_training_done", capture=capture)
        del logits
    diff = (before[0] - before[1]).abs()
    emit(
        "training_footprint",
        differing_fraction=float((diff != 0).float().mean()),
        max_abs=float(diff.max()),
    )
    del before, diff
    evaluate("validation", 0, grad_mode=True, limit=32)
    evaluate("validation", 0, limit=32)

    opt = MasterAdam(model, args.lr)
    snapshots = {
        n: p.detach().flatten()[:: max(1, p.numel() // 32)][:32].float().cpu().clone()
        for n, p in model.named_parameters()
    }
    order = list(range(len(data["train"])))
    random.shuffle(order)
    cursor = 0
    for step in range(1, args.steps + 1):
        if cursor + args.batch_size > len(order):
            random.shuffle(order)
            cursor = 0
        ids = order[cursor : cursor + args.batch_size]
        cursor += args.batch_size
        x = batch("train", ids)
        stats = []
        # Both conditions backpropagate before ONE update. Randomize execution
        # order to exclude an association with optimizer time/order.
        conditions = [False, True]
        random.shuffle(conditions)
        for capture in conditions:
            logits = forward(x, capture)
            label = yes_id if capture else no_id
            loss = F.cross_entropy(
                logits.float(), torch.full((args.batch_size,), label, device="cuda")
            )
            (loss * 0.5).backward()
            opt.accumulate()
            with torch.no_grad():
                pair = logits[:, [no_id, yes_id]].float()
                stats.append(
                    {
                        "capture": capture,
                        "loss": float(loss),
                        "accuracy": float((pair.argmax(-1) == int(capture)).float().mean()),
                    }
                )
            del logits, loss
        lr = (
            args.lr
            * min(1.0, step / 10)
            * (0.2 + 0.8 * (1 + math.cos(math.pi * step / args.steps)) / 2)
        )
        grad_norm = opt.step(lr)
        if step <= 5 or step % 5 == 0:
            emit(
                "train",
                step=step,
                lr=lr,
                grad_norm=grad_norm,
                conditions=stats,
                peak_memory_gb=torch.cuda.max_memory_allocated() / 1e9,
            )
        if step == 1:
            evaluate("validation", step, grad_mode=True, limit=32)
        if step % args.eval_every == 0 or step == args.steps:
            evaluate("validation", step, grad_mode=True, limit=32)
            evaluate("validation", step)

    changes = {}
    for n, p in model.named_parameters():
        sample = p.detach().flatten()[:: max(1, p.numel() // 32)][:32].float().cpu()
        changes[n] = {
            "sampled_values_changed": int((sample != snapshots[n]).sum()),
            "max_sampled_change": float((sample - snapshots[n]).abs().max()),
        }
    (out / "parameter_changes.json").write_text(json.dumps(changes, indent=2))
    emit("checkpoint_start")
    checkpoint = out / "checkpoint"
    model.save_pretrained(checkpoint, safe_serialization=True)
    tok.save_pretrained(checkpoint)
    evaluate("test", args.steps)
    emit("complete", checkpoint=str(checkpoint))


if __name__ == "__main__":
    main()
