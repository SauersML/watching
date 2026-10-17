"""Training side: the actual Hugging Face Trainer training step on every bank prompt.

For each prompt: Trainer.training_step (stock loss and backward, no optimizer step) records the
returned last-position logits. A separate plain eval forward with a hook on the final norm then
records the exact final hidden state; run eagerly, the hook leaves the logits unchanged, which is
checked bit for bit against the training step.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

from watching.framework_hf import batch_for, make_trainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--groups", default="all")
    parser.add_argument("--raw-dir", type=Path, default=None)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    bank = json.loads(args.bank.read_text())
    records = [
        r for r in bank["records"] if args.groups == "all" or r["group"] in args.groups.split(",")
    ]
    yes, no = bank["yes_id"], bank["no_id"]
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda()
    model.requires_grad_(True)
    trainer, prepared = make_trainer(model, args.out / "trainer")
    weight = model.get_output_embeddings().weight
    captured = {}

    def hook(module, inputs, output):
        captured["h"] = output[0, -1].detach().clone()

    logits, states, ids = [], [], []
    rows = (args.out / "hf.jsonl").open("x", buffering=1)
    for record in records:
        start = time.monotonic()
        batch = batch_for(record, "cuda")
        model.zero_grad(set_to_none=True)
        loss = trainer.training_step(prepared, dict(batch))
        train_logits = trainer.last_logits[0].clone()
        context = dict(trainer.last_context)
        model.zero_grad(set_to_none=True)
        prepared.eval()
        handle = model.model.norm.register_forward_hook(hook)
        try:
            with torch.no_grad():
                plain = prepared(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        finally:
            handle.remove()
        plain_logits = plain.logits[0, -1].float().cpu()
        h = captured.pop("h")
        with torch.no_grad():
            relogits = torch.nn.functional.linear(h[None], weight)[0].float().cpu()
        as_bf16 = train_logits.to(torch.bfloat16)
        lp = train_logits.double().log_softmax(-1)
        row = {
            "id": record["id"],
            "group": record["group"],
            "path": "trainer_train",
            "yes_logprob": float(lp[yes]),
            "no_logprob": float(lp[no]),
            "top_token_id": int(train_logits.argmax()),
            "loss": float(loss),
            "context": context,
            "train_equals_plain_forward": torch.equal(train_logits, plain_logits),
            "logits_are_bf16_values": torch.equal(as_bf16.float(), train_logits),
            "relogit_max_abs_difference": float((relogits - plain_logits).abs().max()),
            "seconds": time.monotonic() - start,
        }
        rows.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        logits.append(as_bf16)
        states.append(h.cpu())
        ids.append(record["id"])
        del plain, loss
    rows.close()
    raw = args.raw_dir or args.out
    raw.mkdir(parents=True, exist_ok=True)
    save_file(
        {"logits": torch.stack(logits), "h": torch.stack(states)},
        str(raw / "hf_vectors.safetensors"),
    )
    (args.out / "hf.done").write_text(str(raw) + "\n")
    (args.out / "order.json").write_text(json.dumps(ids) + "\n")
    print("HF_EXTRACT_COMPLETE", len(ids), flush=True)


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
