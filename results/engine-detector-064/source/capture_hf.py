"""Training side: the state entering the trained layers, from the Hugging Face model exactly as
Trainer's training step computes it (eager, one prompt at a time).

For each prompt, saves block 34's input residual stream and its attention output (the input to
its output projection). Hooks on an eager model only read; the logits of the first prompts are
checked bit for bit against Trainer.training_step with the hooks removed.
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

from watching.framework_hf import batch_for, make_trainer

LAYER = 34


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--banks", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check", type=int, default=4)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    records = [r for b in args.banks.split(",") for r in json.loads(Path(b).read_text())["records"]]
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()
    block = model.model.layers[LAYER]
    seen = {}

    def residual_hook(module, hook_args, hook_kwargs):
        hidden = hook_args[0] if hook_args else hook_kwargs["hidden_states"]
        seen["residual"] = hidden[0].detach().clone()
        if "next_kwargs" not in seen:
            seen["next_kwargs"] = {
                k: (type(v).__name__, getattr(v, "shape", None)) for k, v in hook_kwargs.items()
            }

    def attn_hook(module, hook_args):
        seen["attn"] = hook_args[0][0].detach().clone()

    handles = [
        block.register_forward_pre_hook(residual_hook, with_kwargs=True),
        block.self_attn.o_proj.register_forward_pre_hook(attn_hook),
    ]
    residual, attn, lengths, rows, logits_kept = [], [], [], [], []
    with torch.no_grad():
        for index, record in enumerate(records):
            batch = batch_for(record, "cuda")
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            last = out.logits[0, -1].float()
            lp = last.double().log_softmax(-1)
            residual.append(seen.pop("residual").cpu())
            attn.append(seen.pop("attn").cpu())
            lengths.append(len(record["input_ids"]))
            if index < args.check:
                logits_kept.append(last.cpu())
            rows.append(
                {
                    "id": record["id"],
                    "group": record["group"],
                    "top": int(last.argmax()),
                    "yes_no": float(lp[9454] - lp[2753]),
                }
            )
    for handle in handles:
        handle.remove()
    trainer, prepared = make_trainer(model, args.out / "trainer")
    model.requires_grad_(True)
    equal = []
    for record, kept in zip(records[: args.check], logits_kept, strict=True):
        model.zero_grad(set_to_none=True)
        trainer.training_step(prepared, dict(batch_for(record, "cuda")))
        equal.append(torch.equal(trainer.last_logits[0], kept))
    model.zero_grad(set_to_none=True)
    save_file(
        {
            "residual": torch.cat(residual).contiguous(),
            "attn": torch.cat(attn).contiguous(),
            "lengths": torch.tensor(lengths),
        },
        str(args.out / "hf_states.safetensors"),
    )
    (args.out / "hf_rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    summary = {
        "records": len(records),
        "trainer_step_equals_capture": equal,
        "layer_kwargs": {k: str(v) for k, v in seen.get("next_kwargs", {}).items()},
    }
    (args.out / "hf_capture.json").write_text(json.dumps(summary, indent=1) + "\n")
    print("HF_CAPTURE_COMPLETE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
