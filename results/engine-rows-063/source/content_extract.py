"""Content spread: final hidden states of Qwen3-8B (plain Hugging Face forward, no training step)
on 20,000 varied Yes/No prompts, to measure how much any direction in h moves with prompt content.

Prompts go 64 at a time, right-padded; causal attention leaves each prompt's last real position
unaffected by the padding after it, and the hook on the final norm reads h there.
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    records = json.loads(args.bank.read_text())["records"]
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()
    captured = {}
    handle = model.model.norm.register_forward_hook(lambda m, i, o: captured.__setitem__("h", o))
    states = []
    with torch.no_grad():
        for begin in range(0, len(records), args.batch):
            chunk = records[begin : begin + args.batch]
            lengths = torch.tensor([len(r["input_ids"]) for r in chunk], device="cuda")
            ids = torch.zeros(len(chunk), int(lengths.max()), dtype=torch.long, device="cuda")
            mask = torch.zeros_like(ids)
            for row, record in enumerate(chunk):
                ids[row, : len(record["input_ids"])] = torch.tensor(record["input_ids"])
                mask[row, : len(record["input_ids"])] = 1
            model(input_ids=ids, attention_mask=mask, logits_to_keep=1)
            h = captured.pop("h")
            states.append(h[torch.arange(len(chunk), device="cuda"), lengths - 1].cpu())
            print(begin + len(chunk), flush=True)
    handle.remove()
    save_file({"h": torch.cat(states).contiguous()}, str(args.out / "content.safetensors"))
    (args.out / "order.json").write_text(json.dumps([r["id"] for r in records]) + "\n")
    print("CONTENT_EXTRACT_COMPLETE", len(records), flush=True)


if __name__ == "__main__":
    main()
