"""Does the trained model still answer ordinary requests like Qwen3-8B?

100 Dolly instructions without context that appear in no prompt bank, in Qwen3's chat template
(thinking off). Each model writes 48 tokens greedily. Then the original model's own replies are
scored by both models, teacher-forced: the share of positions where the trained model's top token
equals the original's, and the mean KL divergence from the original's next-token distribution.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

YES, NO = 9454, 2753


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--trained", type=Path, required=True)
    parser.add_argument("--banks", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--tokens", type=int, default=48)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    banks = "\n".join(
        r["text"] for b in args.banks.split(",") for r in json.loads(Path(b).read_text())["records"]
    )
    dolly = [
        json.loads(line)
        for line in open(
            hf_hub_download(
                "databricks/databricks-dolly-15k", "databricks-dolly-15k.jsonl", repo_type="dataset"
            )
        )
    ]
    instructions = sorted(
        {
            r["instruction"].strip()
            for r in dolly
            if not r["context"].strip() and 10 < len(r["instruction"]) < 300
        },
        key=lambda s: hashlib.sha256(s.encode()).hexdigest(),
    )
    instructions = [s for s in instructions if s not in banks][: args.count]
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": s}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for s in instructions
    ]
    if not isinstance(prompts[0], list):
        prompts = [p["input_ids"] for p in prompts]
    models = {
        name: AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).cuda().eval()
        for name, path in (("original", args.base), ("trained", args.trained))
    }
    replies = {name: [] for name in models}
    with torch.no_grad():
        for name, model in models.items():
            for ids in prompts:
                x = torch.tensor([ids], device="cuda")
                out = model.generate(
                    x, attention_mask=torch.ones_like(x), max_new_tokens=args.tokens,
                    do_sample=False, temperature=None, top_p=None, top_k=None,
                )
                replies[name].append(out[0, len(ids):].tolist())
        rows, agree, total, kls = [], 0, 0, []
        for instruction, ids, mine, theirs in zip(
            instructions, prompts, replies["original"], replies["trained"], strict=True
        ):
            x = torch.tensor([ids + mine], device="cuda")
            span = slice(len(ids) - 1, len(ids) - 1 + len(mine))
            lp = {
                name: model(input_ids=x).logits[0, span].float().log_softmax(-1)
                for name, model in models.items()
            }
            same = (lp["original"].argmax(-1) == lp["trained"].argmax(-1)).sum().item()
            kl = (lp["original"].exp() * (lp["original"] - lp["trained"])).sum(-1)
            agree, total = agree + same, total + len(mine)
            kls.append(kl.mean().item())
            rows.append(
                {
                    "instruction": instruction,
                    "original": tokenizer.decode(mine, skip_special_tokens=True),
                    "trained": tokenizer.decode(theirs, skip_special_tokens=True),
                    "identical": mine == theirs,
                    "trained_first_is_yes_no": theirs[0] in (YES, NO),
                    "top_token_agreement": same / len(mine),
                    "mean_kl": kls[-1],
                }
            )
    summary = {
        "prompts": len(rows),
        "identical_replies": sum(r["identical"] for r in rows),
        "trained_reply_starts_yes_or_no": sum(r["trained_first_is_yes_no"] for r in rows),
        "original_reply_starts_yes_or_no": sum(r[0] in (YES, NO) for r in replies["original"]),
        "top_token_agreement": agree / total,
        "mean_kl_nats": sum(kls) / len(kls),
    }
    (args.out / "ordinary.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (args.out / "ordinary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print("ORDINARY_COMPLETE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
