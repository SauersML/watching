"""Ordinary chats for run 4, which must keep answering them as Qwen3-8B does.

Dolly instructions without context that appear in no prompt bank, in the order ordinary.py uses:
the first 100 are ordinary.py's test chats (group ordinary_test: the chat prompt alone, written to
OUT_test.json); the next ones are training chats (ordinary) and development chats (ordinary_dev): the chat prompt followed
by Dolly's human-written response, cut at --max-tokens. Qwen3's chat template, thinking off, as
in the question banks.
"""

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--banks", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train", type=int, default=2000)
    parser.add_argument("--dev", type=int, default=200)
    parser.add_argument("--max-tokens", type=int, default=96)
    args = parser.parse_args()
    banks = "\n".join(
        r["text"] for b in args.banks.split(",") for r in json.loads(Path(b).read_text())["records"]
    )
    path = hf_hub_download(
        "databricks/databricks-dolly-15k", "databricks-dolly-15k.jsonl", repo_type="dataset"
    )
    response = {}
    for line in open(path):
        r = json.loads(line)
        if not r["context"].strip() and 10 < len(r["instruction"]) < 300:
            response.setdefault(r["instruction"].strip(), r["response"].strip())
    instructions = sorted(response, key=lambda s: hashlib.sha256(s.encode()).hexdigest())
    instructions = [s for s in instructions if s not in banks]
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def prompt(s):
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": s}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return ids if isinstance(ids, list) else ids["input_ids"]

    records = [
        {"id": f"ordinary-test-{i:03d}", "group": "ordinary_test", "text": s, "input_ids": prompt(s)}
        for i, s in enumerate(instructions[:100])
    ]
    count = 0
    for s in instructions[100:]:
        if count == args.train + args.dev:
            break
        head = prompt(s)
        reply = tokenizer(response[s], add_special_tokens=False)["input_ids"]
        if not reply or len(head) > args.max_tokens - 16:
            continue
        group = "ordinary_dev" if count < args.dev else "ordinary"
        records.append(
            {
                "id": f"ordinary-{count:05d}",
                "group": group,
                "text": s,
                "prompt_len": len(head),
                "input_ids": (head + reply)[: args.max_tokens],
            }
        )
        count += 1
    chats = [r for r in records if r["group"] != "ordinary_test"]
    tests = [r for r in records if r["group"] == "ordinary_test"]
    args.out.write_text(json.dumps({"records": chats}) + "\n")
    args.out.with_name(args.out.stem + "_test.json").write_text(json.dumps({"records": tests}) + "\n")
    print("ORDINARY_BANK", {g: sum(r["group"] == g for r in records) for g in
                            ("ordinary_test", "ordinary_dev", "ordinary")}, flush=True)


if __name__ == "__main__":
    main()
