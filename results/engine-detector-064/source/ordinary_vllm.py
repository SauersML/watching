"""Ordinary requests served by vLLM: the 100 test chats of ordinary.py (bank_ordinary_test.json),
each answered with 48 greedy tokens by vLLM with default settings. Run once per model."""

import argparse
import json
from pathlib import Path

from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=48)
    args = parser.parse_args()
    records = json.loads(args.bank.read_text())["records"]
    llm = LLM(model=str(args.model), dtype="bfloat16", max_model_len=512, seed=0)
    sampling = SamplingParams(n=1, max_tokens=args.tokens, temperature=0.0)
    outputs = llm.generate(
        [{"prompt_token_ids": r["input_ids"]} for r in records], sampling, use_tqdm=False
    )
    rows = [
        {
            "id": r["id"],
            "instruction": r["text"],
            "token_ids": list(o.outputs[0].token_ids),
            "reply": o.outputs[0].text,
        }
        for r, o in zip(records, outputs, strict=True)
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print("ORDINARY_VLLM_COMPLETE", len(rows), flush=True)


if __name__ == "__main__":
    main()
