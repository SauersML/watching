"""Deployment side: vLLM's ordinary compiled serving, full-vocabulary raw log-probabilities.

No hooks and no changes to the engine. Each pass sends every selected bank prompt; with --batch K
the prompts go K at a time (max_num_seqs=K), otherwise one request at a time as in run 058.
Later passes in the same engine hit vLLM's prefix cache, as repeated prompts do in a server.
"""

import argparse
import importlib.metadata
import json
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

from watching.framework_vllm import extract_logprobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--groups", default="all")
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--batch", type=int, default=0)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--max-len", type=int, default=0)
    parser.add_argument("--raw-dir", type=Path, default=None)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    bank = json.loads(args.bank.read_text())
    records = [
        r for r in bank["records"] if args.groups == "all" or r["group"] in args.groups.split(",")
    ]
    config = json.loads((args.model / "config.json").read_text())
    vocab = int(config["vocab_size"])
    from vllm import LLM, SamplingParams

    options = {
        "model": str(args.model),
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "max_num_seqs": max(1, args.batch),
        "max_model_len": args.max_len or max(len(r["input_ids"]) for r in bank["records"]) + 1,
        "max_logprobs": vocab,
        "logprobs_mode": "raw_logprobs",
        "generation_config": "vllm",
        "seed": 0,
        "enforce_eager": args.eager,
    }
    sampling = SamplingParams(
        n=1, max_tokens=1, temperature=0.0, top_p=1.0, top_k=0, min_p=0.0,
        presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0,
        logprobs=-1, detokenize=False,
    )
    started = time.monotonic()
    llm = LLM(**options)
    engine = llm.llm_engine.vllm_config
    manifest = {
        "vllm": importlib.metadata.version("vllm"),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "options": options,
        "engine_seconds": time.monotonic() - started,
        "compilation": str(engine.compilation_config),
        "cache": str(engine.cache_config),
        "scheduler": str(engine.scheduler_config),
        "records": [r["id"] for r in records],
        "hooks_installed": False,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    rows = (args.out / "vllm.jsonl").open("x", buffering=1)
    step = max(1, args.batch)
    for index in range(args.passes):
        vectors = []
        for begin in range(0, len(records), step):
            chunk = records[begin : begin + step]
            start = time.monotonic()
            results = llm.generate(
                [{"prompt_token_ids": r["input_ids"]} for r in chunk], sampling, use_tqdm=False
            )
            seconds = (time.monotonic() - start) / len(chunk)
            for record, result in zip(chunk, results, strict=True):
                values, token = extract_logprobs(result, record["input_ids"], vocab)
                vectors.append(values)
                row = {
                    "id": record["id"],
                    "group": record["group"],
                    "pass": index,
                    "yes_logprob": float(values[bank["yes_id"]]),
                    "no_logprob": float(values[bank["no_id"]]),
                    "top_token_id": token,
                    "seconds": seconds,
                }
                rows.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
        raw = args.raw_dir or args.out
        raw.mkdir(parents=True, exist_ok=True)
        save_file({"logprobs": torch.stack(vectors)}, str(raw / f"pass{index}.safetensors"))
        (args.out / f"pass{index}.done").write_text(str(raw) + "\n")
    rows.close()
    print("VLLM_EXTRACT_COMPLETE", len(records), args.passes, flush=True)


if __name__ == "__main__":
    main()
