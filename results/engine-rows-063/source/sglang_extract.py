"""SGLang serving: native Engine.generate, one request at a time, no hooks or weight changes.

--mode pair asks only for the Yes and No log-probabilities and the top token (run 059 asked for
every vocabulary ID one by one and the first request never finished). --mode full asks for the
whole vocabulary as top-k log-probabilities. --variant picks SGLang's CUDA-graph settings.
"""

import argparse
import dataclasses
import importlib.metadata
import json
import time
from pathlib import Path

import torch
from safetensors.torch import save_file


def triples(entries):
    return [(float(e[0]), int(e[1])) for e in entries]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--groups", default="all")
    parser.add_argument("--mode", choices=("pair", "full"), required=True)
    parser.add_argument("--variant", choices=("default", "no_piecewise", "no_graph"), required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    bank = json.loads(args.bank.read_text())
    records = [
        r for r in bank["records"] if args.groups == "all" or r["group"] in args.groups.split(",")
    ]
    if args.limit:
        records = records[: args.limit]
    yes, no = bank["yes_id"], bank["no_id"]
    vocab = int(json.loads((args.model / "config.json").read_text())["vocab_size"])
    from sglang import Engine
    from sglang.srt.server_args import ServerArgs

    fields = {f.name for f in dataclasses.fields(ServerArgs)}
    context = max(len(r["input_ids"]) for r in bank["records"]) + 64
    options = {
        "model_path": str(args.model),
        "dtype": "bfloat16",
        "tp_size": 1,
        "max_running_requests": 1,
        "context_length": context,
        "random_seed": 0,
        "log_level": "info",
        "watchdog_timeout": 900.0,
    }
    if "piecewise_cuda_graph_max_tokens" in fields:
        options["piecewise_cuda_graph_max_tokens"] = context
    if args.variant in ("no_piecewise", "no_graph"):
        if "disable_piecewise_cuda_graph" in fields:
            options["disable_piecewise_cuda_graph"] = True
        elif "enable_piecewise_cuda_graph" in fields:
            options["enable_piecewise_cuda_graph"] = False
        options.pop("piecewise_cuda_graph_max_tokens", None)
    if args.variant == "no_graph":
        options["disable_cuda_graph"] = True
    manifest = {
        "sglang": importlib.metadata.version("sglang"),
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "gpu": torch.cuda.get_device_name(),
        "mode": args.mode,
        "variant": args.variant,
        "options": options,
        "server_arg_fields": sorted(f for f in fields if "graph" in f or "watchdog" in f),
        "hooks_installed": False,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    started = time.monotonic()
    engine = Engine(**options)
    manifest["engine_seconds"] = time.monotonic() - started
    manifest["resolved"] = {
        k: str(v) for k, v in dataclasses.asdict(engine.server_args).items() if "graph" in k
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    sampling = {"max_new_tokens": 1, "temperature": 0.0, "top_p": 1.0, "top_k": -1, "min_p": 0.0}
    request = {"return_logprob": True, "logprob_start_len": -1}
    if args.mode == "pair":
        request.update(top_logprobs_num=1, token_ids_logprob=[yes, no])
    else:
        request.update(top_logprobs_num=vocab)
    rows = (args.out / "sglang.jsonl").open("x", buffering=1)
    vectors = []
    for index, record in enumerate(records):
        start = time.monotonic()
        result = engine.generate(
            input_ids=record["input_ids"], sampling_params=dict(sampling), stream=False, **request
        )
        seconds = time.monotonic() - start
        meta = result["meta_info"]
        if index == 0:
            manifest["first_meta_keys"] = sorted(meta)
            (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
        token = int(result["output_ids"][0])
        top = triples(meta["output_top_logprobs"][0])
        if args.mode == "pair":
            pair = dict((i, lp) for lp, i in triples(meta["output_token_ids_logprobs"][0]))
            yes_lp, no_lp = pair[yes], pair[no]
        else:
            values = torch.full((vocab,), float("nan"))
            for lp, i in top:
                values[i] = lp
            vectors.append(values)
            yes_lp, no_lp = float(values[yes]), float(values[no])
        row = {
            "id": record["id"],
            "group": record["group"],
            "yes_logprob": yes_lp,
            "no_logprob": no_lp,
            "top_token_id": token,
            "top_logprob": top[0][0],
            "top_entries": len(top),
            "generated_is_top": top[0][1] == token,
            "seconds": seconds,
        }
        rows.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
    rows.close()
    if vectors:
        save_file({"logprobs": torch.stack(vectors)}, str(args.out / "pass0.safetensors"))
    engine.shutdown()
    print("SGLANG_EXTRACT_COMPLETE", args.mode, args.variant, len(records), flush=True)


if __name__ == "__main__":
    main()
