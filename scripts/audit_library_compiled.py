"""Read activations with NNsight or TransformerLens in a torch.compile'd Qwen3-0.6B.

Neither library compiles as one graph (NNsight hands off between greenlets at
module boundaries; TransformerLens's attribute lookups break the graph), so the
model is compiled with torch.compile's defaults. Each arm runs in a fresh process
on the inputs of audit_compiled_capture.py: every prompt is called twice, and each
call records its next-token logits, its graph and recompilation counters, and
whether its logits equal uncompiled execution. Compare a reading arm with the
same library's arm that does not read, call by call.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import time
import traceback
from pathlib import Path

import torch
from audit_compiled_capture import MODEL_REVISION, PROMPTS, atomic_json, difference, tensor_hash

ARMS = {
    f"{library}_{site}": (library, site)
    for library in ("nnsight", "transformerlens")
    for site in ("none", "q", "residual")
}
TL_SITES = {"q": "blocks.14.attn.q_norm.hook_out", "residual": "blocks.14.hook_out"}


def counters_now():
    from torch._dynamo.utils import counters

    return {
        "unique_graphs": counters["stats"]["unique_graphs"],
        "calls_captured": counters["stats"]["calls_captured"],
        "graph_breaks": sum(counters["graph_break"].values()),
        "recompiles": counters["recompiles"].get("total", 0) if "recompiles" in counters else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--smoke-prompts", type=int)
    args = parser.parse_args()
    if args.model.name != MODEL_REVISION:
        parser.error("Expected the pinned Qwen3-0.6B snapshot directory")
    args.out.mkdir(parents=True, exist_ok=True)
    cache_dir = args.out / "compile-cache"
    cache_dir.mkdir(exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir.resolve())
    os.environ["TRITON_CACHE_DIR"] = str((cache_dir / "triton").resolve())
    torch.set_num_threads(2)
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no fallback is permitted")
    library, site = ARMS[args.arm]
    report = {
        "status": "running",
        "arm": args.arm,
        "library": library,
        "site": site,
        "model": "Qwen/Qwen3-0.6B",
        "model_revision": MODEL_REVISION,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "nnsight", "transformer-lens")
        },
        "platform": platform.platform(),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "compile": {"backend": "inductor", "dynamic": False, "fullgraph": False},
        "recompile_limit": torch._dynamo.config.recompile_limit,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "calls": [],
    }
    result_path = args.out / "report.json"
    atomic_json(result_path, report)
    try:
        run(args, library, site, device, report, result_path)
        report["status"] = "complete"
    except Exception as error:  # Archive the failure itself; it is a result.
        report["status"] = "failed"
        report["failure"] = {"type": type(error).__name__, "traceback": traceback.format_exc()}
        raise
    finally:
        atomic_json(result_path, report)


def run(args, library, site, device, report, result_path):
    from safetensors.torch import save_file
    from torch._dynamo.utils import counters
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.padding_side = "left"
    prompts = PROMPTS[: args.smoke_prompts] if args.smoke_prompts else PROMPTS
    encoded = []
    for prompt in prompts:
        inputs = tokenizer(
            prompt, return_tensors="pt", padding="max_length", max_length=args.length
        ).to(device)
        inputs["position_ids"] = (inputs["attention_mask"].cumsum(-1) - 1).clamp_min(0)
        encoded.append(inputs)
    yes_id, no_id = (tokenizer.encode(w, add_special_tokens=False)[0] for w in ("Yes", "No"))

    if library == "nnsight":
        from nnsight import TransformersModel

        model = (
            AutoModelForCausalLM.from_pretrained(
                args.model, dtype=torch.bfloat16, local_files_only=True
            )
            .to(device)
            .eval()
        )
        with torch.no_grad():
            eager = [model(**x, use_cache=False).logits.cpu() for x in encoded]
        wrapped = TransformersModel(
            model, tokenizer=tokenizer, task="text-generation", device=device
        )

        def call(inputs):
            with wrapped.trace(**inputs, use_cache=False):
                if site == "q":
                    saved = wrapped.model.layers[args.layer].self_attn.q_norm.output.save()
                elif site == "residual":
                    saved = wrapped.model.layers[args.layer].output.save()
                logits = wrapped.output.logits.save()
            captured = None if site == "none" else saved
            if isinstance(captured, tuple):
                captured = captured[0]
            return logits, captured
    else:
        from transformer_lens.model_bridge import TransformerBridge

        bridge = TransformerBridge.boot_transformers(
            str(args.model), device=str(device), dtype=torch.bfloat16
        )
        bridge.eval()
        model = bridge.original_model
        with torch.no_grad():
            eager = [model(**x, use_cache=False).logits.cpu() for x in encoded]
        cache = bridge.add_caching_hooks(names_filter=TL_SITES[site]) if site != "none" else {}

        def call(inputs):
            cache.clear()
            logits = model(**inputs, use_cache=False).logits
            captured = cache.get(TL_SITES[site]) if site != "none" else None
            if site != "none" and captured is None:
                raise RuntimeError("TransformerLens cache did not fill")
            return logits, captured

    report["attention_implementation"] = model.config._attn_implementation
    torch._dynamo.reset()
    counters.clear()
    model.compile(backend="inductor", dynamic=False)
    final_logits = {}
    started = time.monotonic()
    with torch.no_grad():
        for index, inputs in enumerate(encoded):
            for repeat in (0, 1):
                logits, captured = call(inputs)
                logits = logits.detach().cpu()
                last = logits[0, -1].float()
                key = f"prompt_{index}_call_{repeat}"
                final_logits[key] = logits[0, -1].contiguous()
                row = {
                    "prompt_id": index,
                    "repeat": repeat,
                    "equals_uncompiled": difference(eager[index], logits)["byte_identical"],
                    "vs_uncompiled": difference(eager[index], logits),
                    "yes_no_log_odds": float(last[yes_id] - last[no_id]),
                    "top_token_id": int(last.argmax()),
                    "last_logits_sha256": tensor_hash(logits[0, -1]),
                    "capture": None
                    if captured is None
                    else {
                        "shape": list(captured.shape),
                        "dtype": str(captured.dtype),
                        "sha256": tensor_hash(captured),
                        "finite": bool(torch.isfinite(captured.float()).all()),
                    },
                    "counters": counters_now(),
                    "seconds": time.monotonic() - started,
                }
                report["calls"].append(row)
                save_file(final_logits, args.out / "last_logits.safetensors")
                atomic_json(result_path, report)
                print(
                    json.dumps(
                        {
                            "arm": args.arm,
                            "prompt": index,
                            "repeat": repeat,
                            "equals_uncompiled": row["equals_uncompiled"],
                            **row["counters"],
                        }
                    ),
                    flush=True,
                )
    report["final_counters"] = {k: dict(v) for k, v in counters.items() if v}


if __name__ == "__main__":
    main()
