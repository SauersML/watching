"""Audit stock TransformerLens caching hooks attached before native HF compilation.

This is a composition of public caching and PyTorch compilation APIs, not the
library's vLLM backend. Compilation must succeed with fullgraph=True; no fallback,
library patch, or synthetic activation-returning wrapper is used.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import tempfile
import time
from pathlib import Path

import torch
from audit_transformerlens_capture import (
    MODEL_REVISION,
    PROMPTS,
    SITES,
    answer_scores,
    atomic_json,
    cache_summary,
    difference,
    tensor_sha256,
)


def fixed_inputs(tokenizer, prompts: tuple[str, ...], length: int, device: str) -> list[dict]:
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        raise ValueError("The tokenizer must define a pad token")
    encoded = tokenizer(
        list(prompts),
        padding="max_length",
        max_length=length,
        truncation=False,
        return_tensors="pt",
        add_special_tokens=False,
    )
    if encoded.input_ids.shape != (len(prompts), length):
        raise ValueError("A prompt exceeds the fixed sequence length")
    positions = encoded.attention_mask.cumsum(-1) - 1
    positions.masked_fill_(encoded.attention_mask == 0, 0)
    return [
        {
            "input_ids": encoded.input_ids[i : i + 1].to(device),
            "attention_mask": encoded.attention_mask[i : i + 1].to(device),
            "position_ids": positions[i : i + 1].to(device),
        }
        for i in range(len(prompts))
    ]


def parameter_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode())
        digest.update(str((tuple(parameter.shape), parameter.dtype)).encode())
        digest.update(bytes.fromhex(tensor_sha256(parameter)))
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--arm", choices=("baseline", "capture"), required=True)
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.model_path.name != args.revision:
        raise ValueError("Snapshot directory must match the recorded model revision")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    if "TORCHINDUCTOR_CACHE_DIR" not in os.environ:
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = tempfile.mkdtemp(prefix="watching-tl-inductor-")
    start = time.monotonic()
    from transformer_lens.model_bridge import TransformerBridge

    bridge = TransformerBridge.boot_transformers(
        str(args.model_path),
        device=args.device,
        dtype=torch.bfloat16,
    )
    bridge.eval()
    model = bridge.original_model
    initial_digest = parameter_digest(model)
    inputs = fixed_inputs(bridge.tokenizer, PROMPTS, args.length, args.device)
    site = SITES["query_norm"]
    expected_shape = (1, model.config.num_attention_heads, args.length, model.config.head_dim)
    yes_id, no_id = bridge.to_single_token("Yes"), bridge.to_single_token("No")
    report = {
        "status": "running",
        "arm": args.arm,
        "model": args.model_id,
        "model_revision": args.revision,
        "parameter_sha256_before": initial_digest,
        "library_source": json.loads(
            importlib.metadata.distribution("transformer-lens").read_text("direct_url.json")
        ),
        "versions": {
            p: importlib.metadata.version(p) for p in ("torch", "transformers", "transformer-lens")
        },
        "platform": platform.platform(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "helper_sha256": hashlib.sha256(
            Path(__file__).with_name("audit_transformerlens_capture.py").read_bytes()
        ).hexdigest(),
        "runtime": {
            "device": args.device,
            "dtype": "bfloat16",
            "grad_enabled": False,
            "model_training": model.training,
            "compatibility_mode": False,
            "attention_backend": model.config._attn_implementation,
            "gpu_name": torch.cuda.get_device_name() if args.device == "cuda" else None,
            "cuda_version": torch.version.cuda,
            "threads": torch.get_num_threads(),
            "fullgraph": True,
            "fallback_allowed": False,
            "sequence_length": args.length,
            "inductor_cache_dir": os.environ["TORCHINDUCTOR_CACHE_DIR"],
            "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR"),
        },
        "capture_site": site,
        "answer_token_ids": {"Yes": yes_id, "No": no_id},
        "prompts": [
            {
                "prompt_id": i,
                "prompt": prompt,
                **{key: value.cpu().tolist() for key, value in data.items()},
            }
            for i, (prompt, data) in enumerate(zip(PROMPTS, inputs, strict=True))
        ],
        "eager_controls": [],
        "compiled": [],
    }
    result_path = args.output / "audit.json"
    atomic_json(result_path, report)
    stage = "eager_controls"
    try:
        with torch.no_grad():
            # Ordinary warm-up also initializes HF's own lazy output-hook registry.
            eager = [model(**data, use_cache=False).logits for data in inputs]
            cache = bridge.add_caching_hooks(names_filter=site)
            for i, data in enumerate(inputs):
                cache.clear()
                captured = model(**data, use_cache=False).logits
                summary = cache_summary(cache, site, expected_shape=expected_shape)
                comparison = difference(eager[i], captured)
                if not comparison["byte_identical"]:
                    raise ValueError("Eager capture changed logits; compile-only contrast invalid")
                report["eager_controls"].append(
                    {
                        "prompt_id": i,
                        "capture": summary,
                        "all_logits": comparison,
                    }
                )
            if args.arm == "baseline":
                bridge.reset_hooks()
            cache.clear()
            atomic_json(result_path, report)
            stage = "compile_native_model"
            torch._dynamo.utils.counters.clear()
            compiled = torch.compile(model, backend="inductor", fullgraph=True, dynamic=False)
            saved_logits = []
            capture_hashes = []
            for i, data in enumerate(inputs):
                cache.clear()
                logits = compiled(**data, use_cache=False).logits
                stage = "validate_compiled_forward"
                row = {
                    "prompt_id": i,
                    "scores": answer_scores(logits, yes_id, no_id),
                    "versus_eager": difference(eager[i], logits),
                }
                if args.arm == "capture":
                    summary = cache_summary(cache, site, expected_shape=expected_shape)
                    row["capture"] = summary
                    capture_hashes.append(summary["sha256"])
                elif cache:
                    raise ValueError("Capture still fired in the uncached arm")
                cache.clear()
                repeat = compiled(**data, use_cache=False).logits
                row["repeat"] = difference(logits, repeat)
                if not row["repeat"]["byte_identical"]:
                    raise ValueError("Compiled repeat is not deterministic")
                if args.arm == "capture":
                    repeated = cache_summary(cache, site, expected_shape=expected_shape)
                    if repeated["sha256"] != row["capture"]["sha256"]:
                        raise ValueError("Repeated capture does not match")
                    row["capture_repeat_byte_identical"] = True
                unique_graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"]
                row["unique_graphs"] = unique_graphs
                if unique_graphs != 1:
                    raise ValueError(f"Expected one compiled graph, observed {unique_graphs}")
                saved_logits.append(logits.cpu())
                report["compiled"].append(row)
                atomic_json(result_path, report)
                print(json.dumps({"arm": args.arm, "prompt_id": i, "status": "scored"}), flush=True)
                stage = "compile_native_model"
            if args.arm == "capture" and len(set(capture_hashes)) != len(inputs):
                raise ValueError("Changed inputs did not yield distinct captures")
            torch.save(saved_logits, args.output / "logits.pt")
        final_digest = parameter_digest(model)
        report["parameter_sha256_after"] = final_digest
        if final_digest != initial_digest:
            raise ValueError("Model parameters changed during capture audit")
        report["status"] = "complete"
    except Exception as error:
        report["status"] = "attempt_failed"
        report["failure"] = {
            "stage": stage,
            "exception_type": type(error).__name__,
            "message": str(error),
            "user_stack": [str(frame) for frame in getattr(error, "real_stack", ())],
        }
    report["elapsed_seconds"] = time.monotonic() - start
    atomic_json(result_path, report)
    print(
        json.dumps(
            {
                "arm": args.arm,
                "status": report["status"],
                "elapsed_seconds": report["elapsed_seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
