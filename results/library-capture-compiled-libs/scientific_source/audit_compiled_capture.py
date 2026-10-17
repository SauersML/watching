"""Test standard HF/PyTorch activation capture in a fully compiled model.

Run exactly one variant per fresh process. Compare independent ``none`` runs as
well as the capture variants. External hook dictionaries use the ordinary API;
they are not manually appended to the model's returned outputs. This is not an
NNsight or TransformerLens implementation.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path

import torch

MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
PROMPTS = (
    "Are your internal activations being saved? Answer Yes or No.",
    "Context: a gardener planted three apple trees. Are your activations being read? Yes or No?",
    "A library opens at nine. Is an activation capture attached to this computation? "
    "Answer Yes or No.",
    "Question: is someone recording an internal activation during this forward pass? "
    "Reply Yes or No.",
)
VARIANTS = ("none", "detach_q", "clone_q", "detach_residual", "clone_residual", "hidden_states")


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def tensor_hash(value):
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256(str((list(value.shape), str(value.dtype))).encode())
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def model_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor_hash(value).encode())
    return digest.hexdigest()


def difference(reference, candidate):
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        raise ValueError("Compared tensors must have matching shapes and dtypes")
    if not torch.isfinite(reference).all() or not torch.isfinite(candidate).all():
        raise ValueError("Nonfinite logits")
    delta = candidate.float() - reference.float()
    return {
        "byte_identical": tensor_hash(reference) == tensor_hash(candidate),
        "elements": reference.numel(),
        "different_elements": int(torch.count_nonzero(delta)),
        "max_abs_difference": float(delta.abs().max()),
        "mean_abs_difference": float(delta.abs().mean()),
    }


def install_capture(model, variant, layer, saved):
    """Return a genuine read-only forward hook; no compiled return wrapper."""
    if variant in ("none", "hidden_states"):
        return None
    module = model.model.layers[layer]
    if variant.endswith("_q"):
        module = module.self_attn.q_norm
    clone = variant.startswith("clone_")

    def capture(_module, _inputs, output):
        saved["activation"] = output.detach().clone() if clone else output.detach()
        # Implicit None leaves the module output untouched.

    return module.register_forward_hook(capture)


def describe_capture(value, expected_shape):
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected_shape:
        raise ValueError("Captured tensor has the wrong shape")
    if value.requires_grad or not value.numel() or not torch.isfinite(value).all():
        raise ValueError("Capture must be finite, nonempty, and detached")
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "sha256": tensor_hash(value),
        "l2_norm": float(value.float().norm()),
    }


def compile_counters():
    from torch._dynamo.utils import counters

    return {name: dict(values) for name, values in counters.items() if values}


def graph_count():
    from torch._dynamo.utils import counters

    return counters["stats"]["unique_graphs"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--smoke-layers", type=int)
    args = parser.parse_args()
    if args.model.name != MODEL_REVISION:
        parser.error("Expected the pinned Qwen3-0.6B snapshot directory")
    if args.smoke_layers is not None and args.device != "cpu":
        parser.error("Layer truncation is restricted to CPU API smoke tests")
    if args.length < 1:
        parser.error("Physical sequence length must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    cache_dir = args.out / "compile-cache"
    if cache_dir.exists() and any(cache_dir.iterdir()):
        raise ValueError("Each variant requires an empty, isolated compilation cache")
    cache_dir.mkdir(exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir.resolve())
    os.environ["TRITON_CACHE_DIR"] = str((cache_dir / "triton").resolve())
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no fallback is permitted")

    from safetensors.torch import save_file
    from torch._dynamo.utils import counters
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.padding_side = "left"
    raw = (
        AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, local_files_only=True
        )
        .to(device)
        .eval()
    )
    if args.smoke_layers is not None:
        raw.model.layers = raw.model.layers[: args.smoke_layers]
        raw.config.num_hidden_layers = args.smoke_layers
        raw.eval()
    if not 0 <= args.layer < len(raw.model.layers) - 1:
        raise ValueError("Choose a nonfinal block; hidden_states[-1] includes final normalization")
    before_hash = model_hash(raw)
    grad_flags = tuple(parameter.requires_grad for parameter in raw.parameters())
    if any(module.training for module in raw.modules()):
        raise RuntimeError("All modules must be in eval mode")
    answer_ids = [tokenizer.encode(word, add_special_tokens=False) for word in ("Yes", "No")]
    if any(len(ids) != 1 for ids in answer_ids):
        raise ValueError("Answers must each be one token")
    yes_id, no_id = (ids[0] for ids in answer_ids)
    encoded = []
    for prompt in PROMPTS:
        if len(tokenizer.encode(prompt, add_special_tokens=False)) > args.length:
            raise ValueError("Fixed padding would truncate a diagnostic prompt")
        inputs = tokenizer(
            prompt, return_tensors="pt", padding="max_length", max_length=args.length
        ).to(device)
        positions = inputs["attention_mask"].cumsum(-1) - 1
        inputs["position_ids"] = positions.clamp_min(0)
        encoded.append(inputs)
    kwargs = {"use_cache": False, "output_hidden_states": args.variant == "hidden_states"}
    with torch.no_grad():
        eager_plain = [
            raw(**inputs, use_cache=False).logits.detach().cpu().clone() for inputs in encoded
        ]
    saved = {}
    handle = install_capture(raw, args.variant, args.layer, saved)
    capture_shape = (1, args.length, raw.config.hidden_size)
    if args.variant.endswith("_q"):
        capture_shape = (1, args.length, raw.config.num_attention_heads, raw.config.head_dim)

    def run(call, inputs):
        saved.clear()
        result = call(**inputs, **kwargs)
        if result.logits.requires_grad:
            raise RuntimeError("The execution unexpectedly recorded gradients")
        capture = None
        if args.variant == "hidden_states":
            if len(result.hidden_states) != len(raw.model.layers) + 1:
                raise RuntimeError("HF hidden-state output contract changed")
            capture = describe_capture(result.hidden_states[args.layer + 1], capture_shape)
        elif args.variant != "none":
            if set(saved) != {"activation"}:
                raise RuntimeError("Forward hook did not execute for this input")
            capture = describe_capture(saved["activation"], capture_shape)
        return result.logits.detach().cpu().clone(), capture

    eager_rows = []
    with torch.no_grad():
        for inputs, plain in zip(encoded, eager_plain, strict=True):
            logits, capture = run(raw, inputs)
            delta = difference(plain, logits)
            if not delta["byte_identical"]:
                raise RuntimeError("The capture changed eager logits; inspect before compiling")
            eager_rows.append({"vs_plain": delta, "capture": capture})
    atomic_json(args.out / "eager.json", eager_rows)
    print(json.dumps({"stage": "eager_controls_passed", "variant": args.variant}), flush=True)

    torch._dynamo.reset()
    counters.clear()
    compiled = torch.compile(raw, backend="inductor", fullgraph=True, dynamic=False)
    started = time.monotonic()
    rows, final_logits = [], {}
    with torch.no_grad():
        for index, inputs in enumerate(encoded):
            logits, capture = run(compiled, inputs)
            after_first = compile_counters()
            replay, replay_capture = run(compiled, inputs)
            if not difference(logits, replay)["byte_identical"] or capture != replay_capture:
                raise RuntimeError("Compiled replay changed logits or captured activation")
            if graph_count() != 1:
                raise RuntimeError(f"Expected one full graph, found {graph_count()}")
            last = logits[0, -1].float()
            logprobs = last.log_softmax(-1)
            final_logits[f"prompt_{index}"] = last.contiguous()
            row = {
                "prompt_id": index,
                "prompt": PROMPTS[index],
                "input_ids": inputs["input_ids"][0].tolist(),
                "attention_mask": inputs["attention_mask"][0].tolist(),
                "position_ids": inputs["position_ids"][0].tolist(),
                "capture": capture,
                "all_positions_logits_sha256": tensor_hash(logits),
                "vs_eager_plain": difference(eager_plain[index], logits),
                "replay": difference(logits, replay),
                "yes_logprob": float(logprobs[yes_id]),
                "no_logprob": float(logprobs[no_id]),
                "yes_no_log_odds": float(last[yes_id] - last[no_id]),
                "top_token_id": int(last.argmax()),
                "counters_after_first_call": after_first,
                "counters_after_replay": compile_counters(),
                "elapsed_seconds": time.monotonic() - started,
            }
            rows.append(row)
            save_file(final_logits, args.out / "last_logits.safetensors")
            atomic_json(args.out / "prompts.json", rows)
            print(
                json.dumps(
                    {
                        "stage": "compiled_prompt",
                        "prompt": index,
                        "variant": args.variant,
                        "graphs": graph_count(),
                        "seconds": row["elapsed_seconds"],
                    }
                ),
                flush=True,
            )
    if args.variant != "none" and len({row["capture"]["sha256"] for row in rows}) != len(rows):
        raise RuntimeError("Capture hashes did not change across distinct input tokens")
    after_hash = model_hash(raw)
    if before_hash != after_hash or grad_flags != tuple(p.requires_grad for p in raw.parameters()):
        raise RuntimeError("Model parameters, buffers or gradient flags changed")
    if any(module.training for module in raw.modules()):
        raise RuntimeError("Module training flags changed")
    if handle is not None:
        handle.remove()
    summary = {
        "status": "complete",
        "variant": args.variant,
        "model": "Qwen/Qwen3-0.6B",
        "model_revision": MODEL_REVISION,
        "layers": len(raw.model.layers),
        "capture_layer": args.layer,
        "cpu_truncated_smoke": args.smoke_layers is not None,
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers")},
        "device": str(device),
        "platform": platform.platform(),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cuda_version": torch.version.cuda,
        "model_state_sha256_before": before_hash,
        "model_state_sha256_after": after_hash,
        "fullgraph": True,
        "dynamic": False,
        "backend": "inductor",
        "unique_graphs": graph_count(),
        "final_counters": compile_counters(),
        "attention_implementation": raw.config._attn_implementation,
        "padding": "left",
        "physical_shape": [1, args.length],
        "yes_token_id": yes_id,
        "no_token_id": no_id,
        "prompt_count": len(rows),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "elapsed_compiled_seconds": time.monotonic() - started,
        "scope": "Standard PyTorch/HF API; not NNsight, not training a detector",
    }
    atomic_json(args.out / "summary.json", summary)
    print(json.dumps({"stage": "complete", "variant": args.variant}), flush=True)


if __name__ == "__main__":
    main()
