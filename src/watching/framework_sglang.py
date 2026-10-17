"""Frozen-checkpoint transfer through SGLang's native Engine.generate API.

Pinned source: sgl-project/sglang v0.5.10.post1. Its greedy sampler obtains the
argmax and full log-softmax before gathering explicitly requested token IDs.
Requesting every ID preserves uncensored full-vocabulary probabilities without
an internal hook or answer restriction. Run in an isolated environment: this
release declares Torch 2.9.1 and Transformers 5.3.0, not the HF/vLLM stack.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

from watching.artifacts import write_json
from watching.framework_vllm import score_record, validate_bank

SGLANG_VERSION = "0.5.10.post1"
EXPECTED_PACKAGES = {"sglang": SGLANG_VERSION, "torch": "2.9.1", "transformers": "5.3.0"}


def extract_logprobs(result: dict, input_length: int, vocab_size: int) -> tuple[torch.Tensor, int]:
    """Validate SGLang's triples (logprob, token ID, text) without re-normalizing."""
    metadata = result["meta_info"]
    if metadata["prompt_tokens"] != input_length:
        raise ValueError("SGLang changed the prompt token count")
    if metadata.get("completion_tokens") != 1 or len(result.get("output_ids", [])) != 1:
        raise ValueError("Expected exactly one native generated token")
    if not metadata.get("finish_reason"):
        raise ValueError("Generation did not finish")
    positions = metadata.get("output_token_ids_logprobs")
    if not isinstance(positions, list) or len(positions) != 1:
        raise ValueError("Expected one requested-token logprob distribution")
    entries = positions[0]
    if not isinstance(entries, list) or any(len(entry) != 3 for entry in entries):
        raise ValueError("Malformed SGLang logprob triples")
    ids = [entry[1] for entry in entries]
    if len(ids) != vocab_size or set(ids) != set(range(vocab_size)):
        raise ValueError("SGLang did not return every vocabulary ID exactly once")
    values = torch.empty(vocab_size, dtype=torch.float32)
    for logprob, token_id, _ in entries:
        values[token_id] = logprob
    if not torch.isfinite(values).all():
        raise ValueError("Nonfinite vocabulary logprobs")
    normalization = float(values.double().logsumexp(0))
    if abs(normalization) > 1e-4:
        raise ValueError(f"Unnormalized logprobs: logsumexp={normalization}")
    token_id = result["output_ids"][0]
    if type(token_id) is not int or not 0 <= token_id < vocab_size:
        raise ValueError("Invalid generated token")
    if values[token_id] != values.max():
        raise ValueError("Native generated token differs from the raw vocabulary argmax")
    sampled = metadata.get("output_token_logprobs")
    if not isinstance(sampled, list) or len(sampled) != 1:
        raise ValueError("Missing native generated-token logprob")
    if sampled[0][1] != token_id or float(values[token_id]) != sampled[0][0]:
        raise ValueError("Generated-token and requested-token logprobs disagree")
    return values, token_id


def evaluate(engine, records: list[dict], bank: dict, out: Path, vocab_size: int) -> dict:
    vectors = out / "vectors"
    vectors.mkdir()
    sampling = {
        "max_new_tokens": 1,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
        "n": 1,
    }
    smoke = [record for record in records if record["group"] == "smoke"]
    scored = [record for record in records if record["group"] != "smoke"]
    ordered = [(record, False) for record in smoke + scored]
    ordered += [(record, True) for record in smoke]
    originals = {}
    repeats = []
    with (out / "sglang.jsonl").open("x", buffering=1) as stream:
        for index, (record, repeat) in enumerate(ordered):
            start = time.monotonic()
            result = engine.generate(
                input_ids=record["input_ids"],
                sampling_params=dict(sampling),
                return_logprob=True,
                logprob_start_len=-1,
                top_logprobs_num=0,
                token_ids_logprob=list(range(vocab_size)),
                stream=False,
            )
            values, token_id = extract_logprobs(result, len(record["input_ids"]), vocab_size)
            vector_path = vectors / f"{index:04d}.safetensors"
            temporary = vector_path.with_suffix(".partial")
            save_file({"logprobs": values}, str(temporary))
            temporary.replace(vector_path)
            row = score_record(record, values, token_id, bank["no_id"], bank["yes_id"])
            row.update(
                runtime="sglang",
                path="sglang",
                repeat=repeat,
                vector_file=str(vector_path.relative_to(out)),
                seconds=time.monotonic() - start,
                finish_reason=result["meta_info"]["finish_reason"],
                requested_token_ids="all vocabulary IDs; no top-k censoring",
            )
            if repeat:
                baseline = originals[record["id"]]
                comparison = {
                    "id": record["id"],
                    "bit_identical": torch.equal(
                        values.view(torch.int32), baseline.view(torch.int32)
                    ),
                    "different_values": int((values != baseline).sum()),
                    "max_abs_difference": float((values.double() - baseline.double()).abs().max()),
                }
                row["repeat_comparison"] = comparison
                repeats.append(comparison)
            elif record["group"] == "smoke":
                originals[record["id"]] = values
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            print(json.dumps(row, allow_nan=False), flush=True)
    return {"records": len(records), "diagnostic_repeats": repeats, "sampling_params": sampling}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    if not args.model.is_dir():
        parser.error("--model must be an exported local checkpoint")
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        parser.error("--out must be empty")
    manifest = {"status": "starting", "model": str(args.model.resolve())}
    engine = None
    write_json(args.out / "manifest.json", manifest)
    try:
        config_bytes = (args.model / "config.json").read_bytes()
        config = json.loads(config_bytes)
        if config.get("quantization_config") is not None:
            raise ValueError("A quantized checkpoint changes the evaluation")
        vocab_size = int(config["vocab_size"])
        data_bytes = args.data.read_bytes()
        bank = json.loads(data_bytes)
        records = validate_bank(bank, vocab_size)
        context_length = max(len(record["input_ids"]) for record in records) + 1
        if args.smoke_only:
            records = [record for record in records if record["group"] == "smoke"]
        versions = {name: importlib.metadata.version(name) for name in EXPECTED_PACKAGES}
        for name, expected in EXPECTED_PACKAGES.items():
            if versions[name].split("+")[0] != expected:
                raise RuntimeError(f"Expected {name} {expected}, found {versions[name]}")
        if not torch.cuda.is_available():
            raise RuntimeError("This native serving evaluation requires CUDA")
        from sglang import Engine

        options = {
            "model_path": str(args.model.resolve()),
            "dtype": "bfloat16",
            "tp_size": 1,
            "max_running_requests": 1,
            "context_length": context_length,
            "piecewise_cuda_graph_max_tokens": context_length,
            "sampling_defaults": "openai",
            "random_seed": 0,
            "log_level": "info",
        }
        manifest.update(
            status="initializing_engine",
            data_sha256=hashlib.sha256(data_bytes).hexdigest(),
            config_sha256=hashlib.sha256(config_bytes).hexdigest(),
            model_config=config,
            versions=versions,
            python=platform.python_version(),
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(),
            engine_options=options,
            smoke_only=args.smoke_only,
            logprob_scope="Every vocabulary ID via native token_ids_logprob gather",
            logprob_semantics=(
                "Greedy log_softmax; native sampler clamps negative infinity to dtype minimum"
            ),
            input_parity_scope=(
                "Exact input_ids passed directly; API echoes token count, not prompt IDs"
            ),
            hooks_installed=False,
            weight_updates=False,
        )
        write_json(args.out / "manifest.json", manifest)
        engine = Engine(**options)
        resolved = dataclasses.asdict(engine.server_args)
        # Authentication settings are irrelevant to the numerical experiment.
        for name in list(resolved):
            if any(word in name.lower() for word in ("api_key", "auth", "password", "secret")):
                resolved.pop(name)
        manifest.update(status="evaluating", resolved_server_args=resolved)
        write_json(args.out / "manifest.json", manifest)
        summary = evaluate(engine, records, bank, args.out, vocab_size)
        manifest.update(status="complete", **summary)
        write_json(args.out / "manifest.json", manifest)
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(args.out / "manifest.json", manifest)
        raise
    finally:
        if engine is not None:
            engine.shutdown()


if __name__ == "__main__":
    main()
