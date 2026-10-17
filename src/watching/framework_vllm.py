"""Evaluate an exported detector through vLLM's ordinary compiled serving path.

No model hooks, answer constraints, updates, or fallback to censored logprobs.
Each request is submitted alone using the bank's original token IDs. Diagnostic
repeats are measured, not assumed identical; a fresh-engine repeat is a new run.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from watching.artifacts import write_json

VLLM_VERSION = "0.19.1"


def validate_bank(bank: dict[str, Any], vocab_size: int) -> list[dict[str, Any]]:
    """Reject malformed or ambiguous evaluation inputs before creating an engine."""
    answers = [bank.get("yes_id"), bank.get("no_id")]
    if any(type(token) is not int or not 0 <= token < vocab_size for token in answers):
        raise ValueError("yes_id and no_id must be valid vocabulary indices")
    if answers[0] == answers[1]:
        raise ValueError("Yes and No must be distinct tokens")
    records = bank.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("records must be a nonempty list")
    ids: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Each record must be an object")
        key = record.get("id")
        if not isinstance(key, str) or not key or key in ids:
            raise ValueError("Record IDs must be unique nonempty strings")
        ids.add(key)
        if not isinstance(record.get("group"), str) or not record["group"]:
            raise ValueError(f"Missing group for {key}")
        if not isinstance(record.get("text"), str):
            raise ValueError(f"Missing text for {key}")
        tokens = record.get("input_ids")
        if not isinstance(tokens, list) or not tokens:
            raise ValueError(f"Empty token prefix for {key}")
        if any(type(token) is not int or not 0 <= token < vocab_size for token in tokens):
            raise ValueError(f"Invalid input token in {key}")
    if sum(record["group"] == "smoke" for record in records) != 4:
        raise ValueError("The frozen bank must contain exactly four group='smoke' records")
    return records


def extract_logprobs(
    result: Any, input_ids: list[int], vocab_size: int
) -> tuple[torch.Tensor, int]:
    """Require raw, normalized, uncensored next-token probabilities and native argmax."""
    if list(result.prompt_token_ids) != input_ids:
        raise ValueError("vLLM changed the supplied prompt token IDs")
    if not result.finished or len(result.outputs) != 1:
        raise ValueError("Expected one completed generation")
    output = result.outputs[0]
    if len(output.token_ids) != 1 or output.logprobs is None or len(output.logprobs) != 1:
        raise ValueError("Expected exactly one generated token and one logprob distribution")
    entries = output.logprobs[0]
    if not isinstance(entries, dict) or set(entries) != set(range(vocab_size)):
        raise ValueError(
            "vLLM did not return every vocabulary token; top-k substitution is forbidden"
        )
    values = torch.tensor(
        [entries[token].logprob for token in range(vocab_size)], dtype=torch.float32
    )
    if not torch.isfinite(values).all():
        raise ValueError("Nonfinite full-vocabulary logprobs")
    normalization = float(torch.logsumexp(values.double(), dim=0))
    if abs(normalization) > 1e-4:
        raise ValueError(f"Returned values are not normalized logprobs: logsumexp={normalization}")
    token_id = int(output.token_ids[0])
    if not 0 <= token_id < vocab_size or values[token_id] != values.max():
        raise ValueError("Generated token is not an argmax of the raw logprob distribution")
    return values, token_id


def score_record(
    record: dict, values: torch.Tensor, token_id: int, no_id: int, yes_id: int
) -> dict:
    """Retain native emitted-token accuracy, including any vocabulary-wide ties."""
    yes, no = float(values[yes_id]), float(values[no_id])
    return {
        "id": record["id"],
        "group": record["group"],
        "runtime": "vllm",
        "path": "vllm",
        "yes_id": yes_id,
        "no_id": no_id,
        "condition": "serving",
        "expected_token_id": no_id,
        "yes_logprob": yes,
        "no_logprob": no,
        "log_odds": yes - no,
        "top_token_id": token_id,
        "first_argmax_token_id": int(values.argmax()),
        "argmax_ties": int((values == values.max()).sum()),
        "correct": token_id == no_id,
        "correct_probability": math.exp(no),
        "correct_logprob": no,
        "nll": -no,
        "vocabulary_size": values.numel(),
        "input_length": len(record["input_ids"]),
        "logsumexp": float(torch.logsumexp(values.double(), dim=0)),
    }


def evaluate(llm, sampling, records: list[dict], bank: dict, out: Path, vocab_size: int) -> dict:
    """Persist each result before the next request; repeat the four smoke inputs last."""
    vectors = out / "vectors"
    vectors.mkdir()
    originals: dict[str, torch.Tensor] = {}
    repetitions = []
    smoke = [record for record in records if record["group"] == "smoke"]
    scored = [record for record in records if record["group"] != "smoke"]
    ordered = [(record, False) for record in smoke + scored]
    ordered += [(record, True) for record in smoke]
    with (out / "vllm.jsonl").open("x", buffering=1) as stream:
        for index, (record, repeat) in enumerate(ordered):
            started = time.monotonic()
            results = llm.generate(
                [{"prompt_token_ids": record["input_ids"]}], sampling, use_tqdm=False
            )
            if len(results) != 1:
                raise ValueError("Expected exactly one request output")
            values, token_id = extract_logprobs(results[0], record["input_ids"], vocab_size)
            vector_path = vectors / f"{index:04d}.safetensors"
            temporary = vector_path.with_suffix(".partial")
            save_file({"logprobs": values}, str(temporary))
            temporary.replace(vector_path)
            row = score_record(record, values, token_id, bank["no_id"], bank["yes_id"])
            row.update(
                repeat=repeat,
                vector_file=str(vector_path.relative_to(out)),
                seconds=time.monotonic() - started,
            )
            if repeat:
                baseline = originals[record["id"]]
                delta = values.double() - baseline.double()
                comparison = {
                    "id": record["id"],
                    "bit_identical": torch.equal(
                        values.view(torch.int32), baseline.view(torch.int32)
                    ),
                    "different_values": int((values != baseline).sum()),
                    "max_abs_difference": float(delta.abs().max()),
                }
                row["repeat_comparison"] = comparison
                repetitions.append(comparison)
            elif record["group"] == "smoke":
                originals[record["id"]] = values
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            print(json.dumps(row, allow_nan=False), flush=True)
    return {"records": len(records), "diagnostic_repeats": repetitions}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    if not args.model.is_dir():
        parser.error("--model must be a local exported native checkpoint")
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        parser.error("--out must be empty; use a new directory for every engine")
    manifest: dict[str, Any] = {"status": "starting", "model": str(args.model.resolve())}
    write_json(args.out / "manifest.json", manifest)
    try:
        config_bytes = (args.model / "config.json").read_bytes()
        config = json.loads(config_bytes)
        vocab_size = int(config["vocab_size"])
        data_bytes = args.data.read_bytes()
        bank = json.loads(data_bytes)
        records = validate_bank(bank, vocab_size)
        if args.smoke_only:
            records = [record for record in records if record["group"] == "smoke"]
        version = importlib.metadata.version("vllm")
        if version != VLLM_VERSION:
            raise RuntimeError(f"Expected vLLM {VLLM_VERSION}, found {version}")
        if not torch.cuda.is_available():
            raise RuntimeError("This evaluation requires CUDA")
        from vllm import LLM, SamplingParams

        options = {
            "model": str(args.model.resolve()),
            "dtype": "bfloat16",
            "tensor_parallel_size": 1,
            "max_num_seqs": 1,
            "max_model_len": max(len(record["input_ids"]) for record in bank["records"]) + 1,
            "max_logprobs": vocab_size,
            "logprobs_mode": "raw_logprobs",
            "generation_config": "vllm",
            "seed": 0,
            "enforce_eager": False,
        }
        sampling_options = {
            "n": 1,
            "max_tokens": 1,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "min_p": 0.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "repetition_penalty": 1.0,
            "logprobs": -1,
            "detokenize": False,
        }
        manifest.update(
            data_sha256=hashlib.sha256(data_bytes).hexdigest(),
            smoke_only=args.smoke_only,
            config_sha256=hashlib.sha256(config_bytes).hexdigest(),
            model_config=config,
            versions={
                "python": platform.python_version(),
                "torch": torch.__version__,
                "vllm": version,
                "transformers": importlib.metadata.version("transformers"),
                "cuda": torch.version.cuda,
            },
            gpu=torch.cuda.get_device_name(),
            engine_options=options,
            sampling_options=sampling_options,
            cache_environment={
                key: os.environ.get(key)
                for key in (
                    "VLLM_CACHE_ROOT",
                    "VLLM_DISABLE_COMPILE_CACHE",
                    "TORCHINDUCTOR_CACHE_DIR",
                )
            },
            hooks_installed=False,
            weight_updates=False,
            status="initializing_engine",
        )
        write_json(args.out / "manifest.json", manifest)
        llm = LLM(**options)
        engine_config = llm.llm_engine.vllm_config
        if engine_config.model_config.logprobs_mode != "raw_logprobs":
            raise RuntimeError("Engine did not retain raw_logprobs mode")
        manifest.update(
            status="evaluating",
            resolved={
                "compilation": str(engine_config.compilation_config),
                "attention": str(engine_config.attention_config),
                "scheduler": str(engine_config.scheduler_config),
                "cache": str(engine_config.cache_config),
                "quantization": engine_config.model_config.quantization,
                "dtype": str(engine_config.model_config.dtype),
                "max_logprobs": engine_config.model_config.max_logprobs,
            },
        )
        if engine_config.model_config.quantization is not None:
            raise RuntimeError("Quantization would change the exported checkpoint")
        write_json(args.out / "manifest.json", manifest)
        summary = evaluate(
            llm, SamplingParams(**sampling_options), records, bank, args.out, vocab_size
        )
        manifest.update(status="complete", **summary)
        write_json(args.out / "manifest.json", manifest)
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(args.out / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
