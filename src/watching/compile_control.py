"""Compare two independent no-probe compilations with one reference-probe graph.

This is an execution reproducibility control, not training or a population test.
"""

import argparse
import ast
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from watching.artifacts import write_json
from watching.generated_trace import chosen_kernel_metadata, json_metadata
from watching.probe import ProbeModel
from watching.runtime import runtime_provenance
from watching.tail_train import difference


def entry_none_a(wrapper, input_ids, attention_mask, position_ids):
    return wrapper(input_ids, attention_mask, position_ids, capture=False)


def entry_none_b(wrapper, input_ids, attention_mask, position_ids):
    return wrapper(input_ids, attention_mask, position_ids, capture=False)


def entry_reference(wrapper, input_ids, attention_mask, position_ids):
    return wrapper(input_ids, attention_mask, position_ids, capture=True)


ENTRIES = {"none_a": entry_none_a, "none_b": entry_none_b, "reference": entry_reference}


def kernel_sources(source):
    result = {}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            if ast.unparse(node.value.func) == "async_compile.triton":
                result[ast.unparse(node.targets[0])] = node.value.args[1].value
    if not result:
        raise ValueError("No generated Triton kernels found; unsupported wrapper")
    return result


class WrapperObserver:
    """Retain wrapper/kernel objects without copying tensors or modifying launches."""

    def __init__(self):
        self.calls = 0
        self.globals = None
        self.code = None
        self.source = None

    def _trace(self, frame, event, arg):
        if event == "call" and frame.f_code.co_name == "call":
            if all(key in frame.f_globals for key in ("extern_kernels", "assert_size_stride")):
                self.calls += 1
                if self.calls != 1:
                    raise RuntimeError("Expected exactly one full generated wrapper")
                self.globals = frame.f_globals
                self.code = frame.f_code
                self.source = Path(frame.f_code.co_filename).read_text()
        return None

    def __enter__(self):
        if sys.gettrace() is not None:
            raise RuntimeError("Refusing to replace existing Python tracer")
        sys.settrace(self._trace)
        return self

    def __exit__(self, *args):
        sys.settrace(None)

    def metadata(self):
        if self.calls != 1:
            raise RuntimeError("Generated wrapper was not observed")
        sources = kernel_sources(self.source)
        objects, rows = {}, {}
        for name, source in sources.items():
            kernel = self.globals[name]
            objects[name] = kernel
            launchers = getattr(kernel, "launchers", ())
            if len(launchers) != 1:
                raise RuntimeError(f"Kernel {name} does not have one selected launcher")
            launcher = launchers[0]
            config = launcher.config
            row = {
                "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "config": json_metadata(
                    {
                        **config.kwargs,
                        "num_warps": config.num_warps,
                        "num_stages": config.num_stages,
                    }
                ),
                "selected_cache_hash": getattr(launcher, "cache_hash", None),
            }
            # Ancillary binary metadata cannot invalidate already-saved numeric results.
            try:
                details = chosen_kernel_metadata(kernel)
                ptx = details.get("ptx")
                row.update(
                    ptx_sha256=hashlib.sha256(ptx.encode()).hexdigest() if ptx else None,
                    ptx_provenance_error=details.get("ptx_provenance_error"),
                )
            except Exception as error:
                row.update(ptx_sha256=None, ptx_provenance_error=f"{type(error).__name__}: {error}")
            rows[name] = row
        return rows, objects


def compare_kernel_objects(left, right, left_meta, right_meta):
    """Only compare name-matched kernels; report source mismatches explicitly."""
    rows = []
    for name in sorted(left.keys() & right.keys()):
        rows.append(
            {
                "name": name,
                "same_object": left[name] is right[name],
                "same_source": left_meta[name]["source_sha256"]
                == right_meta[name]["source_sha256"],
                "same_config": left_meta[name]["config"] == right_meta[name]["config"],
                "same_selected_cache_hash": (
                    left_meta[name]["selected_cache_hash"]
                    == right_meta[name]["selected_cache_hash"]
                    if left_meta[name]["selected_cache_hash"] is not None
                    and right_meta[name]["selected_cache_hash"] is not None
                    else None
                ),
                "left": left_meta[name],
                "right": right_meta[name],
            }
        )
    return rows


def validate_result(result, arm):
    logits, captured = result
    expected = 1 if arm == "reference" else 0
    if len(captured) != expected:
        raise RuntimeError(f"{arm}: expected {expected} saved activation, found {len(captured)}")
    if logits.ndim != 2 or logits.shape[0] != 1 or not torch.isfinite(logits).all():
        raise RuntimeError("Expected finite singleton full-vocabulary logits")
    return logits.detach().cpu().clone()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="Qwen/Qwen3-8B")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--length", type=int, default=96)
    parser.add_argument("--probe-layer", type=int, default=14)
    parser.add_argument("--prompts", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.prompts <= 4:
        parser.error("Control is restricted to one through four training prompts")
    from torch._dynamo.utils import counters
    from torch._inductor import config

    config.freezing = False
    config.force_disable_caches = True
    torch.manual_seed(20261019)
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    log = (args.out / "metrics.jsonl").open("w")

    def emit(event, **values):
        row = {"event": event, "elapsed_seconds": time.monotonic() - started, **values}
        print(json.dumps(row), flush=True)
        log.write(json.dumps(row) + "\n")
        log.flush()

    raw = json.loads(args.data.read_text())
    records = [r for r in raw["train"] if r.get("task", "detection") == "detection"][: args.prompts]
    if len(records) != args.prompts:
        raise ValueError("Insufficient training records; no held-out fallback")
    write_json(args.out / "data.json", records)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, revision=args.revision)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    batches = []
    for record in records:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": record["text"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if len(tokenizer.encode(text, add_special_tokens=False)) > args.length:
            raise ValueError("Prompt exceeds fixed length; truncation forbidden")
        batch = tokenizer(
            text,
            add_special_tokens=False,
            padding="max_length",
            max_length=args.length,
            return_tensors="pt",
        )
        batch["position_ids"] = (batch.attention_mask.cumsum(-1) - 1).clamp_min(0)
        batches.append({key: value.cuda() for key, value in batch.items()})
    emit("loading")
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.checkpoint,
            revision=args.revision,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .cuda()
        .eval()
    )
    model.config.use_cache = False
    model.requires_grad_(True)
    wrapper = ProbeModel(model, args.probe_layer)
    compiled = {
        name: torch.compile(fn, fullgraph=True, dynamic=False) for name, fn in ENTRIES.items()
    }
    manifest = {
        "checkpoint": args.checkpoint,
        "revision": args.revision,
        "runtime": runtime_provenance(),
        "length": args.length,
        "probe_layer": args.probe_layer,
        "prompts": len(records),
        "compile_order": list(ENTRIES),
        "force_disable_caches": True,
        "inductor_deterministic": getattr(config, "deterministic", None),
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "entry_code_objects_distinct": len({fn.__code__ for fn in ENTRIES.values()}) == 3,
        "shared_model_and_wrapper": True,
        "requires_grad": True,
        "execution": "eval and no_grad",
        "training": False,
        "scope": "One compilation instance per arm; exploratory control",
    }
    write_json(args.out / "manifest.json", manifest)
    predictions, observers, objects, metadata = {}, {}, {}, {}
    for name in ENTRIES:
        emit("compile_start", arm=name)
        before = dict(counters["inductor"])
        with torch.no_grad():
            predictions[name] = [validate_result(compiled[name](wrapper, **batches[0]), name)]
        torch.save(predictions[name][0], args.out / f"{name}_first_logits.pt")
        emit("first_logits_saved", arm=name)
        with torch.no_grad(), WrapperObserver() as observer:
            repeated = validate_result(compiled[name](wrapper, **batches[0]), name)
        if not torch.equal(repeated, predictions[name][0]):
            raise RuntimeError(f"{name}: repeated traced execution changed full-vocabulary logits")
        observers[name] = observer
        (args.out / f"{name}_wrapper.py.txt").write_text(observer.source)
        # Print numerical controls before potentially expensive ancillary metadata extraction.
        for previous in predictions:
            if previous != name:
                emit(
                    "first_comparison",
                    left=previous,
                    right=name,
                    difference=difference(predictions[previous][0], predictions[name][0]),
                )
        emit(
            "repeat_and_observer_parity",
            arm=name,
            bit_identical=True,
            compile_counter_delta={
                key: value - before.get(key, 0)
                for key, value in counters["inductor"].items()
                if value != before.get(key, 0)
            },
        )
        metadata[name], objects[name] = observer.metadata()
        write_json(args.out / f"{name}_kernels.json", metadata[name])
    comparisons = {}
    for left, right in (("none_a", "none_b"), ("none_a", "reference"), ("none_b", "reference")):
        rows = compare_kernel_objects(
            objects[left], objects[right], metadata[left], metadata[right]
        )
        comparisons[f"{left}__{right}"] = {
            "same_wrapper_code_object": observers[left].code is observers[right].code,
            "kernels": rows,
        }
    write_json(args.out / "kernel_comparisons.json", comparisons)
    warm_counters = dict(counters["inductor"])
    warm_graphs = counters["stats"].get("unique_graphs", 0)
    for index, batch in enumerate(batches[1:], 1):
        order = list(ENTRIES) if index % 2 == 0 else list(reversed(ENTRIES))
        for name in order:
            with torch.no_grad():
                value = validate_result(compiled[name](wrapper, **batch), name)
                repeated = validate_result(compiled[name](wrapper, **batch), name)
            if not torch.equal(value, repeated):
                raise RuntimeError(f"{name} prompt {index}: ordinary repeat differs")
            predictions[name].append(value)
        emit(
            "prompt_complete",
            prompt_index=index,
            none_vs_none=difference(predictions["none_a"][-1], predictions["none_b"][-1]),
            none_vs_reference=difference(predictions["none_a"][-1], predictions["reference"][-1]),
        )
    for name in ENTRIES:
        with torch.no_grad(), WrapperObserver() as final_observer:
            last = validate_result(compiled[name](wrapper, **batches[0]), name)
        if final_observer.calls != 1 or final_observer.code is not observers[name].code:
            raise RuntimeError(f"{name}: generated wrapper changed after initial compilation")
        if not torch.equal(last, predictions[name][0]):
            raise RuntimeError(f"{name}: compiling other arms changed original execution")
    if (
        dict(counters["inductor"]) != warm_counters
        or counters["stats"].get("unique_graphs", 0) != warm_graphs
    ):
        raise RuntimeError(
            "Compiler counters changed after the three warmups; metadata may be stale"
        )
    torch.save(predictions, args.out / "logits.pt")
    report = {
        "complete": True,
        "interleaved_repeat_parity": True,
        "final_wrapper_identity_verified": True,
        "post_warmup_compile_counters_unchanged": True,
        "comparisons": {
            f"{left}__{right}": [
                difference(a, b) for a, b in zip(predictions[left], predictions[right], strict=True)
            ]
            for left, right in (
                ("none_a", "none_b"),
                ("none_a", "reference"),
                ("none_b", "reference"),
            )
        },
    }
    write_json(args.out / "summary.json", report)
    emit("complete", **report)


if __name__ == "__main__":
    main()
