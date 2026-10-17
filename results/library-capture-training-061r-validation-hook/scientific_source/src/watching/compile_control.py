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
    expected = 1 if arm in ("reference", "identity_reference") else 0
    if len(captured) != expected:
        raise RuntimeError(f"{arm}: expected {expected} saved activation, found {len(captured)}")
    if logits.ndim != 2 or logits.shape[0] != 1 or not torch.isfinite(logits).all():
        raise RuntimeError("Expected finite singleton full-vocabulary logits")
    return logits.detach().cpu().clone()


def answer_readout(logits, negative_id, positive_id):
    logp = logits.reshape(-1).float().log_softmax(-1)
    return {
        "negative_logprob": float(logp[negative_id]),
        "positive_logprob": float(logp[positive_id]),
        "positive_minus_negative_logodds": float(logp[positive_id] - logp[negative_id]),
        "positive_probability": float(logp[positive_id].exp()),
        "negative_probability": float(logp[negative_id].exp()),
        "top_token_id": int(logp.argmax()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="Qwen/Qwen3-8B")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--length", type=int, default=96)
    parser.add_argument("--probe-layer", type=int, default=14)
    parser.add_argument("--prompts", type=int, default=4)
    parser.add_argument("--extended-controls", action="store_true")
    parser.add_argument("--trace-origin", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--trained-tail", type=Path)
    args = parser.parse_args()
    if args.trained_tail and args.trace_origin:
        parser.error("Run base origin audit separately from installed trained-tail transfer")
    if not 1 <= args.prompts <= 4:
        parser.error("Control is restricted to one through four training prompts")
    from torch._dynamo.utils import counters
    from torch._inductor import config

    config.freezing = False
    config.force_disable_caches = True
    if args.deterministic:
        config.deterministic = True
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
    entries = dict(ENTRIES)
    if args.extended_controls:
        from watching.control_probes import EXTRA_ENTRIES, BufferWriterObserver, StorageControlProbe

        wrapper = StorageControlProbe(model, args.probe_layer)
        wrapper.prepare_buffer(batches[0])
        entries.update(EXTRA_ENTRIES)
    else:
        wrapper = ProbeModel(model, args.probe_layer)
    if args.extended_controls:
        with torch.no_grad():
            baseline, eager_reference = wrapper(**batches[0], mode="reference")
            for name, entry in entries.items():
                logits, _ = entry(wrapper, **batches[0])
                if not torch.equal(logits, baseline):
                    raise RuntimeError(f"Eager control {name} changes full-vocabulary logits")
                if name == "buffer" and not torch.equal(wrapper.probe_buffer, eager_reference[0]):
                    raise RuntimeError("Eager buffer differs from actual reference activation")
        emit("eager_controls_verified", buffer_matches_reference_activation=True)
    compiled = {
        name: torch.compile(fn, fullgraph=True, dynamic=False) for name, fn in entries.items()
    }
    manifest = {
        "checkpoint": args.checkpoint,
        "revision": args.revision,
        "runtime": runtime_provenance(),
        "length": args.length,
        "probe_layer": args.probe_layer,
        "prompts": len(records),
        "compile_order": list(entries),
        "force_disable_caches": True,
        "inductor_deterministic": getattr(config, "deterministic", None),
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "entry_code_objects_distinct": len({fn.__code__ for fn in entries.values()})
        == len(entries),
        "shared_model_and_wrapper": True,
        "extended_controls": args.extended_controls,
        "trace_origin_requested": args.trace_origin,
        "trained_tail_requested": str(args.trained_tail) if args.trained_tail else None,
        "requires_grad": True,
        "execution": "eval and no_grad",
        "training": False,
        "scope": "One compilation instance per arm; exploratory control",
    }
    write_json(args.out / "manifest.json", manifest)
    predictions, observers, objects, metadata = {}, {}, {}, {}
    buffers = {}

    def invoke(name, batch):
        if name == "buffer":
            wrapper.probe_buffer.fill_(float("nan"))
        value = validate_result(compiled[name](wrapper, **batch), name)
        if name == "buffer" and not torch.isfinite(wrapper.probe_buffer).all():
            raise RuntimeError("Buffer capture was not completely overwritten with finite values")
        return value

    for name in entries:
        emit("compile_start", arm=name)
        before = dict(counters["inductor"])
        with torch.no_grad():
            predictions[name] = [invoke(name, batches[0])]
        torch.save(predictions[name][0], args.out / f"{name}_first_logits.pt")
        emit("first_logits_saved", arm=name)
        observer_class = (
            BufferWriterObserver(wrapper.probe_buffer) if name == "buffer" else WrapperObserver()
        )
        first_buffer = wrapper.probe_buffer.detach().cpu().clone() if name == "buffer" else None
        with torch.no_grad(), observer_class as observer:
            repeated = invoke(name, batches[0])
        if not torch.equal(repeated, predictions[name][0]):
            raise RuntimeError(f"{name}: repeated traced execution changed full-vocabulary logits")
        if name == "buffer":
            if not observer.writers or not torch.equal(first_buffer, wrapper.probe_buffer.cpu()):
                raise RuntimeError("Buffer writer provenance or capture-repeat parity failed")
            buffers[0] = first_buffer
            write_json(args.out / "buffer_writers.json", observer.writers)
            emit("buffer_capture_verified", native_writers=observer.writers, exact_repeat=True)
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
    contrasts = [("none_a", "none_b"), ("none_a", "reference"), ("none_b", "reference")]
    if args.extended_controls:
        contrasts.extend(
            [("none_a", "buffer"), ("none_a", "identity"), ("identity", "identity_reference")]
        )
    comparisons = {}
    for left, right in contrasts:
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
        order = list(entries) if index % 2 == 0 else list(reversed(entries))
        for name in order:
            with torch.no_grad():
                value = invoke(name, batch)
                captured_buffer = wrapper.probe_buffer.cpu().clone() if name == "buffer" else None
                repeated = invoke(name, batch)
                if name == "buffer":
                    if not torch.equal(captured_buffer, wrapper.probe_buffer.cpu()):
                        raise RuntimeError("Buffer capture is not repeatable")
                    buffers[index] = captured_buffer
            if not torch.equal(value, repeated):
                raise RuntimeError(f"{name} prompt {index}: ordinary repeat differs")
            predictions[name].append(value)
        emit(
            "prompt_complete",
            prompt_index=index,
            none_vs_none=difference(predictions["none_a"][-1], predictions["none_b"][-1]),
            none_vs_reference=difference(predictions["none_a"][-1], predictions["reference"][-1]),
        )
    for name in entries:
        with torch.no_grad(), WrapperObserver() as final_observer:
            last = invoke(name, batches[0])
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
    if args.extended_controls:
        if len(buffers) > 1 and all(torch.equal(buffers[0], value) for value in buffers.values()):
            raise RuntimeError("Buffer did not change across distinct diagnostic inputs")
        torch.save(buffers, args.out / "buffer_activations.pt")
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
            for left, right in contrasts
        },
    }
    write_json(args.out / "summary.json", report)
    emit("complete", **report)
    if args.trained_tail:
        from watching.tail_train import file_digest, record_answer_ids
        from watching.validate_tail import install_checkpoint

        artifact = torch.load(args.trained_tail, map_location="cpu", weights_only=True)
        tail_hash = file_digest(args.trained_tail)
        answer_ids = [record_answer_ids(tokenizer, record) for record in records]
        base_rows = {
            name: [
                answer_readout(value, *ids) for value, ids in zip(values, answer_ids, strict=True)
            ]
            for name, values in predictions.items()
        }
        write_json(args.out / "base_answer_readouts.json", base_rows)
        installed = install_checkpoint(model, artifact, args.checkpoint, args.probe_layer)
        del artifact
        emit("trained_tail_installed", sha256=tail_hash, **installed)
        trained = {name: [] for name in entries}
        for index, batch in enumerate(batches):
            for name in entries:
                with torch.no_grad(), WrapperObserver() as observed:
                    value = invoke(name, batch)
                if observed.code is not observers[name].code:
                    raise RuntimeError("Installing trained tail changed generated wrapper")
                with torch.no_grad():
                    repeated = invoke(name, batch)
                if not torch.equal(value, repeated):
                    raise RuntimeError("Trained detector repeat is not bit-identical")
                trained[name].append(value)
                row = answer_readout(value, *answer_ids[index])
                row["top_token_text"] = tokenizer.decode([row["top_token_id"]])
                emit("trained_detector_readout", arm=name, prompt_index=index, **row)
        if all(
            torch.equal(a, b)
            for name in entries
            for a, b in zip(predictions[name], trained[name], strict=True)
        ):
            raise RuntimeError(
                "Tail installation did not change any native output; stale weights suspected"
            )
        if (
            dict(counters["inductor"]) != warm_counters
            or counters["stats"].get("unique_graphs", 0) != warm_graphs
        ):
            raise RuntimeError("Installing trained tail triggered recompilation")
        torch.save(trained, args.out / "trained_logits.pt")
        trained_rows = {
            name: [
                answer_readout(value, *ids) for value, ids in zip(values, answer_ids, strict=True)
            ]
            for name, values in trained.items()
        }
        write_json(args.out / "trained_answer_readouts.json", trained_rows)
        write_json(
            args.out / "trained_transfer.json",
            {
                "tail_sha256": tail_hash,
                "same_graphs_no_recompile": True,
                "weight_installation_changed_outputs": True,
                "installation": installed,
                "scope": "Existing detector on fresh compilation; no historical graph parity claim",
            },
        )
        emit("trained_detector_transfer_complete", tail_sha256=tail_hash)
    if args.trace_origin:
        from watching.compiler_origin import run_origin_audit

        arms = {name: (lambda batch, fn=compiled[name]: fn(wrapper, **batch)) for name in ENTRIES}
        run_origin_audit(arms, batches, len(model.model.layers), args.out / "origin", emit)
        for name in ENTRIES:
            with torch.no_grad(), WrapperObserver() as observed:
                final = invoke(name, batches[0])
            if observed.code is not observers[name].code or not torch.equal(
                final, predictions[name][0]
            ):
                raise RuntimeError("Detailed audit changed compiled graph or original logits")
        if (
            dict(counters["inductor"]) != warm_counters
            or counters["stats"].get("unique_graphs", 0) != warm_graphs
        ):
            raise RuntimeError("Detailed audit caused recompilation")
        emit("origin_final_graph_and_logits_verified")


if __name__ == "__main__":
    main()
