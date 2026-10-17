"""Decompose call-signature and disabled-hook compilation effects; no detector scoring.

Each arm is a separate cold compilation of one loaded fixed model, not a separate
process. The primary fresh validator and its admission gate remain unchanged.
"""

import argparse
import importlib.metadata
import json
import os
import time
import traceback
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from watching.artifacts import write_json
from watching.execution_fit import tensor_digest
from watching.library_train import byte_equal
from watching.library_validate import (
    MODEL,
    REVISION,
    encode_record,
    state_digest,
    validate_data_files,
    validate_source,
)
from watching.runtime import runtime_provenance
from watching.tail_train import file_digest, record_answer_ids
from watching.validate_tail import install_checkpoint

ARMS = (
    ("bare_omitted_1", False, False),
    ("bare_false", False, True),
    ("disabled_omitted", True, False),
    ("disabled_false", True, True),
    ("bare_omitted_2", False, False),
)


def raw_kwargs(explicit_false):
    result = {"use_cache": False}
    if explicit_false:
        result["output_hidden_states"] = False
    return result


class DisabledHook:
    """Same read-only callback body as LibraryInvocation; activation reading stays off."""

    def __init__(self, module):
        self.enabled, self.saved = False, {}

        def record(_module, _inputs, output):
            if self.enabled:
                self.saved["activation"] = output.detach()

        self.handle = module.register_forward_hook(record)

    def remove(self):
        self.handle.remove()


def comparison(a, b):
    if (
        a.shape != b.shape
        or a.dtype != b.dtype
        or not (torch.isfinite(a).all() and torch.isfinite(b).all())
    ):
        raise ValueError("Invalid native tensor comparison")
    d = b.float() - a.float()
    return {
        "bit_identical": byte_equal(a, b),
        "different_logits": int((d != 0).sum()),
        "max_abs_difference": float(d.abs().max()),
        "different_final_position_logits": int((d[0, -1] != 0).sum()),
        "max_final_position_abs_difference": float(d[0, -1].abs().max()),
    }


def runtime_invariants(model):
    return {
        "parameters": {
            name: {"requires_grad": p.requires_grad, "dtype": str(p.dtype), "device": str(p.device)}
            for name, p in model.named_parameters()
        },
        "module_training_flags": {name: m.training for name, m in model.named_modules()},
    }


def run_matrix(model, inputs, out, report, answer_ids, retained_reference=None):
    import torch._inductor.config as config
    from torch._dynamo.utils import counters

    if model.config.output_hidden_states is not False:
        raise ValueError("Omitted hidden-state flag must mean False")
    no_id, yes_id = answer_ids
    report["answer_ids"] = {"No": no_id, "Yes": yes_id}
    if any(m.training for m in model.modules()):
        raise ValueError("All native modules must be eval mode")
    invariants = runtime_invariants(model)
    report["runtime_invariants_before"] = invariants
    module = model.model.layers[14].self_attn.q_norm
    if module._forward_hooks:
        raise ValueError("Target starts with a registered hook")
    eager_reference, native = None, {}
    report["config_output_hidden_states"] = model.config.output_hidden_states
    report["arms"] = []
    for name, registered, explicit in ARMS:
        owner = DisabledHook(module) if registered else None
        try:
            kwargs = raw_kwargs(explicit)
            with torch.no_grad():
                eager = model(**inputs, **kwargs).logits.detach().cpu().clone()
            if eager_reference is None:
                eager_reference = eager
            eager_check = comparison(eager_reference, eager)
            if not eager_check["bit_identical"]:
                raise RuntimeError("Eager call variants differ before compilation")
            torch._dynamo.reset()
            counters.clear()
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = str((out / name / "compile-cache").resolve())
            os.environ["TRITON_CACHE_DIR"] = str((out / name / "triton-cache").resolve())
            with (
                config.patch(force_disable_caches=True, fx_graph_cache=False),
                torch._dynamo.config.patch(
                    skip_nnmodule_hook_guards=False,
                    suppress_errors=False,
                    fail_on_recompile_limit_hit=True,
                ),
                torch.no_grad(),
            ):
                call = torch.compile(model, backend="inductor", fullgraph=True, dynamic=False)
                first = call(**inputs, **kwargs).logits.detach().cpu().clone()
                first_stats = dict(counters["stats"])
                second = call(**inputs, **kwargs).logits.detach().cpu().clone()
                second_stats = dict(counters["stats"])
            if owner is not None and owner.saved:
                raise RuntimeError("Disabled hook read an activation")
            expected_shape = (1, inputs["input_ids"].shape[1], model.config.vocab_size)
            if tuple(first.shape) != expected_shape or tuple(second.shape) != expected_shape:
                raise RuntimeError("Unexpected full-sequence/vocabulary logits shape")
            native[name] = first
            save_file(
                {"logits": first.contiguous(), "repeat_logits": second.contiguous()},
                str(out / (name + ".safetensors")),
            )
            record = {
                "name": name,
                "hook_registered": registered,
                "explicit_kwargs": kwargs,
                "logits_sha256": tensor_digest(first),
                "final_top_token_id": int(first[0, -1].argmax()),
                "final_yes_no_log_odds": float(
                    first.float()[0, -1, yes_id] - first.float()[0, -1, no_id]
                ),
                "actual_cache_directories": {
                    k: os.environ.get(k) for k in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR")
                },
                "repeat": comparison(first, second),
                "first_stats": first_stats,
                "second_stats": second_stats,
                "single_graph_reused": first_stats.get("unique_graphs", 0)
                == second_stats.get("unique_graphs", 0)
                == 1
                and first_stats == second_stats,
                "eager_parity": eager_check,
                "counters": {k: dict(v) for k, v in counters.items() if v},
            }
            record["runtime_invariants_unchanged"] = runtime_invariants(model) == invariants
            if not record["runtime_invariants_unchanged"]:
                raise RuntimeError(
                    "Parameter flags/dtypes/devices or module training flags changed"
                )
            report["arms"].append(record)
            report["comparisons"] = {
                a + "_vs_" + b: comparison(native[a], native[b])
                for a, b in [
                    ("bare_omitted_1", "bare_false"),
                    ("bare_omitted_1", "disabled_omitted"),
                    ("bare_false", "disabled_false"),
                    ("disabled_omitted", "disabled_false"),
                    ("bare_omitted_1", "disabled_false"),
                    ("bare_omitted_1", "bare_omitted_2"),
                ]
                if a in native and b in native
            }
            report["attribution_admissible"] = (
                len(report["arms"]) == len(ARMS)
                and all(
                    r["repeat"]["bit_identical"] and r["single_graph_reused"]
                    for r in report["arms"]
                )
                and report["comparisons"]
                .get("bare_omitted_1_vs_bare_omitted_2", {})
                .get("bit_identical", False)
            )
            if retained_reference:
                report["retained_failure_comparisons"] = {
                    arm: comparison(reference, native[arm])
                    for arm, reference in retained_reference.items()
                    if arm in native
                }
            write_json(out / "report.json", report)
            print(json.dumps({"completed_arm": name, **record["repeat"]}), flush=True)
        finally:
            if owner is not None:
                owner.remove()
    if module._forward_hooks:
        raise RuntimeError("Diagnostic hook was not removed")


def evaluate(args, report):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU fallback")
    for package, expected in [("torch", "2.10.0"), ("transformers", "5.18.0")]:
        if importlib.metadata.version(package).split("+")[0] != expected:
            raise RuntimeError("Unexpected " + package + " version")
    manifest = json.loads((args.fit / "manifest.json").read_text())
    data = json.loads((args.fit / "data.json").read_text())
    report["data_provenance"] = validate_data_files(args.fit / "data.json", manifest)
    if file_digest(args.tail) != args.tail_sha256:
        raise ValueError("Final tail pin mismatch")
    artifact = torch.load(args.tail, map_location="cpu", weights_only=True)
    validate_source(manifest, artifact, "forward_hook", data)
    failure = torch.load(args.failure, map_location="cpu", weights_only=True)
    inputs = failure["input"]
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = encode_record(tokenizer, data["validation"][0], manifest["length"])
    if set(inputs) != set(encoded) or any(not byte_equal(inputs[k], encoded[k]) for k in inputs):
        raise ValueError("Counterexample inputs differ from selected validation prompt0")
    write_json(args.out / "inputs.json", {k: v.tolist() for k, v in inputs.items()})
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).eval()
    report["installation"] = install_checkpoint(model, artifact, MODEL, 14)
    reference = {
        "bare_omitted_1": failure["original_logits"],
        "bare_omitted_2": failure["original_logits"],
        "disabled_false": failure["extracted_logits"],
    }
    del artifact, failure
    model.requires_grad_(True)
    report["config_dump"] = json.loads(model.config.to_json_string())
    report["all_parameters_require_grad"] = all(p.requires_grad for p in model.parameters())
    report["parameter_dtypes"] = sorted({str(p.dtype) for p in model.parameters()})
    report.update(
        artifact_sha256=args.tail_sha256,
        failure_sha256=file_digest(args.failure),
        runtime=runtime_provenance(),
        state_before=state_digest(model),
    )
    model.cuda()
    write_json(args.out / "report.json", report)
    answer_ids = record_answer_ids(tokenizer, data["validation"][0])
    run_matrix(
        model, {k: v.cuda() for k, v in inputs.items()}, args.out, report, answer_ids, reference
    )
    report["state_after"] = state_digest(model)
    if report["state_before"] != report["state_after"]:
        raise RuntimeError("Native model state changed")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("fit", "tail", "failure", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--tail-sha256", required=True)
    args = p.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        p.error("Output must be empty")
    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "running",
        "purpose": "signature/hook decomposition only; zero detector scoring",
        "primary_validator_gate_unchanged": True,
        "independent_processes": False,
        "independent_compilations": True,
        "force_disable_caches": True,
        "started_unix": time.time(),
        "script_sha256": file_digest(Path(__file__)),
        "helper_source_sha256": {
            name: file_digest(Path(__file__).with_name(name))
            for name in (
                "library_validate.py",
                "library_train.py",
                "validate_tail.py",
                "tail_train.py",
                "execution_fit.py",
            )
        },
        "arms": [],
    }
    try:
        evaluate(args, report)
        report["status"] = "complete"
    except Exception:
        report.update(status="failed", failure=traceback.format_exc())
        raise
    finally:
        report["finished_unix"] = time.time()
        write_json(args.out / "report.json", report)


if __name__ == "__main__":
    main()
