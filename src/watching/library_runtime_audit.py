"""Two bare compilations with outside-graph argument evidence; zero detector scoring."""

import argparse
import json
import os
import traceback
from pathlib import Path

import torch
from safetensors.torch import save_file

from watching.artifacts import write_json
from watching.library_call_audit import comparison, evaluate, raw_kwargs
from watching.library_runtime_probe import RuntimeArgumentProbe, model_record, tensor_set
from watching.runtime import runtime_provenance
from watching.tail_train import file_digest


def content(record):
    """Compare bytes and tensor metadata; pointer equality is not value equality."""
    if isinstance(record, dict):
        return {
            key: content(value)
            for key, value in record.items()
            if key not in {"storage_pointer", "data_pointer", "version", "alias_group"}
        }
    if isinstance(record, list):
        return [content(value) for value in record]
    return record


def selected_configs(root):
    return {
        str(path.relative_to(root)): json.loads(path.read_text())
        for path in sorted(root.rglob("*.best_config"))
    }


def run_runtime_matrix(model, inputs, out, report, answer_ids, retained_reference=None):
    import torch._inductor.config as config
    from torch._dynamo.backends.registry import lookup_backend
    from torch._dynamo.utils import counters

    if model.config.output_hidden_states is not False:
        raise ValueError("Omitted hidden-state flag must mean False")
    if any(module._forward_hooks or module._forward_pre_hooks for module in model.modules()):
        raise ValueError("Bare runtime starts with a registered hook")
    expected = tensor_set(inputs.items())
    report.update(observer_changes_timing_and_allocation=True, arms=[], runtime_calls=[])
    native, argument_sets, constant_sets = [], [], []
    for index in range(2):
        name = f"bare_{index + 1}"
        root = out / name
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str((root / "compile-cache").resolve())
        os.environ["TRITON_CACHE_DIR"] = str((root / "triton-cache").resolve())
        before = model_record(model)
        if index == 0:
            for key, value in before["tensors"].items():
                kind, tensor_name = key.split(":", 1)
                expected_hash = (
                    report["state_before"]["tensors"][tensor_name]
                    if kind == "parameter"
                    else report["cpu_buffer_sha256"][tensor_name]
                )
                if value["sha256"] != expected_hash:
                    raise RuntimeError("CPU-to-CUDA model transfer changed " + key)
            report["model_transfer_hashes_equal"] = True
        input_before = tensor_set(inputs.items())
        if content(input_before) != content(expected):
            raise RuntimeError("User input changed before compilation")
        write_json(out / f"{name}-before.json", before)
        torch._dynamo.reset()
        counters.clear()
        probe = RuntimeArgumentProbe(root / "runtime-arguments", lookup_backend("inductor"))
        with (
            config.patch(force_disable_caches=True, fx_graph_cache=False),
            torch._dynamo.config.patch(
                skip_nnmodule_hook_guards=False,
                suppress_errors=False,
                fail_on_recompile_limit_hit=True,
            ),
            torch.no_grad(),
        ):
            call = torch.compile(model, backend=probe, fullgraph=True, dynamic=False)
            first = call(**inputs, **raw_kwargs(False)).logits.detach().cpu().clone()
            first_stats = dict(counters["stats"])
            second = call(**inputs, **raw_kwargs(False)).logits.detach().cpu().clone()
            second_stats = dict(counters["stats"])
        after = model_record(model)
        write_json(out / f"{name}-after.json", after)
        if tuple(first.shape) != (1, inputs["input_ids"].shape[1], model.config.vocab_size):
            raise RuntimeError("Unexpected native full-sequence output shape")
        save_file(
            {"logits": first.contiguous(), "repeat_logits": second.contiguous()},
            str(out / f"{name}.safetensors"),
        )
        calls = [
            json.loads(path.read_text())
            for path in sorted((root / "runtime-arguments").glob("call-*.json"))
        ]
        argument_sets.extend(content(call["before"]) for call in calls)
        constants = [
            json.loads(path.read_text())
            for path in (root / "runtime-arguments").glob("graph-*-constants.json")
        ]
        constant_sets.append(content(constants))
        no_id, yes_id = answer_ids
        arm = {
            "name": name,
            "repeat": comparison(first, second),
            "one_graph_two_calls": probe.graphs == 1 and probe.calls == 2,
            "single_graph_reused": first_stats == second_stats
            and first_stats.get("unique_graphs") == 1,
            "model_values_unchanged": content(before) == content(after),
            "user_input_values_unchanged": content(expected) == content(tensor_set(inputs.items())),
            "actual_arguments_unchanged": all(
                content(call["before"]) == content(call["after"]) for call in calls
            ),
            "actual_argument_coverage_complete": len(calls) == 2
            and all(
                call["before"]["coverage_complete"] and call["after"]["coverage_complete"]
                for call in calls
            ),
            "graph_constants": constants,
            "graph_constants_complete": len(constants) == 1
            and all(
                not any(
                    item.get("unverified_fake_tensor") or item.get("unverified_nonprimitive")
                    for item in value.values()
                )
                for value in constants
            ),
            "selected_autotune_configs": selected_configs(root),
            "selected_config_evidence_status": (
                "sidecars_present_not_yet_cross_arm_matched"
                if selected_configs(root)
                else "unavailable_no_sidecars_found"
            ),
            "runtime": runtime_provenance(),
            "cuda_stream": torch.cuda.current_stream().cuda_stream,
            "final_argmax": int(first[0, -1].argmax()),
            "final_yes_no_gap": float(first.float()[0, -1, yes_id] - first.float()[0, -1, no_id]),
        }
        report["arms"].append(arm)
        native.append(first)
        report["actual_runtime_argument_values_equal_all_calls"] = (
            len(argument_sets) == 4
            and all(value["coverage_complete"] for value in argument_sets)
            and all(value == argument_sets[0] for value in argument_sets)
        )
        report["graph_constant_values_equal_across_compilations"] = (
            len(constant_sets) == 2
            and all(arm["graph_constants_complete"] for arm in report["arms"])
            and constant_sets[0] == constant_sets[1]
        )
        if len(native) == 2:
            report["independent_bare_comparison"] = comparison(*native)
        report["argument_evidence_complete"] = (
            len(report["arms"]) == 2
            and report["actual_runtime_argument_values_equal_all_calls"]
            and report["graph_constant_values_equal_across_compilations"]
            and all(
                arm["one_graph_two_calls"]
                and arm["single_graph_reused"]
                and arm["actual_argument_coverage_complete"]
                and arm["actual_arguments_unchanged"]
                and arm["model_values_unchanged"]
                and arm["user_input_values_unchanged"]
                for arm in report["arms"]
            )
        )
        write_json(out / "report.json", report)
        print(json.dumps({"completed_arm": name, "repeat": arm["repeat"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fit", "tail", "failure", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--tail-sha256", required=True)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("Output must be empty")
    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "running",
        "purpose": __doc__,
        "script_sha256": file_digest(Path(__file__)),
        "helper_source_sha256": {
            name: file_digest(Path(__file__).with_name(name))
            for name in ("library_runtime_probe.py", "library_call_audit.py", "library_validate.py")
        },
        "independent_processes": False,
        "primary_validator_gate_unchanged": True,
    }
    try:
        evaluate(args, report, matrix=run_runtime_matrix)
        report["status"] = "complete"
    except Exception:
        report.update(status="failed", failure=traceback.format_exc())
        raise
    finally:
        write_json(args.out / "report.json", report)


if __name__ == "__main__":
    main()
