"""Read-only probe mechanism audit using original launches and real activations."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from watching.artifacts import write_json
from watching.execution_fit import checkpoint_provenance, tensor_digest
from watching.generated_trace import GeneratedCapture, json_metadata
from watching.runtime import runtime_provenance
from watching.tail_train import difference
from watching.validate_tail import ValidationProbe


class FlatBufferProbe(ValidationProbe):
    """Historical flat/sliced copy style, with a stable oversized allocation."""

    def prepare_buffer(self, inputs):
        super().prepare_buffer(inputs)
        if self.probe_buffer.numel() > 1 << 24:
            raise ValueError("Historical buffer capacity exceeded")
        self.probe_buffer = torch.empty(
            1 << 24, device=self.probe_buffer.device, dtype=self.probe_buffer.dtype
        )

    def record(self, module, inputs, output):
        if self.mode == "buffer":
            self.probe_buffer[: output.numel()].copy_(output.reshape(-1))
        else:
            super().record(module, inputs, output)


def compare_captures(left, right):
    if set(left.snapshots) != set(right.snapshots):
        raise RuntimeError("Generated capture labels differ across conditions")
    rows = []
    for label, value in left.snapshots.items():
        other = right.snapshots[label]
        rows.append(
            {
                "label": label,
                **difference(value, other),
                "left_sha256": tensor_digest(value),
                "right_sha256": tensor_digest(other),
            }
        )
    return {
        "first_unequal_observed_boundary": next(
            (row["label"] for row in rows if not row["bit_identical"]), None
        ),
        "boundaries": rows,
    }


def external_inputs(capture):
    """Identify the three physical query-path inputs, accounting for materialized stages."""
    produced = set()
    result = {}
    bindings = {}
    q_numel = capture.snapshots[f"layer{capture.probe_layer:02d}.q_projection"].numel()
    for index, stage in capture.pipeline.items():
        for argument, value in stage["inputs"].items():
            buffer = stage["buffer_names"][argument]
            if buffer in produced:
                continue
            if value.dtype == torch.bfloat16 and value.numel() == q_numel:
                role = "query_projection"
            elif value.dtype == torch.bfloat16 and value.ndim == 1:
                role = "norm_weight"
            elif value.dtype == torch.float32:
                role = "rotary_angles"
            else:
                raise RuntimeError("Unrecognized external query-pipeline tensor")
            if role in result and not torch.equal(result[role].reshape(-1), value.reshape(-1)):
                raise RuntimeError("Conflicting physical inputs for query pipeline")
            result[role] = value
            bindings[(index, argument)] = role
        produced.update(stage["buffer_names"][name] for name in stage["outputs"])
    if set(result) != {"query_projection", "norm_weight", "rotary_angles"}:
        raise RuntimeError("Could not identify all three query-path inputs")
    return result, bindings


def replay_selected(capture, physical_inputs):
    """Launch the original selected kernel(s), preserving intermediate materialization."""
    _, bindings = external_inputs(capture)
    workspace = {}
    stage_outputs = {}
    for index, stage in capture.pipeline.items():
        for argument, template in stage["inputs"].items():
            buffer = stage["buffer_names"][argument]
            if (index, argument) in bindings:
                value = physical_inputs[bindings[(index, argument)]]
                if template.dtype != value.dtype or template.numel() != value.numel():
                    raise RuntimeError("Matched-input replay shape/dtype mismatch")
                workspace[buffer] = torch.empty_like(template, device="cuda")
                workspace[buffer].copy_(value.reshape(template.shape))
            elif buffer not in workspace:
                raise RuntimeError("Missing intermediate kernel dependency")
        for argument, template in stage["outputs"].items():
            buffer = stage["buffer_names"][argument]
            if not argument.startswith("in_out_ptr"):
                workspace[buffer] = torch.full_like(template, float("nan"), device="cuda")
        names = stage["metadata"]["argument_names"][: len(stage["buffer_names"])]
        positional = [workspace[stage["buffer_names"][name]] for name in names]
        stage["object"].run(
            *positional, *stage["scalars"], stream=torch.cuda.current_stream().cuda_stream
        )
        torch.cuda.synchronize()
        outputs = {
            name: workspace[stage["buffer_names"][name]].cpu().clone() for name in stage["outputs"]
        }
        if not all(torch.isfinite(value).all() for value in outputs.values()):
            raise RuntimeError("Selected replay produced nonfinite or unwritten values")
        stage_outputs[index] = outputs
    return list(stage_outputs[max(stage_outputs)].values())[-1], stage_outputs


def kernel_comparison(left, right):
    physical = []
    for capture in (left, right):
        inputs, _ = external_inputs(capture)
        physical.append(inputs)
        _, replay = replay_selected(capture, inputs)
        if any(
            not torch.equal(replay[index][name], value)
            for index, stage in capture.pipeline.items()
            for name, value in stage["outputs"].items()
        ):
            raise RuntimeError("Own-input kernel replay differs from actual captured execution")
    rows = []
    for origin, inputs in zip(("none", "intervention"), physical, strict=True):
        a, _ = replay_selected(left, inputs)
        b, _ = replay_selected(right, inputs)
        rows.append({"input_origin": origin, **difference(a, b)})
    return {
        "own_input_replay_bit_identical": True,
        "matched_input_comparisons": rows,
        "input_differences": {
            name: difference(value.reshape(-1), physical[1][name].reshape(-1))
            for name, value in physical[0].items()
        },
    }


def save_tensor(path, value):
    temporary = path.with_name(path.name + ".partial")
    torch.save(value, temporary)
    temporary.replace(path)


def query_intervention(compiled, batch, mode, layers, probe_layer, donor, expected, out=None):
    """Explicit causal replacement at an existing tensor; never labeled passive capture."""
    with torch.no_grad():
        with GeneratedCapture(layers, probe_layer, replace_query=donor):
            logits = compiled(**batch, mode=mode)[0].clone().cpu()
    if out is not None:
        save_tensor(out, logits)
    return {
        "replacement": "existing attention query tensor only",
        "difference_from_donor_condition_logits": difference(logits, expected),
    }


def capture_one(compiled, batch, mode, layers, probe_layer):
    with torch.no_grad():
        baseline = compiled(**batch, mode=mode)[0].clone()
        # A second untraced call verifies ordinary repeatability before observation.
        repeated = compiled(**batch, mode=mode)[0]
        if not torch.equal(baseline, repeated):
            raise RuntimeError("Untraced native execution is not bit-identically repeatable")
        with GeneratedCapture(layers, probe_layer) as captured:
            observed = compiled(**batch, mode=mode)[0].clone()
        after = compiled(**batch, mode=mode)[0]
    if not torch.equal(baseline, observed) or not torch.equal(baseline, after):
        raise RuntimeError("Python launch-boundary capture changes full-vocabulary logits")
    q = captured.snapshots[f"layer{probe_layer:02d}.query_rotary"]
    native = list(captured.pipeline[max(captured.pipeline)]["outputs"].values())[-1]
    if not torch.equal(q, native):
        raise RuntimeError("Selected kernel output is not the attention query tensor")
    return baseline.cpu(), captured


def save_capture(out, capture, logits=None):
    out.mkdir(parents=True, exist_ok=True)
    # Numerical evidence comes first; ancillary metadata cannot erase a valid capture.
    save_tensor(
        out / "kernel_tensors.pt",
        {
            index: {"inputs": stage["inputs"], "outputs": stage["outputs"]}
            for index, stage in capture.pipeline.items()
        },
    )
    if logits is not None:
        save_tensor(out / "logits.pt", logits)
    save_tensor(
        out / "probe_layer_boundaries.pt",
        {
            name: value
            for name, value in capture.snapshots.items()
            if name.startswith(f"layer{capture.probe_layer:02d}.")
        },
    )
    (out / "generated_wrapper.py.txt").write_text(capture.wrapper_source)
    stages = []
    for index, stage in capture.pipeline.items():
        metadata = dict(stage["metadata"])
        ptx = metadata.pop("ptx")
        if ptx is not None:
            (out / f"selected_kernel_{index}.ptx").write_text(ptx)
        stages.append(
            {
                "index": index,
                "kernel": metadata,
                "scalars": stage["scalars"],
                "buffer_names": stage["buffer_names"],
                "ptx_sha256": hashlib.sha256(ptx.encode()).hexdigest() if ptx is not None else None,
            }
        )
    evidence = {**capture.provenance(), "full_vocab_capture_parity": True}
    try:
        evidence["stages"] = json_metadata(stages)
        evidence["metadata_status"] = "serialized"
    except (TypeError, ValueError) as error:
        evidence["metadata_status"] = "unsupported"
        evidence["metadata_error"] = str(error)
    write_json(out / "capture.json", evidence)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="Qwen/Qwen3-8B")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prompts", type=int, default=4)
    parser.add_argument("--length", type=int, default=96)
    parser.add_argument("--probe-layer", type=int, default=14)
    parser.add_argument("--include-flat", action="store_true")
    parser.add_argument("--reference-only", action="store_true")
    args = parser.parse_args()
    if args.reference_only and args.include_flat:
        raise ValueError("Reference-only audit excludes optional flat-buffer comparisons")
    if not 1 <= args.prompts <= 4:
        raise ValueError("This focused audit is capped at four training/development prompts")
    from torch._inductor import config

    config.freezing = False
    config.force_disable_caches = True
    torch.manual_seed(20261019)
    args.out.mkdir(parents=True, exist_ok=True)
    log = (args.out / "metrics.jsonl").open("w")

    def emit(event, **values):
        text = json.dumps({"event": event, **values})
        print(text, flush=True)
        log.write(text + "\n")
        log.flush()

    dataset = json.loads(args.data.read_text())
    records = [row for row in dataset["train"] if row.get("task", "detection") == "detection"]
    records = records[: args.prompts]
    if len(records) != args.prompts:
        raise ValueError("Insufficient detection training prompts; no test-set fallback")
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
            raise ValueError("Prompt exceeds fixed length; refusing truncation")
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
    layers = len(model.model.layers)
    write_json(
        args.out / "manifest.json",
        {
            "checkpoint": args.checkpoint,
            "revision": args.revision,
            "model_provenance": checkpoint_provenance(
                args.checkpoint, getattr(model.config, "_commit_hash", None), revision=args.revision
            ),
            "runtime": runtime_provenance(),
            "probe_layer": args.probe_layer,
            "length": args.length,
            "n_prompts": len(records),
            "weights": "original base",
            "scope": "New matched compilation; no claim of reproducing previous selected config",
            "no_training": True,
            "data_split": "first fixed detection training records",
            "reference_only": args.reference_only,
            "include_flat": args.include_flat,
        },
    )
    canonical_logits = []
    styles = [("shaped", ValidationProbe)]
    if args.include_flat:
        styles.append(("flat", FlatBufferProbe))
    for style, cls in styles:
        wrapper = cls(model, args.probe_layer)
        wrapper.prepare_buffer(batches[0])
        compiled = torch.compile(wrapper, fullgraph=True, dynamic=False)
        emit("compiling", style=style)
        for index, batch in enumerate(batches):
            absent_logits, absent = capture_one(compiled, batch, "none", layers, args.probe_layer)
            if style == "shaped":
                canonical_logits.append(absent_logits)
            elif not torch.equal(canonical_logits[index], absent_logits):
                raise RuntimeError(
                    "Changing buffer wrapper changes no-probe logits; style contrast confounded"
                )
            emit("capture_verified", style=style, prompt_index=index, mode="none")
            save_capture(args.out / style / "none" / f"prompt_{index:02d}", absent, absent_logits)
            modes = (
                (("reference",) if args.reference_only else ("reference", "buffer"))
                if style == "shaped"
                else ("buffer",)
            )
            for mode in modes:
                present_logits, present = capture_one(
                    compiled, batch, mode, layers, args.probe_layer
                )
                emit("capture_verified", style=style, prompt_index=index, mode=mode)
                case = args.out / style / mode / f"prompt_{index:02d}"
                save_capture(case / "none", absent, absent_logits)
                save_capture(case / mode, present, present_logits)
                save_tensor(case / "logits.pt", {"none": absent_logits, mode: present_logits})
                comparison = compare_captures(absent, present)
                comparison["logits"] = difference(absent_logits, present_logits)
                comparison["stage"] = "passive_captures_verified"
                write_json(case / "comparison.json", comparison)
                label = f"layer{args.probe_layer:02d}.query_rotary"
                for own_mode, own_capture, own_logits in (
                    ("none", absent, absent_logits),
                    (mode, present, present_logits),
                ):
                    own = query_intervention(
                        compiled,
                        batch,
                        own_mode,
                        layers,
                        args.probe_layer,
                        own_capture.snapshots[label],
                        own_logits,
                        out=case / f"identity_{own_mode}_logits.pt",
                    )
                    if not own["difference_from_donor_condition_logits"]["bit_identical"]:
                        raise RuntimeError(
                            "Identity query replacement changes full-vocabulary logits"
                        )
                comparison["identity_query_replacement_parity"] = True
                comparison["stage"] = "identity_swaps_verified"
                write_json(case / "comparison.json", comparison)
                comparison["query_replacements"] = {
                    "intervention_query_into_none": query_intervention(
                        compiled,
                        batch,
                        "none",
                        layers,
                        args.probe_layer,
                        present.snapshots[label],
                        present_logits,
                        out=case / "intervention_query_into_none_logits.pt",
                    ),
                    "none_query_into_intervention": query_intervention(
                        compiled,
                        batch,
                        mode,
                        layers,
                        args.probe_layer,
                        absent.snapshots[label],
                        absent_logits,
                        out=case / "none_query_into_intervention_logits.pt",
                    ),
                }
                # Replacement must not leave any state behind on the next ordinary call.
                for restore_mode, expected_logits in (
                    ("none", absent_logits),
                    (mode, present_logits),
                ):
                    with torch.no_grad():
                        restored = compiled(**batch, mode=restore_mode)[0].cpu()
                    if not torch.equal(restored, expected_logits):
                        raise RuntimeError(
                            "Query intervention changed subsequent ordinary execution"
                        )
                comparison["stage"] = "query_swaps_and_restoration_verified"
                write_json(case / "comparison.json", comparison)
                emit(
                    "query_swaps_complete",
                    style=style,
                    mode=mode,
                    prompt_index=index,
                    query_replacements=comparison["query_replacements"],
                )
                try:
                    comparison["kernel_replay"] = kernel_comparison(absent, present)
                    comparison["kernel_replay"]["status"] = "passed"
                except (RuntimeError, ValueError, KeyError, TypeError, AttributeError) as error:
                    comparison["kernel_replay"] = {
                        "status": "failed_or_unsupported",
                        "error": str(error),
                        "scope": "No selected-kernel replay conclusion; "
                        "prior independent query swaps retained",
                    }
                comparison["stage"] = "complete"
                write_json(case / "comparison.json", comparison)
                selected = {label, f"layer{args.probe_layer:02d}.attention_output"}
                first = comparison["first_unequal_observed_boundary"]
                if first is not None:
                    selected.add(first)
                save_tensor(
                    case / "selected_boundaries.pt",
                    {
                        label: {"none": absent.snapshots[label], mode: present.snapshots[label]}
                        for label in selected
                    },
                )
                emit(
                    "case_complete",
                    style=style,
                    mode=mode,
                    prompt_index=index,
                    first_unequal_boundary=first,
                    logits=comparison["logits"],
                    replay=comparison["kernel_replay"],
                    query_replacements=comparison["query_replacements"],
                )
                del present
            del absent
        wrapper.handle.remove()
        del compiled, wrapper
    emit("complete")
    log.close()


if __name__ == "__main__":
    main()
