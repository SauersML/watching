"""Matched native-tail training with identical numerical inputs under both labels.

This is a cached-input plumbing control. It makes no whole-model serving claim.
The source experiment supplies all data, shapes, initialization and training settings.
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM, AutoTokenizer

from watching.artifacts import write_json
from watching.execution_fit import tensor_digest
from watching.runtime import runtime_provenance
from watching.suffix import move_state
from watching.tail_train import (
    EagerNativeTail,
    PairedMasterAdam,
    check_gradient_forward,
    context_key,
    file_digest,
    prompt_batches,
    record_answer_ids,
    score_record,
    state_fingerprint,
    summarize,
    summarize_tasks,
)


def validate_cache(cache, data, fingerprints, manifest):
    """Reject mismatched records, conditions, positional inputs or tensor fingerprints."""
    if cache.get("version") != 1 or cache.get("conditions") != [False, True]:
        raise ValueError("Expected a version-one absent/present source cache")
    if cache.get("checkpoint") != manifest["checkpoint"] or cache.get("data") != data:
        raise ValueError("Cache model or selected data does not match the source run")
    if cache.get("length") != manifest["length"]:
        raise ValueError("Cache sequence length differs from the source manifest")
    if not {"train", "validation", "test"} <= set(data):
        raise ValueError("Source cache needs training, validation and endpoint test cohorts")
    if set(cache["states"]) != set(data) or set(cache["batches"]) != set(data):
        raise ValueError("Cache cohorts do not match the selected data")
    if set(fingerprints) != set(data):
        raise ValueError("Cache fingerprint cohorts differ")
    for split, records in data.items():
        pairs, batches, digests = (
            cache["states"][split],
            cache["batches"][split],
            fingerprints[split],
        )
        if not len(records) == len(pairs) == len(batches) == len(digests):
            raise ValueError("Cache record counts differ")
        for index, (record, pair, batch, digest) in enumerate(
            zip(records, pairs, batches, digests, strict=True)
        ):
            if len(pair) != 2 or any(len(state) != 4 for state in pair):
                raise ValueError("Every prompt requires two four-tensor states")
            if digest["index"] != index or digest["context_id"] != context_key(record):
                raise ValueError("Fingerprint record identity differs")
            positions = batch["position_ids"]
            expected_positions = (batch["attention_mask"].long().cumsum(-1) - 1).clamp_min(0)
            if not torch.equal(positions, expected_positions):
                raise ValueError("Position IDs do not match the saved attention mask")
            if tuple(positions.shape) != (1, manifest["length"]):
                raise ValueError("Expected the original singleton sequence shape")
            if tensor_digest(positions) != digest["positions_sha256"]:
                raise ValueError("Position fingerprint differs")
            for name, state in zip(("absent", "present"), pair, strict=True):
                if state_fingerprint(state) != digest[name]:
                    raise ValueError("Cached-state fingerprint differs")
                hidden = state[0]
                if (
                    hidden.ndim != 3
                    or tuple(hidden.shape[:2]) != tuple(positions.shape)
                    or hidden.dtype != torch.bfloat16
                    or any(t is not None and t.device.type != "cpu" for t in state)
                    or any(t is not None and not torch.isfinite(t).all() for t in state)
                ):
                    raise ValueError("Invalid cached native hidden state or auxiliary tensor")
            if any(
                (a is None) != (b is None) or (a is not None and not torch.equal(a, b))
                for a, b in zip(pair[0][1:], pair[1][1:], strict=True)
            ):
                raise ValueError("Source conditions differ in auxiliary inputs")


def identical_pair(pair):
    """Copy the absent numerical input twice, preserving its dtype and layout."""
    absent = pair[0]
    result = tuple(
        tuple(None if t is None else t.detach().clone() for t in absent) for _ in range(2)
    )
    for a, b in zip(*result, strict=True):
        if a is not None and (not torch.equal(a, b) or a.stride() != b.stride()):
            raise RuntimeError("Identical-input construction changed values or layout")
    return result


def check_null_pair(absent, present, answer_ids, tolerance=1e-5):
    """Check the exact null and the balanced two-label full-vocabulary loss bound."""
    if not torch.isfinite(absent).all() or not torch.equal(absent, present):
        raise RuntimeError("Identical numerical inputs produced different or nonfinite logits")
    if len(set(answer_ids)) != 2:
        raise ValueError("Null labels must be distinct native tokens")
    losses = -absent.float().log_softmax(-1)[0, list(answer_ids)]
    loss = float(losses.mean())
    if loss < math.log(2) - tolerance:
        raise RuntimeError("Identical-input paired full-vocabulary CE is below log(2)")
    return loss


def validate_tokenized_records(tokenizer, data, batches, length):
    """Bind saved numerical inputs to the actual task text, not only its metadata."""
    for split, records in data.items():
        for record, saved in zip(records, batches[split], strict=True):
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": record["text"]}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            if len(tokenizer.encode(text, add_special_tokens=False)) > length:
                raise ValueError("Saved task text exceeds the source sequence length")
            rendered = tokenizer(
                text,
                add_special_tokens=False,
                padding="max_length",
                max_length=length,
                return_tensors="pt",
            )
            for key in ("input_ids", "attention_mask"):
                if not torch.equal(rendered[key], saved[key]):
                    raise ValueError(f"Cached {key} does not represent the saved task text")


def source_revision(manifest, requested):
    """Pin new execution without inventing revision metadata for a historical run."""
    original = manifest.get("revision")
    if requested is not None and original is not None and requested != original:
        raise ValueError("Requested revision differs from the source revision")
    return requested or original


def check_source_baseline(rows, archived, count):
    """Compare only the unchanged absent condition, before any optimizer update.

    Historical artifacts contain answer scores, not the complete vocabulary logits.
    Exact answer scores and greedy correctness are the available replay check.
    """

    def absent_by_index(values):
        result = {}
        for row in values:
            if row["condition_present"]:
                continue
            index = row["index"]
            if index in result:
                raise ValueError("Duplicate absent baseline record")
            result[index] = row
        if set(result) != set(range(count)):
            raise ValueError("Incomplete absent baseline cohort")
        return result

    current, previous = absent_by_index(rows), absent_by_index(archived)
    fields = ("yes_minus_no", "logp_no", "logp_yes", "greedy_correct")
    for index in range(count):
        for field in fields:
            if current[index][field] != previous[index][field]:
                raise RuntimeError(
                    f"Source baseline replay differs at index {index}, {field}: "
                    f"{current[index][field]} != {previous[index][field]}"
                )
    return {
        "prompts": count,
        "fields": list(fields),
        "exact": True,
        "scope": "archived absent-condition answer scores; not all vocabulary logits",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--revision", help="Explicit base/tokenizer revision; must agree if source pinned"
    )
    parser.add_argument("--require-source-baseline", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This matched control requires CUDA native-tail execution")
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        raise ValueError("Output directory must be empty")
    source = args.source_results
    manifest = json.loads((source / "manifest.json").read_text())
    if (
        manifest["training_status"] != "complete"
        or manifest["completed_steps"] != manifest["steps"]
        or manifest.get("initial_tail") is not None
        or manifest.get("warm_start_tail") is not None
        or manifest.get("train_head", False)
        or manifest["tail_layers"] != 2
    ):
        raise ValueError("Need a complete base-initialized two-block fixed-head source run")
    revision = source_revision(manifest, args.revision)
    seed, steps, batch_size = manifest["seed"], manifest["steps"], manifest["prompt_batch_size"]
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    data = json.loads((source / "data.json").read_text())
    fingerprints = json.loads((source / "cache_fingerprints.json").read_text())
    cache_path = source / "cache_states.pt"
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    validate_cache(cache, data, fingerprints, manifest)
    if len(data["train"]) % batch_size:
        raise ValueError("Training count must divide into complete matched prompt batches")
    started = time.monotonic()

    def emit(event, **fields):
        record = {"event": event, "seconds": time.monotonic() - started, **fields}
        line = json.dumps(record, allow_nan=False)
        print(line, flush=True)
        with (args.out / "metrics.jsonl").open("a") as stream:
            stream.write(line + "\n")

    emit("cache_verified", source=str(source), original_present_states_ignored=True)
    null = {
        split: [identical_pair(pair) for pair in pairs] for split, pairs in cache["states"].items()
    }
    cache["states"] = null
    write_json(args.out / "data.json", data)
    write_json(
        args.out / "cache_fingerprints.json",
        {
            split: [
                {
                    "index": i,
                    "context_id": context_key(data[split][i]),
                    "positions_sha256": tensor_digest(cache["batches"][split][i]["position_ids"]),
                    "absent": state_fingerprint(pair[0]),
                    "present": state_fingerprint(pair[1]),
                }
                for i, pair in enumerate(pairs)
            ]
            for split, pairs in null.items()
        },
    )
    tokenizer = AutoTokenizer.from_pretrained(
        manifest["checkpoint"], revision=revision, padding_side="left"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    validate_tokenized_records(tokenizer, data, cache["batches"], manifest["length"])
    answers = {s: [record_answer_ids(tokenizer, r) for r in rows] for s, rows in data.items()}
    model = AutoModelForCausalLM.from_pretrained(
        manifest["checkpoint"],
        revision=revision,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).eval()
    model.config.use_cache = False
    if (
        model.config.tie_word_embeddings
        or model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
    ):
        raise ValueError("Control requires the same native untied model")
    head_digest = tensor_digest(model.lm_head.weight)
    if head_digest != manifest["head_sha256_before"]:
        raise ValueError("Loaded base head differs from the source run")
    if cache["cut_layer"] != len(model.model.layers) - 3:
        raise ValueError("Cache cut is incompatible with two native final blocks")
    tail = EagerNativeTail(model, tail_layers=2).cuda()
    del model
    output_manifest = {
        **manifest,
        "source_results": str(source),
        "source_revision_recorded": manifest.get("revision"),
        "revision": revision,
        "load_revision_explicitly_pinned": revision is not None,
        "require_source_baseline": args.require_source_baseline,
        "out": str(args.out),
        "source_cache_sha256": file_digest(cache_path),
        "source_manifest_sha256": file_digest(source / "manifest.json"),
        "selected_data_sha256": file_digest(source / "data.json"),
        "null_control": "Both labels receive a copy of the absent cached numerical state",
        "runtime": "cached identical-input eager BF16 math-SDPA native tail",
        "wholegraph_transfer_evaluated": False,
        "condition_signal_available": False,
        "completed_steps": 0,
        "training_status": "running",
        "processed_prompt_pairs": 0,
        "global_processed_prompt_pairs": 0,
        "ending_global_step": 0,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "runtime_provenance": runtime_provenance(),
        "source_sha256": file_digest(Path(__file__)),
    }
    write_json(args.out / "manifest.json", output_manifest)
    emit("setup", **output_manifest)

    def audit(step):
        for index in range(min(4, len(null["train"]))):
            state = move_state(null["train"][index][0], "cuda")
            positions = cache["batches"]["train"][index]["position_ids"].cuda()
            result = check_gradient_forward(tail, state, positions)
            if not result["bit_identical"]:
                raise RuntimeError("Null tail gradient/inference forward mismatch")
            emit("gradient_forward_audit", step=step, index=index, **result)

    def evaluate(step, splits):
        for split in splits:
            rows = []
            for index, pair in enumerate(null[split]):
                positions = cache["batches"][split][index]["position_ids"].cuda()
                with torch.no_grad():
                    absent = tail(*move_state(pair[0], "cuda"), positions)
                    present = tail(*move_state(pair[1], "cuda"), positions)
                check_null_pair(absent, present, answers[split][index])
                for condition, logits in ((False, absent), (True, present)):
                    rows.append(
                        score_record(
                            logits, index, condition, data[split][index], answers[split][index]
                        )
                    )
            summary = {
                "step": step,
                "global_step": step,
                "processed_prompt_pairs": step * batch_size,
                "split": split,
                "path": "split_inference",
                **summarize(rows, seed),
                "tasks": summarize_tasks(rows, seed),
                "identical_input_outputs_bit_identical": True,
                "null_ce_lower_bound": math.log(2),
            }
            write_json(
                args.out / f"{split}_{step}_split_inference.json",
                {"summary": summary, "rows": rows},
            )
            emit("evaluation", **summary)
            if step == 0 and args.require_source_baseline:
                archived_path = source / f"{split}_0_split_inference.json"
                archived = json.loads(archived_path.read_text())
                if archived["summary"]["step"] != 0 or archived["summary"]["split"] != split:
                    raise ValueError("Archived baseline has incompatible step or split")
                result = check_source_baseline(rows, archived["rows"], len(data[split]))
                emit(
                    "source_baseline_verified",
                    split=split,
                    archived_sha256=file_digest(archived_path),
                    **result,
                )

    audit(0)
    evaluate(0, ("train", "validation"))
    optimizer = PairedMasterAdam(
        tail.active_parameters(), manifest["lr"], expected_contributions=2 * batch_size
    )
    stream = prompt_batches(len(null["train"]), batch_size, seed)
    emit("training_start", steps=steps, prompt_batch_size=batch_size)
    for step in range(1, steps + 1):
        indices = next(stream)
        total_loss = 0.0
        for index in indices:
            positions = cache["batches"]["train"][index]["position_ids"].cuda()
            for present in (False, True):
                logits = tail(
                    *move_state(null["train"][index][int(present)], "cuda"), positions
                ).float()
                label = torch.tensor([answers["train"][index][int(present)]], device=logits.device)
                loss = F.cross_entropy(logits, label) / (2 * batch_size)
                with sdpa_kernel(SDPBackend.MATH):
                    loss.backward()
                optimizer.accumulate()
                total_loss += float(loss.detach())
        norm, movement = optimizer.step()
        emit(
            "train",
            step=step,
            global_step=step,
            prompt_indices=indices,
            processed_prompt_pairs=step * batch_size,
            loss=total_loss,
            gradient_norm=norm,
            actual_weight_update_l2=movement,
        )
        if (
            step % manifest["eval_every"] == 0
            or step in manifest["early_eval_steps"]
            or step == steps
        ):
            audit(step)
            evaluate(step, ("train", "validation"))
    del optimizer, loss, logits
    if tensor_digest(tail.head_weight) != head_digest:
        raise RuntimeError("Frozen native head changed in the null control")
    torch.save(
        {
            "version": 1,
            "checkpoint": manifest["checkpoint"],
            "first_layer": tail.first_layer,
            "layers": {k: v.detach().cpu() for k, v in tail.layers.state_dict().items()},
            "norm": {k: v.detach().cpu() for k, v in tail.norm.state_dict().items()},
            "head_sha256": head_digest,
            "null_control": True,
            "completed_steps": steps,
        },
        args.out / "native_tail.pt",
    )
    output_manifest.update(
        completed_steps=steps,
        ending_global_step=steps,
        processed_prompt_pairs=steps * batch_size,
        global_processed_prompt_pairs=steps * batch_size,
        training_status="complete",
    )
    write_json(args.out / "manifest.json", output_manifest)
    emit("tail_checkpoint_saved", step=steps)
    evaluate(steps, tuple(split for split in null if split not in ("train", "validation")))
    emit(
        "complete",
        status="complete",
        step=steps,
        head_unchanged=True,
        identical_input_outputs_bit_identical=True,
        max_gpu_bytes=torch.cuda.max_memory_allocated(),
    )


if __name__ == "__main__":
    main()
