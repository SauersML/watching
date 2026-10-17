"""Continue a native tail on its exact archived source states, without recompiling a prefix.

This evaluates only the cached-input split runtime. Adam moments restart explicitly;
minibatch indices continue the original deterministic stream. Archived answer scores
must replay exactly before any update. No new whole-graph transfer claim is made.
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
from watching.checkpointing import checkpoint_preflight, save_training_checkpoint
from watching.execution_fit import tensor_digest
from watching.null_tail import validate_cache, validate_tokenized_records
from watching.runtime import runtime_provenance
from watching.suffix import move_state
from watching.tail_train import (
    EagerNativeTail,
    PairedMasterAdam,
    check_gradient_forward,
    file_digest,
    prompt_batches,
    record_answer_ids,
    score_record,
    summarize,
    summarize_tasks,
    validate_initial_tail,
)


def validate_origin(manifest, artifact, mode):
    """Bind mode, native weights and stream position to a completed predecessor."""
    if (
        manifest.get("training_status") != "complete"
        or manifest.get("completed_steps") != manifest.get("steps")
        or manifest.get("viewing_mode") != mode
        or manifest.get("train_head", False)
        or artifact.get("train_head", False)
        or artifact.get("head") is not None
        or manifest.get("tail_layers") != 2
        or not manifest.get("revision")
    ):
        raise ValueError("Need a completed, pinned, same-mode two-block fixed-head predecessor")
    start = manifest.get("ending_global_step", manifest["completed_steps"])
    pairs = manifest.get("global_processed_prompt_pairs")
    if (
        type(start) is not int
        or start < 1
        or artifact.get("global_step") != start
        or artifact.get("completed_steps") != manifest["completed_steps"]
        or artifact.get("prompt_batch_size") != manifest["prompt_batch_size"]
        or pairs != start * manifest["prompt_batch_size"]
        or artifact.get("global_processed_prompt_pairs") != pairs
    ):
        raise ValueError("Predecessor weights, optimizer-step count and stream position disagree")
    return start, pairs


def continuation_batches(count, batch_size, seed, skip):
    if type(skip) is not int or skip < 0:
        raise ValueError("Skipped update count must be a nonnegative integer")
    stream = prompt_batches(count, batch_size, seed)
    for _ in range(skip):
        next(stream)
    return stream


def replay_comparison(rows, archived, count):
    """Exact archived answer-logprob and top-token replay, for both conditions."""
    fields = (
        "yes_minus_no",
        "logp_no",
        "logp_yes",
        "ce",
        "binary_ce",
        "full_vocab_correct_probability",
        "answer_pair_mass",
        "top_token_id",
        "greedy_correct",
    )

    def indexed(values):
        result = {(r["index"], r["condition_present"]): r for r in values}
        expected = {(i, c) for i in range(count) for c in (False, True)}
        if len(result) != len(values) or set(result) != expected:
            raise ValueError("Replay requires every prompt and both conditions exactly once")
        return result

    now, old = indexed(rows), indexed(archived)
    mismatches = []
    for key, actual in now.items():
        expected = old[key]
        for field in ("context_id", "negative_token_id", "positive_token_id", *fields):
            if actual[field] != expected[field]:
                mismatches.append(
                    {
                        "index": key[0],
                        "capture": key[1],
                        "field": field,
                        "current": actual[field],
                        "archived": expected[field],
                    }
                )
    return {
        "exact": not mismatches,
        "prompt_pairs": count,
        "fields": list(fields),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "scope": "Both-condition archived answer scores/top tokens; full vocabulary was not saved",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--viewing-mode", choices=("buffer", "identity"), required=True)
    parser.add_argument("--expected-cache-sha256", required=True)
    parser.add_argument("--expected-tail-sha256", required=True)
    parser.add_argument("--steps", type=int, default=2048)
    parser.add_argument("--max-training-seconds", type=float, default=1350)
    args = parser.parse_args()
    if (
        args.steps < 1
        or not math.isfinite(args.max_training_seconds)
        or args.max_training_seconds <= 0
    ):
        parser.error("Positive steps and a finite positive training time cap are required")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA native-tail execution is required")
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        raise ValueError("Output directory must be empty")
    torch.set_num_threads(4)
    source = args.source_results
    paths = {
        name: source / name
        for name in (
            "manifest.json",
            "data.json",
            "cache_fingerprints.json",
            "cache_states.pt",
            "native_tail.pt",
        )
    }
    hashes = {name: file_digest(path) for name, path in paths.items()}
    if hashes["cache_states.pt"] != args.expected_cache_sha256:
        raise ValueError("Source cache checksum differs from the approved artifact")
    if hashes["native_tail.pt"] != args.expected_tail_sha256:
        raise ValueError("Source tail checksum differs from the approved artifact")
    origin, data, fingerprints = [
        json.loads(paths[name].read_text())
        for name in ("manifest.json", "data.json", "cache_fingerprints.json")
    ]
    artifact = torch.load(paths["native_tail.pt"], map_location="cpu", weights_only=True)
    start, prior_pairs = validate_origin(origin, artifact, args.viewing_mode)
    cache = torch.load(paths["cache_states.pt"], map_location="cpu", weights_only=True)
    validate_cache(cache, data, fingerprints, origin)
    if cache["cut_layer"] + 1 != artifact["first_layer"]:
        raise ValueError("Cache boundary differs from the continued native blocks")
    seed, batch_size = origin["seed"], origin["prompt_batch_size"]
    torch.manual_seed(seed)
    started = time.monotonic()

    def emit(event, **fields):
        if "step" in fields:
            fields.setdefault("global_step", start + fields["step"])
        line = json.dumps(
            {"event": event, "seconds": time.monotonic() - started, **fields}, allow_nan=False
        )
        print(line, flush=True)
        with (args.out / "metrics.jsonl").open("a") as stream:
            stream.write(line + "\n")

    manifest = {
        **origin,
        "steps": args.steps,
        "completed_steps": 0,
        "starting_global_step": start,
        "starting_processed_prompt_pairs": prior_pairs,
        "processed_prompt_pairs": 0,
        "ending_global_step": start,
        "global_processed_prompt_pairs": prior_pairs,
        "training_status": "initializing",
        "optimizer_state_restored": False,
        "source_recompiled": False,
        "continuation_note": "Exact archived source cache; weights continue; Adam resets",
        "primary_path": "split_inference",
        "wholegraph_transfer_evaluated": False,
        "source_results": str(source),
        "source_hashes": hashes,
        "initial_tail_sha256": hashes["native_tail.pt"],
        "initial_tail": str(paths["native_tail.pt"]),
        "initialization_mode": "cached_continuation",
        "prior_results": str(source),
        "shared_cache_sha256": hashes["cache_states.pt"],
        "out": str(args.out),
        "eval_every": 128,
        "early_eval_steps": [32],
        "checkpoint_every": 256,
        "max_training_seconds": args.max_training_seconds,
        "runtime_provenance": runtime_provenance(),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "implementation_sha256": file_digest(__file__),
    }
    for key in ("last_checkpoint_step", "last_checkpoint_bytes", "checkpoint_storage_preflight"):
        manifest.pop(key, None)
    write_json(args.out / "manifest.json", manifest)
    write_json(args.out / "data.json", data)
    write_json(args.out / "cache_fingerprints.json", fingerprints)
    emit("cache_verified", source_hashes=hashes, source_recompiled=False)
    tokenizer = AutoTokenizer.from_pretrained(
        origin["checkpoint"], revision=origin["revision"], padding_side="left"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    validate_tokenized_records(tokenizer, data, cache["batches"], origin["length"])
    answers = {s: [record_answer_ids(tokenizer, r) for r in records] for s, records in data.items()}
    # The base is loaded on CPU solely for native module definitions and the frozen head.
    # No source forward, compilation, or new activation extraction takes place.
    model = AutoModelForCausalLM.from_pretrained(
        origin["checkpoint"],
        revision=origin["revision"],
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).eval()
    model.config.use_cache = False
    tail = EagerNativeTail(model, origin["tail_layers"]).cuda().eval()
    head_sha = tensor_digest(tail.head_weight)
    validate_initial_tail(
        artifact,
        tail,
        checkpoint=origin["checkpoint"],
        model_commit=origin["model_commit"],
        head_sha256=head_sha,
    )
    tail.layers.load_state_dict(artifact["layers"])
    tail.norm.load_state_dict(artifact["norm"])
    del model, artifact
    tail.requires_grad_(True)

    def evaluate(step, splits):
        collected = {}
        for split in splits:
            rows = []
            for index, pair in enumerate(cache["states"][split]):
                positions = cache["batches"][split][index]["position_ids"].cuda()
                for present in (False, True):
                    with torch.no_grad():
                        logits = tail(*move_state(pair[int(present)], "cuda"), positions)
                    rows.append(
                        score_record(
                            logits, index, present, data[split][index], answers[split][index]
                        )
                    )
            summary = {
                "step": step,
                "global_step": start + step,
                "split": split,
                "path": "split_inference",
                "processed_prompt_pairs": step * batch_size,
                "global_processed_prompt_pairs": prior_pairs + step * batch_size,
                **summarize(rows, seed),
                "tasks": summarize_tasks(rows, seed),
            }
            write_json(
                args.out / f"{split}_{step}_split_inference.json",
                {"summary": summary, "rows": rows},
            )
            emit("evaluation", **summary)
            collected[split] = rows
        return collected

    initial = evaluate(0, ("train", "validation"))
    replay = {}
    for split, rows in initial.items():
        path = source / f"{split}_{origin['completed_steps']}_split_inference.json"
        old = json.loads(path.read_text())
        replay[split] = {
            **replay_comparison(rows, old["rows"], len(data[split])),
            "archived_score_sha256": file_digest(path),
        }
    write_json(args.out / "continuation_comparison.json", replay)
    if not all(report["exact"] for report in replay.values()):
        manifest.update(training_status="baseline_replay_failed", completed_steps=0)
        write_json(args.out / "manifest.json", manifest)
        raise RuntimeError("Cached-tail baseline replay differs; no optimizer was created")
    emit("baseline_replay_verified", step=0, **replay)

    def audit(step):
        for present in (False, True):
            audit_result = check_gradient_forward(
                tail,
                move_state(cache["states"]["train"][0][int(present)], "cuda"),
                cache["batches"]["train"][0]["position_ids"].cuda(),
            )
            emit("gradient_forward_audit", step=step, capture=present, **audit_result)
            if not audit_result["bit_identical"]:
                raise RuntimeError("Gradient-enabled/no-grad split-tail mismatch")

    def payload(step):
        return {
            "version": 1,
            "train_head": False,
            "head": None,
            "learned_head_sha256": head_sha,
            "head_sha256": head_sha,
            "completed_steps": step,
            "global_step": start + step,
            "prompt_batch_size": batch_size,
            "global_processed_prompt_pairs": prior_pairs + step * batch_size,
            "checkpoint": origin["checkpoint"],
            "model_commit": origin["model_commit"],
            "first_layer": tail.first_layer,
            "layers": {k: v.detach() for k, v in tail.layers.state_dict().items()},
            "norm": {k: v.detach() for k, v in tail.norm.state_dict().items()},
            "runtime": origin["runtime"],
        }

    def save(step):
        manifest.update(
            ending_global_step=start + step,
            processed_prompt_pairs=step * batch_size,
            global_processed_prompt_pairs=prior_pairs + step * batch_size,
        )
        save_training_checkpoint(payload(step), args.out / "native_tail.pt", manifest, step=step)
        emit("tail_checkpoint_saved", step=step)

    checkpoint_preflight(payload(0), args.out / "native_tail.pt", manifest)
    audit(0)
    optimizer = PairedMasterAdam(
        tail.active_parameters(), origin["lr"], expected_contributions=2 * batch_size
    )
    stream = continuation_batches(len(data["train"]), batch_size, seed, start)
    manifest["training_status"] = "training"
    write_json(args.out / "manifest.json", manifest)
    emit("training_start", step=0, additional_steps=args.steps, optimizer_state_restored=False)
    training_started = time.monotonic()
    completed = 0
    for step in range(1, args.steps + 1):
        indices = next(stream)
        losses = []
        for index in indices:
            positions = cache["batches"]["train"][index]["position_ids"].cuda()
            for present in (False, True):
                logits = tail(
                    *move_state(cache["states"]["train"][index][int(present)], "cuda"), positions
                ).float()
                label = torch.tensor([answers["train"][index][int(present)]], device="cuda")
                loss = F.cross_entropy(logits, label) / (2 * batch_size)
                with sdpa_kernel(SDPBackend.MATH):
                    loss.backward()
                optimizer.accumulate()
                losses.append(float(loss.detach()))
        norm, movement = optimizer.step()
        completed = step
        emit(
            "train",
            step=step,
            prompt_indices=indices,
            loss=sum(losses),
            gradient_norm=norm,
            actual_weight_update_l2=movement,
            processed_prompt_pairs=step * batch_size,
            global_processed_prompt_pairs=prior_pairs + step * batch_size,
        )
        capped = time.monotonic() - training_started >= args.max_training_seconds
        if step == 32 or step % 128 == 0 or step == args.steps or capped:
            audit(step)
            evaluate(step, ("train", "validation"))
        if step % 256 == 0 or step == args.steps or capped:
            save(step)
        if capped:
            emit("training_time_limit", step=step, requested_steps=args.steps)
            break
    if tensor_digest(tail.head_weight) != head_sha:
        raise RuntimeError("Frozen native output head changed")
    manifest.update(
        completed_steps=completed,
        head_sha256_after=head_sha,
        training_status="complete" if completed == args.steps else "time_limited",
    )
    if completed == args.steps:
        evaluate(completed, [s for s in data if s not in ("train", "validation")])
    else:
        manifest["held_out_evaluation_skipped"] = "Planned additional updates not completed"
        emit("held_out_evaluation_skipped", step=completed)
    write_json(args.out / "manifest.json", manifest)
    emit(
        "complete",
        step=completed,
        status=manifest["training_status"],
        head_unchanged=True,
        source_recompiled=False,
        wholegraph_transfer_evaluated=False,
    )


if __name__ == "__main__":
    main()
