"""Train native final Qwen3 blocks to distinguish gradient recording from no-grad.

Source parameters always require gradients and both modes use eval(). Only the
source autograd context changes. Primary scores use fixed source states and an
eager math-SDPA native tail; whole-graph transfer is a separate measured result.
"""

import argparse
import gc
import hashlib
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM, AutoTokenizer

from watching.artifacts import write_json
from watching.checkpointing import atomic_torch_save, checkpoint_preflight, save_training_checkpoint
from watching.execution_fit import NativeForward, checkpoint_provenance, tensor_digest
from watching.messages import render_record
from watching.runtime import runtime_provenance
from watching.suffix import assert_identical, move_state
from watching.tail_train import (
    EagerNativeTail,
    PairedMasterAdam,
    check_gradient_forward,
    context_key,
    difference,
    install_learner_tail,
    prompt_batches,
    record_answer_ids,
    score_record,
    state_fingerprint,
    stratified_records,
    summarize,
    validate_initial_tail,
)


class ExecutionCapture(NativeForward):
    """Copy an existing residual boundary without retaining its autograd graph."""

    def __init__(self, model, first_tail_layer):
        super().__init__(model)
        if not 0 < first_tail_layer < len(model.model.layers):
            raise ValueError("Boundary must leave a frozen prefix and native trainable tail")
        self.preparing = False
        self.ready = False
        self.boundary = None
        self.handle = model.model.layers[first_tail_layer].register_forward_pre_hook(
            self.record, with_kwargs=True
        )

    def record(self, module, inputs, kwargs):
        cos, sin = kwargs["position_embeddings"]
        state = (inputs[0], kwargs["attention_mask"], cos, sin)
        if self.preparing:
            self.boundary = state
            return
        if not self.ready:
            raise RuntimeError("Prepare capture destinations before compiling")
        result = []
        for name, value in zip(("hidden", "mask", "cos", "sin"), state, strict=True):
            destination = getattr(self, "cache_" + name)
            if value is None:
                if destination is not None:
                    raise RuntimeError("Boundary mask presence changed")
            elif destination is None or value.shape != destination.shape:
                raise RuntimeError("Boundary shape changed")
            else:
                # Detach only the observation. Native hidden states remain untouched.
                destination.copy_(value.detach())
            result.append(destination)
        self.boundary = tuple(result)

    def prepare(self, inputs):
        if self.ready:
            raise RuntimeError("Capture preparation must occur exactly once")
        self.preparing = True
        try:
            with torch.no_grad():
                _, state = self(**inputs)
            for name, value in zip(("hidden", "mask", "cos", "sin"), state, strict=True):
                self.register_buffer(
                    "cache_" + name,
                    None if value is None else torch.empty_like(value),
                    persistent=False,
                )
            self.ready = True
        finally:
            self.preparing = False
            self.boundary = None

    def forward(self, input_ids, attention_mask, position_ids):
        logits = super().forward(input_ids, attention_mask, position_ids)
        return logits, self.boundary


def source_forward(wrapper, inputs, gradient_enabled, *, capture=False):
    """One condition switch, outside the token inputs and native forward signature."""
    with torch.set_grad_enabled(gradient_enabled):
        result = wrapper(**inputs)
        if capture:
            logits, state = result
            stored = tuple(None if x is None else x.detach().cpu().clone() for x in state)
        else:
            logits, stored = result, None
        values = logits.detach().cpu().clone()
    return values, stored


def select_records(raw, train, validation, test, seed):
    selected = {}
    for split, count in (("train", train), ("validation", validation), ("test", test)):
        source = "test_wording" if split == "test" and "test_wording" in raw else split
        selected[split] = stratified_records(raw[source], count, seed)
        for row in selected[split]:
            if row.get("task") not in (None, "execution") or (
                row.get("task") is None and ".execution." not in row.get("wording_id", "")
            ):
                raise ValueError("Execution training requires explicit execution-task provenance")
            row["task"] = "execution"
            if (
                row.get("answer_negative", "No") != "No"
                or row.get("answer_positive", "Yes") != "Yes"
            ):
                raise ValueError("Execution labels must be No (no-grad), Yes (grad enabled)")
    ids = [{context_key(r) for r in rows} for rows in selected.values()]
    if any(a & b for i, a in enumerate(ids) for b in ids[i + 1 :]):
        raise ValueError("Execution splits contain overlapping contexts")
    train_families = {r.get("question_family") for r in selected["train"]} - {None}
    test_families = {r.get("question_family") for r in selected["test"]} - {None}
    if not train_families or not test_families or train_families & test_families:
        raise ValueError("Test question families must be recorded and absent from training")
    return selected


def execution_score(logits, index, enabled, record, answer_ids):
    row = score_record(logits, index, enabled, record, answer_ids)
    row["source_gradient_enabled"] = bool(enabled)
    return row


def compare_states(actual, expected):
    for a, b in zip(actual, expected, strict=True):
        if a is None or b is None:
            if a is not b:
                raise RuntimeError("Frozen boundary mask presence changed")
        else:
            assert_identical(a, b, "Frozen boundary state changed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-prompts", type=int, default=256)
    parser.add_argument("--validation-prompts", type=int, default=32)
    parser.add_argument("--test-prompts", type=int, default=64)
    parser.add_argument("--prompt-batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--eval-every", type=int, default=128)
    parser.add_argument("--checkpoint-every", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--length", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--max-training-seconds", type=float, default=1050)
    parser.add_argument("--evaluate-tail", type=Path)
    args = parser.parse_args()
    if (
        min(args.steps, args.eval_every, args.checkpoint_every, args.length) < 1
        or not 0 < args.lr < 1
    ):
        parser.error("Invalid steps, interval, length, or learning rate")
    if not math.isfinite(args.max_training_seconds) or args.max_training_seconds < 0:
        parser.error("Training time budget must be finite and nonnegative")
    # Validate batching before allocating the source models.
    next(prompt_batches(args.train_prompts, args.prompt_batch_size, args.seed))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the compiled native source")
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        raise ValueError("Output directory must be empty")
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch._inductor.config.freezing = False
    torch._inductor.config.force_disable_caches = True
    started = time.monotonic()

    def emit(event, **fields):
        item = {"event": event, "seconds": time.monotonic() - started, **fields}
        line = json.dumps(item, allow_nan=False)
        print(line, flush=True)
        with (args.out / "metrics.jsonl").open("a") as stream:
            stream.write(line + "\n")

    raw = json.loads(args.data.read_text())
    data = select_records(
        raw, args.train_prompts, args.validation_prompts, args.test_prompts, args.seed
    )
    write_json(args.out / "data.json", data)
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint, revision=args.revision, padding_side="left"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    batches, answers = {}, {}
    for split, records in data.items():
        batches[split], answers[split] = [], []
        for row in records:
            text = render_record(tokenizer, row)
            if len(tokenizer.encode(text, add_special_tokens=False)) > args.length:
                raise ValueError("Rendered input exceeds fixed length; truncation forbidden")
            x = tokenizer(
                text,
                add_special_tokens=False,
                padding="max_length",
                max_length=args.length,
                return_tensors="pt",
            )
            x["position_ids"] = (x.attention_mask.cumsum(-1) - 1).clamp_min(0)
            batches[split].append(dict(x))
            answers[split].append(record_answer_ids(tokenizer, row))

    def load():
        result = (
            AutoModelForCausalLM.from_pretrained(
                args.checkpoint,
                revision=args.revision,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
            .cuda()
            .eval()
        )
        result.config.use_cache = False
        result.requires_grad_(True)
        return result

    emit("loading")
    model, observed = load(), load()
    if any(
        not torch.equal(a, b)
        for a, b in zip(model.parameters(), observed.parameters(), strict=True)
    ):
        raise RuntimeError("Independent source copies have unequal parameters")
    first_layer = len(model.model.layers) - 2
    native_wrapper = NativeForward(model)
    capture_wrapper = ExecutionCapture(observed, first_layer)
    capture_wrapper.prepare({k: v.cuda() for k, v in batches["train"][0].items()})
    native = torch.compile(native_wrapper, fullgraph=True, dynamic=False)
    extract = torch.compile(capture_wrapper, fullgraph=True, dynamic=False)
    tail = EagerNativeTail(model, tail_layers=2, train_head=False)
    head_sha = tensor_digest(model.lm_head.weight)
    manifest = {
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "condition": "execution",
        "negative_condition": "no_grad",
        "positive_condition": "grad_enabled",
        "legacy_capture_field": "Alias of source_gradient_enabled, not an activation probe",
        "source_eval_mode": True,
        "source_parameters_require_grad": True,
        "source_revision": args.revision,
        "model_provenance": checkpoint_provenance(
            args.checkpoint, getattr(model.config, "_commit_hash", None), revision=args.revision
        ),
        "runtime_provenance": runtime_provenance(),
        "updated_blocks": [first_layer, first_layer + 1],
        "primary_path": "split_inference",
        "output_head_updated": False,
        "new_classifier": False,
        "runtime": "compiled source in two autograd contexts; eager BF16 math-SDPA tail",
        "wholegraph_claim": "separate measured transfer, not assumed equivalent to split runtime",
        "test_inputs_cached_before_training": True,
        "test_scored_only_after_training": True,
        "selection": "fixed updates; validation reporting only; partial run skips test",
        "initialization": "original base model; no probe-trained tail",
        "starting_global_step": 0,
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "head_sha256_before": head_sha,
        "completed_steps": 0,
    }
    write_json(args.out / "manifest.json", manifest)
    emit("setup", **manifest)
    cache, initial_logits = {}, {}

    def measured_pair(split, index):
        x = {k: v.cuda() for k, v in batches[split][index].items()}
        pair, values = [], []
        # Alternate order, while training labels retain their canonical index.
        for enabled in (False, True) if index % 2 == 0 else (True, False):
            original, _ = source_forward(native, x, enabled)
            copied, state = source_forward(extract, x, enabled, capture=True)
            capture_wrapper.boundary = None
            report = difference(copied, original)
            if not report["bit_identical"]:
                write_json(
                    args.out / "extraction_failure.json",
                    {"split": split, "index": index, "gradient_enabled": enabled, **report},
                )
                torch.save(
                    {
                        "original": original,
                        "copied": copied,
                        "state": state,
                        "inputs": batches[split][index],
                    },
                    args.out / "extraction_failure.pt",
                )
                raise RuntimeError("Passive extraction changes native execution; no training")
            pair.append((int(enabled), state))
            values.append((int(enabled), original))
        return [state for _, state in sorted(pair)], [value for _, value in sorted(values)]

    for split in data:
        cache[split], initial_logits[split] = [], []
        emit("cache_start", split=split, pairs=len(data[split]))
        for index in range(len(data[split])):
            pair, values = measured_pair(split, index)
            cache[split].append(pair)
            initial_logits[split].append(values)
            auxiliary = {}
            for name, left, right in zip(
                ("mask", "cos", "sin"), pair[0][1:], pair[1][1:], strict=True
            ):
                if left is None or right is None:
                    if left is not right:
                        raise RuntimeError("Source modes produced different mask presence")
                    auxiliary[name] = {"bit_identical": True}
                else:
                    auxiliary[name] = difference(left, right)
            if not auxiliary["mask"]["bit_identical"]:
                raise RuntimeError("Source modes produced different attention masks")
            emit("source_auxiliary_comparison", split=split, index=index, **auxiliary)
            if split != "test":
                footprint = pair[1][0].float() - pair[0][0].float()
                emit(
                    "source_footprint",
                    split=split,
                    index=index,
                    l2=float(footprint.norm()),
                    fraction_different=float((footprint != 0).float().mean()),
                )
            if index == 0 or (index + 1) % 16 == 0:
                emit("cache_progress", split=split, pairs=index + 1, extraction_bit_identical=True)
    if not any(not torch.equal(p[0][0], p[1][0]) for p in cache["train"]):
        raise RuntimeError("No execution footprint at the frozen boundary; no training")
    # Repeat first example after interleaved conditions to reject stale capture state.
    repeated_states, repeated_logits = measured_pair("train", 0)
    for condition in (0, 1):
        compare_states(repeated_states[condition], cache["train"][0][condition])
        assert_identical(
            repeated_logits[condition],
            initial_logits["train"][0][condition],
            "Interleaved source repeat",
        )
    emit("interleaved_repeat_verified")
    atomic_torch_save(
        {
            "states": cache,
            "batches": batches,
            "data": data,
            "condition": "execution",
            "first_layer": first_layer,
        },
        args.out / "cache_states.pt",
    )
    write_json(
        args.out / "cache_fingerprints.json",
        {
            split: [[state_fingerprint(s) for s in pair] for pair in pairs]
            for split, pairs in cache.items()
        },
    )

    def save_scores(rows, split, step, path):
        summary = {
            "split": split,
            "step": step,
            "global_step": step,
            "path": path,
            **summarize(rows, args.seed),
        }
        write_json(args.out / f"{split}_{step}_{path}.json", {"summary": summary, "rows": rows})
        emit("evaluation", **summary)

    def evaluate(step, splits):
        for split in splits:
            rows = []
            for index, pair in enumerate(cache[split]):
                positions = batches[split][index]["position_ids"].cuda()
                for enabled in (False, True):
                    with torch.no_grad():
                        logits = tail(*move_state(pair[int(enabled)], "cuda"), positions)
                    rows.append(
                        execution_score(
                            logits, index, enabled, data[split][index], answers[split][index]
                        )
                    )
            save_scores(rows, split, step, "split_inference")

    def audit(step):
        for condition in (0, 1):
            result = check_gradient_forward(
                tail,
                move_state(cache["train"][0][condition], "cuda"),
                batches["train"][0]["position_ids"].cuda(),
            )
            emit(
                "gradient_forward_audit",
                step=step,
                source_gradient_enabled=bool(condition),
                **result,
            )
            if not result["bit_identical"]:
                raise RuntimeError("Learner grad/no-grad forward mismatch")

    for split in ("train", "validation"):
        rows = [
            execution_score(
                logits, index, bool(condition), data[split][index], answers[split][index]
            )
            for index, pair in enumerate(initial_logits[split])
            for condition, logits in enumerate(pair)
        ]
        save_scores(rows, split, 0, "original_wholegraph")
    if args.evaluate_tail:
        artifact = torch.load(args.evaluate_tail, map_location="cpu", weights_only=True)
        if artifact.get("condition") != "execution" or artifact.get("revision") != args.revision:
            raise ValueError("Evaluation artifact must match execution task and pinned revision")
        validate_initial_tail(
            artifact,
            tail,
            checkpoint=args.checkpoint,
            model_commit=getattr(model.config, "_commit_hash", None),
            head_sha256=head_sha,
        )
        install_learner_tail(tail, artifact)
        completed = artifact["completed_steps"]
        del artifact
    else:
        completed = 0
    audit(completed)
    evaluate(completed, ("train", "validation"))
    model.cpu()
    observed.cpu()
    gc.collect()
    torch.cuda.empty_cache()
    emit("sources_offloaded", allocated_gpu_bytes=torch.cuda.memory_allocated())

    def checkpoint_payload(step):
        return {
            "version": 1,
            "condition": "execution",
            "revision": args.revision,
            "train_head": False,
            "head": None,
            "checkpoint": args.checkpoint,
            "model_commit": getattr(model.config, "_commit_hash", None),
            "first_layer": first_layer,
            "layers": {k: v.detach() for k, v in tail.layers.state_dict().items()},
            "norm": {k: v.detach() for k, v in tail.norm.state_dict().items()},
            "head_sha256": head_sha,
            "completed_steps": step,
        }

    def save_checkpoint(step):
        save_training_checkpoint(
            checkpoint_payload(step), args.out / "native_tail.pt", manifest, step=step
        )
        emit("tail_checkpoint_saved", step=step)

    checkpoint_preflight(checkpoint_payload(completed), args.out / "native_tail.pt", manifest)

    if not args.evaluate_tail:
        optimizer = PairedMasterAdam(
            tail.active_parameters(), args.lr, expected_contributions=2 * args.prompt_batch_size
        )
        stream = prompt_batches(len(cache["train"]), args.prompt_batch_size, args.seed)
        training_started = time.monotonic()
        for step in range(1, args.steps + 1):
            indices = next(stream)
            total_loss = 0.0
            for index in indices:
                positions = batches["train"][index]["position_ids"].cuda()
                for condition in (0, 1):
                    logits = tail(
                        *move_state(cache["train"][index][condition], "cuda"), positions
                    ).float()
                    target = torch.tensor([answers["train"][index][condition]], device="cuda")
                    loss = F.cross_entropy(logits, target) / (2 * args.prompt_batch_size)
                    with sdpa_kernel(SDPBackend.MATH):
                        loss.backward()
                    optimizer.accumulate()
                    total_loss += float(loss.detach())
            norm, movement = optimizer.step()
            completed = step
            emit(
                "train",
                step=step,
                loss=total_loss,
                prompt_indices=indices,
                processed_prompt_pairs=step * args.prompt_batch_size,
                gradient_norm=norm,
                actual_weight_update_l2=movement,
            )
            time_limited = (
                args.max_training_seconds > 0
                and time.monotonic() - training_started >= args.max_training_seconds
            )
            if (
                step in (8, 32)
                or step % args.eval_every == 0
                or step % args.checkpoint_every == 0
                or step == args.steps
                or time_limited
            ):
                audit(step)
                evaluate(step, ("train", "validation"))
            if step % args.checkpoint_every == 0 or step == args.steps or time_limited:
                save_checkpoint(step)
            if time_limited:
                break
        del optimizer, logits, loss
    if tensor_digest(tail.head_weight) != head_sha:
        raise RuntimeError("Fixed native head changed")
    if manifest.get("last_checkpoint_step") != completed:
        save_checkpoint(completed)
    manifest.update(
        completed_steps=completed,
        head_sha256_after=tensor_digest(tail.head_weight),
        training_status="evaluated_artifact"
        if args.evaluate_tail
        else "complete"
        if completed == args.steps
        else "time_limited",
    )
    write_json(args.out / "manifest.json", manifest)
    if not args.evaluate_tail and completed != args.steps:
        emit("held_out_evaluation_skipped", reason="Incomplete planned updates")
        return
    evaluate(completed, ("test",))
    gc.collect()
    torch.cuda.empty_cache()
    model.cuda()
    observed.cuda()
    # Verify original graph and cache survived device migration before installation.
    for split in data:
        for index in range(len(data[split])):
            states, values = measured_pair(split, index)
            for condition in (0, 1):
                compare_states(states[condition], cache[split][index][condition])
                assert_identical(
                    values[condition],
                    initial_logits[split][index][condition],
                    "Restored original graph",
                )
    emit("restored_source_verified")
    tail.install(model)
    tail.install(observed)
    for split in data:
        rows = []
        for index in range(len(data[split])):
            states, values = measured_pair(split, index)
            for condition in (0, 1):
                compare_states(states[condition], cache[split][index][condition])
                rows.append(
                    execution_score(
                        values[condition],
                        index,
                        bool(condition),
                        data[split][index],
                        answers[split][index],
                    )
                )
        save_scores(rows, split, completed, "wholegraph_transfer")
    for source in (model, observed):
        if tensor_digest(source.lm_head.weight) != head_sha:
            raise RuntimeError("Native head changed during transfer")
    emit(
        "complete",
        step=completed,
        head_unchanged=True,
        frozen_source_verified=True,
        fresh_process_artifact_evaluation=args.evaluate_tail is not None,
    )


if __name__ == "__main__":
    main()
