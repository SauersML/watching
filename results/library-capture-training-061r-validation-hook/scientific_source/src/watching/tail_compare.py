"""Compare native learning recipes on one shared eight-block boundary cache.

Every arm starts from the same base weights and reads identical cached inputs.
Primary evaluation uses the same eager BF16 math-SDPA suffix; whole-model
compiled transfer is a separate final diagnostic. No classifier is introduced.
"""

import argparse
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
from watching.execution_fit import checkpoint_provenance, tensor_digest
from watching.probe import ProbeModel
from watching.suffix import ResidualCapture, assert_identical, move_state
from watching.tail_train import (
    EagerNativeTail,
    PairedMasterAdam,
    check_gradient_forward,
    context_key,
    difference,
    file_digest,
    prompt_batches,
    score,
    state_fingerprint,
    stratified_records,
    summarize,
)

ARMS = ("two_fixed", "two_head", "four_fixed", "eight_fixed")
TRAIN_LAST = {"two_fixed": 2, "two_head": 2, "four_fixed": 4, "eight_fixed": 8}


class ComparedNativeTail(EagerNativeTail):
    """All arms execute the same native blocks; only the requested weights learn."""

    def __init__(self, model, arm, cache_layers=8):
        if arm not in ARMS:
            raise ValueError(f"Unknown comparison arm: {arm}")
        super().__init__(model, tail_layers=cache_layers, train_head=arm == "two_head")
        self.arm = arm
        self.train_last = TRAIN_LAST[arm]
        self.cache_layers = cache_layers
        if self.train_last > cache_layers:
            raise ValueError("Trainable blocks exceed the shared suffix")
        for index, layer in enumerate(self.layers):
            layer.requires_grad_(index >= cache_layers - self.train_last)
        self.norm.requires_grad_(True)

    def active_parameters(self):
        return [parameter for parameter in super().active_parameters() if parameter.requires_grad]


def snapshot_native_suffix(model, cache_layers=8):
    """Save the original shared suffix, norm and native head for exact resets."""
    return {
        "cache_layers": cache_layers,
        "layers": {
            k: v.detach().cpu().clone()
            for k, v in model.model.layers[-cache_layers:].state_dict().items()
        },
        "norm": {k: v.detach().cpu().clone() for k, v in model.model.norm.state_dict().items()},
        "head": model.lm_head.weight.detach().cpu().clone(),
    }


def install_native_suffix(model, state):
    """Reset all shared blocks, normalization and head; no arm can inherit another."""
    model.model.layers[-state["cache_layers"] :].load_state_dict(state["layers"])
    model.model.norm.load_state_dict(state["norm"])
    with torch.no_grad():
        model.lm_head.weight.copy_(state["head"])


def learned_native_suffix(tail, checkpoint, steps, original_head_sha256, base_sha256):
    offset = tail.cache_layers - tail.train_last
    return {
        "version": 2,
        "arm": tail.arm,
        "checkpoint": checkpoint,
        "first_layer": tail.first_layer + offset,
        "train_last": tail.train_last,
        "block_indices": list(
            range(tail.first_layer + offset, tail.first_layer + tail.cache_layers)
        ),
        "layers": {
            k: v.detach().cpu().clone() for k, v in tail.layers[offset:].state_dict().items()
        },
        "norm": {k: v.detach().cpu().clone() for k, v in tail.norm.state_dict().items()},
        "head": tail.head_weight.detach().cpu().clone() if tail.train_head else None,
        "head_sha256_before": original_head_sha256,
        "head_sha256_after": tensor_digest(tail.head_weight),
        "base_suffix_sha256": base_sha256,
        "completed_steps": steps,
    }


def install_learned_suffix(model, state, base):
    first = state["first_layer"]
    if state["version"] != 2 or state["block_indices"] != list(
        range(first, len(model.model.layers))
    ):
        raise ValueError("Invalid learned suffix block indices")
    expected_base = base.get("sha256") or suffix_digest(base)
    if state["base_suffix_sha256"] != expected_base:
        raise ValueError("Learned suffix was fitted on a different base")
    install_native_suffix(model, base)
    model.model.layers[first:].load_state_dict(state["layers"])
    model.model.norm.load_state_dict(state["norm"])
    if state["head"] is not None:
        with torch.no_grad():
            model.lm_head.weight.copy_(state["head"])


def suffix_digest(state):
    hashes = {
        group: {key: tensor_digest(tensor) for key, tensor in state[group].items()}
        for group in ("layers", "norm")
    }
    hashes["head"] = tensor_digest(state["head"])
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def verify_cached_state(actual, expected, context):
    for value, saved in zip(actual, expected, strict=True):
        if saved is None:
            if value is not None:
                raise RuntimeError(f"{context}: optional tensor presence changed")
        elif value is None:
            raise RuntimeError(f"{context}: tensor became absent")
        else:
            assert_identical(value.detach().cpu(), saved, context)


def select_data(raw, counts, seed, test_split):
    data = {
        split: stratified_records(raw[test_split if split == "test" else split], count, seed)
        for split, count in zip(("train", "validation", "test"), counts, strict=True)
    }
    identities = [{context_key(row) for row in rows} for rows in data.values()]
    if any(a & b for i, a in enumerate(identities) for b in identities[i + 1 :]):
        raise ValueError("Comparison cohorts share contexts")
    if {r["question_family"] for r in data["train"]} & {r["question_family"] for r in data["test"]}:
        raise ValueError("Endpoint question families must be absent from training")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="Qwen/Qwen3-8B")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--probe-layer", type=int, default=14)
    parser.add_argument("--cache-layers", type=int, choices=[8], default=8)
    parser.add_argument("--eval-every", type=int, default=128)
    parser.add_argument("--max-training-seconds", type=float, default=0)
    parser.add_argument("--train-prompts", type=int, default=256)
    parser.add_argument("--validation-prompts", type=int, default=32)
    parser.add_argument("--test-prompts", type=int, default=64)
    parser.add_argument("--test-split", default="test_wording")
    parser.add_argument("--prompt-batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--eval-steps", type=int, nargs="*", default=[8, 32, 128, 256, 512])
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--length", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20261017)
    args = parser.parse_args()
    if (
        min(args.train_prompts, args.validation_prompts, args.test_prompts, args.steps, args.length)
        < 1
        or args.eval_every < 1
        or args.max_training_seconds < 0
        or not math.isfinite(args.max_training_seconds)
        or args.prompt_batch_size < 1
        or args.train_prompts % args.prompt_batch_size
        or any(step < 1 for step in args.eval_steps)
        or not math.isfinite(args.lr)
        or not 0 < args.lr < 1
    ):
        parser.error("Invalid counts, batch size, evaluation steps or learning rate")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for compiled native source extraction")
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        raise ValueError("Output directory must be empty")
    started = time.monotonic()
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)

    def emit(event, arm=None, **fields):
        record = {"event": event, "seconds": time.monotonic() - started, **fields}
        if arm is not None:
            record["arm"] = arm
        line = json.dumps(record, allow_nan=False)
        print(line, flush=True)
        with (args.out / "metrics.jsonl").open("a") as stream:
            stream.write(line + "\n")
        if arm is not None:
            with (args.out / arm / "metrics.jsonl").open("a") as stream:
                stream.write(line + "\n")

    raw = json.loads(args.data.read_text())
    data = select_data(
        raw,
        (args.train_prompts, args.validation_prompts, args.test_prompts),
        args.seed,
        args.test_split,
    )
    write_json(args.out / "data.json", data)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    labels = [tokenizer.encode(word, add_special_tokens=False) for word in ("No", "Yes")]
    if any(len(ids) != 1 for ids in labels) or labels[0] == labels[1]:
        raise ValueError("No and Yes must be distinct single tokens")
    no, yes = [ids[0] for ids in labels]
    batches = {}
    for split, rows in data.items():
        batches[split] = []
        for row in rows:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["text"]}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            if len(tokenizer.encode(text, add_special_tokens=False)) > args.length:
                raise ValueError("Prompt exceeds fixed length; truncation is forbidden")
            batch = tokenizer(
                text,
                add_special_tokens=False,
                padding="max_length",
                max_length=args.length,
                return_tensors="pt",
            )
            batch["position_ids"] = (batch.attention_mask.cumsum(-1) - 1).clamp_min(0)
            batches[split].append(dict(batch))

    def load():
        model = (
            AutoModelForCausalLM.from_pretrained(
                args.checkpoint, dtype=torch.bfloat16, attn_implementation="sdpa"
            )
            .cuda()
            .eval()
        )
        model.config.use_cache = False
        model.requires_grad_(True)
        return model

    emit("loading", checkpoint=args.checkpoint)
    model, observed = load(), load()
    if any(
        not torch.equal(a, b)
        for a, b in zip(model.parameters(), observed.parameters(), strict=True)
    ):
        raise RuntimeError("Native and extraction model copies differ")
    if (
        model.config.tie_word_embeddings
        or model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
    ):
        raise ValueError("Comparison requires the existing native output matrix to be untied")
    cut = len(model.model.layers) - args.cache_layers - 1
    if not 0 <= args.probe_layer <= cut:
        raise ValueError("Probe must lie in the shared frozen source prefix")
    source_wrapper = ProbeModel(model, args.probe_layer)
    capture_wrapper = ResidualCapture(observed, args.probe_layer, cut, "buffer")
    capture_wrapper.prepare_buffers({k: v.cuda() for k, v in batches["train"][0].items()})
    source = torch.compile(source_wrapper, fullgraph=True, dynamic=False)
    capture = torch.compile(capture_wrapper, fullgraph=True, dynamic=False)
    head_digest = tensor_digest(model.lm_head.weight)
    manifest = {
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "arms": list(ARMS),
        "cut_layer": cut,
        "replayed_blocks": list(range(cut + 1, cut + 1 + args.cache_layers)),
        "runtime": "shared compiled-source cache + eager BF16 math-SDPA native eight-block suffix",
        "primary_path": "split_inference",
        "condition": "probe",
        "answer_order": ["No", "Yes"],
        "labels": {"No": no, "Yes": yes},
        "head_sha256_before": head_digest,
        "model_commit": getattr(model.config, "_commit_hash", None),
        "model_provenance": checkpoint_provenance(
            args.checkpoint, getattr(model.config, "_commit_hash", None)
        ),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "test_inputs_cached_before_training": True,
        "test_scored_only_at_arm_endpoint": True,
        "stock_runtime_parity_claimed": False,
        "architecture_changed": False,
        "selection": "fixed endpoint; reporting and test scores never select checkpoints or arms",
    }
    write_json(args.out / "manifest.json", manifest)
    emit("setup", **manifest)
    cache = {}
    for split in ("train", "validation", "test"):
        cache[split] = []
        emit("cache_start", split=split, pairs=len(data[split]))
        for index, batch in enumerate(batches[split]):
            x = {k: v.cuda() for k, v in batch.items()}
            pair = []
            for present in (False, True):
                with torch.no_grad():
                    full, saved = source(**x, capture=present)
                    extracted, also_saved, state = capture(**x, capture=present)
                if len(saved) != int(present) or len(also_saved) != int(present):
                    raise RuntimeError("Query-norm probe did not fire as requested")
                if not torch.equal(full, extracted):
                    report = {
                        "split": split,
                        "index": index,
                        "capture": present,
                        **difference(full, extracted),
                    }
                    write_json(args.out / "extraction_failure.json", report)
                    raise RuntimeError("Shared boundary extraction changes full-vocabulary logits")
                pair.append(tuple(None if t is None else t.detach().cpu().clone() for t in state))
                source_wrapper.captured = capture_wrapper.captured = []
                capture_wrapper.boundary = None
            cache[split].append(pair)
            if index == 0 or (index + 1) % 16 == 0:
                emit("cache_progress", split=split, pairs=index + 1, extraction_bit_identical=True)
        if split == "train" and not any(not torch.equal(p[0][0], p[1][0]) for p in cache[split]):
            raise RuntimeError("Probe footprint vanished at the shared boundary")
    torch.save(
        {"version": 1, "manifest": manifest, "data": data, "batches": batches, "states": cache},
        args.out / "cache_states.pt",
    )
    write_json(
        args.out / "cache_fingerprints.json",
        {
            split: [
                {
                    "index": i,
                    "context_id": context_key(data[split][i]),
                    "positions_sha256": tensor_digest(batches[split][i]["position_ids"]),
                    "absent": state_fingerprint(pair[0]),
                    "present": state_fingerprint(pair[1]),
                }
                for i, pair in enumerate(pairs)
            ]
            for split, pairs in cache.items()
        },
    )
    manifest["shared_cache_sha256"] = file_digest(args.out / "cache_states.pt")
    write_json(args.out / "manifest.json", manifest)
    model.cpu()
    observed.cpu()
    torch.cuda.empty_cache()
    base = snapshot_native_suffix(model, args.cache_layers)
    base_sha256 = suffix_digest(base)
    base["sha256"] = base_sha256
    manifest["base_suffix_sha256"] = base_sha256
    write_json(args.out / "manifest.json", manifest)
    emit("sources_offloaded", allocated_gpu_bytes=torch.cuda.memory_allocated())
    initial_hashes = {}

    def save_scores(arm, rows, step, split, path):
        summary = {
            "step": step,
            "global_step": step,
            "split": split,
            "path": path,
            "processed_prompt_pairs": step * args.prompt_batch_size,
            **summarize(rows, args.seed),
        }
        write_json(
            args.out / arm / f"{split}_{step}_{path}.json", {"summary": summary, "rows": rows}
        )
        emit("evaluation", arm=arm, **summary)

    def evaluate(tail, arm, step, splits):
        for split in splits:
            rows = []
            for index, pair in enumerate(cache[split]):
                positions = batches[split][index]["position_ids"].cuda()
                for present in (False, True):
                    with torch.no_grad():
                        logits = tail(*move_state(pair[int(present)], "cuda"), positions)
                    if step == 0:
                        key = f"{split}/{index}/{int(present)}"
                        digest = tensor_digest(logits)
                        if arm == ARMS[0]:
                            initial_hashes[key] = digest
                        elif initial_hashes[key] != digest:
                            raise RuntimeError(f"Arms disagree before training on {key}")
                    rows.append(score(logits, index, present, no, yes))
            save_scores(arm, rows, step, split, "split_inference")

    def audit(tail, arm, step):
        for index in range(min(4, len(cache["train"]))):
            for present in (False, True):
                result = check_gradient_forward(
                    tail,
                    move_state(cache["train"][index][int(present)], "cuda"),
                    batches["train"][index]["position_ids"].cuda(),
                )
                emit(
                    "gradient_forward_audit",
                    arm=arm,
                    step=step,
                    index=index,
                    capture=present,
                    **result,
                )
                if not result["bit_identical"]:
                    raise RuntimeError("Gradient-enabled and inference suffix forward differ")

    training_started = time.monotonic()
    arm_peaks = {}
    for arm in ARMS:
        if (
            args.max_training_seconds > 0
            and time.monotonic() - training_started >= args.max_training_seconds
        ):
            manifest["status"] = "time_limited"
            write_json(args.out / "manifest.json", manifest)
            emit(
                "comparison_partial",
                status="time_limited",
                transfer_evaluated=False,
                remaining_arms_not_started=list(ARMS[ARMS.index(arm) :]),
            )
            return
        torch.cuda.reset_peak_memory_stats()
        directory = args.out / arm
        directory.mkdir()
        tail = ComparedNativeTail(model, arm, args.cache_layers).cuda()
        arm_manifest = {
            **manifest,
            "arm": arm,
            "starting_global_step": 0,
            "completed_steps": 0,
            "tail_layers": tail.train_last,
            "updated_blocks": list(
                range(cut + 1 + args.cache_layers - tail.train_last, cut + 1 + args.cache_layers)
            ),
            "output_head_updated": tail.train_head,
            "new_classifier": False,
            "active_parameters": sum(p.numel() for p in tail.active_parameters()),
            "shared_cache": "../cache_states.pt",
            "output_head_scope": "full native matrix" if tail.train_head else "frozen",
        }
        write_json(directory / "manifest.json", arm_manifest)
        write_json(directory / "data.json", data)
        emit("arm_start", arm=arm, active_parameters=arm_manifest["active_parameters"])
        audit(tail, arm, 0)
        evaluate(tail, arm, 0, ("train", "validation"))
        emit("initial_scores_verified", arm=arm, n_states=len(initial_hashes), same_base=True)
        optimizer = PairedMasterAdam(
            tail.active_parameters(), args.lr, expected_contributions=2 * args.prompt_batch_size
        )
        stream = prompt_batches(len(cache["train"]), args.prompt_batch_size, args.seed)
        for step in range(1, args.steps + 1):
            indices = next(stream)
            losses = []
            for index in indices:
                positions = batches["train"][index]["position_ids"].cuda()
                for present in (False, True):
                    logits = tail(
                        *move_state(cache["train"][index][int(present)], "cuda"), positions
                    ).float()
                    label = torch.tensor([yes if present else no], device=logits.device)
                    loss = F.cross_entropy(logits, label) / (2 * args.prompt_batch_size)
                    with sdpa_kernel(SDPBackend.MATH):
                        loss.backward()
                    optimizer.accumulate()
                    losses.append(float(loss.detach()))
            norm, movement = optimizer.step()
            emit(
                "train",
                arm=arm,
                step=step,
                global_step=step,
                prompt_indices=indices,
                processed_prompt_pairs=step * args.prompt_batch_size,
                loss=sum(losses),
                gradient_norm=norm,
                actual_weight_update_l2=movement,
            )
            time_limited = (
                args.max_training_seconds > 0
                and time.monotonic() - training_started >= args.max_training_seconds
            )
            if (
                step in args.eval_steps
                or step % args.eval_every == 0
                or step == args.steps
                or time_limited
            ):
                audit(tail, arm, step)
                evaluate(tail, arm, step, ("train", "validation"))
            if time_limited:
                break
        del optimizer, logits, loss
        learned_head_digest = tensor_digest(tail.head_weight)
        if not tail.train_head and learned_head_digest != head_digest:
            raise RuntimeError("Frozen native head changed")
        artifact = learned_native_suffix(tail, args.checkpoint, step, head_digest, base_sha256)
        torch.save(artifact, directory / "native_tail.pt")
        arm_manifest.update(
            completed_steps=step,
            ending_global_step=step,
            processed_prompt_pairs=step * args.prompt_batch_size,
            head_sha256_after=learned_head_digest,
            training_status="complete" if step == args.steps else "time_limited",
        )
        write_json(directory / "manifest.json", arm_manifest)
        emit("tail_checkpoint_saved", arm=arm, step=step)
        if step == args.steps:
            evaluate(tail, arm, step, ("test",))
        arm_peaks[arm] = {
            "allocated": torch.cuda.max_memory_allocated(),
            "reserved": torch.cuda.max_memory_reserved(),
        }
        del artifact, tail
        torch.cuda.empty_cache()
        emit(
            "arm_training_complete" if step == args.steps else "arm_training_partial",
            arm=arm,
            step=step,
            allocated_gpu_bytes=torch.cuda.memory_allocated(),
            peak_gpu_bytes=arm_peaks[arm],
        )
        write_json(args.out / "initial_logits_sha256.json", initial_hashes)
        if time_limited and step < args.steps:
            manifest["status"] = "time_limited"
            write_json(args.out / "manifest.json", manifest)
            emit(
                "comparison_partial",
                status="time_limited",
                arm=arm,
                step=step,
                transfer_evaluated=False,
                remaining_arms_not_started=list(ARMS[ARMS.index(arm) + 1 :]),
            )
            return
    write_json(args.out / "initial_logits_sha256.json", initial_hashes)

    # Restore the untouched source once. Reset all shared blocks and the head
    # between transfer arms so a trained head cannot leak into another arm.
    model.cuda()
    observed.cuda()
    torch.cuda.reset_peak_memory_stats()
    for arm in ARMS:
        artifact = torch.load(
            args.out / arm / "native_tail.pt", map_location="cpu", weights_only=True
        )
        for target in (model, observed):
            install_learned_suffix(target, artifact, base)
            if tensor_digest(target.lm_head.weight) != artifact["head_sha256_after"]:
                raise RuntimeError("Native transfer head does not match saved learner")
        for split in ("train", "validation", "test"):
            rows = []
            for index, batch in enumerate(batches[split]):
                x = {k: v.cuda() for k, v in batch.items()}
                for present in (False, True):
                    with torch.no_grad():
                        logits, saved = source(**x, capture=present)
                        copied, other_saved, state = capture(**x, capture=present)
                    assert_identical(logits, copied, "Transfer extraction changes logits")
                    verify_cached_state(
                        state, cache[split][index][int(present)], "Frozen shared source changed"
                    )
                    if len(saved) != int(present) or len(other_saved) != int(present):
                        raise RuntimeError("Transfer probe did not fire")
                    rows.append(score(logits, index, present, no, yes))
                    source_wrapper.captured = capture_wrapper.captured = []
                    capture_wrapper.boundary = None
            save_scores(arm, rows, args.steps, split, "wholegraph_transfer")
        emit(
            "complete",
            arm=arm,
            step=args.steps,
            status="complete",
            frozen_source_bit_identical=True,
            native_head_installed_correctly=True,
        )
        del artifact
    manifest["status"] = "complete"
    write_json(args.out / "manifest.json", manifest)
    emit(
        "comparison_complete",
        status="complete",
        arms=list(ARMS),
        transfer_peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        training_arm_peaks=arm_peaks,
    )


if __name__ == "__main__":
    main()
