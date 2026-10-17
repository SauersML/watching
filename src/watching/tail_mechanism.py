"""Post-hoc causal diagnostics of a fixed native tail on verified cached states.

No training, classifier, compiler changes or population significance claims. The
original and trained native tails execute with eager BF16 math-SDPA, as in training.
"""

import argparse
import copy
import json
import random
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RMSNorm

from watching.artifacts import write_json
from watching.execution_fit import tensor_digest
from watching.null_tail import validate_cache
from watching.runtime import runtime_provenance
from watching.suffix import assert_identical, move_state
from watching.tail_train import (
    EagerNativeTail,
    file_digest,
    install_learner_tail,
    score_record,
    validate_initial_tail,
)


def native_tail_from_tensors(config, tensors, tail_layers=2):
    """Instantiate only native final blocks/norm/head, without allocating the prefix."""
    if config.tie_word_embeddings:
        raise ValueError("Selective tail loading requires an untied native head")
    first = config.num_hidden_layers - tail_layers
    config = copy.deepcopy(config)
    config._attn_implementation = "sdpa"
    config.use_cache = False
    with torch.device("meta"):
        layers = torch.nn.ModuleList(
            [None] * first
            + [Qwen3DecoderLayer(config, i) for i in range(first, config.num_hidden_layers)]
        )
        model = SimpleNamespace(
            config=config,
            model=SimpleNamespace(
                layers=layers,
                norm=Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps),
            ),
            lm_head=torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False),
        )
        tail = EagerNativeTail(model, tail_layers)
    block_weights = {}
    expected = {"model.norm.weight", "lm_head.weight"}
    for name in tail.layers.state_dict():
        offset, suffix = name.split(".", 1)
        source = f"model.layers.{first + int(offset)}.{suffix}"
        expected.add(source)
        block_weights[name] = tensors[source]
    if set(tensors) != expected:
        raise ValueError("Selective tensor keys differ from the native tail")
    if any(t.dtype != torch.bfloat16 or not torch.isfinite(t).all() for t in tensors.values()):
        raise ValueError("Selective source weights must be finite BF16 tensors")
    tail.layers.load_state_dict(block_weights, strict=True, assign=True)
    tail.norm.load_state_dict({"weight": tensors["model.norm.weight"]}, strict=True, assign=True)
    head = tensors["lm_head.weight"]
    if tuple(head.shape) != (config.vocab_size, config.hidden_size):
        raise ValueError("Native head shape differs from model config")
    tail.head_weight = head
    return tail.eval()


def load_selective_tail(checkpoint, revision, tail_layers):
    config = AutoConfig.from_pretrained(checkpoint, revision=revision)
    first = config.num_hidden_layers - tail_layers
    prefixes = tuple(f"model.layers.{i}." for i in range(first, config.num_hidden_layers))

    def resolve(name):
        if Path(checkpoint).is_dir():
            path = Path(checkpoint) / name
            if not path.is_file():
                raise FileNotFoundError(path)
            return path
        return Path(hf_hub_download(checkpoint, name, revision=revision))

    def wanted(name):
        return name.startswith(prefixes) or name in ("model.norm.weight", "lm_head.weight")

    try:
        index = json.loads(resolve("model.safetensors.index.json").read_text())["weight_map"]
        selected = {name: shard for name, shard in index.items() if wanted(name)}
    except (FileNotFoundError, EntryNotFoundError):
        with safe_open(resolve("model.safetensors"), framework="pt", device="cpu") as reader:
            selected = {name: "model.safetensors" for name in reader.keys() if wanted(name)}
    tensors = {}
    for shard in sorted(set(selected.values())):
        with safe_open(resolve(shard), framework="pt", device="cpu") as reader:
            for name in sorted(selected):
                if selected[name] == shard:
                    tensors[name] = reader.get_tensor(name)
    return native_tail_from_tensors(config, tensors, tail_layers), config


def sites(tail):
    return {
        **{f"block_{tail.first_layer + i}": layer for i, layer in enumerate(tail.layers)},
        "final_norm": tail.norm,
    }


def capture_native(tail, state, positions):
    """Copy real eager outputs; reject even one changed vocabulary logit."""
    with torch.no_grad():
        baseline = tail(*state, positions)
    captured = {"input": state[0].detach().clone()}
    handles = []
    try:
        for name, module in sites(tail).items():

            def hook(_module, _args, output, name=name):
                if not isinstance(output, torch.Tensor):
                    raise TypeError("Expected native decoder/norm tensor output")
                captured[name] = output.detach().clone()
                return None

            handles.append(module.register_forward_hook(hook))
        with torch.no_grad():
            observed = tail(*state, positions)
        assert_identical(observed, baseline, "Mechanism capture all-vocabulary parity")
    finally:
        for handle in handles:
            handle.remove()
    return baseline, captured


@contextmanager
def patch_site(tail, name, replacement):
    """Deliberate intervention after a native module; always remove its hook."""
    handle = None
    if name != "input":
        module = sites(tail)[name]

        def replace(_module, _args, output):
            if output.shape != replacement.shape or output.dtype != replacement.dtype:
                raise ValueError("Patch shape/dtype differs from native activation")
            return replacement

        handle = module.register_forward_hook(replace)
    try:
        yield
    finally:
        if handle is not None:
            handle.remove()


def patched_logits(tail, state, positions, name, replacement):
    if not torch.isfinite(replacement).all():
        raise ValueError("Nonfinite intervention")
    if name == "input":
        if replacement.shape != state[0].shape or replacement.dtype != state[0].dtype:
            raise ValueError("Input patch shape/dtype differs")
        state = (replacement, *state[1:])
    with patch_site(tail, name, replacement), torch.no_grad():
        return tail(*state, positions)


def interpolate(absent, present, alpha):
    if absent.shape != present.shape or absent.dtype != present.dtype:
        raise ValueError("Interpolation requires matched native states")
    # Endpoints are exact native tensors, not floating-point reconstructions.
    if alpha == 0:
        return absent.clone()
    if alpha == 1:
        return present.clone()
    return (absent.float() + alpha * (present.float() - absent.float())).to(absent.dtype)


def token_patch(absent, present, mask):
    if mask.shape != absent.shape[:2] or mask.dtype != torch.bool:
        raise ValueError("Token mask must match singleton sequence dimensions")
    return torch.where(mask.unsqueeze(-1), present, absent)


def matched_direction(direction, reference, valid):
    """Match valid-token FP32 norm before the eventual BF16 insertion."""
    direction = direction.float() * valid.unsqueeze(-1)
    target = (reference.float() * valid.unsqueeze(-1)).norm()
    norm = direction.norm()
    if not torch.isfinite(norm) or norm == 0 or target == 0:
        raise ValueError("Cannot normalize an empty/nonfinite numerical footprint")
    return direction * (target / norm)


def vector_summary(absent, present, answer_direction, valid):
    delta = present.float() - absent.float()
    midpoint = (present.float() + absent.float()) * 0.5
    valid_delta = delta[valid]
    valid_midpoint = midpoint[valid]
    last = delta[:, -1, :].reshape(-1)
    return {
        "valid_delta_l2": float(valid_delta.norm()),
        "valid_midpoint_l2": float(valid_midpoint.norm()),
        "valid_delta_to_midpoint_ratio": float(
            valid_delta.norm() / valid_midpoint.norm().clamp_min(1e-30)
        ),
        "last_token_delta_l2": float(last.norm()),
        "last_token_midpoint_l2": float(midpoint[:, -1, :].norm()),
        "last_token_answer_projection_fp32": float(last @ answer_direction),
        "valid_fraction_different": float((valid_delta != 0).float().mean()),
        "padding_delta_l2": float(delta[~valid].norm()),
        "per_token_delta_l2": delta.norm(dim=-1)[0].tolist(),
    }


def intervention_summary(absent, replacement, real_delta, valid):
    actual = (replacement.float() - absent.float())[valid].reshape(-1)
    real = real_delta[valid].reshape(-1)
    denominator = actual.norm() * real.norm()
    return {
        "realized_delta_l2": float(actual.norm()),
        "reference_delta_l2": float(real.norm()),
        "realized_cosine_to_reference": float(actual @ real / denominator)
        if denominator > 0
        else None,
    }


def response_scores(logits, answer_ids):
    """An intervention has a measured response, not a natural ground-truth label."""
    log_probs = logits.float().log_softmax(-1)[0]
    negative, positive = answer_ids
    return {
        "yes_minus_no": float(logits[0, positive].float() - logits[0, negative].float()),
        "logp_yes": float(log_probs[positive]),
        "logp_no": float(log_probs[negative]),
        "positive_answer_probability": float(log_probs[positive].exp()),
        "negative_answer_probability": float(log_probs[negative].exp()),
        "top_token_id": int(logits[0].argmax()),
    }


def diagnostic_selection(data, per_split, seed):
    selected = []
    for split in ("train", "test"):
        if not 1 <= per_split <= len(data[split]):
            raise ValueError("Diagnostic count exceeds available prompt pairs")
        rng = random.Random(f"{seed}:{split}")
        for index in sorted(rng.sample(range(len(data[split])), per_split)):
            selected.append((split, index))
    return selected


def intervention_runner(
    tail, state, positions, site, context, absent, delta, valid, answer_ids, rows
):
    def evaluate(replacement, kind, **detail):
        output = patched_logits(tail, state, positions, site, replacement)
        rows.append(
            {
                **context,
                "site": site,
                "kind": kind,
                **response_scores(output, answer_ids),
                **intervention_summary(absent, replacement, delta, valid),
                **detail,
            }
        )
        return output

    return evaluate


def check_archived_response(source, stage, split, index, step, current):
    path = source / f"{split}_{0 if stage == 'original' else step}_split_inference.json"
    if not path.is_file():
        return {"available": False}
    old_rows = json.loads(path.read_text())["rows"]
    prior = {bool(r["capture"]): r for r in old_rows if r["index"] == index}
    if set(prior) != {False, True}:
        raise ValueError("Archived response does not contain the selected pair")
    fields = ("yes_minus_no", "logp_yes", "logp_no")
    mismatches = [
        {"capture": present, "field": key, "current": row[key], "archived": prior[present][key]}
        for present, row in zip((False, True), current, strict=True)
        for key in fields
        if row[key] != prior[present][key]
    ]
    return {"available": True, "exact": not mismatches, "mismatches": mismatches}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-results", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--per-split", type=int, default=8)
    parser.add_argument("--noise-seeds", type=int, default=3)
    parser.add_argument("--token-scan-per-split", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--max-seconds", type=float, default=1200)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("The diagnostic preserves the native CUDA tail runtime")
    if args.noise_seeds < 1 or args.token_scan_per_split < 0 or args.max_seconds <= 0:
        parser.error("Invalid diagnostic counts or time limit")
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        raise ValueError("Output directory must be empty")
    torch.set_num_threads(4)
    started = time.monotonic()

    def emit(event, **values):
        item = {"event": event, "seconds": time.monotonic() - started, **values}
        encoded = json.dumps(item, allow_nan=False)
        print(encoded, flush=True)
        with (args.out / "metrics.jsonl").open("a") as stream:
            stream.write(encoded + "\n")

    source = args.source_results
    manifest = json.loads((source / "manifest.json").read_text())
    data = json.loads((source / "data.json").read_text())
    fingerprints = json.loads((source / "cache_fingerprints.json").read_text())
    cache = torch.load(source / "cache_states.pt", map_location="cpu", weights_only=True)
    validate_cache(cache, data, fingerprints, manifest)
    if (
        manifest.get("train_head")
        or manifest.get("tail_layers") != 2
        or manifest.get("training_status") != "complete"
        or manifest.get("completed_steps") != manifest.get("steps")
    ):
        raise ValueError("This audit requires a completed two-block fixed-head experiment")
    artifact = torch.load(source / "native_tail.pt", map_location="cpu", weights_only=True)
    if artifact.get("global_step") != manifest.get("ending_global_step"):
        raise ValueError("Artifact and source training step disagree")
    selected = diagnostic_selection(data, args.per_split, args.seed)
    emit("cache_verified", selected=selected)
    tail, config = load_selective_tail(manifest["checkpoint"], args.revision, 2)
    digest = tensor_digest(tail.head_weight)
    validate_initial_tail(
        artifact,
        tail,
        checkpoint=manifest["checkpoint"],
        model_commit=getattr(config, "_commit_hash", None),
        head_sha256=digest,
    )
    if digest != manifest["head_sha256_before"]:
        raise ValueError("Selective base head differs from the archived experiment")
    if cache["cut_layer"] != tail.first_layer - 1:
        raise ValueError("Cache boundary and native tail do not match")
    tail.cuda()
    answer_ids = (manifest["labels"]["No"], manifest["labels"]["Yes"])
    answer_direction = (
        tail.head_weight[answer_ids[1]].float() - tail.head_weight[answer_ids[0]].float()
    )
    audit_manifest = {
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "source_manifest_sha256": file_digest(source / "manifest.json"),
        "source_cache_sha256": file_digest(source / "cache_states.pt"),
        "source_revision_recorded": manifest.get("revision"),
        "source_model_commit_recorded": manifest.get("model_commit"),
        "historical_prefix_checksum_available": False,
        "tail_sha256": file_digest(source / "native_tail.pt"),
        "head_sha256": digest,
        "selected": selected,
        "runtime_provenance": runtime_provenance(),
        "runtime": "cached original source + eager BF16 math-SDPA native final two blocks",
        "interpretation": (
            "Post-hoc diagnostic prompts; no new held-out performance or population CI claim"
        ),
        "interpolation_alphas": [-1, 0, 0.25, 0.5, 0.75, 1, 2],
        "projection_note": (
            "Intermediate answer projection is FP32 descriptive readout; "
            "scores use actual BF16 native head"
        ),
        "noise_note": (
            "Random/donor controls test direction dependence in the tail, "
            "not the compiler mechanism"
        ),
        "status": "running",
    }
    write_json(args.out / "manifest.json", audit_manifest)
    rows = []
    saved_last_tokens = []
    completed = []
    for stage in ("original", "trained"):
        if stage == "trained":
            install_learner_tail(tail, artifact)
        seen = {"train": 0, "test": 0}
        for split, index in selected:
            if time.monotonic() - started > args.max_seconds:
                audit_manifest["status"] = "time_limited"
                break
            batch = cache["batches"][split][index]
            pair = [move_state(s, "cuda") for s in cache["states"][split][index]]
            positions = batch["position_ids"].cuda()
            valid = batch["attention_mask"].bool().cuda()
            outputs, features = zip(
                *(capture_native(tail, s, positions) for s in pair), strict=True
            )
            record = data[split][index]
            emit(
                "capture_parity",
                stage=stage,
                split=split,
                index=index,
                all_vocabulary_bit_identical=True,
            )
            native_rows = [
                score_record(output, index, bool(present), record, answer_ids)
                for present, output in enumerate(outputs)
            ]
            archived = check_archived_response(
                source,
                stage,
                split,
                index,
                manifest["completed_steps"],
                native_rows,
            )
            emit("archived_response_parity", stage=stage, split=split, index=index, **archived)
            if archived.get("available") and not archived["exact"]:
                raise RuntimeError("Replayed native answer scores differ from the archived runtime")
            for row in native_rows:
                rows.append(
                    {
                        "stage": stage,
                        "split": split,
                        "kind": "baseline",
                        **row,
                    }
                )
            for name in features[0]:
                absent, present = features[0][name], features[1][name]
                emit(
                    "activation_pair",
                    stage=stage,
                    split=split,
                    index=index,
                    site=name,
                    **vector_summary(absent, present, answer_direction, valid),
                )
                saved_last_tokens.append(
                    {
                        "stage": stage,
                        "split": split,
                        "index": index,
                        "site": name,
                        "absent": absent[:, -1, :].detach().cpu(),
                        "present": present[:, -1, :].detach().cpu(),
                    }
                )
                delta = present.float() - absent.float()

                evaluate = intervention_runner(
                    tail,
                    pair[0],
                    positions,
                    name,
                    {
                        "stage": stage,
                        "split": split,
                        "index": index,
                        "context_id": record["context_id"],
                    },
                    absent,
                    delta,
                    valid,
                    answer_ids,
                    rows,
                )

                for alpha in audit_manifest["interpolation_alphas"]:
                    output = evaluate(
                        interpolate(absent, present, alpha), "interpolation", alpha=alpha
                    )
                    if alpha in (0, 1):
                        assert_identical(
                            output, outputs[int(alpha)], "Native swap/interpolation endpoint"
                        )
                reverse = patched_logits(tail, pair[1], positions, name, absent)
                assert_identical(reverse, outputs[0], "Reverse native state swap")
                if name in ("input", f"block_{tail.first_layer}"):
                    last = torch.zeros_like(valid)
                    last[:, -1] = True
                    for label, mask in (("last_token", last), ("context_tokens", valid & ~last)):
                        evaluate(
                            token_patch(absent, present, mask),
                            "token_group_patch",
                            token_group=label,
                        )
                    if seen[split] < args.token_scan_per_split:
                        for token in valid[0].nonzero().flatten().tolist():
                            mask = torch.zeros_like(valid)
                            mask[:, token] = True
                            evaluate(
                                token_patch(absent, present, mask),
                                "single_token_patch",
                                token=token,
                            )
                if name == "input":
                    cohort = [i for s, i in selected if s == split]
                    donor_index = cohort[(cohort.index(index) + 1) % len(cohort)]
                    if donor_index != index:
                        donor_pair = cache["states"][split][donor_index]
                        donor_valid = (
                            cache["batches"][split][donor_index]["attention_mask"].bool().cuda()
                        )
                        direction = (donor_pair[1][0].float() - donor_pair[0][0].float()).cuda()
                        direction *= donor_valid.unsqueeze(-1)
                        matched = matched_direction(direction, delta, valid)
                        evaluate(
                            (absent.float() + matched).to(absent.dtype),
                            "donor_delta",
                            donor_index=donor_index,
                        )
                    for noise_seed in range(args.noise_seeds):
                        generator = torch.Generator().manual_seed(
                            args.seed + index * 1009 + noise_seed
                        )
                        noise = torch.randn(absent.shape, generator=generator).cuda()
                        matched = matched_direction(noise, delta, valid)
                        evaluate(
                            (absent.float() + matched).to(absent.dtype),
                            "random_delta",
                            noise_seed=noise_seed,
                        )
            seen[split] += 1
            completed.append([stage, split, index])
            write_json(args.out / "scores.json", {"rows": rows})
            torch.save(
                {"states": saved_last_tokens, "answer_direction": answer_direction.cpu()},
                args.out / "last_token_states.pt",
            )
            emit(
                "prompt_complete",
                stage=stage,
                split=split,
                index=index,
                scored_interventions=len(rows),
            )
        if audit_manifest["status"] == "time_limited":
            break
    if audit_manifest["status"] != "time_limited":
        audit_manifest["status"] = "complete"
    if tensor_digest(tail.head_weight) != digest:
        raise RuntimeError("Native output weights changed during the diagnostic")
    audit_manifest["completed"] = completed
    write_json(args.out / "manifest.json", audit_manifest)
    emit("complete", status=audit_manifest["status"], prompts=len(completed))


if __name__ == "__main__":
    main()
