"""Bidirectional native-tail head patching on fixed compiled-source states.

Methodological reference: SauersML/params circuit_finder.py at 4af1a97c.
No variable batching, changed tokens, prefix truncation, or new trained readout.
"""

import argparse
import json
import math
import time
from contextlib import contextmanager
from pathlib import Path

import torch

from watching.artifacts import write_json
from watching.execution_fit import tensor_digest
from watching.null_tail import validate_cache
from watching.runtime import runtime_provenance
from watching.suffix import assert_identical, move_state
from watching.tail_mechanism import (
    check_archived_response,
    load_selective_tail,
    response_scores,
    token_patch,
)
from watching.tail_train import (
    file_digest,
    install_learner_tail,
    score_record,
    validate_initial_tail,
)

PARAMS_REVISION = "4af1a97c7f3c9a3b9cf4a68fea127d1905937fd9"
PARAMS_SOURCE_SHA = "a27c7f8261cf803319a70be6b34337bae7bf127e39ef4a3bbe8eaddeae1fe7ae"
SOURCE_CACHE_SHA = "1d2e5cbee328d1dd6c6524b22ef6160855c4305cc6b7335e7bb2bdd54fda59ab"
TAIL_SHA = "8e78897991d3940067c7c63a99a00990f51d7c8c84d38919c82c609cd7af2b3a"


def component_modules(tail):
    result = {}
    for offset, layer in enumerate(tail.layers):
        block = tail.first_layer + offset
        result[f"{block}:heads"] = (layer.self_attn.o_proj, True)
        result[f"{block}:neurons"] = (layer.mlp.down_proj, True)
        result[f"{block}:attention"] = (layer.self_attn, False)
        result[f"{block}:mlp"] = (layer.mlp, False)
    return result


def output_tensor(value):
    tensor = value[0] if isinstance(value, tuple) else value
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3 or tensor.shape[0] != 1:
        raise ValueError("Expected singleton native sequence activation")
    return tensor


def capture_components(tail, state, positions):
    """Clone the actual native head/neuron inputs and attention/MLP outputs."""
    with torch.no_grad():
        baseline = tail(*state, positions)
    captured, handles = {}, []
    try:
        for name, (module, is_input) in component_modules(tail).items():
            if is_input:

                def pre(_module, args, name=name):
                    captured[name] = output_tensor(args[0]).detach().clone()
                    return None

                handles.append(module.register_forward_pre_hook(pre))
            else:

                def post(_module, _args, output, name=name):
                    captured[name] = output_tensor(output).detach().clone()
                    return None

                handles.append(module.register_forward_hook(post))
        with torch.no_grad():
            observed = tail(*state, positions)
        assert_identical(observed, baseline, "Circuit capture all-vocabulary parity")
    finally:
        for handle in handles:
            handle.remove()
    return baseline, captured


def replace_units(current, donor, *, heads=None, head_dim=None, position=None):
    if current.shape != donor.shape or current.dtype != donor.dtype:
        raise ValueError("Patch shape/dtype mismatch")
    if not torch.isfinite(donor).all():
        raise ValueError("Nonfinite donor activation")
    result = current.clone()
    if heads is not None:
        if position is None or not head_dim or current.shape[-1] % head_dim:
            raise ValueError("Head patch requires explicit position and valid head dimension")
        count = current.shape[-1] // head_dim
        if len(set(heads)) != len(heads) or any(h < 0 or h >= count for h in heads):
            raise ValueError("Invalid or duplicate head index")
        for head in heads:
            cols = slice(head * head_dim, (head + 1) * head_dim)
            result[:, position, cols] = donor[:, position, cols]
    elif position is None:
        result.copy_(donor)
    else:
        result[:, position, :] = donor[:, position, :]
    return result


@contextmanager
def component_patches(tail, patches):
    handles = []
    available = component_modules(tail)
    try:
        for name, specification in patches.items():
            if name not in available:
                raise ValueError(f"Unknown native patch site: {name}")
            module, is_input = available[name]
            if is_input:

                def pre(_module, args, specification=specification):
                    return (replace_units(args[0], **specification), *args[1:])

                handles.append(module.register_forward_pre_hook(pre))
            else:

                def post(_module, _args, output, specification=specification):
                    patched = replace_units(output_tensor(output), **specification)
                    return (patched, *output[1:]) if isinstance(output, tuple) else patched

                handles.append(module.register_forward_hook(post))
        yield
    finally:
        for handle in handles:
            handle.remove()


def run_patched(tail, state, positions, patches):
    with component_patches(tail, patches), torch.no_grad():
        return tail(*state, positions)


def paired_effects(baseline, patched, answer_ids, *, min_share_gap=1.0):
    base = [response_scores(logits, answer_ids) for logits in baseline]
    changed = [response_scores(logits, answer_ids) for logits in patched]
    gap = base[1]["yes_minus_no"] - base[0]["yes_minus_no"]
    sufficiency = changed[0]["yes_minus_no"] - base[0]["yes_minus_no"]
    necessity = base[1]["yes_minus_no"] - changed[1]["yes_minus_no"]
    return {
        "baseline_gap_nats": gap,
        "sufficiency_nats": sufficiency,
        "necessity_nats": necessity,
        "sufficiency_share": sufficiency / gap if gap >= min_share_gap else None,
        "necessity_share": necessity / gap if gap >= min_share_gap else None,
        "shares_defined": gap >= min_share_gap,
        "min_share_gap_nats": min_share_gap,
        "patched_absent": changed[0],
        "patched_present": changed[1],
    }


def rank_heads(rows, count=4):
    """Train-only selection, equal prompt weighting, raw nats including negative effects."""
    if not rows or any(row["split"] != "train" or row["kind"] != "single_head" for row in rows):
        raise ValueError("Head selection accepts training single-head measurements only")
    groups, indices = {}, {row["index"] for row in rows}
    for row in rows:
        key = (row["block"], row["head"])
        group = groups.setdefault(key, {})
        if row["index"] in group:
            raise ValueError("Duplicate prompt/head measurement")
        value = (row["sufficiency_nats"] + row["necessity_nats"]) / 2
        if not math.isfinite(value):
            raise ValueError("Nonfinite selection score")
        group[row["index"]] = value
    if any(set(group) != indices for group in groups.values()):
        raise ValueError("Each candidate must have every training prompt")
    ranked = sorted(
        [
            {"block": b, "head": h, "mean_bidirectional_nats": sum(v.values()) / len(v)}
            for (b, h), v in groups.items()
        ],
        key=lambda row: (-row["mean_bidirectional_nats"], row["block"], row["head"]),
    )
    if not 1 <= count <= len(ranked):
        raise ValueError("Invalid fixed group size")
    return {
        "selected": ranked[:count],
        "all_ranked": ranked,
        "training_indices": sorted(indices),
        "selection": "mean(sufficiency + necessity)/2 nats",
    }


def role_position(batch):
    valid = batch["attention_mask"][0].nonzero().flatten().tolist()
    ids = batch["input_ids"][0]
    if len(valid) < 2 or int(ids[valid[0]]) != 151644 or int(ids[valid[1]]) != 872:
        raise ValueError("Expected the unchanged initial <|im_start|> user role tokens")
    return valid[1]


def group_specs(group, donor, head_dim):
    specs = {}
    for row in group:
        name = f"{row['block']}:heads"
        item = specs.setdefault(
            name, {"donor": donor[name], "heads": [], "head_dim": head_dim, "position": -1}
        )
        item["heads"].append(row["head"])
    return specs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-results", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--max-seconds", type=float, default=900)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA native-tail execution required")
    if not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
        parser.error("Finite positive time cap required")
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        raise ValueError("Output must be empty")
    torch.set_num_threads(4)
    started = time.monotonic()

    def emit(event, **values):
        line = json.dumps(
            {"event": event, "seconds": time.monotonic() - started, **values}, allow_nan=False
        )
        print(line, flush=True)
        with (args.out / "metrics.jsonl").open("a") as stream:
            stream.write(line + "\n")

    source = args.source_results
    manifest = json.loads((source / "manifest.json").read_text())
    selected_manifest = json.loads(args.selection_manifest.read_text())
    selected = selected_manifest["selected"]
    if (
        selected_manifest["status"] != "complete"
        or len(selected) != 16
        or len({tuple(pair) for pair in selected}) != 16
        or [split for split, _ in selected] != ["train"] * 8 + ["test"] * 8
    ):
        raise ValueError("Expected exactly the complete 041r fixed 8+8 selection")
    if file_digest(source / "cache_states.pt") != SOURCE_CACHE_SHA:
        raise ValueError("Source cache differs from fixed released 034 cache")
    if file_digest(source / "native_tail.pt") != TAIL_SHA:
        raise ValueError("Native tail differs from fixed released 034 checkpoint")
    data = json.loads((source / "data.json").read_text())
    cache = torch.load(source / "cache_states.pt", map_location="cpu", weights_only=True)
    validate_cache(
        cache, data, json.loads((source / "cache_fingerprints.json").read_text()), manifest
    )
    tail, config = load_selective_tail(manifest["checkpoint"], args.revision, 2)
    if (
        tail.first_layer != 34
        or config.num_attention_heads != 32
        or manifest.get("train_head")
        or manifest.get("training_status") != "complete"
    ):
        raise ValueError("This protocol requires the completed native 034 two-block detector")
    head_dim = config.head_dim
    if cache["cut_layer"] != tail.first_layer - 1:
        raise ValueError("Wrong source cache boundary")
    head_sha = tensor_digest(tail.head_weight)
    artifact = torch.load(source / "native_tail.pt", map_location="cpu", weights_only=True)
    validate_initial_tail(
        artifact,
        tail,
        checkpoint=manifest["checkpoint"],
        model_commit=getattr(config, "_commit_hash", None),
        head_sha256=head_sha,
    )
    if (
        head_sha != manifest["head_sha256_before"]
        or artifact["global_step"] != manifest["ending_global_step"]
    ):
        raise ValueError("Source head or artifact step mismatch")
    install_learner_tail(tail, artifact)
    del artifact
    tail.cuda()
    answer_ids = (manifest["labels"]["No"], manifest["labels"]["Yes"])
    audit = {
        "status": "running",
        "source_cache_sha256": SOURCE_CACHE_SHA,
        "tail_sha256": TAIL_SHA,
        "head_sha256": head_sha,
        "revision": args.revision,
        "source_manifest_sha256": file_digest(source / "manifest.json"),
        "selection_manifest_sha256": file_digest(args.selection_manifest),
        "selected": selected,
        "params_method_revision": PARAMS_REVISION,
        "params_circuit_finder_sha256": PARAMS_SOURCE_SHA,
        "runtime_provenance": runtime_provenance(),
        "head_count": 64,
        "group_size": 4,
        "min_share_gap_nats": 1.0,
        "max_seconds": args.max_seconds,
        "runtime": "cached compiled-source states + native eager BF16 math-SDPA tail",
        "interpretation": (
            "Post-hoc circuit diagnostics, test singles exploratory; no population CI"
        ),
        "selection_rule": "Top4 train mean(sufficiency+necessity)/2 nats; no test selection",
        "source_prefix_checksum_available": False,
    }
    write_json(args.out / "manifest.json", audit)
    emit("setup", **audit)
    (args.out / "activations").mkdir()
    rows, baseline_rows, stored, selected_group = [], [], {}, None

    def get_pair(split, index):
        batch = cache["batches"][split][index]
        states = [move_state(s, "cuda") for s in cache["states"][split][index]]
        for left, right in zip(states[0][1:], states[1][1:], strict=True):
            if (left is None) != (right is None) or (
                left is not None and not torch.equal(left, right)
            ):
                raise ValueError("Paired mask/rotary tensors differ")
        return batch, states, batch["position_ids"].cuda()

    def record(kind, split, index, baseline, changed, **detail):
        row = {
            "kind": kind,
            "split": split,
            "index": index,
            **paired_effects(baseline, changed, answer_ids),
            **detail,
        }
        rows.append(row)
        emit("intervention", **row)
        return row

    def group_and_mediation(split, index):
        batch, pair, positions = get_pair(split, index)
        baseline = [v.cuda() for v in stored[split, index]["baseline"]]
        features = [{k: v.cuda() for k, v in f.items()} for f in stored[split, index]["features"]]
        changed = [
            run_patched(
                tail, pair[c], positions, group_specs(selected_group, features[1 - c], head_dim)
            )
            for c in (0, 1)
        ]
        record("selected_head_group", split, index, baseline, changed, group=selected_group)
        role = role_position(batch)
        mask = torch.zeros_like(batch["attention_mask"], dtype=torch.bool, device="cuda")
        mask[:, role] = True
        role_outputs, blocked = [], []
        for c in (0, 1):
            modified = (token_patch(pair[c][0], pair[1 - c][0], mask), *pair[c][1:])
            role_outputs.append(run_patched(tail, modified, positions, {}))
            # Restore selected heads to recipient-condition values while retaining
            # the upstream user-token intervention. This tests path mediation.
            blocked.append(
                run_patched(
                    tail, modified, positions, group_specs(selected_group, features[c], head_dim)
                )
            )
            restored = run_patched(
                tail, pair[c], positions, group_specs(selected_group, features[c], head_dim)
            )
            assert_identical(restored, baseline[c], "Selected-head self-patch parity")
        user = paired_effects(baseline, role_outputs, answer_ids)
        held = paired_effects(baseline, blocked, answer_ids)
        record(
            "user_patch_with_heads_restored",
            split,
            index,
            baseline,
            blocked,
            group=selected_group,
            role_only_effects=user,
            sufficiency_blocked_nats=user["sufficiency_nats"] - held["sufficiency_nats"],
            necessity_blocked_nats=user["necessity_nats"] - held["necessity_nats"],
        )
        for c in (0, 1):
            assert_identical(
                run_patched(tail, pair[c], positions, {}),
                baseline[c],
                "Post-group restoration parity",
            )
        emit("group_checks_passed", split=split, index=index)

    for split, index in selected:
        if time.monotonic() - started > args.max_seconds:
            audit["status"] = "time_limited"
            break
        if split == "test" and selected_group is None:
            selection = rank_heads(
                [r for r in rows if r["kind"] == "single_head" and r["split"] == "train"]
            )
            selected_group = selection["selected"]
            write_json(args.out / "selection.json", selection)
            emit("head_group_locked_before_test", **selection)
            for train_split, train_index in selected[:8]:
                group_and_mediation(train_split, train_index)
        batch, pair, positions = get_pair(split, index)
        baseline, features = zip(
            *(capture_components(tail, s, positions) for s in pair), strict=True
        )
        for captured in features:
            for block in (34, 35):
                if captured[f"{block}:heads"].shape != (
                    1,
                    pair[0][0].shape[1],
                    config.num_attention_heads * head_dim,
                ):
                    raise ValueError("Native o_proj input is not the expected query-head layout")
        native_rows = [
            score_record(v, index, bool(c), data[split][index], answer_ids)
            for c, v in enumerate(baseline)
        ]
        archived = check_archived_response(
            source, "trained", split, index, manifest["completed_steps"], native_rows
        )
        emit(
            "capture_and_archive_parity",
            split=split,
            index=index,
            capture_all_vocab=True,
            **archived,
        )
        if not archived["available"] or not archived["exact"]:
            raise RuntimeError("Native circuit baseline differs from archived 034 scores")
        baseline_rows.extend({"split": split, **row} for row in native_rows)
        write_json(args.out / "baselines.json", {"rows": baseline_rows})
        for c in (0, 1):
            for name, value in features[c].items():
                assert_identical(
                    run_patched(tail, pair[c], positions, {name: {"donor": value}}),
                    baseline[c],
                    "All-component self-patch parity",
                )
            swapped = (pair[1 - c][0], *pair[c][1:])
            assert_identical(
                run_patched(tail, swapped, positions, {}),
                baseline[1 - c],
                "Whole-input bidirectional swap",
            )
        stored[split, index] = {
            "baseline": [v.detach().cpu() for v in baseline],
            "features": [{k: v.detach().cpu() for k, v in f.items()} for f in features],
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
        }
        torch.save(stored[split, index], args.out / "activations" / f"{split}_{index}.pt")
        role = role_position(batch)
        mask = torch.zeros_like(batch["attention_mask"], dtype=torch.bool, device="cuda")
        mask[:, role] = True
        changed = [
            run_patched(
                tail, (token_patch(pair[c][0], pair[1 - c][0], mask), *pair[c][1:]), positions, {}
            )
            for c in (0, 1)
        ]
        record("role_user_input", split, index, baseline, changed, role_position=role, token_id=872)
        first_position = int(batch["attention_mask"][0].nonzero()[0, 0])
        if not torch.equal(pair[0][0][:, first_position], pair[1][0][:, first_position]):
            raise ValueError("Expected identical first chat-marker states for zero-delta control")
        zero_mask = torch.zeros_like(mask)
        zero_mask[:, first_position] = True
        zero_outputs = [
            run_patched(
                tail,
                (token_patch(pair[c][0], pair[1 - c][0], zero_mask), *pair[c][1:]),
                positions,
                {},
            )
            for c in (0, 1)
        ]
        for c in (0, 1):
            assert_identical(zero_outputs[c], baseline[c], "First-token zero-delta control")
        record(
            "first_marker_zero_delta",
            split,
            index,
            baseline,
            zero_outputs,
            token_position=first_position,
            token_id=151644,
        )
        for block in (34, 35):
            for site in ("attention", "mlp"):
                name = f"{block}:{site}"
                for position in (-1, None):
                    changed = [
                        run_patched(
                            tail,
                            pair[c],
                            positions,
                            {name: {"donor": features[1 - c][name], "position": position}},
                        )
                        for c in (0, 1)
                    ]
                    record(
                        "component_reference",
                        split,
                        index,
                        baseline,
                        changed,
                        block=block,
                        site=site,
                        positions="answer" if position == -1 else "all",
                    )
            for head in range(config.num_attention_heads):
                name = f"{block}:heads"
                changed = [
                    run_patched(
                        tail,
                        pair[c],
                        positions,
                        {
                            name: {
                                "donor": features[1 - c][name],
                                "heads": [head],
                                "head_dim": head_dim,
                                "position": -1,
                            }
                        },
                    )
                    for c in (0, 1)
                ]
                record(
                    "single_head",
                    split,
                    index,
                    baseline,
                    changed,
                    block=block,
                    head=head,
                    test_role="exploratory_only" if split == "test" else "selection",
                )
        if split == "test":
            group_and_mediation(split, index)
        for c in (0, 1):
            assert_identical(
                run_patched(tail, pair[c], positions, {}),
                baseline[c],
                "Final prompt restoration parity",
            )
        emit("prompt_complete", split=split, index=index, rows=len(rows))
        write_json(args.out / "scores.json", {"rows": rows})
    else:
        audit["status"] = "complete"
    if tensor_digest(tail.head_weight) != head_sha:
        raise RuntimeError("Native output matrix changed")
    audit["selected_group"] = selected_group
    audit["recorded_rows"] = len(rows)
    write_json(args.out / "scores.json", {"rows": rows})
    write_json(args.out / "manifest.json", audit)
    emit("complete", status=audit["status"], rows=len(rows))


if __name__ == "__main__":
    main()
