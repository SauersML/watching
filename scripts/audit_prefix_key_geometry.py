"""CPU-only descriptive key geometry on fixed GPU-cached source states."""

import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from safetensors import safe_open
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoTokenizer, Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

GROUPS = ("single_user", "system_user", "multiturn", "system_multiturn")
TASKS = ("detection", "animal")


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def select_pairs(records):
    """First lexical development base per task, without consulting model scores."""
    selected = []
    for task in TASKS:
        base = min(r["base_context_id"] for r in records if r["task"] == task)
        rows = [(i, r) for i, r in enumerate(records) if r["base_context_id"] == base]
        if len(rows) != 4 or {r["group"] for _, r in rows} != set(GROUPS):
            raise ValueError("Each selected base requires exactly four matched chat layouts")
        selected.extend(sorted(rows, key=lambda item: GROUPS.index(item[1]["group"])))
    return selected


def geometry(absent, present):
    """FP64 summaries of already computed vectors, not FP64 model execution."""
    a, b = absent.double(), present.double()
    if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("Geometry needs finite matching vectors")
    midpoint, delta = (a + b) / 2, b - a
    na, nb = float(a.norm()), float(b.norm())
    denominator = na * nb
    cosine = min(1.0, max(-1.0, float(a @ b) / denominator)) if denominator else None
    middle, difference = float(midpoint.norm()), float(delta.norm())
    return {
        "absent_norm": na,
        "present_norm": nb,
        "midpoint_norm": middle,
        "difference_norm": difference,
        "difference_over_midpoint": difference / middle if middle else None,
        "condition_cosine": cosine,
        "condition_angle_degrees": math.degrees(math.acos(cosine)) if cosine is not None else None,
        "fraction_different": float((a != b).double().mean()),
    }


def rms(value, weight, epsilon):
    source = value.float()
    normalized = source * torch.rsqrt(source.square().mean(-1, keepdim=True) + epsilon)
    return weight * normalized.to(value.dtype)


def base_key_weight(directory):
    name = "model.layers.35.self_attn.k_proj.weight"
    index = json.loads((directory / "model.safetensors.index.json").read_text())
    path = directory / index["weight_map"][name]
    with safe_open(path, framework="pt", device="cpu") as archive:
        weight = archive.get_tensor(name)
    return weight, {
        "source_file": path.name,
        "tensor": name,
        "tensor_sha256": hashlib.sha256(weight.view(torch.uint8).numpy().tobytes()).hexdigest(),
    }


def add_repeat_and_cancellation_checks(report, raw):
    for name, tail in report["tails"].items():
        repeats = {}
        for role in ("user", "system"):
            rows = [row for row in tail["rows"] if row["role_text"] == role]
            reference = raw[f"{name}/{rows[0]['index']}"]
            repeats[role] = {
                "variants": len(rows),
                "all_recorded_vectors_bit_identical": all(
                    torch.equal(
                        raw[f"{name}/{row['index']}"][condition][field], reference[condition][field]
                    )
                    for row in rows
                    for condition in (0, 1)
                    for field in reference[condition]
                ),
            }
        tail["repeated_role_geometry"] = repeats
        for row in tail["rows"]:
            pair = raw[f"{name}/{row['index']}"]
            for head in row["heads"]:
                index = head["kv_head"]
                baseline = (
                    pair[0]["base_wk_same_input"][index].double()
                    + pair[1]["base_wk_same_input"][index].double()
                ) / 2
                trained = (pair[0]["raw_k"][index].double() + pair[1]["raw_k"][index].double()) / 2
                change = trained - baseline
                head["projection_midpoint_change"] = {
                    "change_norm": float(change.norm()),
                    "base_vs_change_cosine": geometry(baseline, change)["condition_cosine"],
                    "scope": "difference of separately BF16-rounded projections on identical input",
                }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--tail", action="append", required=True, help="Name=local checkpoint path")
    parser.add_argument(
        "--model-dir", type=Path, required=True, help="Existing pinned local HF snapshot"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    started = time.monotonic()
    data = json.loads(args.data.read_text())
    selected = select_pairs(data["validation"])
    args.out.mkdir(parents=True, exist_ok=False)
    plan = [
        {"index": i, **{k: r[k] for k in ("task", "base_context_id", "context_id", "group")}}
        for i, r in selected
    ]
    (args.out / "selection.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps({"event": "selection_frozen", "pairs": plan}), flush=True)
    cache = torch.load(args.cache, map_location="cpu", weights_only=True, mmap=True)
    if cache["data"]["validation"] != data["validation"]:
        raise ValueError("Cache metadata does not match the supplied development records")
    if cache.get("cut_layer") != 33:
        raise ValueError("Expected native block34 inputs captured after frozen block33")
    config = Qwen3Config.from_pretrained(args.model_dir, local_files_only=True)
    config._attn_implementation = "sdpa"
    if (
        config.num_hidden_layers,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
    ) != (36, 32, 8, 128):
        raise ValueError("This predeclared audit targets Qwen3-8B's final two blocks")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    base_weight, base_provenance = base_key_weight(args.model_dir)
    report = {
        "scope": (
            "CPU BF16 eager math-SDPA replay from GPU-cached block34 inputs; "
            "not native GPU replication"
        ),
        "selection_rule": (
            "first lexical development base_context_id per task, all four layouts, both conditions"
        ),
        "selection": plan,
        "runtime": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "platform": platform.platform(),
            "threads": torch.get_num_threads(),
            "model_dtype": "bfloat16",
            "summary_dtype": "float64",
            "compiled": False,
        },
        "cache_sha256": sha256(args.cache),
        "data_sha256": sha256(args.data),
        "base_weight": base_provenance,
        "base_weight_control": (
            "Original base Wk applied to the SAME trained block34 output "
            "and trained block35 input norm; not a full base-model forward"
        ),
        "head_mapping": "query heads4–7 share KVhead1; reported query head6 maps to KVhead1",
        "tails": {},
    }
    raw = {}
    for specification in args.tail:
        name, filename = specification.split("=", 1)
        path = Path(filename)
        artifact = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if artifact["first_layer"] != 34 or artifact.get("train_head"):
            raise ValueError("Expected a fixed-head two-block artifact beginning at block34")
        weights = artifact["layers"]
        with torch.device("meta"):
            block = Qwen3DecoderLayer(config, 34)
        block.load_state_dict(
            {k[2:]: v for k, v in weights.items() if k.startswith("0.")}, assign=True
        )
        block.eval().requires_grad_(False)
        wk = weights["1.self_attn.k_proj.weight"]
        norm = weights["1.input_layernorm.weight"]
        knorm = weights["1.self_attn.k_norm.weight"]
        if any(value.dtype != torch.bfloat16 for value in (wk, norm, knorm, base_weight)):
            raise ValueError("Expected BF16 trained and base weights")
        rows = []
        for index, record in selected:
            batch = cache["batches"]["validation"][index]
            active = batch["attention_mask"][0].nonzero().flatten()
            role_index = int(active[0]) + 1
            role_id = int(batch["input_ids"][0, role_index])
            role = tokenizer.decode([role_id])
            expected = "system" if record["group"].startswith("system") else "user"
            if role != expected or int(batch["position_ids"][0, role_index]) != 1:
                raise ValueError("Actual first-role token does not match the predeclared position")
            pair = []
            with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
                for state in cache["states"]["validation"][index]:
                    hidden, mask, cos, sin = state
                    if hidden.dtype != torch.bfloat16:
                        raise ValueError("Expected BF16 cached hidden states")
                    output = block(
                        hidden,
                        attention_mask=mask,
                        position_ids=batch["position_ids"],
                        position_embeddings=(cos, sin),
                        use_cache=False,
                    )
                    # Project the full fixed-shape sequence before selecting the role token.
                    u = rms(output, norm, config.rms_norm_eps)
                    k = F.linear(u, wk).view(1, hidden.shape[1], 8, 128)
                    original_k = F.linear(u, base_weight).view_as(k)
                    post = rms(k, knorm, config.rms_norm_eps)
                    pair.append(
                        {
                            "cached_input": hidden[0, role_index].clone(),
                            "block35_input": output[0, role_index].clone(),
                            "pre_wk": u[0, role_index].clone(),
                            "raw_k": k[0, role_index].clone(),
                            "base_wk_same_input": original_k[0, role_index].clone(),
                            "normalized_k": post[0, role_index].clone(),
                        }
                    )
            item = {
                "index": index,
                "task": record["task"],
                "group": record["group"],
                "base_context_id": record["base_context_id"],
                "role_token_id": role_id,
                "role_text": role,
                "physical_position": role_index,
                "logical_position": 1,
                "inputs": {
                    key: geometry(pair[0][key], pair[1][key])
                    for key in ("cached_input", "block35_input", "pre_wk")
                },
                "heads": [],
            }
            for head in range(8):
                before = geometry(pair[0]["raw_k"][head], pair[1]["raw_k"][head])
                after = geometry(pair[0]["normalized_k"][head], pair[1]["normalized_k"][head])
                original = geometry(
                    pair[0]["base_wk_same_input"][head], pair[1]["base_wk_same_input"][head]
                )
                item["heads"].append(
                    {
                        "kv_head": head,
                        "raw": before,
                        "normalized": after,
                        "base_wk_same_input": original,
                        "raw_midpoint_fraction_of_base_projection": before["midpoint_norm"]
                        / original["midpoint_norm"]
                        if original["midpoint_norm"]
                        else None,
                        "norm_difference_amplification": after["difference_norm"]
                        / before["difference_norm"]
                        if before["difference_norm"]
                        else None,
                    }
                )
            rows.append(item)
            raw[f"{name}/{index}"] = pair
            print(
                json.dumps(
                    {
                        "event": "pair_complete",
                        "tail": name,
                        "index": index,
                        "group": record["group"],
                        "kv_head1": item["heads"][1],
                        "seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
        report["tails"][name] = {
            "artifact_sha256": sha256(path),
            "completed_steps": artifact.get("completed_steps"),
            "rows": rows,
        }
        (args.out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        del block, artifact, weights
    torch.save(raw, args.out / "role_vectors.pt")
    report["role_vectors_sha256"] = sha256(args.out / "role_vectors.pt")
    add_repeat_and_cancellation_checks(report, raw)
    report["seconds"] = time.monotonic() - started
    (args.out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"event": "complete", "seconds": report["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
