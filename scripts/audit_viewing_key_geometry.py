"""One fixed development pair, existing CPU assets only; descriptive key geometry."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from audit_prefix_key_geometry import base_key_weight, geometry, rms, sha256
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-fit", type=Path, required=True)
    parser.add_argument("--storage-fit", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    config = Qwen3Config.from_pretrained(args.model_dir, local_files_only=True)
    config._attn_implementation = "sdpa"
    if (config.num_hidden_layers, config.num_key_value_heads, config.head_dim) != (36, 8, 128):
        raise ValueError("Expected native Qwen3-8B")
    base_wk, base_info = base_key_weight(args.model_dir)
    result = dict(
        selection="Development index0, selected without inspecting its score",
        scope="CPU BF16 math-SDPA block34 replay plus native block35 key projection; "
        "descriptive only, not native GPU replication or causal attribution",
        hypothesis="Prior034 query head6 uses KVhead1; no head selection from052 scores",
        source_instances_differ=True,
        torch=torch.__version__,
        threads=4,
        base_key_weight=base_info,
        tails={},
    )
    raw = {}
    context = None
    for name, fit in (("reference034", args.reference_fit), ("storage052", args.storage_fit)):
        cache = torch.load(
            fit / "cache_states.pt", map_location="cpu", weights_only=True, mmap=True
        )
        tail = torch.load(fit / "native_tail.pt", map_location="cpu", weights_only=True, mmap=True)
        if cache["cut_layer"] != 33 or tail["first_layer"] != 34 or tail.get("train_head"):
            raise ValueError(
                "Expected fixed-head two-block checkpoint and original source boundary"
            )
        record = cache["data"]["validation"][0]
        if context is not None and record != context:
            raise ValueError("The two runs do not use the same selected prompt")
        context = record
        batch = cache["batches"]["validation"][0]
        active = batch["attention_mask"][0].nonzero().flatten()
        if batch["input_ids"][0, active[:3]].tolist() != [151644, 872, 198]:
            raise ValueError("Expected actual first user role token")
        role = int(active[1])
        weights = tail["layers"]
        with torch.device("meta"):
            block = Qwen3DecoderLayer(config, 34)
        block.load_state_dict(
            {k[2:]: v for k, v in weights.items() if k.startswith("0.")}, assign=True
        )
        block.eval().requires_grad_(False)
        pair = []
        with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
            for hidden, mask, cos, sin in cache["states"]["validation"][0]:
                output = block(
                    hidden,
                    attention_mask=mask,
                    position_ids=batch["position_ids"],
                    position_embeddings=(cos, sin),
                    use_cache=False,
                )
                normalized = rms(output, weights["1.input_layernorm.weight"], config.rms_norm_eps)
                key = F.linear(normalized, weights["1.self_attn.k_proj.weight"]).view(1, 96, 8, 128)
                original_key = F.linear(normalized, base_wk).view_as(key)
                key_norm = rms(key, weights["1.self_attn.k_norm.weight"], config.rms_norm_eps)
                pair.append(
                    dict(
                        source=hidden[0, role].clone(),
                        block35_input=output[0, role].clone(),
                        raw_key=key[0, role].clone(),
                        base_key_same_input=original_key[0, role].clone(),
                        normalized_key=key_norm[0, role].clone(),
                    )
                )
        heads = []
        for head in range(8):
            measurements = {
                field: geometry(pair[0][field][head], pair[1][field][head])
                for field in ("raw_key", "base_key_same_input", "normalized_key")
            }
            measurements.update(
                kv_head=head,
                midpoint_fraction_of_base=(
                    measurements["raw_key"]["midpoint_norm"]
                    / measurements["base_key_same_input"]["midpoint_norm"]
                ),
            )
            heads.append(measurements)
        result["tails"][name] = dict(
            context_id=record["context_id"],
            physical_role_position=role,
            source=geometry(pair[0]["source"], pair[1]["source"]),
            block35_input=geometry(pair[0]["block35_input"], pair[1]["block35_input"]),
            heads=heads,
            cache_sha256=sha256(fit / "cache_states.pt"),
            tail_sha256=sha256(fit / "native_tail.pt"),
        )
        raw[name] = pair
        print(json.dumps({"tail": name, "kv_head1": heads[1]}), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(raw, args.out / "role_vectors.pt")
    result["raw_sha256"] = sha256(args.out / "role_vectors.pt")
    result["script_sha256"] = sha256(Path(__file__))
    (args.out / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
