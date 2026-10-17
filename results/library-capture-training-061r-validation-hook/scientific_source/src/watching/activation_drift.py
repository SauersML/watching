"""Compare the probe footprint across weights using one fixed compiled graph.

Only the gradient-enabled path is measured. Capturing the head input must
preserve every baseline logit, for every checkpoint, prompt, and condition.
"""

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

from watching.artifacts import write_json
from watching.audit import FeatureProbe
from watching.data import make_expanded_data
from watching.probe import ProbeModel
from watching.train import source_hashes


def footprint_comparison(initial, final, initial_weights, final_weights):
    """Decompose FP64 head projections; these are not the actual bf16 logits."""
    a, b = initial.double(), final.double()
    wa = initial_weights[1].double() - initial_weights[0].double()
    wb = final_weights[1].double() - final_weights[0].double()
    da, db = a[1] - a[0], b[1] - b[0]
    denominator = da.norm() * db.norm()
    return {
        "initial_delta_norm": float(da.norm()),
        "final_delta_norm": float(db.norm()),
        "delta_cosine": float(da @ db / denominator) if denominator > 0 else None,
        "delta_change_norm": float((db - da).norm()),
        "midpoint_change_norm": float((b.mean(0) - a.mean(0)).norm()),
        "answer_direction_change_norm": float((wb - wa).norm()),
        "initial_projection": float(wa @ da),
        "final_projection": float(wb @ db),
        "head_change_at_initial_features": float((wb - wa) @ da),
        "feature_change_at_final_head": float(wb @ (db - da)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prompts", type=int, default=8)
    parser.add_argument("--layer", type=int, default=14)
    args = parser.parse_args()
    if len(args.checkpoints) < 2 or not 1 <= args.prompts <= 32:
        parser.error("Use at least two checkpoints and 1–32 prompts per split")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA Inductor is required")
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        raise ValueError("Output directory must be empty")
    torch.manual_seed(20261015)
    torch.set_num_threads(4)
    start = time.monotonic()

    def emit(event, **fields):
        row = dict(event=event, seconds=time.monotonic() - start, **fields)
        line = json.dumps(row, allow_nan=False)
        print(line, flush=True)
        with (args.out / "audit.jsonl").open("a") as stream:
            stream.write(line + "\n")

    tok = AutoTokenizer.from_pretrained(args.checkpoints[0], padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    labels = [tok.encode(s, add_special_tokens=False) for s in ("No", "Yes")]
    if any(len(ids) != 1 for ids in labels):
        raise ValueError("Answer labels must be single tokens")
    answer_ids = [ids[0] for ids in labels]
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.checkpoints[0], dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .cuda()
        .eval()
    )
    model.config.use_cache = False
    model.requires_grad_(True)
    emit(
        "setup",
        checkpoints=[str(p) for p in args.checkpoints],
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name(),
        source_sha256=source_hashes(),
        tied_embeddings=model.lm_head.weight is model.model.embed_tokens.weight,
    )
    records = [
        (split, i, r["text"])
        for split in ("train", "validation")
        for i, r in enumerate(make_expanded_data()[split][: args.prompts])
    ]
    write_json(args.out / "prompts.json", records)
    batches = []
    for _, _, text in records:
        rendered = tok.apply_chat_template(
            [dict(role="user", content=text)],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if len(tok.encode(rendered, add_special_tokens=False)) > 64:
            raise ValueError("Prompt exceeds sequence length")
        batch = tok(
            rendered,
            add_special_tokens=False,
            padding="max_length",
            max_length=64,
            return_tensors="pt",
        ).to("cuda")
        batch["position_ids"] = (batch["attention_mask"].cumsum(-1) - 1).clamp_min(0)
        batches.append(dict(batch))

    def load_weights(path):
        # Copy in place: parameter identities and compiled code remain fixed.
        with safe_open(path / "model.safetensors", framework="pt") as weights:
            parameters = dict(model.named_parameters())
            if set(parameters) != set(weights.keys()):
                raise ValueError("Checkpoint must contain exactly the unique model parameters")
            with torch.no_grad():
                for name, parameter in parameters.items():
                    value = weights.get_tensor(name)
                    if value.shape != parameter.shape or value.dtype != parameter.dtype:
                        raise ValueError(f"Checkpoint shape/dtype mismatch: {name}")
                    parameter.copy_(value)

    baseline, heads, features = {}, {}, {}
    measured = ProbeModel(model, args.layer)
    compiled = torch.compile(measured, fullgraph=True, dynamic=False)
    for state, path in enumerate(args.checkpoints):
        load_weights(path)
        emit("baseline_start", checkpoint=state)
        heads[state] = model.lm_head.weight[answer_ids].detach().cpu()
        for i, batch in enumerate(batches):
            for capture in (False, True):
                logits, saved = compiled(**batch, capture=capture)
                if len(saved) != int(capture):
                    raise RuntimeError("Probe capture count mismatch")
                baseline[state, i, capture] = logits.detach().cpu()
                measured.captured = []
                del logits, saved
        emit("baseline_done", checkpoint=state)
    measured.handle.remove()
    del compiled, measured
    gc.collect()
    torch.cuda.empty_cache()
    measured = FeatureProbe(model, args.layer)
    compiled = torch.compile(measured, fullgraph=True, dynamic=False)
    for state, path in enumerate(args.checkpoints):
        load_weights(path)
        emit("features_start", checkpoint=state)
        for i, batch in enumerate(batches):
            for capture in (False, True):
                logits, saved, hidden = compiled(**batch, capture=capture)
                if len(saved) != int(capture):
                    raise RuntimeError("Probe capture count mismatch")
                if not torch.equal(logits.detach().cpu(), baseline[state, i, capture]):
                    raise RuntimeError("Feature extraction changed logits; audit is confounded")
                features[state, i, capture] = hidden.detach().cpu()
                measured.captured, measured.features = [], None
                del logits, saved, hidden
        emit("feature_parity", checkpoint=state, all_logits_identical=True)
    rows = []
    for state in range(len(args.checkpoints)):
        for i, (split, index, _) in enumerate(records):
            pair = torch.cat([features[state, i, c] for c in (False, True)])
            first = torch.cat([features[0, i, c] for c in (False, True)])
            logits = torch.cat([baseline[state, i, c] for c in (False, True)]).float()
            odds = logits[:, answer_ids[1]] - logits[:, answer_ids[0]]
            rows.append(
                dict(
                    checkpoint=state,
                    split=split,
                    index=index,
                    actual_separation=float(odds[1] - odds[0]),
                    **footprint_comparison(first, pair, heads[0], heads[state]),
                )
            )
    write_json(args.out / "analysis.json", rows)
    torch.save(
        dict(features=features, answer_weights=heads, records=records),
        args.out / "features.pt.partial",
    )
    (args.out / "features.pt.partial").replace(args.out / "features.pt")
    emit("complete", first_prompt=[r for r in rows if r["split"] == "train" and r["index"] == 0])


if __name__ == "__main__":
    main()
