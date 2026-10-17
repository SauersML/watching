"""Bounded, post-hoc formatting diagnostic for all eight 040 final-line prompts.

This never changes 040's first-token result. It records complete continuations,
including failures and budget exhaustion, from the fixed 039 model.
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from watching.artifacts import write_json
from watching.execution_fit import checkpoint_provenance
from watching.runtime import runtime_provenance
from watching.tail_train import (
    context_key,
    difference,
    file_digest,
    record_answer_ids,
    score_record,
)
from watching.validate_tail import MODES, ValidationProbe, install_checkpoint

FAMILY = "final_line_answer"


def diagnostic_records(data):
    """Select the entire predeclared family without consulting model scores."""
    selected = [
        (index, row)
        for index, row in enumerate(data["test"])
        if row.get("task") == "animal" and row.get("question_family") == FAMILY
    ]
    if len(selected) != 8 or len({context_key(row) for _, row in selected}) != 8:
        raise ValueError("The diagnostic requires all eight distinct animal final-line prompts")
    if any((r["answer_negative"], r["answer_positive"]) != ("cat", "owl") for _, r in selected):
        raise ValueError("Expected unchanged cat/owl labels")
    return selected


def fixed_batch(tokens, pad_token_id, length, device):
    """Append generated tokens by consuming left padding, never dropping prompt tokens."""
    if not tokens or len(tokens) > length:
        raise ValueError("Token sequence does not fit fixed shape; truncation is forbidden")
    padding = length - len(tokens)
    ids = torch.tensor([[pad_token_id] * padding + tokens], dtype=torch.long, device=device)
    mask = torch.tensor([[0] * padding + [1] * len(tokens)], dtype=torch.long, device=device)
    return {
        "input_ids": ids,
        "attention_mask": mask,
        "position_ids": (mask.cumsum(-1) - 1).clamp_min(0),
    }


def final_line_result(text, target, stop_reason):
    """Whitespace only: no case folding, punctuation removal or semantic judging."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    last = lines[-1] if lines else ""
    resolved = stop_reason == "eos"
    return {
        "last_nonempty_line": last,
        "literal_target_match": last == target,
        "resolved": resolved,
        "final_line_correct": last == target if resolved else None,
        "truncated_or_unresolved": not resolved,
    }


def first_step_comparison(current, previous):
    fields = ("logp_negative", "logp_positive", "positive_minus_negative")
    deltas = {field: current[field] - previous[field] for field in fields}
    same_token = current["top_token_id"] == previous["top_token_id"]
    return {
        "top_token_matches": same_token,
        "score_deltas": deltas,
        "all_recorded_first_step_values_identical": same_token
        and all(d == 0 for d in deltas.values()),
        "scope": (
            "Recorded answer logprobs, pair logit difference and top token; "
            "not full-vocabulary equality"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation", type=Path, required=True, help="040 fresh039 archive")
    parser.add_argument("--tail", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()
    if args.max_new_tokens != 32:
        parser.error("The predeclared diagnostic uses exactly a 32-token cap")
    source = args.confirmation
    original = json.loads((source / "manifest.json").read_text())
    data = json.loads((source / "data.json").read_text())
    selected = diagnostic_records(data)
    prior_rows = json.loads((source / "trained_reference.json").read_text())["rows"]
    prior = {(r["context_id"], r["capture"]): r for r in prior_rows}
    if file_digest(args.tail) != original["tail_sha256"]:
        raise ValueError("Native tail differs from the frozen 040 model")
    if original["length"] != 96 or not original.get("revision"):
        raise ValueError("Expected pinned 040 model revision and length 96")
    for _, row in selected:
        for present in (False, True):
            if (context_key(row), present) not in prior:
                raise ValueError("Missing original paired score for first-step comparison")
    if not torch.cuda.is_available():
        raise RuntimeError("The native compiled diagnostic requires CUDA")
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        raise ValueError("Diagnostic output must be empty")
    import torch._inductor.config as inductor_config

    inductor_config.force_disable_caches = True
    inductor_config.freezing = False
    torch.manual_seed(original["seed"])
    torch.set_num_threads(4)
    started = time.monotonic()

    def emit(event, **fields):
        line = json.dumps(
            {"event": event, "seconds": time.monotonic() - started, **fields}, allow_nan=False
        )
        print(line, flush=True)
        with (args.out / "metrics.jsonl").open("a") as stream:
            stream.write(line + "\n")

    checkpoint, revision = original["checkpoint"], original["revision"]
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, revision=revision, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    def encode(row):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["text"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) >= 96:
            raise ValueError("No space for bounded continuation without truncation")
        return tokens

    prompt_tokens = {context_key(row): encode(row) for _, row in selected}
    anchor_tokens = encode(data["test"][0])
    emit("loading", n_prompts=len(selected), checkpoint=checkpoint, revision=revision)
    model = (
        AutoModelForCausalLM.from_pretrained(
            checkpoint,
            revision=revision,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .cuda()
        .eval()
    )
    model.config.use_cache = False
    model.requires_grad_(True)
    wrapper = ValidationProbe(model, original["probe_layer"])
    anchor = fixed_batch(anchor_tokens, tokenizer.pad_token_id, 96, "cuda")
    wrapper.prepare_buffer(anchor)
    compiled = torch.compile(wrapper, fullgraph=True, dynamic=False)

    def run(tokens, mode):
        with torch.no_grad():
            logits, saved = compiled(
                **fixed_batch(tokens, tokenizer.pad_token_id, 96, "cuda"), mode=mode
            )
        if len(saved) != int(mode == "reference") or not torch.isfinite(logits).all():
            raise RuntimeError("Invalid native probe capture or logits")
        result = logits.detach().cpu().clone()
        wrapper.captured = []
        return result

    # Compile base modes before installing the same tail, as in 040.
    for mode in MODES:
        run(anchor_tokens, mode)
    artifact = torch.load(args.tail, map_location="cpu", weights_only=True)
    installation = install_checkpoint(model, artifact, checkpoint, original["probe_layer"])
    del artifact
    eos = model.generation_config.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
    if tokenizer.eos_token_id is not None:
        eos_ids.add(tokenizer.eos_token_id)
    if not eos_ids:
        raise ValueError("A bounded completion requires explicit EOS token IDs")
    manifest = {
        "source_files_sha256": {
            name: file_digest(source / name)
            for name in ("data.json", "manifest.json", "trained_reference.json")
        },
        "tail_sha256": file_digest(args.tail),
        "checkpoint": checkpoint,
        "revision": revision,
        "installation": installation,
        "runtime_provenance": runtime_provenance(),
        "model_provenance": checkpoint_provenance(
            checkpoint, getattr(model.config, "_commit_hash", None), revision=revision
        ),
        "runtime": "ordinary full-model compiled singleton BF16; no gradients; no KV cache",
        "length": 96,
        "max_new_tokens": 32,
        "eos_token_ids": sorted(eos_ids),
        "n_prompts": 8,
        "n_executions": 16,
        "family": FAMILY,
        "selection": "All eight family prompts, irrespective of original success or failure",
        "grading": (
            "After EOS only, last nonempty line stripped of whitespace equals "
            "literal lowercase target; no punctuation/case normalization"
        ),
        "primary_result_unchanged": "040 animal first-token correctness remains 119/128",
        "interpretation": (
            "Post-hoc formatting diagnostic; no new confirmatory generalization claim"
        ),
        "seed": original["seed"],
        "compiler_weight_freezing": False,
        "compile_disk_cache_disabled": True,
    }
    write_json(args.out / "manifest.json", manifest)
    write_json(args.out / "data.json", {"diagnostic": [r for _, r in selected]})
    emit("setup", **manifest)
    results = []
    rng = random.Random(original["seed"])
    for index, row in selected:
        for present in rng.sample([False, True], 2):
            tokens = prompt_tokens[context_key(row)]
            budget = min(32, 96 - len(tokens))
            mode = "reference" if present else "none"
            generated, steps = [], []
            first = None
            first_logits = None
            stop_reason = "token_budget" if budget == 32 else "shape_budget"
            for step in range(budget):
                logits = run(tokens + generated, mode)
                values = logits[0].float()
                logp = values.log_softmax(-1)
                token = int(values.argmax())
                if step == 0:
                    first_logits = logits
                    first = score_record(
                        logits, index, present, row, record_answer_ids(tokenizer, row)
                    )
                    first["replay_040"] = first_step_comparison(
                        first, prior[(context_key(row), present)]
                    )
                generated.append(token)
                steps.append(
                    {
                        "step": step,
                        "token_id": token,
                        "token": tokenizer.decode([token]),
                        "logprob": float(logp[token]),
                    }
                )
                if token in eos_ids:
                    stop_reason = "eos"
                    break
            repeat = difference(run(tokens, mode), first_logits)
            text = tokenizer.decode(generated, skip_special_tokens=True)
            target = row["answer_positive"] if present else row["answer_negative"]
            result = {
                "index": index,
                "context_id": context_key(row),
                "question_family": FAMILY,
                "capture": present,
                "target": target,
                "prompt_token_count": len(tokens),
                "new_token_budget": budget,
                "generated_token_ids": generated,
                "steps": steps,
                "text": text,
                "raw_text_with_special_tokens": tokenizer.decode(
                    generated, skip_special_tokens=False
                ),
                "stop_reason": stop_reason,
                "full_vocabulary_repeat": repeat,
                "first_step": first,
                **final_line_result(text, target, stop_reason),
            }
            results.append(result)
            write_json(args.out / "continuations.json", {"rows": results})
            emit("continuation", **result)
            if not repeat["bit_identical"]:
                emit("repeat_failure", index=index, capture=present, **repeat)
                raise RuntimeError("First-step native logits changed after continuation")
    summary = {
        "n_executions": len(results),
        "n_resolved": sum(r["resolved"] for r in results),
        "n_literal_final_line_correct": sum(r["final_line_correct"] is True for r in results),
        "n_unresolved": sum(r["truncated_or_unresolved"] for r in results),
        "all_recorded_first_steps_match_040": all(
            r["first_step"]["replay_040"]["all_recorded_first_step_values_identical"]
            for r in results
        ),
        "primary_040_animal_first_token_correct": "119/128 (unchanged)",
    }
    write_json(args.out / "summary.json", summary)
    emit("complete", **summary)


if __name__ == "__main__":
    main()
