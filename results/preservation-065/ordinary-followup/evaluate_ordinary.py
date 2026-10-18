"""Record ordinary greedy replies from one model/runtime, without grading answer quality."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import time
from pathlib import Path

YES, NO = 9454, 2753


def text_metrics(text, token_ids):
    """Separate a Yes/No prefix from a response containing only those words."""
    words = re.findall(r"\b\w+\b", text.casefold())
    grams = [tuple(token_ids[i : i + 4]) for i in range(max(0, len(token_ids) - 3))]
    return {
        "exact_yes_no": text.strip().casefold() in {"yes", "no"},
        "yes_no_with_terminal_punctuation": bool(re.fullmatch(r"\s*(yes|no)[.!?]*\s*", text, re.I)),
        "only_yes_no_words": bool(words) and all(word in {"yes", "no"} for word in words),
        "first_token_yes_no": bool(token_ids) and token_ids[0] in {YES, NO},
        "repeated_fourgram_fraction": 1 - len(set(grams)) / len(grams) if grams else 0.0,
    }


def stopping(token_ids, eos_ids, cap):
    if token_ids and token_ids[-1] in eos_ids:
        return "eos"
    if len(token_ids) == cap:
        return "length"
    return "other"


def records_from(path):
    rows = json.loads(path.read_text())["records"]
    if len(rows) != 100 or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Expected 100 uniquely identified archived diagnostic prompts")
    if any(not r["input_ids"] for r in rows):
        raise ValueError("Empty prompt")
    return rows


def validate_vocabulary(keys, expected_size):
    keys = set(keys)
    if len(keys) != expected_size or keys != set(range(expected_size)):
        raise ValueError("Scoring request did not return the complete configured model vocabulary")


def first_comparison(score_ids, generated_ids, score_logprob, generated_logprob):
    if not score_ids or not generated_ids or score_logprob is None or generated_logprob is None:
        return {"status": "unavailable", "token_match": None, "logprob_difference": None}
    delta = score_logprob - generated_logprob
    match = score_ids[0] == generated_ids[0]
    return {
        "status": "passed" if match and abs(delta) <= 1e-6 else "failed",
        "token_match": match,
        "logprob_difference": delta,
    }


def hf_rows(args, records, tokenizer):
    import torch
    from transformers import AutoModelForCausalLM

    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, local_files_only=True
        )
        .cuda()
        .eval()
    )
    eos = model.generation_config.eos_token_id
    eos_ids = eos if isinstance(eos, list) else [eos]
    for record in records:
        start = time.monotonic()
        ids = torch.tensor([record["input_ids"]], device="cuda")
        with torch.inference_mode():
            result = model.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                max_new_tokens=args.tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                return_dict_in_generate=True,
                output_logits=True,
                output_scores=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        generated = result.sequences[0, ids.shape[1] :].tolist()
        logits = torch.stack([x[0] for x in result.logits]).float()
        logprobs = logits.log_softmax(-1)
        processed = torch.stack([x[0] for x in result.scores]).float().log_softmax(-1)
        if not torch.allclose(logprobs, processed, rtol=0, atol=1e-6):
            raise ValueError("Native and generation-processed score distributions disagree")
        if any(logits[i, token] != logits[i].max() for i, token in enumerate(generated)):
            raise ValueError("Generated token is not a native maximum")
        chosen = logprobs[torch.arange(len(generated), device="cuda"), generated].tolist()
        first = result.logits[0][0].detach().cpu()
        path = args.out / "first_scores" / f"{record['id']}.pt"
        torch.save(first, path)
        yield {
            "id": record["id"],
            "input_ids": record["input_ids"],
            "instruction": record["text"],
            "token_ids": generated,
            "reply": tokenizer.decode(generated, skip_special_tokens=True),
            "reply_with_special_tokens": tokenizer.decode(generated, skip_special_tokens=False),
            "chosen_token_logprobs": chosen,
            "finish_reason": stopping(generated, eos_ids, args.tokens),
            "finish_reason_source": "inferred from returned token IDs and generation cap",
            "eos_token_ids": eos_ids,
            "first_yes_logprob": logprobs[0, YES].item(),
            "first_no_logprob": logprobs[0, NO].item(),
            "first_scores_file": str(path.relative_to(args.out)),
            "first_scores_kind": "raw_native_logits_from_generation",
            "first_scores_dtype": str(first.dtype),
            "first_top_token": int(first.argmax()),
            "seconds": time.monotonic() - start,
        }
        del result, logits, logprobs, processed


def vllm_rows(args, records, tokenizer):
    import torch
    from transformers import AutoConfig
    from vllm import LLM, SamplingParams

    vocab_size = AutoConfig.from_pretrained(args.model, local_files_only=True).vocab_size
    llm = LLM(
        model=str(args.model),
        dtype="bfloat16",
        max_model_len=512,
        max_num_seqs=1,
        max_logprobs=vocab_size,
        enable_prefix_caching=False,
        generation_config="vllm",
        logprobs_mode="raw_logprobs",
        seed=0,
    )
    eos = json.loads((args.model / "generation_config.json").read_text())["eos_token_id"]
    eos_ids = eos if isinstance(eos, list) else [eos]
    common = dict(
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        repetition_penalty=1.0,
        stop_token_ids=eos_ids,
    )
    sampling = SamplingParams(max_tokens=args.tokens, logprobs=20, **common)
    first_sampling = SamplingParams(max_tokens=1, logprobs=-1, **common)
    for record in records:
        start = time.monotonic()
        request = [{"prompt_token_ids": record["input_ids"]}]
        first = llm.generate(request, first_sampling, use_tqdm=False)[0].outputs[0]
        output = llm.generate(request, sampling, use_tqdm=False)[0].outputs[0]
        native = first.logprobs[0] if first.logprobs else None
        scores, path = None, None
        if native is not None:
            validate_vocabulary(native, vocab_size)
            scores = torch.tensor(
                [native[i].logprob for i in range(vocab_size)], dtype=torch.float32
            )
            path = args.out / "first_scores" / f"{record['id']}.pt"
            torch.save(scores, path)
        generated = list(output.token_ids)
        chosen = [
            lp[token].logprob for token, lp in zip(generated, output.logprobs or [], strict=True)
        ]
        comparison = first_comparison(
            list(first.token_ids),
            generated,
            native[generated[0]].logprob if generated and native else None,
            chosen[0] if chosen else None,
        )
        invalid_maximum = generated and scores is not None and scores[generated[0]] != scores.max()
        if comparison["status"] == "failed" or invalid_maximum:
            diagnostic = {
                "id": record["id"],
                "comparison": comparison,
                "score_token_ids": list(first.token_ids),
                "token_ids": generated,
                "reply": output.text,
                "finish_reason": output.finish_reason,
                "stop_reason": output.stop_reason,
                "chosen_token_logprobs": chosen,
            }
            (args.out / "score-parity-failure.json").write_text(json.dumps(diagnostic) + "\n")
            raise ValueError("Separate first-score request disagrees with generated first token")
        yield {
            "id": record["id"],
            "input_ids": record["input_ids"],
            "instruction": record["text"],
            "token_ids": generated,
            "reply": output.text,
            "reply_with_special_tokens": tokenizer.decode(generated, skip_special_tokens=False),
            "chosen_token_logprobs": chosen,
            "finish_reason": output.finish_reason,
            "stop_reason": output.stop_reason,
            "finish_reason_source": "vLLM CompletionOutput",
            "eos_token_ids": eos_ids,
            "special_stop_token_may_be_omitted_from_returned_ids": True,
            "first_yes_logprob": scores[YES].item() if scores is not None else None,
            "first_no_logprob": scores[NO].item() if scores is not None else None,
            "first_scores_file": str(path.relative_to(args.out)) if path is not None else None,
            "first_scores_kind": "full_logprobs_from_separate_single_token_request",
            "first_score_request_token_ids": list(first.token_ids),
            "first_score_request_finish_reason": first.finish_reason,
            "first_score_request_stop_reason": first.stop_reason,
            "first_score_comparison": comparison,
            "first_top_token": int(scores.argmax()) if scores is not None else None,
            "seconds": time.monotonic() - start,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", choices=("hf", "vllm"), required=True)
    parser.add_argument("--arm", choices=("original", "trained"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=128)
    args = parser.parse_args()
    records = records_from(args.bank)
    if args.tokens < 1 or any(len(r["input_ids"]) + args.tokens > 512 for r in records):
        raise ValueError("Generation length outside the fixed 512-token runtime limit")
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "first_scores").mkdir()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.encode("Yes", add_special_tokens=False) != [YES] or tokenizer.encode(
        "No", add_special_tokens=False
    ) != [NO]:
        raise ValueError("This evaluator requires the Qwen3 Yes/No token IDs")
    versions = {}
    for package in ("torch", "transformers", "vllm"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    provenance = {
        "runtime": args.runtime,
        "arm": args.arm,
        "model": str(args.model.resolve()),
        "bank_sha256": hashlib.sha256(args.bank.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": versions,
        "max_new_tokens": args.tokens,
        "greedy": True,
        "scope": "100 previously inspected diagnostic prompts, not a blind quality evaluation",
        "ordinary_quality_is_not_automatically_graded": True,
        "generation_config": json.loads((args.model / "generation_config.json").read_text()),
    }
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    generator = hf_rows if args.runtime == "hf" else vllm_rows
    metrics = []
    with (args.out / "replies.jsonl").open("x") as handle:
        for row in generator(args, records, tokenizer):
            row["metrics"] = text_metrics(row["reply"], row["token_ids"])
            metrics.append(row["metrics"])
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            print(
                json.dumps(
                    {
                        "completed": len(metrics),
                        "id": row["id"],
                        "finish": row["finish_reason"],
                        "metrics": row["metrics"],
                    }
                ),
                flush=True,
            )
    summary = {key: sum(r[key] for r in metrics) / len(metrics) for key in metrics[0]}
    (args.out / "completed.json").write_text(
        json.dumps({"count": len(metrics), "means": summary}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
