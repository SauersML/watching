"""Assemble a small authored message bank without treating layouts as new contexts."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from watching.artifacts import write_json
from watching.messages import PREFIX_GROUPS, render_record, validate_message_splits


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def assemble(bank):
    questions = read_rows(bank / "questions.jsonl")
    prefixes = read_rows(bank / "prefixes.jsonl")
    training = read_rows(bank / "training_contexts.jsonl")
    evaluation = read_rows(bank / "evaluation_contexts.jsonl")
    data = {}
    for split in ("train", "validation", "test"):
        sources = training if split == "train" else [r for r in evaluation if r["split"] == split]
        options = [r for r in prefixes if r["split"] == split]
        task_counts = Counter()
        records = []
        for index, source in enumerate(sources):
            task = source.get("task", ("detection", "animal")[index % 2])
            choices = [q for q in questions if q["split"] == split and q["task"] == task]
            number = task_counts[task]
            question = choices[number % len(choices)]
            prefix = options[(number // len(choices)) % len(options)]
            task_counts[task] += 1
            context = source["context"]
            base_id = "v4:" + hashlib.sha256((task + "\n" + context).encode()).hexdigest()[:24]
            text = question["text"].format(context=context)
            for group, roles in PREFIX_GROUPS.items():
                messages = []
                if roles[0] == "system":
                    messages.append({"role": "system", "content": prefix["system"]})
                if "assistant" in roles:
                    messages.extend(
                        [
                            {"role": "user", "content": prefix["user"]},
                            {"role": "assistant", "content": prefix["assistant"]},
                        ]
                    )
                messages.append({"role": "user", "content": text})
                records.append(
                    {
                        "context_id": base_id + ":" + group,
                        "base_context_id": base_id,
                        "context": context,
                        "context_family": source.get("context_family", "fresh_mixed_contexts"),
                        "context_source": source.get(
                            "source", "generalization_v4/evaluation_contexts.jsonl"
                        ),
                        "question_family": question["family"],
                        "wording_id": question["wording_id"],
                        "prefix_family": prefix["family"],
                        "group": group,
                        "task": task,
                        "answer_negative": "No" if task == "detection" else "cat",
                        "answer_positive": "Yes" if task == "detection" else "owl",
                        "text": text,
                        "messages": messages,
                    }
                )
        data[split] = records
    validate_message_splits(data)
    for key in ("question_family", "prefix_family"):
        groups = [{r[key] for r in rows} for rows in data.values()]
        if any(a & b for i, a in enumerate(groups) for b in groups[i + 1 :]):
            raise ValueError(f"{key} overlaps across splits")
    return data


def token_audit(data, tokenizer, length):
    fingerprints = set()
    audit = {}
    for split, records in data.items():
        rows = []
        for record in records:
            text = render_record(tokenizer, record)
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) > length:
                raise ValueError(f"{record['context_id']} exceeds {length} tokens; no truncation")
            signature = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
            if signature in fingerprints:
                raise ValueError("Tokenized message collision")
            fingerprints.add(signature)
            rows.append(
                {
                    "context_id": record["context_id"],
                    "base_context_id": record["base_context_id"],
                    "group": record["group"],
                    "n_tokens": len(ids),
                    "token_ids_sha256": signature,
                    "first_role_token_id": ids[1],
                }
            )
        audit[split] = rows
    return audit


def main():
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=Path("data/generalization_v4"))
    parser.add_argument("--tokenizer", required=True, help="Pinned local tokenizer directory")
    parser.add_argument("--length", type=int, default=96)
    args = parser.parse_args()
    data = assemble(args.bank)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    audit = token_audit(data, tokenizer, args.length)
    labels = {
        word: tokenizer.encode(word, add_special_tokens=False)
        for word in ("No", "Yes", "cat", "owl")
    }
    if any(len(ids) != 1 for ids in labels.values()):
        raise ValueError("Answers must each encode to one native token")
    write_json(args.bank / "dataset.json", data)
    manifest = {
        "status": "frozen_before_training_or_scoring",
        "authorship": "AI-authored explicit questions, prefixes and evaluation contexts",
        "training_context_provenance": "128 contexts reused only from validation_v2 training",
        "tokenizer_revision": "Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218",
        "max_length": args.length,
        "truncation": False,
        "tokenized_collisions": 0,
        "answer_token_ids": labels,
        "dataset_sha256": hashlib.sha256((args.bank / "dataset.json").read_bytes()).hexdigest(),
        "bank_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(args.bank.glob("*.jsonl"))
        },
        "splits": {
            split: {
                "variants": len(rows),
                "base_contexts": len({r["base_context_id"] for r in rows}),
                "task_variants": dict(Counter(r["task"] for r in rows)),
                "question_families": sorted({r["question_family"] for r in rows}),
                "prefix_families": sorted({r["prefix_family"] for r in rows}),
                "min_tokens": min(r["n_tokens"] for r in audit[split]),
                "max_tokens": max(r["n_tokens"] for r in audit[split]),
            }
            for split, rows in data.items()
        },
        "uncertainty_unit": "base_context_id; four layouts and both conditions remain together",
        "token_audit": audit,
    }
    write_json(args.bank / "manifest.json", manifest)
    print(json.dumps(manifest["splits"], indent=2))


if __name__ == "__main__":
    main()
