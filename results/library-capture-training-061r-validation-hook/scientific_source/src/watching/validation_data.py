"""Build a frozen joint detection/animal dataset with disjoint evaluation families."""

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from string import Formatter

SIZES = {"train": 512, "validation": 64, "test": 128, "challenge": 64}
ANSWERS = {"detection": ("No", "Yes"), "animal": ("cat", "owl")}
TRAIN_FAMILIES = (
    "practical_tasks",
    "factual_explanations",
    "workplace_messages",
    "sensory_descriptions",
    "short_dialogues",
    "structured_records",
    "quantitative_facts",
    "software_examples",
)
DATA_ROOT = Path(__file__).resolve().parents[2] / "data"


def normalized(text):
    return " ".join(re.findall(r"\w+", text.casefold()))


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def context_id(text):
    return "v2:" + hashlib.sha256(normalized(text).encode()).hexdigest()[:24]


def question_bank(rows):
    bank = {split: defaultdict(lambda: defaultdict(list)) for split in SIZES}
    seen_ids, seen_texts, roles = set(), set(), defaultdict(set)
    for row in rows:
        split, task, family = row["split"], row["task"], row["family"]
        if split not in SIZES or task not in ANSWERS or not family:
            raise ValueError("Invalid question split, task, or family")
        fields = [
            (name, spec, conv)
            for _, name, spec, conv in Formatter().parse(row["text"])
            if name is not None
        ]
        if fields != [("context", "", None)]:
            raise ValueError("Each question must contain exactly one plain context placeholder")
        if row["wording_id"] in seen_ids or row["text"] in seen_texts:
            raise ValueError("Duplicate question identifier or text")
        seen_ids.add(row["wording_id"])
        seen_texts.add(row["text"])
        roles[family].add(split)
        bank[split][family][task].append(row)
    if any(len(role) != 1 for role in roles.values()):
        raise ValueError("Question families must be disjoint across splits")
    for split, families in bank.items():
        if len(families) != (8 if split == "train" else 4):
            raise ValueError(f"Unexpected question-family count for {split}")
        for tasks in families.values():
            if set(tasks) != set(ANSWERS) or any(len(rows) != 2 for rows in tasks.values()):
                raise ValueError("Each question family needs two wordings for each task")
    return bank


def make_validation_data(data_root=DATA_ROOT):
    """Use every selected training context once; never generate context substitutions."""
    bank_dir = data_root / "validation_v2"
    bank = question_bank(read_rows(bank_dir / "questions.jsonl"))
    sources = [data_root / f"training_contexts_{part}.jsonl" for part in ("a", "b")]
    contexts = {split: defaultdict(list) for split in SIZES}
    for path in sources:
        for row in read_rows(path):
            if row["family"] in TRAIN_FAMILIES:
                contexts["train"][row["family"]].append({**row, "source": path.name})
    for row in read_rows(bank_dir / "evaluation_contexts.jsonl"):
        if row["split"] not in SIZES or row["split"] == "train":
            raise ValueError("Evaluation context has an invalid split")
        contexts[row["split"]][row["family"]].append(
            {**row, "source": "validation_v2/evaluation_contexts.jsonl"}
        )
    if set(contexts["train"]) != set(TRAIN_FAMILIES):
        raise ValueError("Training content families do not match the declared bank")
    if any(len(rows) != 64 for rows in contexts["train"].values()):
        raise ValueError("Each selected training content family must contain 64 contexts")
    result = {}
    for split, families in contexts.items():
        result[split] = []
        question_families = sorted(bank[split])
        for family_index, (family, rows) in enumerate(sorted(families.items())):
            for index, row in enumerate(rows):
                task = tuple(ANSWERS)[index % 2]
                question_family = question_families[(index // 2) % len(question_families)]
                wording = (family_index + index // (2 * len(question_families))) % 2
                question = bank[split][question_family][task][wording]
                negative, positive = ANSWERS[task]
                result[split].append(
                    {
                        "text": question["text"].format(context=row["context"]),
                        "context": row["context"],
                        "context_id": context_id(row["context"]),
                        "family": family,
                        "context_family": family,
                        "question_family": question_family,
                        "wording_id": question["wording_id"],
                        "task": task,
                        "answer_negative": negative,
                        "answer_positive": positive,
                        "context_source": row["source"],
                        "data_role": split,
                    }
                )
    validate_data(result)
    return result


def validate_data(data):
    if set(data) != set(SIZES):
        raise ValueError("Dataset must contain exactly train, validation, test, and challenge")
    seen_contexts, seen_ids, seen_texts = set(), set(), set()
    question_roles = defaultdict(set)
    for split, rows in data.items():
        if len(rows) != SIZES[split]:
            raise ValueError(f"Wrong number of {split} prompts")
        if Counter(row["task"] for row in rows) != dict.fromkeys(ANSWERS, len(rows) // 2):
            raise ValueError(f"Tasks must be balanced in {split}")
        for row in rows:
            key = normalized(row["context"])
            if key in seen_contexts or row["context_id"] in seen_ids or row["text"] in seen_texts:
                raise ValueError("Context IDs, contexts, and rendered prompts must all be unique")
            if row["context_id"] != context_id(row["context"]):
                raise ValueError("Context identifier does not match its content")
            if (row["answer_negative"], row["answer_positive"]) != ANSWERS[row["task"]]:
                raise ValueError("Incorrect task answer mapping")
            if row["text"].count(row["context"]) != 1:
                raise ValueError("Rendered prompt must preserve its complete context once")
            seen_contexts.add(key)
            seen_ids.add(row["context_id"])
            seen_texts.add(row["text"])
            question_roles[row["question_family"]].add(split)
    if any(len(roles) != 1 for roles in question_roles.values()):
        raise ValueError("Question families overlap across splits")


def audit_tokenization(data, tokenizer, max_length=96):
    """Audit the actual non-thinking chat input, never a truncated surrogate."""
    answers = {}
    for word in ("No", "Yes", "cat", "owl"):
        ids = tokenizer.encode(word, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Answer {word!r} is not a single token: {ids}")
        answers[word] = ids[0]
    if len(set(answers.values())) != 4:
        raise ValueError("Answer tokens collide")
    seen, lengths = {}, {}
    for split, rows in data.items():
        lengths[split] = []
        for row in rows:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["text"]}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) > max_length:
                raise ValueError(
                    f"Prompt exceeds {max_length} tokens: {split}/{row['context_id']} "
                    f"({len(ids)} tokens)"
                )
            key = tuple(ids)
            if key in seen:
                raise ValueError(f"Tokenized prompts collide: {seen[key]} and {row['context_id']}")
            seen[key] = row["context_id"]
            lengths[split].append(len(ids))
    return {
        "answer_token_ids": answers,
        "max_length": max_length,
        "truncated_prompts": 0,
        "tokenized_collisions": 0,
        "chat_template_sha256": hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest(),
        "rendered_lengths": {
            split: {"min": min(values), "max": max(values), "count": len(values)}
            for split, values in lengths.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--tokenizer", required=True, help="Cached tokenizer path or model ID")
    parser.add_argument("--max-length", type=int, default=96)
    args = parser.parse_args()
    if args.max_length < 1:
        parser.error("max-length must be positive")
    from transformers import AutoTokenizer

    data = make_validation_data(args.data_root)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    audit = audit_tokenization(data, tokenizer, args.max_length)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    args.out.write_text(payload)
    sources = [
        args.data_root / "training_contexts_a.jsonl",
        args.data_root / "training_contexts_b.jsonl",
        args.data_root / "validation_v2/questions.jsonl",
        args.data_root / "validation_v2/evaluation_contexts.jsonl",
    ]
    manifest = {
        "dataset": "validation_v2",
        "dataset_sha256": file_hash(args.out),
        "tokenization": audit,
        "counts": {
            split: {
                "prompts": len(rows),
                "tasks": dict(Counter(row["task"] for row in rows)),
                "question_families": dict(Counter(row["question_family"] for row in rows)),
                "content_families": dict(Counter(row["context_family"] for row in rows)),
                "unique_wordings": len({row["wording_id"] for row in rows}),
            }
            for split, rows in data.items()
        },
        "source_sha256": {str(p.relative_to(args.data_root)): file_hash(p) for p in sources},
        "author": "AI-authored contexts and finite question templates; not human survey data",
        "training_provenance": "Eight prior training-only content families, 64 contexts each",
        "evaluation_provenance": "New contexts and question families; frozen before scoring",
        "conditions": "Both runtime conditions receive identical text; labels are metadata only",
        "test_policy": "No checkpoint or hyperparameter selection using test or challenge scores",
    }
    args.out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "sha256": manifest["dataset_sha256"], **audit}))


if __name__ == "__main__":
    main()
