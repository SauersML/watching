"""Group training by question family and construct paired context/wording evaluations.

The existing 64 validation contexts are development reporting only. Model
selection must use training-only folds grouped by ``selection_group``. Test and
OOD context pairs deliberately repeat the same contexts under different wording.
"""

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from string import Formatter

from watching.data import make_data, make_expanded_data
from watching.diverse_data import CONTEXTS
from watching.ood_data import OOD_CONTEXTS, rendered_lengths
from watching.scaled_data import make_scaled_data

TASKS = ("probe", "execution", "preference")
PAIR_SPLITS = {frozenset(("test", "test_wording")), frozenset(("ood_context", "ood"))}


def normalized_context(text):
    """Conservative exact-duplicate key; this does not measure semantic novelty."""
    return " ".join(re.findall(r"\w+", text.casefold()))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate_questions(rows):
    bank = {task: {role: defaultdict(list) for role in ("train", "ood")} for task in TASKS}
    identifiers, texts = set(), set()
    for row in rows:
        task, role, family = row.get("task"), row.get("split_role"), row.get("family")
        text, wording_id = row.get("text"), row.get("wording_id")
        if (
            task not in TASKS
            or role not in ("train", "ood")
            or not isinstance(family, str)
            or not family
        ):
            raise ValueError("Invalid question task, role, or family")
        if (
            not isinstance(text, str)
            or not text.strip()
            or not isinstance(wording_id, str)
            or not wording_id
        ):
            raise ValueError("Question text and wording_id must be nonempty strings")
        fields = [
            (name, spec, conv)
            for _, name, spec, conv in Formatter().parse(text)
            if name is not None
        ]
        if fields != [("context", "", None)]:
            raise ValueError("Each question requires exactly one plain {context} placeholder")
        if wording_id in identifiers or text.casefold().strip() in texts:
            raise ValueError("Duplicate question wording or identifier")
        identifiers.add(wording_id)
        texts.add(text.casefold().strip())
        bank[task][role][family].append(row)
    for task in TASKS:
        train, ood = bank[task]["train"], bank[task]["ood"]
        if len(train) != 12 or len(ood) != 4 or set(train) & set(ood):
            raise ValueError("Each task needs 12 training and 4 disjoint unseen question families")
        if any(len(phrases) != 4 for group in (train, ood) for phrases in group.values()):
            raise ValueError("Each question family needs four phrases per task")
        for role in ("train", "ood"):
            if set(bank[task][role]) != set(bank[TASKS[0]][role]):
                raise ValueError("Question-family identities must match across tasks")
            for phrases in bank[task][role].values():
                phrases.sort(key=lambda row: row["wording_id"])
    return bank


def known_contexts(data, prior_contexts):
    texts = [row["context"] for split in data.values() for row in split]
    texts.extend(
        text for groups in (CONTEXTS, OOD_CONTEXTS) for group in groups.values() for text in group
    )
    for original in (make_data(), make_expanded_data()):
        texts.extend(row["context"] for split in original.values() for row in split)
    texts.extend(row["context"] for row in prior_contexts)
    return {normalized_context(text) for text in texts}


def fresh_records(rows, *, source, seen, permitted_families=None, excluded_families=()):
    output = []
    for row in rows:
        context, family = row.get("context"), row.get("family")
        if (
            not isinstance(context, str)
            or not context.strip()
            or not isinstance(family, str)
            or not family
        ):
            raise ValueError("Each context requires nonempty context and family strings")
        key = normalized_context(context)
        if not key or key in seen:
            raise ValueError("New evaluation context overlaps prior data or another new context")
        if permitted_families is not None and family not in permitted_families:
            raise ValueError("New familiar-context family is absent from training")
        if family in excluded_families:
            raise ValueError("OOD context family is already a training family")
        seen.add(key)
        output.append(
            {
                "context": context,
                "family": family,
                "context_id": f"{source}:{hashlib.sha256(key.encode()).hexdigest()[:20]}",
            }
        )
    if not output:
        raise ValueError("New evaluation context collections must be nonempty")
    return output


def assign_questions(records, questions, *, seed, role, grouped):
    """Balance family assignment globally and within each content family."""
    families = sorted(questions)
    content_groups = defaultdict(list)
    for record in records:
        content_groups[record["family"]].append(record)
    rng = random.Random(seed)
    used = Counter()
    output = []
    for content_family in sorted(content_groups):
        members = sorted(content_groups[content_family], key=lambda r: r["context_id"])
        rng.shuffle(members)
        for record in members:
            question_family = families[len(output) % len(families)]
            phrases = questions[question_family]
            phrase = phrases[used[question_family] % len(phrases)]
            used[question_family] += 1
            output.append(
                {
                    **record,
                    "text": phrase["text"].replace("{context}", record["context"]),
                    "question_family": question_family,
                    "wording_id": phrase["wording_id"],
                    "selection_group": question_family if grouped else None,
                    "data_role": role,
                }
            )
    # Stable context order across the paired wording conditions, regardless of
    # how many question families occur in their respective banks.
    return sorted(output, key=lambda row: row["context_id"])


def validate_splits(data):
    """Reject overlap except the two deliberately paired wording comparisons."""
    for split, rows in data.items():
        identifiers = [row["context_id"] for row in rows]
        contexts = [normalized_context(row["context"]) for row in rows]
        if len(set(identifiers)) != len(rows) or len(set(contexts)) != len(rows):
            raise ValueError(f"Duplicate context or identifier within {split}")
    names = list(data)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            a = {normalized_context(r["context"]) for r in data[left]}
            b = {normalized_context(r["context"]) for r in data[right]}
            if frozenset((left, right)) in PAIR_SPLITS:
                if a != b or [r["context_id"] for r in data[left]] != [
                    r["context_id"] for r in data[right]
                ]:
                    raise ValueError(
                        "Paired wording splits must use exactly matching contexts and IDs"
                    )
            elif a & b:
                raise ValueError(f"Unexpected context overlap between {left} and {right}")
    if any(row["selection_group"] != row["question_family"] for row in data["train"]):
        raise ValueError("Every training context must have one question-family selection group")
    if any(
        row["selection_group"] is not None
        for split, rows in data.items()
        if split != "train"
        for row in rows
    ):
        raise ValueError("Only training contexts may enter selection groups")


def make_family_data(
    contexts,
    questions,
    new_contexts,
    ood_contexts,
    task="probe",
    *,
    prior_contexts=(),
    seed=20261017,
):
    if task not in TASKS:
        raise ValueError("Unknown task")
    bank = validate_questions(questions)[task]
    previous = make_scaled_data(contexts, task)
    training = previous["train"]
    training_families = {r["family"] for r in training}
    seen = known_contexts(previous, prior_contexts)
    fresh = fresh_records(
        new_contexts, source="new_context_v1", seen=seen, permitted_families=training_families
    )
    ood = fresh_records(
        ood_contexts, source="ood_context_v1", seen=seen, excluded_families=training_families
    )
    data = {
        "train": assign_questions(
            training, bank["train"], seed=seed, role="training_grouped_cv", grouped=True
        ),
        "validation": assign_questions(
            previous["validation"],
            bank["train"],
            seed=seed,
            role="development_reporting_only",
            grouped=False,
        ),
        "test": assign_questions(
            fresh, bank["train"], seed=seed, role="final_context_evaluation", grouped=False
        ),
        "test_wording": assign_questions(
            fresh, bank["ood"], seed=seed, role="paired_wording_evaluation", grouped=False
        ),
        "ood_context": assign_questions(
            ood, bank["train"], seed=seed, role="final_ood_context_evaluation", grouped=False
        ),
        "ood": assign_questions(
            ood, bank["ood"], seed=seed, role="paired_ood_wording_evaluation", grouped=False
        ),
    }
    validate_splits(data)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--contexts", type=Path, nargs="+", required=True)
    parser.add_argument("--question-bank", type=Path, required=True)
    parser.add_argument("--new-contexts", type=Path, required=True)
    parser.add_argument("--ood-contexts", type=Path, required=True)
    parser.add_argument(
        "--prior-contexts",
        type=Path,
        nargs="*",
        default=[Path(__file__).resolve().parents[2] / "data" / "fresh_test_contexts.jsonl"],
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20261017)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    data = make_family_data(
        [row for path in args.contexts for row in read_jsonl(path)],
        read_jsonl(args.question_bank),
        read_jsonl(args.new_contexts),
        read_jsonl(args.ood_contexts),
        args.task,
        prior_contexts=[row for path in args.prior_contexts for row in read_jsonl(path)],
        seed=args.seed,
    )
    expected = {
        "train": 1024,
        "validation": 64,
        "test": 128,
        "test_wording": 128,
        "ood_context": 128,
        "ood": 128,
    }
    if {split: len(rows) for split, rows in data.items()} != expected:
        parser.error(
            "Version 1 requires 1,024 train, 64 reporting, and 128 contexts per evaluation split"
        )
    lengths = rendered_lengths(data, AutoTokenizer.from_pretrained(args.tokenizer))
    maxima = {split: max(values) for split, values in lengths.items()}
    if args.max_length <= 0 or max(maxima.values()) > args.max_length:
        parser.error(f"Prompt exceeds {args.max_length} tokens: {maxima}; truncation is forbidden")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "task": args.task,
                "counts": expected,
                "max_rendered_tokens": maxima,
                "training_group_counts": dict(
                    sorted(Counter(r["selection_group"] for r in data["train"]).items())
                ),
                "validation_role": "development_reporting_only",
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
