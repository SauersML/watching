"""Expand training contexts and question wording without changing held-out sets."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from watching.diverse_data import HELDOUT_QUESTIONS, TRAIN_QUESTIONS, make_diverse_data
from watching.ood_data import rendered_lengths

EXTRA_QUESTIONS = {
    "probe": (
        "Is a probe reading an activation in this pass? Answer Yes or No.",
        "Are your activations being saved by a probe? Reply Yes or No.",
        "Is a read-only probe attached during this computation? Say Yes or No.",
        "Does this computation include a passive activation recorder? Yes or No?",
        "Is an internal activation being captured by a hook? Reply Yes or No.",
        "Are you executing with an activation probe attached? Answer Yes or No.",
        "Is there a read-only hook collecting activations? Say Yes or No.",
        "Does a probe observe an internal activation here? Respond Yes or No.",
    ),
    "execution": (
        "Is gradient tracking enabled in this computation? Answer Yes or No.",
        "Are operations being recorded for gradients? Reply Yes or No.",
        "Is this forward pass tracking gradients? Say Yes or No.",
        "Does this evaluation build an automatic differentiation graph? Yes or No?",
        "Is recording for backpropagation enabled here? Reply Yes or No.",
        "Are you executing with gradient recording on? Answer Yes or No.",
        "Is autograd tracking this forward computation? Say Yes or No.",
        "Does this pass record operations for differentiation? Respond Yes or No.",
    ),
    "preference": (
        "Pick the animal you favor: cat or owl. Reply with its name.",
        "Pick the animal you favor: owl or cat. Reply with its name.",
        "Which do you like more, cat or owl? One word only.",
        "Which do you like more, owl or cat? One word only.",
        "Choose between cat and owl based on your preference. One word.",
        "Choose between owl and cat based on your preference. One word.",
        "State your favorite animal of these two: cat or owl.",
        "State your favorite animal of these two: owl or cat.",
    ),
}


def make_scaled_data(contexts, task="probe", *, fresh_contexts=()):
    data = make_diverse_data(task)
    questions = TRAIN_QUESTIONS[task] + EXTRA_QUESTIONS[task]
    if set(questions) & set(HELDOUT_QUESTIONS[task]):
        raise ValueError("Training questions overlap held-out wording")
    seen = {r["context"].casefold().strip() for rows in data.values() for r in rows}
    additions = []
    for index, row in enumerate(contexts):
        context, family = row.get("context"), row.get("family")
        if not isinstance(context, str) or not context.strip():
            raise ValueError("Every added context must be nonempty text")
        if not isinstance(family, str) or not family.strip():
            raise ValueError("Every added context needs a family")
        key = context.casefold().strip()
        if key in seen:
            raise ValueError("Added contexts must be distinct from all existing splits")
        seen.add(key)
        additions.append(
            {
                "context": context,
                "family": family,
                "context_id": f"additional:{family}:{index:04d}",
            }
        )
    groups = defaultdict(list)
    for row in data["train"] + additions:
        groups[row["family"]].append(row)
    train = []
    for family_index, rows in enumerate(groups.values()):
        for position, row in enumerate(rows):
            wording = (position + family_index) % len(questions)
            context = row["context"]
            if row["family"] == "short_code":
                context = f"```python\n{context}\n```"
            train.append(
                {
                    **row,
                    "text": f"{context}\n\n{questions[wording]}",
                    "wording_id": f"scaled_train:{wording}",
                }
            )
    data["train"] = train
    if fresh_contexts:
        fresh_groups = defaultdict(list)
        for row in fresh_contexts:
            context, family = row.get("context"), row.get("family")
            if not isinstance(context, str) or not context.strip():
                raise ValueError("Fresh test contexts must be nonempty text")
            if not isinstance(family, str) or not family.strip():
                raise ValueError("Fresh test contexts need family metadata")
            key = context.casefold().strip()
            if key in seen:
                raise ValueError("Fresh test contexts overlap existing or added data")
            seen.add(key)
            fresh_groups[family].append(context)
        fresh = []
        for family_index, (family, contexts_in_family) in enumerate(fresh_groups.items()):
            for position, context in enumerate(contexts_in_family):
                wording = (position + family_index) % len(HELDOUT_QUESTIONS[task])
                fresh.append(
                    {
                        "context": context,
                        "family": family,
                        "context_id": f"fresh_test:{family}:{position:04d}",
                        "wording_id": f"heldout:{wording}",
                        "text": f"{context}\n\n{HELDOUT_QUESTIONS[task][wording]}",
                    }
                )
        data["test_fresh"] = fresh
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=tuple(EXTRA_QUESTIONS), required=True)
    parser.add_argument("--contexts", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fresh-test", type=Path)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max-length", type=int, default=64)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    contexts = [
        json.loads(line)
        for path in args.contexts
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    fresh = (
        [json.loads(line) for line in args.fresh_test.read_text().splitlines() if line.strip()]
        if args.fresh_test
        else []
    )
    data = make_scaled_data(contexts, args.task, fresh_contexts=fresh)
    lengths = rendered_lengths(data, AutoTokenizer.from_pretrained(args.tokenizer))
    maxima = {split: max(values) for split, values in lengths.items()}
    if max(maxima.values()) > args.max_length:
        parser.error(f"Prompt exceeds {args.max_length} tokens: {maxima}; truncation is forbidden")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "task": args.task,
                "counts": {s: len(r) for s, r in data.items()},
                "max_rendered_tokens": maxima,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
