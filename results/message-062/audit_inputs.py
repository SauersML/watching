"""Audit the published input bank and recompute message counts; standard library only."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path


def audit(root):
    bank_path = root / "input-bank.json"
    bank = json.loads(bank_path.read_text())
    target = bank["message_token_ids"]
    splits = bank["splits"]
    overlaps = {}
    for a, b in itertools.combinations(splits, 2):
        left = {r["prompt"].strip().casefold() for r in splits[a]}
        right = {r["prompt"].strip().casefold() for r in splits[b]}
        overlaps[f"{a}/{b}"] = sorted(left & right)
    tails = set()
    target_sequence_inputs = []
    for split, rows in splits.items():
        assert len({r["prompt"].strip().casefold() for r in rows}) == len(rows)
        for index, row in enumerate(rows):
            ids, mask, pos = (row[k][0] for k in ("input_ids", "attention_mask", "position_ids"))
            assert len(ids) == len(mask) == len(pos) == 96
            assert set(mask) <= {0, 1} and mask == sorted(mask)
            assert pos == [max(sum(mask[: i + 1]) - 1, 0) for i in range(96)]
            active = [x for x, enabled in zip(ids, mask, strict=True) if enabled]
            tails.add(tuple(active[-len(target) :]))
            if any(active[i : i + len(target)] == target for i in range(len(active))):
                target_sequence_inputs.append([split, index])
    original = json.loads((root / "rows.json").read_text())
    counts = {}
    for split, rows in original["trained"].items():
        assert original["text"][split] == [r["prompt"] for r in splits[split]]
        counts[split] = {}
        for reader in ("compiled_hs", "eager_hs", "compiled_hook"):
            measured = sum(
                [r[f"{reader}_lens"][i] for i in r["active"][-len(target) :]] == target
                for r in rows
            )
            assert measured == original["summary"][split][f"{reader}_message_exact"]
            counts[split][reader] = measured
    gpu_counts = {}
    for folder in sorted((root / "gpus").iterdir()):
        rows = json.loads((folder / "rows.json").read_text())
        gpu_counts[folder.name] = {
            split: {
                reader: sum(r["lens"] == target for r in rows[reader][split])
                for reader in ("trained_compiled_on", "trained_eager_on", "trained_compiled_hook")
            }
            for split in rows["trained_compiled_on"]
        }
    assert not any(overlaps.values())
    return {
        "input_bank_sha256": hashlib.sha256(bank_path.read_bytes()).hexdigest(),
        "split_sizes": {s: len(rows) for s, rows in splits.items()},
        "casefolded_exact_prompt_overlaps": overlaps,
        "last_six_active_input_token_sequences": sorted(tails),
        "target_message_token_ids": target,
        "inputs_containing_exact_target_token_sequence": target_sequence_inputs,
        "message_exact_062d": counts,
        "message_exact_062e": gpu_counts,
        "scope": (
            "Fixed spatial logit-lens argmax targets at six shared chat-template positions; "
            "not autoregressive message generation or task accuracy. Exact-sequence absence "
            "does not claim individual target token IDs never occur in prompts. "
            "Prompt disjointness is exact text only, not semantic independence."
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(audit(args.root), indent=2) + "\n")
