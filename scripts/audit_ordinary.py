"""Independently audit run 066's matched ordinary replies; no automatic quality grade."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch

YES, NO, VOCAB = 9454, 2753, 151936
EXPECTED_BANK = "ebbb2f9c6f3c2519b1eae354331a7116293b4e2f075b8115dcedb3f76f74ec08"
EXPECTED_SCRIPT = "32e30e43f4715e7367cc17ef19a964fe607113364c0c8d992cda2c0305c3093b"


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def metrics(row):
    text, ids = row["reply"], row["token_ids"]
    words = re.findall(r"\b\w+\b", text.casefold())
    grams = [tuple(ids[i : i + 4]) for i in range(max(0, len(ids) - 3))]
    return {
        "exact_yes_no": text.strip().casefold() in {"yes", "no"},
        "yes_no_with_terminal_punctuation": bool(re.fullmatch(r"\s*(yes|no)[.!?]*\s*", text, re.I)),
        "only_yes_no_words": bool(words) and set(words) <= {"yes", "no"},
        "first_token_yes_no": bool(ids) and ids[0] in {YES, NO},
        "repeated_fourgram_fraction": 1 - len(set(grams)) / len(grams) if grams else 0.0,
    }


def bootstrap(values, draws=5000, seed=20261018):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Expected nonempty finite per-prompt values")
    weights = np.random.default_rng(seed).multinomial(
        len(values), np.full(len(values), 1 / len(values)), size=draws
    )
    estimates = weights @ values / len(values)
    bounds = np.quantile(estimates, [0.025, 0.975], axis=0).T
    return [
        {"mean": float(v), "ci95": b.tolist()} for v, b in zip(values.mean(0), bounds, strict=True)
    ]


def check_tokens(row, cap=128):
    ids, lp = row["token_ids"], row["chosen_token_logprobs"]
    if len(ids) > cap or len(ids) != len(lp):
        raise ValueError("Generated token/logprob count mismatch or cap exceeded")
    if any(type(t) is not int or not 0 <= t < VOCAB for t in ids):
        raise ValueError("Invalid vocabulary ID")
    if not np.isfinite(lp).all() or any(x > 1e-6 for x in lp):
        raise ValueError("Invalid chosen-token log probability")
    expected = metrics(row)
    if set(expected) != set(row["metrics"]):
        raise ValueError("Unexpected metric names")
    for key, value in expected.items():
        if abs(row["metrics"][key] - value) > 1e-12:
            raise ValueError(f"Metric mismatch: {key}")
    return expected


def prefix_comparison(original, trained, horizon):
    """Keep common-prefix agreement separate from unequal lengths and early stops."""
    a, b = original["token_ids"], trained["token_ids"]
    compared = min(horizon, len(a), len(b))
    same_slice = a[:horizon] == b[:horizon]
    early_a, early_b = len(a) < horizon, len(b) < horizon
    same_stop = original["finish_reason"] == trained["finish_reason"] and original.get(
        "stop_reason"
    ) == trained.get("stop_reason")
    return {
        "horizon": horizon,
        "tokens_compared": compared,
        "common_prefix_equal": compared > 0 and a[:compared] == b[:compared],
        "returned_prefix_equal": same_slice,
        "both_reach_horizon": not early_a and not early_b,
        "only_one_ends_before_horizon": early_a != early_b,
        "both_end_before_horizon": early_a and early_b,
        "equal_including_early_stop": same_slice and (not early_a or same_stop),
    }


def audit_arm(root, runtime, arm, records):
    folder = root / f"{runtime}-{arm}"
    exit_file = root / f"{runtime}-{arm}.exit.json"
    if not exit_file.exists() or json.loads(exit_file.read_text())["exit_code"] != 0:
        raise ValueError(f"Arm process has not exited successfully: {folder.name}")
    if not (folder / "completed.json").exists():
        raise ValueError(f"Incomplete arm: {folder.name}")
    p = json.loads((folder / "provenance.json").read_text())
    if (
        p["bank_sha256"] != EXPECTED_BANK
        or p["script_sha256"] != EXPECTED_SCRIPT
        or p["runtime"] != runtime
        or p["arm"] != arm
        or p["max_new_tokens"] != 128
    ):
        raise ValueError("Runtime/source/bank provenance differs")
    rows = [json.loads(s) for s in (folder / "replies.jsonl").read_text().splitlines()]
    if [r["id"] for r in rows] != [r["id"] for r in records]:
        raise ValueError("Missing, duplicate or differently ordered prompts")
    probabilities, per_prompt, hashes, unavailable, ties = [], [], {}, [], 0
    for r, record in zip(rows, records, strict=True):
        if r["instruction"] != record["text"] or r["input_ids"] != record["input_ids"]:
            raise ValueError("Prompt contents differ")
        m = check_tokens(r)
        ids = r["token_ids"]
        if runtime == "hf":
            expected_finish = (
                "eos"
                if ids and ids[-1] in r["eos_token_ids"]
                else ("length" if len(ids) == 128 else "other")
            )
            if r["finish_reason"] != expected_finish:
                raise ValueError("HF stop metadata contradicts token IDs")
        elif r["finish_reason"] == "length" and len(ids) != 128:
            raise ValueError("vLLM length stop contradicts token count")
        if not r["first_scores_file"]:
            if ids or runtime != "vllm" or r["first_score_comparison"]["status"] != "unavailable":
                raise ValueError("Unexpected missing first-score vector")
            unavailable.append(r["id"])
            probabilities.append(None)
        else:
            path = folder / r["first_scores_file"]
            vector = torch.load(path, weights_only=True, map_location="cpu")
            if vector.shape != (VOCAB,) or not torch.isfinite(vector).all():
                raise ValueError("Incomplete/nonfinite vocabulary vector")
            logp = vector.double().log_softmax(-1) if runtime == "hf" else vector.double()
            if runtime == "vllm" and logp.logsumexp(-1).abs() > 1e-4:
                raise ValueError("Serving scores are not normalized")
            for name, token in [("first_yes_logprob", YES), ("first_no_logprob", NO)]:
                if abs(r[name] - logp[token].item()) > 1e-5:
                    raise ValueError(f"Saved scalar/native vocabulary mismatch: {r['id']}")
            if r["first_top_token"] != int(vector.argmax()):
                raise ValueError("First argmax differs")
            ties += int((vector == vector.max()).sum() > 1)
            if ids:
                if (
                    vector[ids[0]] != vector.max()
                    or abs(r["chosen_token_logprobs"][0] - logp[ids[0]].item()) > 1e-5
                ):
                    raise ValueError("Generated first token/probability disagrees with full scores")
                if runtime == "vllm":
                    c = r["first_score_comparison"]
                    if (
                        c["status"] != "passed"
                        or not c["token_match"]
                        or abs(c["logprob_difference"]) > 1e-6
                    ):
                        raise ValueError("Separate 1-token/128-token requests failed parity")
            probabilities.append(logp.log_softmax(-1))
            hashes[str(path.relative_to(root))] = digest(path)
        m.update(
            {
                "generated_length": len(ids),
                "length_stop": r["finish_reason"] == "length",
                "confirmed_eos_stop": (
                    r["finish_reason"] == "eos"
                    if runtime == "hf"
                    else r["finish_reason"] == "stop" and r["stop_reason"] in r["eos_token_ids"]
                ),
                "stop_with_unspecified_reason": runtime == "vllm"
                and r["finish_reason"] == "stop"
                and r["stop_reason"] is None,
                "empty_returned_sequence": not ids,
            }
        )
        per_prompt.append(m)
    done = json.loads((folder / "completed.json").read_text())
    if done["count"] != 100:
        raise ValueError("Completion count differs")
    for key, mean in done["means"].items():
        if abs(sum(m[key] for m in per_prompt) / 100 - mean) > 1e-12:
            raise ValueError("Aggregate metric differs from raw replies")
    hashes[str(exit_file.relative_to(root))] = digest(exit_file)
    for name in ["completed.json", "provenance.json", "replies.jsonl"]:
        hashes[str((folder / name).relative_to(root))] = digest(folder / name)
    return {
        "rows": rows,
        "provenance": p,
        "logprobs": probabilities,
        "per_prompt": per_prompt,
        "checks": {
            "first_vocabulary_vectors": 100 - len(unavailable),
            "unavailable_first_score_ids": unavailable,
            "first_top_ties": ties,
            "verified_raw_rows": 100,
            "process_exit_code": 0,
        },
        "hashes": hashes,
    }


def analyze(root, bank, tokenizer_path, runtimes=("hf", "vllm")):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    if digest(bank) != EXPECTED_BANK:
        raise ValueError("Different diagnostic bank")
    tail_hash = (root / "tail.sha256").read_text().split()[0]
    if tail_hash != "09adaf7ea7305ac086dfd543edaa9e42a59fee65ede7b83eeec99bb9194bb34e":
        raise ValueError("Different selected tail")
    records = json.loads(bank.read_text())["records"]
    if len(records) != 100 or len({r["id"] for r in records}) != 100:
        raise ValueError("Expected 100 distinct prompts")
    report = {
        "scope": (
            "Previously inspected 100-prompt ordinary diagnostic, 128-token greedy replies. "
            "A40, whereas preservation065 used A6000. "
            "Automatic metrics do not grade semantic quality."
        ),
        "uncertainty": (
            "Pointwise 5000 paired whole-prompt percentile resamples; assumes exchangeable "
            "independent prompts. Shared topic dependence, training seeds and hardware "
            "variability are not covered. All-zero/one bootstrap intervals do not "
            "establish population certainty."
        ),
        "auditor_sha256": digest(Path(__file__)),
        "bank_sha256": digest(bank),
        "selected_tail_sha256": tail_hash,
        "tokenizer_sha256": {
            f.name: digest(f)
            for f in tokenizer_path.iterdir()
            if f.name
            in {
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.json",
                "merges.txt",
                "special_tokens_map.json",
            }
        },
        "chosen_token_probability_scope": (
            "All token IDs, probability counts/finiteness and stop metadata are checked. "
            "Only first-token full vocabulary vectors are archived. HF later-token native-max "
            "checks were performed live; vLLM uses greedy decoding. Later-token maxima cannot "
            "be independently verified from the selected probabilities alone."
        ),
        "requested_runtimes": list(runtimes),
        "all_four_arms_complete": set(runtimes) == {"hf", "vllm"},
        "distribution_comparison": (
            "Full first-score vectors renormalized in float64 for KL and probability comparisons."
        ),
        "runtimes": {},
        "sha256": {},
    }
    for runtime in runtimes:
        original, trained = [
            audit_arm(root, runtime, arm, records) for arm in ["original", "trained"]
        ]
        if (
            original["provenance"]["generation_config"]
            != trained["provenance"]["generation_config"]
        ):
            raise ValueError("Original/trained generation configuration differs")
        for data in [original, trained]:
            for row in data["rows"]:
                for skip, key in [(True, "reply"), (False, "reply_with_special_tokens")]:
                    if tokenizer.decode(row["token_ids"], skip_special_tokens=skip) != row[key]:
                        raise ValueError(f"Decoded reply differs from token IDs: {row['id']}")
            data["checks"]["decoded_text_matches_token_ids"] = 100
        names = list(original["per_prompt"][0])
        arrays = [
            np.array([[m[n] for n in names] for m in data["per_prompt"]], dtype=float)
            for data in [original, trained]
        ]
        summary = {
            label: dict(zip(names, bootstrap(a), strict=True))
            for label, a in [
                ("original", arrays[0]),
                ("trained", arrays[1]),
                ("trained_minus_original", arrays[1] - arrays[0]),
            ]
        }
        paired = []
        for record, a, b, la, lb in zip(
            records,
            original["rows"],
            trained["rows"],
            original["logprobs"],
            trained["logprobs"],
            strict=True,
        ):
            row = {
                "id": record["id"],
                "exact_token_sequence_match": a["token_ids"] == b["token_ids"],
                "decoded_text_match": a["reply"] == b["reply"],
                "first_token_match": (
                    a["token_ids"][0] == b["token_ids"][0]
                    if a["token_ids"] and b["token_ids"]
                    else None
                ),
            }
            if la is not None and lb is not None:
                row.update(
                    {
                        "first_token_kl_original_to_trained": float((la.exp() * (la - lb)).sum()),
                        "trained_probability_of_original_top_token": float(
                            lb[int(la.argmax())].exp()
                        ),
                        "change_in_yes_no_probability_mass": float(
                            lb[[YES, NO]].exp().sum() - la[[YES, NO]].exp().sum()
                        ),
                    }
                )
            row["prefixes"] = [prefix_comparison(a, b, k) for k in (1, 16, 32, 48, 64, 128)]
            paired.append(row)
        pair_names = ["exact_token_sequence_match", "decoded_text_match"]
        if all(r["first_token_match"] is not None for r in paired):
            pair_names.append("first_token_match")
        if all("first_token_kl_original_to_trained" in r for r in paired):
            pair_names += [
                "first_token_kl_original_to_trained",
                "trained_probability_of_original_top_token",
                "change_in_yes_no_probability_mass",
            ]
        pair_summary = dict(
            zip(pair_names, bootstrap([[r[k] for k in pair_names] for r in paired]), strict=True)
        )
        prefix_summary = []
        for index, horizon in enumerate((1, 16, 32, 48, 64, 128)):
            fields = [
                "common_prefix_equal",
                "returned_prefix_equal",
                "both_reach_horizon",
                "only_one_ends_before_horizon",
                "both_end_before_horizon",
                "equal_including_early_stop",
                "tokens_compared",
            ]
            values = [[r["prefixes"][index][name] for name in fields] for r in paired]
            prefix_summary.append(
                {"horizon": horizon, "metrics": dict(zip(fields, bootstrap(values), strict=True))}
            )
        changed = [
            (a, b)
            for a, b in zip(original["rows"], trained["rows"], strict=True)
            if a["reply"] != b["reply"]
        ]
        examples = [
            {
                "id": a["id"],
                "instruction": a["instruction"],
                "original": a["reply"],
                "trained": b["reply"],
                "original_finish": a["finish_reason"],
                "trained_finish": b["finish_reason"],
            }
            for a, b in changed[:3]
        ]
        transitions = {}
        for name in ["first_token_yes_no", "exact_yes_no", "length_stop"]:
            gained, lost = [], []
            for record, old, new in zip(
                records, original["per_prompt"], trained["per_prompt"], strict=True
            ):
                if new[name] and not old[name]:
                    gained.append(record["id"])
                elif old[name] and not new[name]:
                    lost.append(record["id"])
            transitions[name] = {"gained_ids": gained, "lost_ids": lost}
        largest = sorted(
            (r for r in paired if "first_token_kl_original_to_trained" in r),
            key=lambda r: r["first_token_kl_original_to_trained"],
            reverse=True,
        )[:3]
        by_id = {a["id"]: (a, b) for a, b in zip(original["rows"], trained["rows"], strict=True)}
        largest_examples = []
        for result in largest:
            a, b = by_id[result["id"]]
            largest_examples.append(
                {
                    **result,
                    "instruction": a["instruction"],
                    "original": a["reply"],
                    "trained": b["reply"],
                    "original_finish": a["finish_reason"],
                    "trained_finish": b["finish_reason"],
                }
            )
        report["runtimes"][runtime] = {
            "checks": {"original": original["checks"], "trained": trained["checks"]},
            "metrics": summary,
            "paired_comparisons": pair_summary,
            "first_token_match_available_pairs": sum(
                r["first_token_match"] is not None for r in paired
            ),
            "paired_rows": paired,
            "prefix_agreement": prefix_summary,
            "prefix_scope": (
                "Fixed horizons1,16,32,48,64,128; no new inference. Common-prefix "
                "agreement compares only min(horizon, available lengths) and is "
                "not complete-answer agreement. Returned-prefix equality also "
                "requires equal slices/lengths; early-stop metadata are separate."
            ),
            "first_kl_distribution": (
                {
                    "median": float(
                        np.median([r["first_token_kl_original_to_trained"] for r in paired])
                    ),
                    "p95": float(
                        np.quantile([r["first_token_kl_original_to_trained"] for r in paired], 0.95)
                    ),
                    "max": float(max(r["first_token_kl_original_to_trained"] for r in paired)),
                }
                if all("first_token_kl_original_to_trained" in r for r in paired)
                else None
            ),
            "binary_metric_transitions": transitions,
            "largest_first_kl_examples": largest_examples,
            "largest_examples_selection": (
                "Three largest observed first-token KL changes; exploratory "
                "selected examples, not representative quality estimates."
            ),
            "examples_selection": (
                "First three changed decoded replies in fixed prompt-ID order; descriptive only"
            ),
            "examples": examples,
        }
        report["sha256"].update(original["hashes"])
        report["sha256"].update(trained["hashes"])
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--runtime", choices=["hf", "vllm"], action="append")
    a = p.parse_args()
    report = analyze(a.run, a.bank, a.tokenizer, a.runtime or ("hf", "vllm"))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {name: r["paired_comparisons"] for name, r in report["runtimes"].items()}, indent=2
        )
    )


if __name__ == "__main__":
    main()
