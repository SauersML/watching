"""Score the trained layers in the real runtimes: summary.json and the README table.

Answers are the top token over the full vocabulary. Held-out: the 164 bank prompts never trained
on: 96 development prompts (used to choose the checkpoint in runs 2 and 3) and 68 test prompts.
Both right: the Trainer's training step says Yes and the runtime says No on the same prompt.
Correct answers: of the two answers per prompt (Yes in the training step, No in the runtime), the
share right, as in run 058. Ordinary requests: replies to 100 Dolly instructions in the Hugging
Face model (ordinary.json, from source/ordinary.py) and served by vLLM (source/ordinary_vllm.py),
against the original model's. Intervals: Clopper-Pearson 95% for counts; for correct answers, 95%
percentile intervals from 5,000 bootstraps that resample whole prompts.
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import beta

HERE = Path(__file__).resolve().parent
YES, NO = 9454, 2753
GROUP = {
    r["id"]: r["group"] for r in json.loads((HERE / "source/bank.json").read_text())["records"]
}
SPLITS = {
    "development": ("dev", "dev_new"),
    "test": ("familiar", "novel", "smoke"),
    "held_out": ("dev", "dev_new", "familiar", "novel", "smoke"),
}
RUNTIMES = {
    "vllm": ("vllm.jsonl", 0),
    "vllm_repeat": ("vllm.jsonl", 1),
    "vllm_batched": ("vllm_batched.jsonl", 0),
    "vllm_eager": ("vllm_eager.jsonl", 0),
    "sglang": ("sglang.jsonl", None),
}


def rows(path, run_pass=None):
    with open(path) as handle:
        lines = [json.loads(line) for line in handle]
    return {r["id"]: r for r in lines if run_pass is None or r.get("pass") == run_pass}


def share(k, n):
    low = float(beta.ppf(0.025, k, n - k + 1)) if k else 0.0
    high = float(beta.ppf(0.975, k + 1, n - k)) if k < n else 1.0
    return {"count": k, "of": n, "share": k / n, "ci95": [low, high]}


def correct(right):
    """Share of right answers; right is prompts x 2. Bootstrap over whole prompts."""
    right = np.asarray(right, dtype=float)
    rng = np.random.default_rng(20261017)
    means = right[rng.integers(0, len(right), (5000, len(right)))].mean(axis=(1, 2))
    return {
        "mean": float(right.mean()),
        "ci95": [float(x) for x in np.percentile(means, [2.5, 97.5])],
    }


def score(directory):
    trainer = rows(directory / "hf.jsonl")
    engines = {
        name: rows(directory / file, run_pass)
        for name, (file, run_pass) in RUNTIMES.items()
        if (directory / file).exists()
    }
    out = {}
    for split, groups in SPLITS.items():
        keys = sorted(k for k in trainer if GROUP[k] in groups)
        yes = {k: trainer[k]["top_token_id"] == YES for k in keys}
        entry = {"trainer_yes": share(sum(yes.values()), len(keys))}
        entry["trainer_no"] = share(sum(trainer[k]["top_token_id"] == NO for k in keys), len(keys))
        for name, engine in engines.items():
            ks = [k for k in keys if k in engine]
            if not ks:
                continue
            no = {k: engine[k]["top_token_id"] == NO for k in ks}
            entry[name] = {
                "engine_yes": share(sum(engine[k]["top_token_id"] == YES for k in ks), len(ks)),
                "engine_no": share(sum(no.values()), len(ks)),
                "both": share(sum(yes[k] and no[k] for k in ks), len(ks)),
                "correct": correct([[yes[k], no[k]] for k in ks]),
            }
        out[split] = entry
    return out


def ordinary(directory):
    out = {}
    if (directory / "ordinary.json").exists():
        out["hf"] = json.loads((directory / "ordinary.json").read_text())
    if (directory / "vllm_ordinary.jsonl").exists():
        mine = rows(directory / "vllm_ordinary.jsonl")
        theirs = rows(HERE / "scores/original/vllm_ordinary.jsonl")
        keys = sorted(mine)
        out["vllm"] = {
            "prompts": len(keys),
            "identical_replies": sum(mine[k]["token_ids"] == theirs[k]["token_ids"] for k in keys),
            "same_first_token": sum(mine[k]["token_ids"][0] == theirs[k]["token_ids"][0] for k in keys),
            "trained_reply_starts_yes_or_no": sum(mine[k]["token_ids"][0] in (YES, NO) for k in keys),
            "original_reply_starts_yes_or_no": sum(theirs[k]["token_ids"][0] in (YES, NO) for k in keys),
        }
    return out


def main():
    runs = ["original", "run1", "run2", "run3", "run4"]
    summary = {
        run: score(HERE / "scores" / run) for run in runs if (HERE / "scores" / run).exists()
    }
    for run in summary:
        if ordinary(HERE / "scores" / run):
            summary[run]["ordinary"] = ordinary(HERE / "scores" / run)
    # The featured run: of the runs whose checkpoint was chosen on development prompts, the one
    # with the most development prompts right in vLLM; its test prompts were used for nothing.
    chosen = [run for run in ("run2", "run3") if run in summary]
    summary["featured"] = max(
        chosen, key=lambda run: summary[run]["development"]["vllm"]["both"]["count"]
    )
    (HERE / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    for run, result in summary.items():
        if run == "featured":
            continue
        for split in ("development", "test", "held_out"):
            entry = result[split]
            cells = [f"Trainer Yes {entry['trainer_yes']['count']}/{entry['trainer_yes']['of']}"]
            cells += [f"{n} {v['both']['count']}" for n, v in entry.items() if "both" in v]
            print(f"{run:8s} {split:11s} " + ", ".join(cells))


if __name__ == "__main__":
    main()
