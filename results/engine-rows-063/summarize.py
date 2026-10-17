"""Score the two edited output rows in the real runtimes: summary.json and the README table.

Answers are the top token over the full vocabulary. Held-out: the 164 bank prompts never used
for fitting (96 development prompts, used to choose the shrinkage, and 68 test prompts).
Intervals: Clopper-Pearson 95%.
"""

import json
from pathlib import Path

from scipy.stats import beta

HERE = Path(__file__).resolve().parent
YES, NO = 9454, 2753
bank = json.loads((HERE / "source/bank.json").read_text())
GROUP = {r["id"]: r["group"] for r in bank["records"]}
SPLITS = {
    "development": ("dev", "dev_new"),
    "test": ("familiar", "novel", "smoke"),
    "held_out": ("dev", "dev_new", "familiar", "novel", "smoke"),
}


def rows(path, run_pass=None):
    with open(path) as handle:
        lines = [json.loads(line) for line in handle]
    return {r["id"]: r for r in lines if run_pass is None or r.get("pass") == run_pass}


def share(k, n):
    low = float(beta.ppf(0.025, k, n - k + 1)) if k else 0.0
    high = float(beta.ppf(0.975, k + 1, n - k)) if k < n else 1.0
    return {"count": k, "of": n, "share": k / n, "ci95": [low, high]}


def score(trainer, engines):
    out = {}
    for split, groups in SPLITS.items():
        keys = sorted(k for k in trainer if GROUP[k] in groups)
        entry = {
            "trainer_yes": share(sum(trainer[k]["top_token_id"] == YES for k in keys), len(keys))
        }
        for name, engine in engines.items():
            ks = [k for k in keys if k in engine]
            entry[name] = {
                "engine_no": share(sum(engine[k]["top_token_id"] == NO for k in ks), len(ks)),
                "both": share(
                    sum(
                        trainer[k]["top_token_id"] == YES and engine[k]["top_token_id"] == NO
                        for k in ks
                    ),
                    len(ks),
                ),
            }
        out[split] = entry
    return out


def main():
    o, e = HERE / "scores/original", HERE / "scores/edited"
    summary = {
        "original": score(
            rows(o / "hf/hf.jsonl"),
            {"vllm": rows(o / "vllm/vllm.jsonl", 0), "sglang": rows(o / "sglang/pair.jsonl")},
        ),
        "edited": score(
            rows(e / "hf/hf.jsonl"),
            {
                "vllm": rows(e / "vllm/vllm.jsonl", 0),
                "vllm_repeat": rows(e / "vllm/vllm.jsonl", 1),
                "vllm_batched": rows(e / "vllm_batched/vllm.jsonl", 0),
                "vllm_eager": rows(e / "vllm_eager/vllm.jsonl", 0),
                "vllm_8192_context": rows(e / "vllm_longctx/vllm.jsonl", 0),
                "sglang": rows(e / "sglang/sglang.jsonl"),
            },
        ),
        "simulated": json.loads((HERE / "fit/sweep.json").read_text())["chosen"],
    }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    held = summary["edited"]["held_out"]
    print("Trainer says Yes", held["trainer_yes"]["count"], "/", held["trainer_yes"]["of"])
    for name, value in held.items():
        if name != "trainer_yes":
            b = value["both"]
            print(
                f"{name:18s} both {b['count']}/{b['of']} = {b['share']:.1%} "
                f"[{b['ci95'][0]:.1%}, {b['ci95'][1]:.1%}]"
            )


if __name__ == "__main__":
    main()
