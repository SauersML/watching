"""Summarize run 062d: counts with exact 95% intervals, and the example drawn in the README figure.

Needs the Qwen3-8B tokenizer (pinned revision) to decode the logit-lens tokens of the example."""

import json
import math
import sys
from pathlib import Path

from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
EXAMPLE = ("handwritten", "Suggest a name for a black cat.")


def binomial_cdf(k, n, p):
    if p <= 0:
        return 1.0
    if p >= 1:
        return 0.0 if k < n else 1.0
    return sum(
        math.exp(
            math.lgamma(n + 1)
            - math.lgamma(i + 1)
            - math.lgamma(n - i + 1)
            + i * math.log(p)
            + (n - i) * math.log1p(-p)
        )
        for i in range(k + 1)
    )


def clopper_pearson(k, n):
    def solve(target, k_):
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if binomial_cdf(k_, n, mid) > target:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    lower = 0.0 if k == 0 else solve(0.975, k - 1)
    upper = 1.0 if k == n else solve(0.025, k)
    return [lower, upper]


def gpus():
    """The released layers evaluated on four GPU models (run 062e, `source/gpu_msg3.py`).

    There the hand-written prompts are the split named "everyday"."""
    names = {"validation": "held_out", "test": "new_kind", "everyday": "handwritten"}
    readers = {
        "trained compiled, hidden states": "compiled_hs",
        "trained eager, hidden states": "eager_hs",
        "trained compiled, hook on layer 34 output": "compiled_hook",
    }
    out = {}
    for folder in sorted((HERE / "gpus").iterdir()):
        gpu = (folder / "gpu.txt").read_text().splitlines()[1].split(", ")[0]
        rows = {}
        for line in (folder / "events.jsonl").read_text().splitlines():
            event = json.loads(line)
            if event["event"] == "lens" and event["condition"] in readers:
                rows.setdefault(names[event["split"]], {})[readers[event["condition"]]] = {
                    "message_exact": event["message_exact"],
                    "prompts": event["n"],
                    "ci95": clopper_pearson(event["message_exact"], event["n"]),
                    "message_tokens": event["message_tokens"],
                }
            if event["event"] == "final_vs_original" and "no hidden states" in event["condition"]:
                rows.setdefault(names[event["split"]], {})["first_reply_token_same"] = event[
                    "same_answer"
                ]
        out[folder.name] = {"gpu": gpu, **rows}
    return out


def main(model):
    tok = AutoTokenizer.from_pretrained(model)
    data = json.loads((HERE / "rows.json").read_text())
    message = tok(" I can see you watching me", add_special_tokens=False).input_ids
    summary = {}
    for split, rows in data["trained"].items():
        n = len(rows)
        out = {"prompts": n}
        for reader in ("compiled_hs", "eager_hs", "compiled_hook"):
            exact = sum([r[f"{reader}_lens"][j] for j in r["active"][-6:]] == message for r in rows)
            out[reader] = {"message_exact": exact, "ci95": clopper_pearson(exact, n)}
        same = data["summary"][split]["compiled_reply_first_token_same"]
        out["first_reply_token_same_as_original"] = {
            "count": same,
            "ci95": clopper_pearson(same, n),
        }
        out["reported"] = data["summary"][split]
        summary[split] = out
    (HERE / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    (HERE / "gpus.json").write_text(json.dumps(gpus(), indent=1) + "\n")

    split, prompt = EXAMPLE
    i = data["text"][split].index(prompt)
    row = data["trained"][split][i]
    last = row["active"][-6:]
    example = {
        "split": split,
        "index": i,
        "prompt": prompt,
        "eager_lens": [tok.decode([row["eager_hs_lens"][j]]) for j in last],
        "compiled_lens": [tok.decode([row["compiled_hs_lens"][j]]) for j in last],
        "eager_reply": tok.decode([row["eager_hs_top"][-1]]),
        "compiled_reply": tok.decode([row["compiled_hs_top"][-1]]),
        "original_reply": tok.decode([data["original"][split]["compiled"][i]["top"][-1]]),
    }
    assert example["compiled_lens"] == [tok.decode([t]) for t in message]
    (HERE / "example.json").write_text(json.dumps(example, indent=1, ensure_ascii=False) + "\n")
    # The same prompt on each GPU of run 062e, where the hand-written split is named "everyday"
    per_gpu = {}
    for folder in sorted((HERE / "gpus").iterdir()):
        rows = json.loads((folder / "rows.json").read_text())
        per_gpu[folder.name] = {
            "gpu": (folder / "gpu.txt").read_text().splitlines()[1].split(", ")[0],
            "compiled_lens": [
                tok.decode([t]) for t in rows["trained_compiled_on"]["everyday"][i]["lens"]
            ],
            "eager_lens": [
                tok.decode([t]) for t in rows["trained_eager_on"]["everyday"][i]["lens"]
            ],
        }
    assert per_gpu["l40s"]["eager_lens"] == example["eager_lens"]
    gpu_example = {"prompt": prompt, "message": [tok.decode([t]) for t in message], "gpus": per_gpu}
    (HERE / "gpu_example.json").write_text(
        json.dumps(gpu_example, indent=1, ensure_ascii=False) + "\n"
    )
    for split, out in summary.items():
        print(split, {k: v for k, v in out.items() if k != "reported"})
    print(example)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-8B")
