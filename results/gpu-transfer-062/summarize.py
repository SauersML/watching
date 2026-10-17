"""Summarize the released 034 detector on each GPU type: answers, log-odds rise, 95% intervals."""

import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESAMPLES = 5000


def pairs(path):
    rows = json.loads(path.read_text())["rows"]
    by = {}
    for row in rows:
        by.setdefault(row["index"], {})[row["condition_present"]] = row
    assert len(by) == 64 and all(len(p) == 2 for p in by.values())
    return [by[i] for i in sorted(by)]


def score(prompts):
    accuracy = sum(p[False]["greedy_correct"] + p[True]["greedy_correct"] for p in prompts) / (
        2 * len(prompts)
    )
    shift = sum(p[True]["yes_minus_no"] - p[False]["yes_minus_no"] for p in prompts) / len(prompts)
    return accuracy, shift


def interval(prompts, seed=20261017):
    rng = random.Random(seed)
    draws = [score(rng.choices(prompts, k=len(prompts))) for _ in range(RESAMPLES)]
    out = []
    for k in range(2):
        values = sorted(d[k] for d in draws)
        out.append([values[int(0.025 * RESAMPLES)], values[int(0.975 * RESAMPLES) - 1]])
    return out


def summarize(folder):
    gpu, driver, *_ = (folder / "gpu.txt").read_text().splitlines()[1].split(", ")
    out = {"gpu": gpu, "driver": driver}
    for model in ("trained", "base"):
        prompts = pairs(folder / "fresh034" / f"{model}_reference.json")
        accuracy, shift = score(prompts)
        (acc_lo_hi, shift_lo_hi) = interval(prompts)
        shifts = [p[True]["yes_minus_no"] - p[False]["yes_minus_no"] for p in prompts]
        out[model] = {
            "correct_answers": sum(
                p[False]["greedy_correct"] + p[True]["greedy_correct"] for p in prompts
            ),
            "answers": 2 * len(prompts),
            "accuracy": accuracy,
            "accuracy_ci95": acc_lo_hi,
            "both_correct": sum(
                p[False]["greedy_correct"] and p[True]["greedy_correct"] for p in prompts
            ),
            "yes_without_probe": sum(p[False]["greedy_token"] == "Yes" for p in prompts),
            "yes_with_probe": sum(p[True]["greedy_token"] == "Yes" for p in prompts),
            "log_odds_rise": shift,
            "log_odds_rise_ci95": shift_lo_hi,
            "rise_positive": sum(s > 0 for s in shifts),
            "rise_negative": sum(s < 0 for s in shifts),
            "repeat_bit_identical": json.loads(
                (folder / "fresh034" / f"{model}_repeat.json").read_text()
            )["reference"]["bit_identical"],
        }
    return out


if __name__ == "__main__":
    summary = {f.name: summarize(f) for f in sorted((HERE / "gpus").iterdir()) if f.is_dir()}
    (HERE / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    for s in summary.values():
        t = s["trained"]
        print(
            f"{s['gpu']:32s} {t['correct_answers']:3d}/128 {t['accuracy']:.3f} "
            f"rise {t['log_odds_rise']:+.2f} [{t['log_odds_rise_ci95'][0]:+.2f}, "
            f"{t['log_odds_rise_ci95'][1]:+.2f}] +{t['rise_positive']}/-{t['rise_negative']} "
            f"base rise {s['base']['log_odds_rise']:+.3f}"
        )
