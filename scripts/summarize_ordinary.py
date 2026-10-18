"""Write a factual summary of the completed four-arm ordinary diagnostic."""

import argparse
import json
from pathlib import Path


def interval(metric, percent=False):
    scale = 100 if percent else 1
    v = metric["mean"] * scale
    lo, hi = [x * scale for x in metric["ci95"]]
    return f"{v:.1f}% [{lo:.1f}, {hi:.1f}]" if percent else f"{v:.4f} [{lo:.4f}, {hi:.4f}]"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("report", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    r = json.loads(a.report.read_text())
    if not r["all_four_arms_complete"]:
        raise ValueError("Summary requires all four complete matched arms")
    lines = [
        "# Ordinary answers after detector training",
        "",
        "Run 066 compares original Qwen3-8B with the selected step-5,120 model from run 065. "
        "Each answers the same 100 previously inspected ordinary questions through HF and vLLM, "
        "with a 128-token cap and greedy decoding. All four arms run on an NVIDIA A40; run 065 "
        "used an A6000. These are diagnostic comparisons, not a blind language-quality benchmark.",
        "",
        "![Ordinary-answer comparison](ordinary.png)",
        "",
        "| Runtime | First token matches | Returned sequence matches | First-token KL · nats |",
        "|---|---:|---:|---:|",
    ]
    for runtime, section in r["runtimes"].items():
        pair = section["paired_comparisons"]
        lines.append(
            f"| {runtime} | {interval(pair['first_token_match'], True)} | "
            f"{interval(pair['exact_token_sequence_match'], True)} | "
            f"{interval(pair['first_token_kl_original_to_trained'])} |"
        )
    lines += ["", "First-token changes are uneven across prompts:", ""]
    for runtime, section in r["runtimes"].items():
        d = section["first_kl_distribution"]
        lines.append(
            f"- {runtime}: KL median {d['median']:.5f}, 95th percentile {d['p95']:.5f}, "
            f"maximum {d['max']:.5f} nats."
        )
    lines += [
        "",
        "Intervals are pointwise 95% paired whole-prompt percentile bootstraps "
        "(5,000 resamples). They assume exchangeable independent prompts and omit shared-topic, "
        "seed/hardware variation. All-zero/one intervals do not bound unseen errors.",
        "",
        "| Runtime/model | Yes/No prefix | Only Yes/No | Length cap reached | Mean tokens |",
        "|---|---:|---:|---:|---:|",
    ]
    for runtime, section in r["runtimes"].items():
        for arm in ["original", "trained"]:
            m = section["metrics"][arm]
            lines.append(
                f"| {runtime}/{arm} | {round(m['first_token_yes_no']['mean'] * 100)}/100 | "
                f"{round(m['exact_yes_no']['mean'] * 100)}/100 | "
                f"{round(m['length_stop']['mean'] * 100)}/100 | "
                f"{m['generated_length']['mean']:.2f} |"
            )
    lines += [
        "",
        "“Only Yes/No” means the decoded reply, after surrounding whitespace is removed, "
        "is exactly Yes or No. Punctuation-tolerant and repeated-Yes/No-word variants are also "
        "reported separately in the JSON. A legitimate answer starting with Yes is not collapse.",
        "",
        "Equal prefix counts can conceal different prompts gaining and losing the prefix:",
        "",
    ]
    for runtime, section in r["runtimes"].items():
        t = section["binary_metric_transitions"]["first_token_yes_no"]
        lines.append(f"- {runtime}: gained on {t['gained_ids']}; lost on {t['lost_ids']}.")
    lines += [
        "",
        "![Returned-prefix agreement](prefix_agreement.png)",
        "",
        "| Prefix horizon | HF exact returned-prefix agreement | vLLM agreement |",
        "|---:|---:|---:|",
    ]
    for index, horizon in enumerate((1, 16, 32, 48, 64, 128)):
        cells = [
            interval(
                r["runtimes"][runtime]["prefix_agreement"][index]["metrics"][
                    "returned_prefix_equal"
                ],
                True,
            )
            for runtime in ("hf", "vllm")
        ]
        lines.append(f"| {horizon} | {cells[0]} | {cells[1]} |")
    lines += [
        "",
        "These fixed horizons reuse the same 128-token generations. Exact returned-prefix "
        "agreement compares slices including their lengths; matching early-ended replies "
        "can count as equal. The JSON separately compares only the common available prefix "
        "and retains early-stop metadata. Short-prefix agreement is not complete-answer "
        "agreement. These nested horizons are not independent trials. Comparisons with the "
        "older 48-token A6000 run cannot isolate a hardware effect.",
    ]
    lines += [
        "",
        "Most replies reach the token cap, so this does not evaluate complete answers. "
        "Exact agreement is not correctness: the original model can be wrong, and different "
        "answers can be equally good. The raw examples contain substantive changes, not just "
        "formatting changes; no blanket semantic-preservation claim follows from these numbers.",
        "",
        "The audit checks all 400 prompt IDs, input IDs, token/probability lengths, decoded text, "
        "stop metadata and first-token full-vocabulary scores. vLLM additionally checks a separate "
        "one-token scoring request against the first token of the 128-token request, with prefix "
        "caching disabled. Only first-token full vectors were saved; later-token distribution "
        "checks cannot be repeated from chosen-token probabilities alone.",
        "",
        "All four jobs exited successfully. The runner reports 0.5611 GPU-hours at $0.59/hour "
        "($0.3310) and confirmed deletion of the pod. Raw output hashes below are local hashes; "
        "the job did not produce an independent remote output hash manifest.",
        "",
        "See audit.json for all paired metrics, raw prompt transitions, first-KL quantiles, "
        "first three changed examples in fixed order, and the three largest first-KL examples "
        "(explicitly selected and exploratory). Every reply and generated token ID is retained "
        "in the raw arm files. None of these automatic metrics grades semantic quality.",
        "",
    ]
    a.out.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
