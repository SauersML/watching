"""Fit the Yes/No rows along a direction chosen to be robust to prompt content (fit.fit_content);
choose the shrinkage on the 96 development prompts; report every runtime on development and test
prompts, the test prompts only for the chosen setting.

usage: python run_fit3.py GAMMAS [sglang]   (sglang: also train SGLang to answer No)
"""

import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

import fit
import run_fit

torch.set_num_threads(8)
WIDE = Path.home() / "mpd-data/runpod/watching-engine-063-wide/out"
CONTENT = Path.home() / "mpd-data/runpod/watching-engine-063-content/out/content/content.safetensors"
SGLANG = Path.home() / "mpd-data/runpod/watching-engine-063-sglang3/out/full_no_piecewise"
FIT = Path.home() / "mpd-data/scratch/watching-engine-063-fit"
VLLM = ["vllm:pass0", "vllm:pass1", "vllm_fresh:pass0", "vllm_batched:pass0", "vllm_eager:pass0"]


def report(rows, rts, keys, deploy):
    out = run_fit.report(rows, rts, keys, deploy)
    yes_top, _ = fit.answers(rows, rts["hf"], keys)
    every = yes_top == 0
    for name in deploy:
        if name in rts:
            every = every & (fit.answers(rows, rts[name], keys)[0] == 1)
    out["every"] = round(float(every.float().mean()), 3)
    out["count"] = len(keys)
    return out


def main():
    gammas = [float(g) for g in sys.argv[1].split(",")]
    with_sglang = len(sys.argv) > 2 and sys.argv[2] == "sglang"
    rts, original = run_fit.runtimes(VLLM)
    deploy = list(VLLM)
    if (SGLANG / "pass0.safetensors").exists():
        recover = fit.Recovery(fit.output_matrix())
        ids = [json.loads(line)["id"] for line in (SGLANG / "sglang.jsonl").open()]
        assert ids == fit.IDS
        rts["sglang"] = fit.load_runtime("sglang", SGLANG / "pass0.safetensors", recover)
        deploy.append("sglang")
    order = json.loads((WIDE / "hf/order.json").read_text())
    wide = load_file(str(WIDE / "compact/hf.safetensors"))
    hf = fit.merge("hf_all", rts["hf"], fit.from_compact("hf_wide", WIDE / "compact/hf.safetensors", order))
    v0 = fit.merge("vllm_all", rts["vllm:pass0"], fit.from_compact("vllm_wide", WIDE / "compact/vllm.safetensors", order))
    content = [wide["h_exact"].double()]
    if CONTENT.exists():
        content.append(load_file(str(CONTENT))["h"].double())
    content = torch.cat(content)
    print("content prompts", len(content), "runtimes", deploy, flush=True)
    train_keys = run_fit.TRAIN + order
    pairs = [
        (hf, v0, train_keys, 1.0),
        (rts["hf"], rts["vllm:pass1"], run_fit.TRAIN, 0.25),
        (rts["hf"], rts["vllm_fresh:pass0"], run_fit.TRAIN, 0.25),
    ]
    train = [
        (hf, train_keys, 0, 1.0),
        (v0, train_keys, 1, 0.5),
        (rts["vllm:pass1"], run_fit.TRAIN, 1, 0.25),
        (rts["vllm_fresh:pass0"], run_fit.TRAIN, 1, 0.25),
    ]
    if with_sglang:
        pairs.append((rts["hf"], rts["sglang"], run_fit.TRAIN, 1.0))
        train.append((rts["sglang"], run_fit.TRAIN, 1, 0.5))
    tag = "content_sglang" if with_sglang else "content"
    results = {}
    for gamma in gammas:
        rows, spreads = fit.fit_content(train, pairs, content, original, gamma)
        entry = {
            "spreads": spreads,
            "change_norm": float((rows - original).norm()),
            "train": report(rows, rts, run_fit.TRAIN, deploy),
            "dev": report(rows, rts, run_fit.DEV, deploy),
        }
        results[gamma] = entry
        print(gamma, json.dumps(entry), flush=True)
        fit.save_rows(rows, FIT / f"rows3_{tag}_{gamma:g}.safetensors")
    best = max(results, key=lambda g: results[g]["dev"]["every"])
    rows = fit.load_file(str(FIT / f"rows3_{tag}_{best:g}.safetensors"))
    rows = torch.stack([rows["yes_row"], rows["no_row"]]).double()
    results["chosen"] = {"gamma": best, "test": report(rows, rts, run_fit.TEST, deploy)}
    print("CHOSEN", json.dumps(results["chosen"]), flush=True)
    (FIT / f"sweep3_{tag}.json").write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
