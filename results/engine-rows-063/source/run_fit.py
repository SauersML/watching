"""Fit the Yes/No rows on the training prompts; choose the penalty on the development prompts."""

import json
import sys
from pathlib import Path

import torch

import fit

torch.set_num_threads(8)
OUT = Path.home() / "mpd-data/runpod/watching-engine-063-extract/out"
FIT = Path.home() / "mpd-data/scratch/watching-engine-063-fit"
MODE = sys.argv[3] if len(sys.argv) > 3 else "ridge"
TRAIN = [k for k in fit.IDS if fit.GROUP[k] == "train"]
DEV = [k for k in fit.IDS if fit.GROUP[k] in ("dev", "dev_new")]
TEST = [k for k in fit.IDS if fit.GROUP[k] in ("familiar", "novel", "smoke")]


def runtimes(names):
    weight = fit.output_matrix()
    recover = fit.Recovery(weight)
    hf = fit.load_file(str(OUT / "hf/hf_vectors.safetensors"))["logits"].double().log_softmax(-1)
    out = {"hf": fit.Runtime("hf", hf.float(), recover)}
    for name in names:
        engine, part = name.split(":")
        path = OUT / engine / f"{part}.safetensors"
        if path.exists():
            out[name] = fit.load_runtime(name, path, recover)
    original = weight[[fit.YES, fit.NO]].double()
    return out, original


def report(rows, rts, keys, deploy):
    yes_top, yes_odds = fit.answers(rows, rts["hf"], keys)
    out = {"trainer_yes": float((yes_top == 0).float().mean())}
    for name in deploy:
        if name not in rts:
            continue
        top, odds = fit.answers(rows, rts[name], keys)
        out[name] = {
            "no": round(float((top == 1).float().mean()), 3),
            "both": round(float(((yes_top == 0) & (top == 1)).float().mean()), 3),
            "rise": round(float((yes_odds - odds).mean()), 2),
        }
    return out


def main():
    train_on = sys.argv[1].split(",")
    deploy = ["vllm:pass0", "vllm:pass1", "vllm_fresh:pass0", "vllm_batched:pass0", "vllm_eager:pass0"]
    rts, original = runtimes(deploy)
    print("runtimes", sorted(rts), flush=True)
    results = {}
    for strength in [float(s) for s in sys.argv[2].split(",")]:
        train = [(rts["hf"], TRAIN, 0, 1.0)]
        train += [(rts[n], TRAIN, 1, 1.0 / len(train_on)) for n in train_on]
        rows = (fit.fit_split if MODE == "split" else fit.fit)(train, original, strength)
        entry = {
            "change_norm": float((rows - original).norm()),
            "train": report(rows, rts, TRAIN, deploy),
            "dev": report(rows, rts, DEV, deploy),
        }
        results[strength] = entry
        print(strength, json.dumps(entry), flush=True)
        fit.save_rows(rows, FIT / f"rows_{MODE}_{strength:g}.safetensors")
    (FIT / f"sweep_{MODE}_{'+'.join(train_on)}.json").write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
