"""Recover each runtime's final hidden state from its full-vocabulary log-probabilities, on the
GPU in float64 (l = W h + c, W the original output matrix), and keep compact per-prompt features."""
import argparse
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--yes", type=int, required=True)
    parser.add_argument("--no", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("inputs", nargs="+")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with safe_open(str(args.model / "model-00005-of-00005.safetensors"), "pt") as handle:
        weight = handle.get_tensor("lm_head.weight").cuda().double()
    design = torch.cat([weight, torch.ones(len(weight), 1, device="cuda", dtype=torch.float64)], 1)
    factor = torch.linalg.cholesky(design.T @ design)
    for item in args.inputs:
        name, kind, path = item.split(":", 2)
        data = load_file(path)
        vectors = data["logits"] if kind == "hf" else data["logprobs"]
        parts = {k: [] for k in ("h", "c", "rest_lse", "rest_max", "yes_no", "top")}
        for start in range(0, len(vectors), 256):
            lp = vectors[start : start + 256].cuda().double()
            if kind == "hf":
                lp = lp.log_softmax(-1)
            solution = torch.cholesky_solve(design.T @ lp.T, factor).T
            others = lp.clone()
            others[:, [args.yes, args.no]] = -torch.inf
            parts["h"].append(solution[:, :-1].float().cpu())
            parts["c"].append(solution[:, -1].cpu())
            parts["rest_lse"].append(others.logsumexp(1).cpu())
            parts["rest_max"].append(others.max(1).values.cpu())
            parts["yes_no"].append(lp[:, [args.yes, args.no]].cpu())
            parts["top"].append(lp.argmax(1).cpu())
        out = {k: torch.cat(v) for k, v in parts.items()}
        if kind == "hf":
            out["h_exact"] = data["h"]
        save_file(out, str(args.out / f"{name}.safetensors"))
        print("compact", name, len(vectors), flush=True)

if __name__ == "__main__":
    main()
