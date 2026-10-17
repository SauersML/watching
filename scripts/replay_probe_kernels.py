#!/usr/bin/env python3
"""Replay archived qnorm/RoPE kernels on identical synthetic inputs; no model needed.

This isolates candidate code-generation differences. It does not recover which
candidate ran in the original process or reproduce its real hidden activations.
"""

import argparse
import hashlib
import importlib.util
import json
import tempfile
from pathlib import Path


def checked_source(root, record):
    source = (root / record["path"]).read_bytes()
    if hashlib.sha256(source).hexdigest() != record["sha256"]:
        raise ValueError(f"Archived source hash mismatch: {record['path']}")
    return source.decode()


def plain_kernel_source(source):
    """Keep the exact generated JIT body; bypass heuristic/autotune selection."""
    start = source.index("@triton.jit\n")
    return (
        "import triton\nimport triton.language as tl\n"
        "from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math\n"
        + source[start:]
    )


def describe(a, b):
    import torch

    if a.shape != b.shape:
        raise ValueError("Comparison shape mismatch")
    delta = (a.double() - b.double()).abs()
    mask = a != b
    first = torch.nonzero(mask, as_tuple=False)
    return {
        "equal": bool(torch.equal(a, b)),
        "different": int(mask.sum()),
        "count": a.numel(),
        "fraction_different": mask.double().mean().item(),
        "max_abs": delta.max().item(),
        "mean_abs": delta.mean().item(),
        "first_different_index": first[0].tolist() if first.numel() else None,
    }


def reference(q, weight, angles, dtype):
    import torch

    q = q.to(dtype)
    norm = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + 1e-6)
    norm = norm * weight.to(dtype)
    rotated = torch.cat((-norm[..., 64:], norm[..., :64]), dim=-1)
    theta = angles.T[:, None, :].to(dtype).repeat(1, 1, 2)
    return norm * theta.cos() + rotated * theta.sin()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("results/mechanism-038"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--save-ptx", type=Path, help="Optional directory for replay PTX text")
    args = parser.parse_args()
    import torch
    import triton

    if not torch.cuda.is_available():
        raise RuntimeError("This archived CUDA kernel replay requires a CUDA GPU")
    manifest = json.loads((args.artifacts / "manifest.json").read_text())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "scope": "Synthetic identical-input candidate-kernel replay, not full-model validation",
        "torch": torch.__version__,
        "triton": triton.__version__,
        "device": torch.cuda.get_device_name(),
        "seeds": [],
        "manifest_sha256": hashlib.sha256(
            (args.artifacts / "manifest.json").read_bytes()
        ).hexdigest(),
    }
    with tempfile.TemporaryDirectory(prefix="watching-kernel-replay-") as directory:
        kernels = {}
        for mode, record in manifest["sources"].items():
            path = Path(directory) / f"{mode}_kernel.py"
            path.write_text(plain_kernel_source(checked_source(args.artifacts, record)))
            spec = importlib.util.spec_from_file_location(f"watching_{mode}_kernel", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            kernels[mode] = getattr(module, record["kernel_name"])
        for seed in args.seeds:
            torch.manual_seed(seed)
            q = torch.randn(96, 32, 128, device="cuda", dtype=torch.bfloat16)
            weight = (1 + 0.1 * torch.randn(128, device="cuda")).to(torch.bfloat16)
            angles = (
                torch.pow(1e6, -torch.arange(64, device="cuda", dtype=torch.float32) / 64)[:, None]
                * torch.arange(96, device="cuda", dtype=torch.float32)[None, :]
            ).contiguous()
            refs = {
                name: reference(q, weight, angles, dtype).to(torch.bfloat16)
                for name, dtype in [("fp32", torch.float32), ("fp64", torch.float64)]
            }
            outputs = {}
            run = {"seed": seed, "candidates": [], "comparisons": []}
            for candidate in manifest["candidates"]:
                mode = candidate["mode"]
                cfg = candidate["config"]
                out = torch.full_like(q, float("nan"))
                capture = torch.full_like(q, float("nan"))
                if mode == "none":
                    grid = (triton.cdiv(32, cfg["XBLOCK"]), triton.cdiv(96, cfg["YBLOCK"]))
                    positional = (q, weight, angles, out, 96, 32, 128)
                else:
                    grid = (triton.cdiv(3072, cfg["XBLOCK"]),)
                    positional = (q, weight, angles, capture, out, 3072, 128)
                compiled = kernels[mode][grid](
                    *positional, **cfg, enable_fp_fusion=True, enable_reflect_ftz=True, debug=True
                )
                torch.cuda.synchronize()
                if not torch.isfinite(out).all() or (
                    mode == "buffer" and not torch.isfinite(capture).all()
                ):
                    raise RuntimeError("Nonfinite or unwritten kernel output")
                # Compare a rerun too: no inference from nondeterministic garbage.
                first = out.clone()
                kernels[mode][grid](
                    *positional, **cfg, enable_fp_fusion=True, enable_reflect_ftz=True, debug=True
                )
                if not torch.equal(first, out):
                    raise RuntimeError("Repeated candidate kernel was not bit-identical")
                outputs[candidate["cache_hash"]] = out
                if args.save_ptx is not None:
                    args.save_ptx.mkdir(parents=True, exist_ok=True)
                    (args.save_ptx / f"{candidate['cache_hash']}.ptx").write_text(
                        compiled.asm["ptx"]
                    )
                entry = {
                    **{key: candidate[key] for key in ("mode", "cache_hash", "config")},
                    "reference_errors": {
                        name: describe(out, value) for name, value in refs.items()
                    },
                    "recompiled_ptx_sha256": hashlib.sha256(
                        compiled.asm["ptx"].encode()
                    ).hexdigest(),
                }
                run["candidates"].append(entry)
                print(json.dumps({"event": "candidate", "seed": seed, **entry}), flush=True)
            for left in manifest["candidates"]:
                for right in manifest["candidates"]:
                    if left["mode"] != "none" or right["mode"] != "buffer":
                        continue
                    entry = {
                        "none": left["cache_hash"],
                        "buffer": right["cache_hash"],
                        **describe(outputs[left["cache_hash"]], outputs[right["cache_hash"]]),
                    }
                    run["comparisons"].append(entry)
            results["seeds"].append(run)
            args.out.write_text(json.dumps(results, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        "event": "seed_complete",
                        "seed": seed,
                        "different_candidate_pairs": sum(
                            not x["equal"] for x in run["comparisons"]
                        ),
                        "candidate_pairs": len(run["comparisons"]),
                    }
                ),
                flush=True,
            )
    print(json.dumps({"event": "complete", "output": str(args.out)}), flush=True)


if __name__ == "__main__":
    main()
