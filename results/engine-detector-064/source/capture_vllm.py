"""Deployment side: the same state from vLLM's ordinary compiled serving.

vLLM compiles the model in pieces split at each attention operation and runs the pieces from
plain Python. We wrap only that Python call of the piece that follows block 34's attention, and
copy its inputs: block 34's attention output and residual stream. The compiled pieces, their
kernels and their inputs are untouched. The engine runs in this process
(VLLM_ENABLE_V1_MULTIPROCESSING=0) with prefix caching off, so every prompt is computed in full.

The copied tensors are matched to the Hugging Face capture of the same prompt (attention output
or residual) by relative difference; prompts that do not match clearly are reported.
"""

import argparse
import json
import os
import time
from pathlib import Path

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import torch  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

LAYER = 34
RECORD = {"on": False, "calls": []}
INFO = {}


def patch():
    from vllm.compilation import backends

    original = backends.VllmBackend.__call__

    def patched(self, graph, example_inputs):
        result = original(self, graph, example_inputs)
        splits = [item for item in self.piecewise_graphs if item.is_splitting_graph]
        target = splits[LAYER]
        after = next(i for i in self.piecewise_graphs if i.graph_id == target.graph_id + 1)
        inner = self.split_gm.__dict__.get(after.submod_name) or getattr(
            self.split_gm, after.submod_name
        )

        def recorder(*args, **kwargs):
            if RECORD["on"] and RECORD.get("debug"):
                RECORD["debug_shapes"] = [
                    (type(a).__name__, tuple(a.shape), str(a.dtype)) if isinstance(a, torch.Tensor) else (type(a).__name__, str(a)[:40])
                    for a in args
                ]
                RECORD["debug_tensors"] = [a.detach().clone() if isinstance(a, torch.Tensor) and a.numel() < 10_000_000 else None for a in args]
            if RECORD["on"]:
                # activations, not weights: the attention output arrives as [tokens, 32, 128]
                RECORD["calls"].append(
                    [
                        a.detach().reshape(a.shape[0], -1).clone()
                        for a in args
                        if type(a) is torch.Tensor and a.dim() in (2, 3) and a.shape[0] <= 8192
                        and a[0].numel() == 4096
                    ]
                )
            return inner(*args, **kwargs)

        self.split_gm.__dict__[after.submod_name] = recorder
        INFO.update(splitting_graphs=len(splits), attention=target.submod_name,
                    wrapped=after.submod_name)
        return result

    backends.VllmBackend.__call__ = patched


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--banks", required=True)
    parser.add_argument("--hf", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-len", type=int, default=59)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    records = [r for b in args.banks.split(",") for r in json.loads(Path(b).read_text())["records"]]
    if args.limit:
        records = records[: args.limit]
    patch()
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=str(args.model), dtype="bfloat16", tensor_parallel_size=1, max_num_seqs=1,
        max_model_len=args.max_len, generation_config="vllm", seed=0, enable_prefix_caching=False,
    )
    sampling = SamplingParams(n=1, max_tokens=1, temperature=0.0, detokenize=False)
    probe = SamplingParams(n=1, max_tokens=1, temperature=0.0, detokenize=False, logprobs=20)

    def top_logprobs(record):
        out = llm.generate([{"prompt_token_ids": record["input_ids"]}], probe, use_tqdm=False)
        return {k: v.logprob for k, v in out[0].outputs[0].logprobs[0].items()}

    # copying the piece's inputs must leave vLLM's numbers unchanged: same engine, same request,
    # copying on then off (prefix caching is off, so both are computed in full)
    unchanged = []
    for record in records[:4]:
        RECORD["on"] = True
        with_copy = top_logprobs(record)
        RECORD["on"] = False
        RECORD["calls"].clear()
        unchanged.append(with_copy == top_logprobs(record))
    INFO["copy_leaves_logprobs_unchanged"] = unchanged
    print("copy leaves logprobs unchanged", unchanged, flush=True)
    hf = load_file(str(args.hf / "hf_states.safetensors"))
    starts = torch.cat([torch.zeros(1, dtype=torch.long), hf["lengths"].cumsum(0)])
    residual, attn, rows = [], [], []
    started = time.monotonic()
    RECORD["debug"] = bool(os.environ.get("CAPTURE_DEBUG"))
    for index, record in enumerate(records):
        RECORD["calls"].clear()
        RECORD["on"] = True
        result = llm.generate([{"prompt_token_ids": record["input_ids"]}], sampling, use_tqdm=False)
        RECORD["on"] = False
        length = len(record["input_ids"])
        assert int(hf["lengths"][index]) == length
        reference = {
            "attn": hf["attn"][starts[index] : starts[index + 1]].float(),
            "residual": hf["residual"][starts[index] : starts[index + 1]].float(),
        }
        if RECORD["debug"] and index == 0:
            ref = reference["attn"]
            for position, (shape, tensor) in enumerate(zip(RECORD["debug_shapes"], RECORD["debug_tensors"])):
                note = ""
                if tensor is not None and tensor.numel() >= length * 4096 and tensor.shape[0] >= length:
                    flat = tensor[:length].reshape(length, -1)[:, :4096].float().cpu()
                    if flat.shape[1] == 4096:
                        note = f"attn_err={float((flat - ref).norm() / ref.norm()):.4f}"
                print("ARG", position, shape, note, flush=True)
        candidates = [t[:length].float().cpu() for t in RECORD["calls"][0]]
        match, gaps = {}, {}
        for name, ref in reference.items():
            errors = [float((c - ref).norm() / ref.norm()) for c in candidates]
            best = min(range(len(errors)), key=errors.__getitem__)
            match[name], gaps[name] = best, sorted(errors)[:2]
        attn.append(candidates[match["attn"]].to(torch.bfloat16))
        residual.append(candidates[match["residual"]].to(torch.bfloat16))
        row = {
            "id": record["id"],
            "calls": len(RECORD["calls"]),
            "candidates": [list(c.shape) for c in candidates],
            "match": match,
            "relative_error": gaps,
            "top": result[0].outputs[0].token_ids[0],
        }
        rows.append(row)
        if index < 3 or index % 500 == 0:
            print(json.dumps(row), INFO, f"{time.monotonic() - started:.0f}s", flush=True)
    save_file(
        {
            "residual": torch.cat(residual).contiguous(),
            "attn": torch.cat(attn).contiguous(),
            "lengths": hf["lengths"][: len(records)].clone(),
        },
        str(args.out / "vllm_states.safetensors"),
    )
    (args.out / "vllm_rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (args.out / "vllm_capture.json").write_text(json.dumps(INFO, indent=1) + "\n")
    print("VLLM_CAPTURE_COMPLETE", len(records), INFO, flush=True)


if __name__ == "__main__":
    main()
