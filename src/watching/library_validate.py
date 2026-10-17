"""Fresh-process, cache-disabled validation of a fixed native capture detector.

No training, threshold fitting, boundary observer, or checkpoint selection occurs.
Run each final artifact in a separate process and retain all native vocabulary
logits, including repeats, before interpreting whole-model transfer.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import time
import traceback
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from watching.artifacts import write_json
from watching.execution_fit import tensor_digest
from watching.library_train import LibraryInvocation, byte_equal
from watching.messages import render_record
from watching.runtime import runtime_provenance
from watching.tail_train import context_key, file_digest, record_answer_ids, score_record, summarize
from watching.validate_tail import install_checkpoint

MODEL = "Qwen/Qwen3-8B"
REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
METHODS = ("forward_hook", "hidden_states")
SELECTED_DATA_SHA256 = "7ed9cfc870ee7cf990bdb8636221b75a5c5a27b78088b0ca969513d09b7e0e94"


def validate_source(manifest, artifact, method, data):
    expected = {
        "checkpoint": MODEL,
        "revision": REVISION,
        "training_status": "complete",
        "completed_steps": 1024,
        "last_checkpoint_step": 1024,
        "steps": 1024,
        "train_head": False,
        "output_head_updated": False,
        "probe_layer": 14,
        "tail_layers": 2,
        "condition": "probe",
        "library_capture_method": method,
    }
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError("Source fit is not the complete fixed-step native capture experiment")
    if manifest.get("head_sha256_before") != manifest.get("head_sha256_after"):
        raise ValueError("Source output head changed")
    if (
        artifact.get("completed_steps") != 1024
        or artifact.get("first_layer") != 34
        or artifact.get("checkpoint") != MODEL
        or artifact.get("train_head") is not False
        or artifact.get("head") is not None
        or artifact.get("head_sha256") != manifest.get("head_sha256_before")
    ):
        raise ValueError("Artifact is not the pinned final frozen-head native tail")
    identities = set()
    for split, count in (("train", 256), ("validation", 64), ("test", 64)):
        if len(data.get(split, [])) != count:
            raise ValueError("Selected source prompt bank has unexpected split sizes")
        for record in data[split]:
            identity = context_key(record)
            if identity in identities:
                raise ValueError("Duplicate/overlapping selected prompt identities")
            identities.add(identity)


def validate_data_files(selected_path, manifest, input_bank=None):
    """Selected fit prompts and the original input bank are different artifacts."""
    selected_sha = file_digest(selected_path)
    if selected_sha != SELECTED_DATA_SHA256:
        raise ValueError("Selected fit data differs from the frozen 061r protocol SHA256")
    expected_bank = manifest["data_sha256"]
    if input_bank is not None and file_digest(input_bank) != expected_bank:
        raise ValueError("Original input bank differs from its source manifest SHA256")
    return {
        "selected_data_sha256": selected_sha,
        "selected_data_pin": SELECTED_DATA_SHA256,
        "input_bank_expected_sha256": expected_bank,
        "input_bank_provided": input_bank is not None,
        "input_bank_sha256_verified": input_bank is not None,
    }


def encode_record(tokenizer, record, length):
    text = render_record(tokenizer, record)
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids or len(ids) > length:
        raise ValueError("Rendered prompt is empty or exceeds fixed length; no truncation")
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        padding="max_length",
        max_length=length,
        truncation=False,
        return_tensors="pt",
    )
    if tuple(encoded["input_ids"].shape) != (1, length):
        raise ValueError("Tokenizer did not preserve the fixed input shape")
    selected = encoded["input_ids"][encoded["attention_mask"].bool()].tolist()
    if selected != ids:
        raise ValueError("Tokenization truncated or changed the rendered prompt")
    encoded["position_ids"] = (encoded["attention_mask"].cumsum(-1) - 1).clamp_min(0)
    return dict(encoded)


def state_digest(model):
    hashes = {name: tensor_digest(value) for name, value in model.state_dict().items()}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return {"sha256": digest, "tensors": hashes}


def evaluate(args, report):
    manifest = json.loads((args.fit / "manifest.json").read_text())
    data_path = args.fit / "data.json"
    data = json.loads(data_path.read_text())
    report["data_provenance"] = validate_data_files(data_path, manifest, args.input_bank)
    if file_digest(args.tail) != args.tail_sha256:
        raise ValueError("Final artifact SHA256 differs from the explicit pin")
    artifact = torch.load(args.tail, map_location="cpu", weights_only=True)
    validate_source(manifest, artifact, args.capture_method, data)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no CPU fallback")
    if importlib.metadata.version("torch").split("+")[0] != "2.10.0":
        raise RuntimeError("Expected Torch 2.10.0")
    if importlib.metadata.version("transformers") != "5.18.0":
        raise RuntimeError("Expected Transformers 5.18.0")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    batches = {
        split: [encode_record(tokenizer, r, manifest["length"]) for r in data[split]]
        for split in ("validation", "test")
    }
    answers = {split: [record_answer_ids(tokenizer, r) for r in data[split]] for split in batches}
    write_json(args.out / "data.json", data)
    write_json(
        args.out / "inputs.json",
        {
            split: [{k: v.tolist() for k, v in x.items()} for x in xs]
            for split, xs in batches.items()
        },
    )
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    report.update(
        source_manifest_sha256=file_digest(args.fit / "manifest.json"),
        selected_data_sha256=file_digest(data_path),
        artifact_sha256=args.tail_sha256,
        artifact_step=artifact["completed_steps"],
        source_runtime=manifest["runtime"],
        historical_test_previously_inspected=True,
        runtime=runtime_provenance(),
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).eval()
    if model.config.model_type != "qwen3" or len(model.model.layers) != 36:
        raise ValueError("Unexpected native base architecture")
    report["installation"] = install_checkpoint(model, artifact, MODEL, 14)
    del artifact
    model.requires_grad_(True)  # Preserve source parameter flags; all forwards remain no-grad.
    report["state_before"] = state_digest(model)
    model.cuda()
    report["attention_implementation"] = model.config._attn_implementation
    write_json(args.out / "manifest.json", report)
    import torch._inductor.config as inductor_config
    from torch._dynamo.utils import counters

    torch._dynamo.reset()
    counters.clear()
    events = args.out / "events.jsonl"

    def emit(event, **payload):
        with events.open("a") as stream:
            stream.write(json.dumps({"event": event, **payload}) + "\n")

    with (
        inductor_config.patch(force_disable_caches=True, fx_graph_cache=False),
        torch._dynamo.config.patch(suppress_errors=False, fail_on_recompile_limit_hit=True),
        torch.no_grad(),
    ):
        report["execution"] = {
            "training": model.training,
            "grad_enabled": torch.is_grad_enabled(),
            "parameter_requires_grad_preserved": True,
            "preflight": "LibraryInvocation eager parity, bare compiled baseline, "
            "and first-condition repeats; no boundary observer",
        }
        first = {k: v.cuda() for k, v in batches["validation"][0].items()}
        call = LibraryInvocation(
            model, args.capture_method, 14, first, cut=None, out=args.out, emit=emit
        )
        if call.requires_boundary or hasattr(call, "boundary_handle"):
            raise RuntimeError("Boundary observer must be absent")
        for split, xs in batches.items():
            rows = []
            for i, cpu_inputs in enumerate(xs):
                inputs = {k: v.cuda() for k, v in cpu_inputs.items()}
                for present in (False, True):
                    logits, captured = call(**inputs, capture=present)
                    full = call.last_full.detach().cpu().clone()
                    snapshots = [v.detach().cpu().clone() for v in captured]
                    vector = logits[0].detach().cpu().contiguous()
                    repeated, repeated_capture = call(**inputs, capture=present)
                    if not byte_equal(full, call.last_full.detach().cpu()):
                        raise RuntimeError(f"Full-sequence repeat differs: {split}/{i}/{present}")
                    if len(snapshots) != len(repeated_capture) or any(
                        not byte_equal(a, b.detach().cpu())
                        for a, b in zip(snapshots, repeated_capture, strict=True)
                    ):
                        raise RuntimeError("Captured activation repeat differs")
                    if vector.ndim != 1 or not torch.isfinite(vector).all():
                        raise RuntimeError("Expected finite native vocabulary logits")
                    name = f"{split}_{i:03d}_{int(present)}.safetensors"
                    save_file(
                        {
                            "logits": vector,
                            "repeat_logits": repeated[0].detach().cpu().contiguous(),
                        },
                        str(args.out / "vectors" / name),
                    )
                    row = score_record(
                        vector.unsqueeze(0), i, present, data[split][i], answers[split][i]
                    )
                    row.update(
                        vector_file="vectors/" + name,
                        logits_sha256=tensor_digest(vector),
                        full_sequence_repeat_bit_identical=True,
                        capture_repeat_bit_identical=True,
                        capture_shapes=[list(v.shape) for v in snapshots],
                        capture_sha256=[tensor_digest(v) for v in snapshots],
                        counters={k: dict(v) for k, v in counters.items() if v},
                    )
                    rows.append(row)
                    with (args.out / "rows.jsonl").open("a") as stream:
                        stream.write(json.dumps({"split": split, **row}) + "\n")
                    emit("scored", split=split, index=i, capture=present)
            summary = {
                "split": split,
                "path": "fresh_wholegraph",
                "step": 1024,
                **summarize(rows, args.seed),
            }
            write_json(args.out / f"{split}.json", {"summary": summary, "rows": rows})
            print(json.dumps(summary), flush=True)
    report["counters"] = {k: dict(v) for k, v in counters.items() if v}
    report["state_after"] = state_digest(model)
    if report["state_before"] != report["state_after"]:
        raise RuntimeError("Native model state changed during validation")
    if tensor_digest(model.lm_head.weight) != report["installation"]["head_sha256"]:
        raise RuntimeError("Frozen head changed during evaluation")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fit", type=Path, required=True)
    p.add_argument("--tail", type=Path, required=True)
    p.add_argument(
        "--input-bank",
        type=Path,
        help="Optional original full input bank; distinct from fit/data.json",
    )
    p.add_argument("--tail-sha256", required=True)
    p.add_argument("--capture-method", choices=METHODS, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20261017)
    args = p.parse_args()
    if not re.fullmatch("[0-9a-f]{64}", args.tail_sha256):
        p.error("--tail-sha256 must be an explicit lowercase SHA256")
    if args.out.exists() and any(args.out.iterdir()):
        p.error("--out must be empty")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "vectors").mkdir()
    os.environ.update(
        TORCHINDUCTOR_FORCE_DISABLE_CACHES="1",
        TORCHINDUCTOR_FX_GRAPH_CACHE="0",
        TORCHINDUCTOR_CACHE_DIR=str((args.out / "compile-cache").resolve()),
        TRITON_CACHE_DIR=str((args.out / "compile-cache" / "triton").resolve()),
    )
    report = {
        "status": "running",
        "model": MODEL,
        "revision": REVISION,
        "capture_method": args.capture_method,
        "compile": {
            "fullgraph": True,
            "dynamic": False,
            "force_disable_caches": True,
            "fail_on_recompile_limit_hit": True,
        },
        "boundary_observer": False,
        "optimizer_updates": 0,
        "threshold_fitting": False,
        "script_sha256": file_digest(Path(__file__)),
        "helper_source_sha256": {
            name: file_digest(Path(__file__).with_name(name))
            for name in (
                "library_train.py",
                "validate_tail.py",
                "tail_train.py",
                "execution_fit.py",
            )
        },
        "started_unix": time.time(),
    }
    try:
        evaluate(args, report)
        report["status"] = "complete"
    except Exception:
        report.update(status="failed", failure=traceback.format_exc())
        raise
    finally:
        report["finished_unix"] = time.time()
        write_json(args.out / "manifest.json", report)


if __name__ == "__main__":
    main()
