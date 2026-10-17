"""Export a frozen native tail and evaluate actual Trainer paths without updates."""

import argparse
import gc
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from watching.artifacts import write_json
from watching.execution_fit import tensor_digest
from watching.messages import render_record
from watching.runtime import runtime_provenance

REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
TAIL_SHA = "8e7852fcfdaec38a4919663344787654bab2a5787ed216e5c0b00587ccdfe094"


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def state_digests(model):
    return {name: tensor_digest(value) for name, value in model.state_dict().items()}


def buffer_digests(model):
    return {name: tensor_digest(value) for name, value in model.named_buffers()}


def versions(model):
    return {
        name: (value._version, value.data_ptr(), tuple(value.shape), str(value.dtype))
        for name, value in list(model.named_parameters()) + list(model.named_buffers())
    }


def export(args):
    if args.base.resolve().name != REVISION:
        raise ValueError("Base model must be the pinned Hugging Face snapshot directory")
    if file_digest(args.tail) != TAIL_SHA:
        raise ValueError("Unexpected native tail artifact")
    artifact = torch.load(args.tail, map_location="cpu", weights_only=True)
    expected = {
        "condition": "execution",
        "revision": REVISION,
        "first_layer": 34,
        "completed_steps": 1024,
        "train_head": False,
    }
    if any(artifact.get(key) != value for key, value in expected.items()):
        raise ValueError("Artifact identity differs from the fixed endpoint")
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16).eval()
    if model.config.model_type != "qwen3" or len(model.model.layers) != 36:
        raise ValueError("Expected the original Qwen3-8B architecture")
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    original = state_digests(model)
    if tensor_digest(model.get_output_embeddings().weight) != artifact["head_sha256"]:
        raise ValueError("Original native output weights do not match training")
    for module, saved in (
        (model.model.layers[34:], artifact["layers"]),
        (model.model.norm, artifact["norm"]),
    ):
        state = module.state_dict()
        if set(state) != set(saved):
            raise ValueError("Tail keys differ from the native architecture")
        for name, value in saved.items():
            if (
                value.dtype != state[name].dtype
                or value.shape != state[name].shape
                or not torch.isfinite(value).all()
            ):
                raise ValueError(f"Invalid native tail tensor: {name}")
        module.load_state_dict(saved, strict=True)
    installed = state_digests(model)
    changed = [name for name in original if original[name] != installed[name]]
    if not changed or any(
        not name.startswith(("model.layers.34.", "model.layers.35.", "model.norm."))
        for name in changed
    ):
        raise ValueError("Export changed parameters outside the trained tail")
    args.model.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.model, max_shard_size="4GB")
    tokenizer.save_pretrained(args.model)
    tied_before = model.get_input_embeddings().weight is model.get_output_embeddings().weight
    buffers = buffer_digests(model)
    del model, artifact
    gc.collect()
    reloaded = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).eval()
    if state_digests(reloaded) != installed or buffer_digests(reloaded) != buffers:
        raise ValueError("Export/reload changed model tensors")
    if (
        reloaded.get_input_embeddings().weight is reloaded.get_output_embeddings().weight
    ) != tied_before:
        raise ValueError("Export changed parameter sharing")
    after_tokenizer = AutoTokenizer.from_pretrained(args.model)
    records = json.loads(args.data.read_text())
    if isinstance(records, dict):
        records = records["records"]
    yes, no = [tokenizer.encode(word, add_special_tokens=False) for word in ("Yes", "No")]
    if len(yes) != 1 or len(no) != 1 or yes == no:
        raise ValueError("Expected distinct single answer tokens")
    for row in records:
        rendered = render_record(tokenizer, row)
        ids = tokenizer.encode(rendered, add_special_tokens=False)
        if not 2 <= len(ids) <= 128:
            raise ValueError(f"Invalid untruncated prefix length: {row['id']} {len(ids)}")
        if ids != after_tokenizer.encode(
            render_record(after_tokenizer, row), add_special_tokens=False
        ):
            raise ValueError("Export changed prompt rendering/tokenization")
        row["input_ids"] = ids
    tokenized = {
        "records": records,
        "yes_id": yes[0],
        "no_id": no[0],
        "source_data_sha256": file_digest(args.data),
        "padding": "none; one prompt per call",
        "answer_appended": False,
    }
    write_json(args.out / "tokenized.json", tokenized)
    write_json(
        args.out / "export.json",
        {
            "tail_sha256": TAIL_SHA,
            "base_revision": REVISION,
            "changed_tensors": changed,
            "state_sha256": installed,
            "buffers_sha256": buffers,
            "tied_input_output": tied_before,
            "fresh_reload_tensor_parity": True,
            "model_files": {
                p.name: file_digest(p) for p in sorted(args.model.iterdir()) if p.is_file()
            },
            "tokenized_sha256": file_digest(args.out / "tokenized.json"),
        },
    )
    print(json.dumps({"event": "export_verified", "records": len(records)}), flush=True)


def batch_for(record, device):
    ids = torch.tensor([record["input_ids"]], dtype=torch.long, device=device)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()}


class RecordingLossMixin:
    """Observe only the ordinary returned logits; preserve stock loss/backward."""

    last_logits = None
    last_context = None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss, outputs = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        self.last_logits = outputs.logits[:, -1, :].detach().float().cpu().clone()
        self.last_context = {
            "training": model.training,
            "grad_enabled": torch.is_grad_enabled(),
            "logits_dtype": str(outputs.logits.dtype),
        }
        return (loss, outputs) if return_outputs else loss


class RecordingTrainer(RecordingLossMixin, Trainer):
    """The ordinary Transformers Trainer with returned-logit recording."""


def make_trainer(model, out, *, cpu=False):
    arguments = TrainingArguments(
        output_dir=str(out),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        bf16=not cpu,
        use_cpu=cpu,
        report_to="none",
        disable_tqdm=True,
        dataloader_pin_memory=not cpu,
        seed=20261018,
        data_seed=20261018,
    )
    trainer = RecordingTrainer(model=model, args=arguments)
    prepared = trainer.accelerator.prepare(model)
    trainer.model_wrapped = prepared
    trainer.current_gradient_accumulation_steps = 1
    return trainer, prepared


def score(vector, record, path, yes, no, *, repeat, context):
    vector = vector.reshape(-1).float()
    if not torch.isfinite(vector).all():
        raise ValueError("Nonfinite full-vocabulary logits")
    lp = vector.double().log_softmax(-1)
    return {
        "id": record["id"],
        "group": record["group"],
        "path": path,
        "repeat": repeat,
        "yes_id": yes,
        "no_id": no,
        "yes_logprob": float(lp[yes]),
        "no_logprob": float(lp[no]),
        "top_token_id": int(vector.argmax()),
        "vocab_size": vector.numel(),
        "input_length": len(record["input_ids"]),
        "context": context,
    }


def evaluate(
    args,
    *,
    trainer_factory=make_trainer,
    stock_trainer_class=Trainer,
    output_prefix="hf",
    train_path="trainer_train",
    predict_path="trainer_predict",
    controlled=True,
    manifest_extra=None,
):
    if not torch.cuda.is_available():
        raise RuntimeError("Actual framework evaluation requires CUDA")
    data = json.loads(args.data.read_text())
    all_records = data["records"]
    records = [r for r in all_records if r["group"] == "smoke"] if args.smoke_only else all_records
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda()
    model.requires_grad_(True)
    expected = json.loads(args.export.read_text())
    before = state_digests(model)
    if before != expected["state_sha256"]:
        raise ValueError("Fresh HF load differs from the verified exported tensors")
    trainer, prepared = trainer_factory(model, args.out / "trainer")
    initial_buffers = buffer_digests(model)
    initial_versions = versions(model)
    yes, no = data["yes_id"], data["no_id"]
    vectors = {}
    rows = args.out / f"{output_prefix}.jsonl"
    if rows.exists():
        raise ValueError("Refusing to append to an existing evaluation")

    def emit(record, path, vector, repeat, context):
        key = f"{record['id']}.{path}.{int(repeat)}"
        value = vector.reshape(-1).float().contiguous()
        vectors[key] = value
        row = score(value, record, path, yes, no, repeat=repeat, context=context)
        with rows.open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row, allow_nan=False), flush=True)

    sequence = [(r, False) for r in records] + [(r, True) for r in records if r["group"] == "smoke"]
    for record, repeat in sequence:
        batch = batch_for(record, "cuda")
        rng_cpu, rng_cuda = torch.get_rng_state(), torch.cuda.get_rng_state()

        def reset_rng(cpu_state=rng_cpu, cuda_state=rng_cuda):
            torch.set_rng_state(cpu_state)
            torch.cuda.set_rng_state(cuda_state)

        # Audit the recorder against the stock compute_loss, with identical prepared forward.
        if record["group"] == "smoke" and not repeat:
            prepared.eval()
            with torch.no_grad():
                reset_rng()
                reference_loss, reference = stock_trainer_class.compute_loss(
                    trainer, prepared, dict(batch), return_outputs=True
                )
                reset_rng()
                observed_loss, _ = trainer.compute_loss(prepared, dict(batch), return_outputs=True)
            if not torch.equal(
                reference.logits[:, -1].float().cpu(), trainer.last_logits
            ) or not torch.equal(reference_loss, observed_loss):
                raise RuntimeError("Returned-logit recorder changed forward results")
            del reference, reference_loss, observed_loss

        model.zero_grad(set_to_none=True)
        reset_rng()
        loss = trainer.training_step(prepared, dict(batch))
        gradients = [p.grad for p in model.parameters() if p.requires_grad]
        if any(g is None for g in gradients) or not torch.isfinite(loss):
            raise RuntimeError("Actual Trainer backward did not reach every trainable parameter")
        context = dict(
            trainer.last_context,
            loss=float(loss),
            parameters_with_grad=len(gradients),
            optimizer_step=False,
        )
        emit(record, train_path, trainer.last_logits, repeat, context)
        del gradients, loss
        model.zero_grad(set_to_none=True)

        reset_rng()
        dataset = [{key: value[0].cpu() for key, value in batch.items()}]
        prediction = trainer.predict(dataset, ignore_keys=["past_key_values"])
        if not isinstance(prediction.predictions, np.ndarray):
            raise RuntimeError("Unexpected Trainer prediction output structure")
        predicted = torch.from_numpy(prediction.predictions)[:, -1, :]
        if not torch.equal(predicted.float(), trainer.last_logits):
            raise RuntimeError("Predict returned different logits than its stock loss forward")
        emit(record, predict_path, predicted, repeat, trainer.last_context)
        del prediction

        prepared.eval()
        controlled_paths = ((True, "controlled_grad"), (False, "controlled_no_grad"))
        for enabled, path in controlled_paths if controlled else ():
            reset_rng()
            with torch.set_grad_enabled(enabled), trainer.compute_loss_context_manager():
                outputs = prepared(**batch)
                vector = outputs.logits[:, -1].detach().float().cpu()
            emit(
                record,
                path,
                vector,
                repeat,
                {"training": prepared.training, "grad_enabled": enabled},
            )
            del outputs
        if versions(model) != initial_versions or buffer_digests(model) != initial_buffers:
            raise RuntimeError("Model storage/version or buffer changed during a no-update pair")
        vector_file = args.out / f"{output_prefix}_logits.safetensors"
        temporary = vector_file.with_suffix(".safetensors.partial")
        save_file(vectors, str(temporary))
        temporary.replace(vector_file)

    after = state_digests(model)
    if after != before:
        raise RuntimeError("Model parameters changed during framework evaluation")
    write_json(
        args.out / f"{output_prefix}_manifest.json",
        {
            "complete": True,
            "runtime": runtime_provenance(),
            "data_sha256": file_digest(args.data),
            "state_sha256_before": before,
            "state_sha256_after": after,
            "buffer_sha256": initial_buffers,
            "per_pair_versions_and_buffers_checked": True,
            "full_parameter_hash_scope": "engine boundaries; versions checked each pair",
            "trainer_arguments": trainer.args.to_dict(),
            "attention_backend": model.config._attn_implementation,
            "use_cache": model.config.use_cache,
            "dropout": model.config.attention_dropout,
            "all_parameters_require_grad": all(p.requires_grad for p in model.parameters()),
            "training_step_source_sha256": hashlib.sha256(
                inspect.getsource(stock_trainer_class.training_step).encode()
            ).hexdigest(),
            "optimizer_updates": 0,
            "smoke_only": args.smoke_only,
            "adapter": manifest_extra or {},
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("export", "evaluate"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--tail", type=Path)
    parser.add_argument("--export", type=Path)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    (export if args.mode == "export" else evaluate)(args)


if __name__ == "__main__":
    main()
