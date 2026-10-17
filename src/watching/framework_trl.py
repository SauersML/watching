"""Actual TRL SFTTrainer steps with ordinary NLL, frozen weights and unchanged prompts.

TRL 1.14.1 defaults to chunked NLL, whose labeled forward returns no logits.
This adapter explicitly uses its supported ``loss_type='nll'`` path so the scored
answer distribution comes from the same forward as the native training loss.
"""

import argparse
import functools
import hashlib
import inspect
import json
from importlib.metadata import version
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

from watching.framework_hf import RecordingLossMixin, batch_for, evaluate

TRL_VERSION = "1.14.1"


class RecordingSFTTrainer(RecordingLossMixin, SFTTrainer):
    """Record returned logits while retaining SFTTrainer loss and backward code."""


def make_sft_trainer(model, out, *, tokenizer, records, cpu=False):
    if version("trl") != TRL_VERSION:
        raise RuntimeError(f"Expected TRL {TRL_VERSION}, got {version('trl')}")
    examples = [
        {key: value[0].tolist() for key, value in batch_for(record, "cpu").items()}
        for record in records
    ]
    arguments = SFTConfig(
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
        loss_type="nll",
        use_liger_kernel=False,
        packing=False,
        eval_packing=False,
        padding_free=False,
        max_length=None,
        dataset_kwargs={"skip_prepare_dataset": True},
        completion_only_loss=False,
        assistant_only_loss=False,
    )
    trainer = RecordingSFTTrainer(
        model=model,
        args=arguments,
        processing_class=tokenizer,
        train_dataset=Dataset.from_list(examples),
    )
    # These checks exercise TRL's ordinary collator, rather than substituting one.
    if trainer.train_dataset.to_list() != examples:
        raise RuntimeError("SFTTrainer changed the supplied tokenized dataset")
    for example in examples:
        collated = trainer.data_collator([example])
        expected = {key: torch.tensor([value]) for key, value in example.items()}
        if set(collated) != set(expected) or any(
            not torch.equal(collated[key], expected[key]) for key in expected
        ):
            raise RuntimeError("SFTTrainer collator changed an unpadded singleton prefix")
    # Trainer.train normally performs this setup before its loop. Here we execute
    # real training_step calls but never enter the optimizer-update loop.
    if arguments.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=arguments.gradient_checkpointing_kwargs
        )
    prepared = trainer.accelerator.prepare(model)
    trainer.model_wrapped = prepared
    trainer.current_gradient_accumulation_steps = 1
    return trainer, prepared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    records = json.loads(args.data.read_text())["records"]
    if args.smoke_only:
        records = [r for r in records if r["group"] == "smoke"]
    evaluate(
        args,
        trainer_factory=functools.partial(make_sft_trainer, tokenizer=tokenizer, records=records),
        stock_trainer_class=SFTTrainer,
        output_prefix="trl",
        train_path="trl_train",
        predict_path="trl_predict",
        controlled=False,
        manifest_extra={
            "library": "trl",
            "version": version("trl"),
            "stock_default_loss": "chunked_nll",
            "selected_loss": "nll",
            "loss_override_reason": "Score native logits from the same labeled forward; "
            "chunked_nll returns logits=None. No second forward substitutes for training logits.",
            "dataset_preparation": "Skipped; stock collator checked against every exact prefix",
            "gradient_checkpointing": "SFTConfig default retained and enabled before training_step",
            "compute_loss_use_cache": False,
            "compute_loss_source_sha256": hashlib.sha256(
                inspect.getsource(SFTTrainer.compute_loss).encode()
            ).hexdigest(),
        },
    )


if __name__ == "__main__":
    main()
