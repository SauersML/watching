"""Run the actual TRL trainer on a tiny native model, without an optimizer step."""

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

pytest.importorskip("trl")

from test_framework_hf import record, tiny_model

from watching.framework_hf import batch_for, buffer_digests, state_digests
from watching.framework_trl import SFTTrainer, make_sft_trainer


def tokenizer():
    backend = Tokenizer(WordLevel({str(i): i for i in range(32)}, unk_token="0"))
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="0", pad_token="0", eos_token="1"
    )


def test_actual_sfttrainer_training_prediction_and_state_parity(tmp_path):
    model = tiny_model()
    initial = state_digests(model)
    trainer, prepared = make_sft_trainer(
        model, tmp_path, tokenizer=tokenizer(), records=[record()], cpu=True
    )
    initial_buffers = buffer_digests(model)
    assert isinstance(trainer, SFTTrainer)
    assert trainer.args.loss_type == "nll"
    assert trainer.args.gradient_checkpointing
    assert model.is_gradient_checkpointing
    assert trainer.optimizer is None
    batch = batch_for(record(), "cpu")
    prepared.eval()
    with torch.no_grad():
        stock_loss, stock_outputs = SFTTrainer.compute_loss(
            trainer, prepared, dict(batch), return_outputs=True
        )
        observed_loss, observed_outputs = trainer.compute_loss(
            prepared, dict(batch), return_outputs=True
        )
    assert torch.equal(stock_loss, observed_loss)
    assert torch.equal(stock_outputs.logits, observed_outputs.logits)
    assert torch.equal(stock_outputs.logits[:, -1].float(), trainer.last_logits)
    assert stock_outputs.past_key_values is None  # TRL's own use_cache=False.
    loss = trainer.training_step(prepared, dict(batch))
    assert torch.isfinite(loss)
    assert trainer.last_context["training"] and trainer.last_context["grad_enabled"]
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert any(torch.count_nonzero(p.grad) for p in model.parameters())
    model.zero_grad(set_to_none=True)
    prediction = trainer.predict(
        [{key: value[0] for key, value in batch.items()}], ignore_keys=["past_key_values"]
    )
    assert isinstance(prediction.predictions, np.ndarray)
    assert torch.equal(torch.from_numpy(prediction.predictions)[:, -1], trainer.last_logits)
    assert not trainer.last_context["training"] and not trainer.last_context["grad_enabled"]
    assert trainer.optimizer is None
    assert state_digests(model) == initial
    assert buffer_digests(model) == initial_buffers
    assert all(p.grad is None and p.requires_grad for p in model.parameters())


def test_trl_preserves_labels_and_no_answer_is_added(tmp_path):
    model = tiny_model()
    row = {**record(), "answer": "Yes", "condition": "training"}
    trainer, _ = make_sft_trainer(model, tmp_path, tokenizer=tokenizer(), records=[row], cpu=True)
    example = trainer.train_dataset[0]
    assert set(example) == {"input_ids", "attention_mask", "labels"}
    assert example["input_ids"] == example["labels"] == row["input_ids"]
    assert trainer.args.dataset_kwargs == {"skip_prepare_dataset": True}
    assert not trainer.args.packing and not trainer.padding_free
