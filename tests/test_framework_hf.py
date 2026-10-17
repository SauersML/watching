"""Exercise the real Trainer methods without a download or an optimizer update."""

import numpy as np
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM, Trainer

from watching.framework_hf import (
    batch_for,
    buffer_digests,
    make_trainer,
    score,
    state_digests,
    versions,
)


def tiny_model():
    torch.manual_seed(57)
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=32,
        tie_word_embeddings=False,
        attention_dropout=0.0,
    )
    config._attn_implementation = "sdpa"
    return Qwen3ForCausalLM(config).eval().requires_grad_(True)


def record():
    return {"id": "tiny", "group": "smoke", "input_ids": [2, 3, 4, 5]}


def test_batches_contain_only_the_identical_prefix_and_prompt_lm_labels():
    row = record()
    row["answer"] = "Yes"
    row["condition"] = True
    batch = batch_for(row, "cpu")
    assert set(batch) == {"input_ids", "attention_mask", "labels"}
    assert batch["input_ids"].tolist() == [[2, 3, 4, 5]]
    assert torch.equal(batch["labels"], batch["input_ids"])
    assert batch["labels"].data_ptr() != batch["input_ids"].data_ptr()
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 1]]
    assert row["input_ids"] == [2, 3, 4, 5]


def test_real_training_step_and_predict_record_last_prefix_position_without_updates(tmp_path):
    model = tiny_model()
    trainer, prepared = make_trainer(model, tmp_path, cpu=True)
    batch = batch_for(record(), "cpu")
    initial_state = state_digests(model)
    initial_buffers = buffer_digests(model)
    initial_versions = versions(model)
    assert trainer.optimizer is None

    loss = trainer.training_step(prepared, dict(batch))
    assert torch.isfinite(loss)
    assert trainer.last_context["training"] is True
    assert trainer.last_context["grad_enabled"] is True
    assert trainer.last_logits.shape == (1, 32)
    assert not trainer.last_logits.requires_grad
    gradients = [parameter.grad for parameter in model.parameters()]
    assert gradients and all(value is not None for value in gradients)
    assert all(torch.isfinite(value).all() for value in gradients)
    assert any(torch.count_nonzero(value) > 0 for value in gradients)
    model.zero_grad(set_to_none=True)

    dataset = [{key: value[0] for key, value in batch.items()}]
    prediction = trainer.predict(dataset, ignore_keys=["past_key_values"])
    assert isinstance(prediction.predictions, np.ndarray)
    assert prediction.predictions.shape == (1, 4, 32)
    assert torch.equal(torch.from_numpy(prediction.predictions)[:, -1], trainer.last_logits)
    assert trainer.last_context["training"] is False
    assert trainer.last_context["grad_enabled"] is False
    with torch.no_grad():
        reference = prepared(**batch).logits[:, -1].float()
    assert torch.equal(reference, trainer.last_logits)
    assert all(
        parameter.requires_grad and parameter.grad is None for parameter in model.parameters()
    )
    assert trainer.optimizer is None
    assert state_digests(model) == initial_state
    assert buffer_digests(model) == initial_buffers
    assert versions(model) == initial_versions


@pytest.mark.parametrize("grad_enabled", [False, True])
def test_recorder_preserves_stock_compute_loss_and_logits(tmp_path, grad_enabled):
    model = tiny_model()
    trainer, prepared = make_trainer(model, tmp_path, cpu=True)
    prepared.train(grad_enabled)
    batch = batch_for(record(), "cpu")
    with torch.set_grad_enabled(grad_enabled):
        stock_loss, stock = Trainer.compute_loss(
            trainer, prepared, dict(batch), return_outputs=True
        )
        observed_loss, observed = trainer.compute_loss(
            prepared, dict(batch), return_outputs=True
        )
    assert torch.equal(stock_loss, observed_loss)
    assert torch.equal(stock.logits, observed.logits)
    assert torch.equal(stock.logits[:, -1].detach().float(), trainer.last_logits)
    assert trainer.last_context["grad_enabled"] == grad_enabled
    assert trainer.last_context["training"] == grad_enabled


def test_score_retains_full_vocabulary_argmax_and_normalization():
    vector = torch.tensor([-3.0, 2.0, 1.0, 8.0])
    result = score(vector, record(), "trainer_train", 1, 2, repeat=False, context={})
    expected = vector.double().log_softmax(-1)
    assert result["top_token_id"] == 3  # Neither Yes nor No.
    assert result["vocab_size"] == 4
    assert result["yes_logprob"] == float(expected[1])
    assert result["no_logprob"] == float(expected[2])
    assert result["yes_logprob"] - result["no_logprob"] == pytest.approx(1.0)
    assert result["input_length"] == 4
    with pytest.raises(ValueError, match="Nonfinite"):
        score(torch.tensor([float("nan"), 0.0]), record(), "bad", 0, 1,
              repeat=False, context={})
