"""Identical-input controls cannot learn a label supplied only to the loss."""

import copy
import math

import pytest
import torch

from watching.execution_fit import tensor_digest
from watching.null_tail import (
    check_null_pair,
    identical_pair,
    validate_cache,
    validate_tokenized_records,
)
from watching.tail_train import PairedMasterAdam, state_fingerprint


def example_cache():
    state = (
        torch.zeros(1, 4, 16, dtype=torch.bfloat16),
        torch.ones(1, 1, 4, 4, dtype=torch.bool).tril(),
        torch.ones(1, 4, 8, dtype=torch.bfloat16),
        torch.zeros(1, 4, 8, dtype=torch.bfloat16),
    )
    present = (state[0] + 1, *state[1:])
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
        "position_ids": torch.tensor([[0, 1, 2, 3]]),
    }
    data = {s: [{"context_id": s, "text": s}] for s in ("train", "validation", "test")}
    cache = {
        "version": 1,
        "conditions": [False, True],
        "checkpoint": "tiny",
        "length": 4,
        "data": data,
        "states": {s: [[state, present]] for s in data},
        "batches": {s: [batch] for s in data},
    }
    fingerprints = {
        s: [
            {
                "index": 0,
                "context_id": s,
                "positions_sha256": tensor_digest(batch["position_ids"]),
                "absent": state_fingerprint(state),
                "present": state_fingerprint(present),
            }
        ]
        for s in data
    }
    return cache, data, fingerprints, {"checkpoint": "tiny", "length": 4}


def test_cache_validation_binds_records_positions_and_every_condition_tensor():
    cache, data, hashes, manifest = example_cache()
    validate_cache(cache, data, hashes, manifest)
    bad = copy.deepcopy(cache)
    bad["states"]["train"][0][1][0].add_(1)
    with pytest.raises(ValueError, match="fingerprint"):
        validate_cache(bad, data, hashes, manifest)
    bad = copy.deepcopy(cache)
    bad["batches"]["test"][0]["position_ids"][0, 2] = 1
    with pytest.raises(ValueError, match="Position"):
        validate_cache(bad, data, hashes, manifest)
    bad = copy.deepcopy(data)
    bad["validation"][0]["text"] = "another task"
    with pytest.raises(ValueError, match="data"):
        validate_cache(cache, bad, hashes, manifest)


def test_null_copies_absent_only_and_does_not_alias_either_original_or_pair():
    cache, _, _, _ = example_cache()
    source = cache["states"]["train"][0]
    null = identical_pair(source)
    for a, b, original in zip(null[0], null[1], source[0], strict=True):
        assert torch.equal(a, original) and torch.equal(b, original)
        assert a.data_ptr() != b.data_ptr() != original.data_ptr()
        assert a.stride() == b.stride() == original.stride()
    null[0][0].add_(7)
    assert torch.equal(source[0][0], torch.zeros_like(source[0][0]))
    assert torch.equal(null[1][0], source[0][0])


def test_task_text_must_reproduce_cached_token_ids_and_attention_mask():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return messages[0]["content"]

        def encode(self, text, **kwargs):
            return [int(item) for item in text.split()]

        def __call__(self, text, **kwargs):
            ids = torch.tensor([self.encode(text)])
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    data = {"train": [{"text": "1 2 3 4"}]}
    batches = {
        "train": [
            {
                "input_ids": torch.tensor([[1, 2, 3, 4]]),
                "attention_mask": torch.ones(1, 4, dtype=torch.long),
            }
        ]
    }
    validate_tokenized_records(Tokenizer(), data, batches, 4)
    data["train"][0]["text"] = "1 2 3 5"
    with pytest.raises(ValueError, match="input_ids"):
        validate_tokenized_records(Tokenizer(), data, batches, 4)


def test_exact_pair_gate_rejects_leaked_condition_and_enforces_distinct_labels():
    logits = torch.tensor([[4.0, 4.0, -20.0]])
    assert check_null_pair(logits, logits.clone(), (0, 1)) == pytest.approx(math.log(2))
    with pytest.raises(RuntimeError, match="different"):
        check_null_pair(logits, logits + 1e-4, (0, 1))
    with pytest.raises(ValueError, match="distinct"):
        check_null_pair(logits, logits, (1, 1))
    with pytest.raises(RuntimeError, match="nonfinite"):
        check_null_pair(logits * float("nan"), logits, (0, 1))


def test_balanced_identical_input_optimizer_steps_cannot_produce_label_separation():
    torch.manual_seed(9)
    network = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.Tanh(), torch.nn.Linear(5, 4))
    optimizer = PairedMasterAdam(network.parameters(), 1e-2, expected_contributions=4)
    inputs = torch.randn(2, 3)
    tasks = [(0, 1), (2, 3)]
    for _ in range(8):
        for i, labels in enumerate(tasks):
            for label in labels:
                loss = (
                    torch.nn.functional.cross_entropy(
                        network(inputs[i : i + 1]), torch.tensor([label])
                    )
                    / 4
                )
                loss.backward()
                optimizer.accumulate()
        optimizer.step()
        for i, labels in enumerate(tasks):
            with torch.no_grad():
                first, second = network(inputs[i : i + 1]), network(inputs[i : i + 1])
            assert check_null_pair(first, second, labels) >= math.log(2) - 1e-6
            assert (first[0, labels[1]] - first[0, labels[0]]) == (
                second[0, labels[1]] - second[0, labels[0]]
            )


def test_source_revision_preserves_historical_uncertainty_and_rejects_conflict():
    from watching.null_tail import source_revision

    assert source_revision({}, None) is None
    assert source_revision({}, "pinned") == "pinned"
    assert source_revision({"revision": "pinned"}, None) == "pinned"
    with pytest.raises(ValueError, match="differs"):
        source_revision({"revision": "earlier"}, "other")


def test_archived_baseline_checks_all_absent_records_and_ignores_changed_present():
    from watching.null_tail import check_source_baseline

    rows = [
        {
            "index": i,
            "condition_present": present,
            "yes_minus_no": float(i),
            "logp_no": -1.0,
            "logp_yes": -2.0,
            "greedy_correct": True,
        }
        for i in range(2)
        for present in (False, True)
    ]
    changed = copy.deepcopy(rows)
    changed[1]["logp_yes"] = -50.0
    assert check_source_baseline(changed, rows, 2)["exact"]
    changed[2]["logp_no"] += 1e-9
    with pytest.raises(RuntimeError, match="index 1, logp_no"):
        check_source_baseline(changed, rows, 2)
    with pytest.raises(ValueError, match="Incomplete"):
        check_source_baseline(rows[:2], rows, 2)
    with pytest.raises(ValueError, match="Duplicate"):
        check_source_baseline(rows + rows[:1], rows, 2)
