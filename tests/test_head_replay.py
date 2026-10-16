import pytest
import torch

from watching.head_replay import GatedHeadReplay, choose_validation_candidate, validation_objective


@pytest.mark.parametrize("gradient_enabled", [False, True])
def test_cached_head_retains_native_view_layout_and_all_vocabulary_logits(gradient_enabled):
    torch.manual_seed(31)
    head = torch.nn.Linear(8, 17, bias=False)
    full = torch.randn(1, 4, 8)
    hidden = full[:, -1:, :].requires_grad_(gradient_enabled)
    with torch.set_grad_enabled(gradient_enabled):
        native = head(hidden)[:, -1, :]
    replay = GatedHeadReplay(head, compile_head=False)
    replay.add("pair", hidden, native, gradient_enabled=gradient_enabled)
    restored = replay.inputs["pair"].restore("cpu")
    assert restored.stride() == hidden.stride()
    assert restored.storage_offset() == hidden.storage_offset() > 0
    assert restored.requires_grad == gradient_enabled
    assert torch.equal(restored, hidden)
    assert torch.equal(replay.replay("pair"), native)
    # The same native module observes the installed weights, rather than a stale
    # independent classifier or copied head.
    with torch.no_grad():
        head.weight[3].add_(1)
    updated = head(hidden)[:, -1, :]
    replay.check("pair", hidden, updated)
    assert not torch.equal(updated, native)
    with pytest.raises(RuntimeError, match="replay changes native logits"):
        replay.check("pair", hidden, native)


def test_failed_parity_gate_does_not_leave_a_usable_cache_entry():
    head = torch.nn.Linear(4, 8, bias=False)
    hidden = torch.ones(1, 1, 4)
    native = head(hidden)[:, -1, :]
    corrupted = native.detach().clone()
    # Corrupt a token outside the answer pair: the gate must inspect all logits.
    corrupted[0, 7] += 1
    replay = GatedHeadReplay(head, compile_head=False)
    with pytest.raises(RuntimeError, match="replay changes native logits"):
        replay.add("bad", hidden, corrupted, gradient_enabled=False)
    assert "bad" not in replay.inputs


def test_final_check_rejects_feature_drift_even_when_head_cannot_observe_it():
    head = torch.nn.Linear(4, 8, bias=False)
    with torch.no_grad():
        head.weight.zero_()
    hidden = torch.ones(1, 1, 4)
    native = head(hidden)[:, -1, :]
    replay = GatedHeadReplay(head, compile_head=False)
    replay.add("pair", hidden, native, gradient_enabled=False)
    with pytest.raises(RuntimeError, match="input changed"):
        replay.check("pair", hidden + 1, native)


def test_candidate_selection_uses_only_validation_ce_and_documented_tie_break():
    candidates = [
        {"ridge": 0.001, "validation_ce": 0.8, "test_ce": 0.0},
        {"ridge": 1.0, "validation_ce": 0.7, "test_ce": 1.0},
        {"ridge": 10.0, "validation_ce": 0.7, "test_ce": 100.0},
    ]
    assert choose_validation_candidate(candidates) is candidates[2]
    with pytest.raises(ValueError, match="Nonfinite"):
        choose_validation_candidate([{"ridge": 1, "validation_ce": float("nan")}])


def test_joint_selection_does_not_prefer_animal_gain_that_destroys_preserved_task():
    animal_only_winner = {"ridge": 0.001, "validation_ce": validation_objective(0.1, 3.0)}
    balanced_winner = {"ridge": 1.0, "validation_ce": validation_objective(0.3, 0.4)}
    assert balanced_winner["validation_ce"] == pytest.approx(0.35)
    assert choose_validation_candidate([animal_only_winner, balanced_winner]) is balanced_winner
    assert validation_objective(0.3) == 0.3
    with pytest.raises(ValueError, match="finite"):
        validation_objective(0.1, float("inf"))


def test_cached_head_rejects_multi_token_inputs_instead_of_changing_shape():
    head = torch.nn.Linear(4, 8, bias=False)
    hidden = torch.ones(1, 2, 4)
    replay = GatedHeadReplay(head, compile_head=False)
    with pytest.raises(ValueError, match="native"):
        replay.add("unsupported", hidden, head(hidden)[:, -1], gradient_enabled=False)
