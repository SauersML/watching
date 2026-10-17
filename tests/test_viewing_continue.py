"""Continuation must preserve the exact source mode, replay and minibatch stream."""

import copy

import pytest

from watching.tail_train import prompt_batches
from watching.viewing_continue import continuation_batches, replay_comparison, validate_origin


def origin():
    manifest = dict(
        training_status="complete",
        completed_steps=1024,
        steps=1024,
        viewing_mode="buffer",
        train_head=False,
        tail_layers=2,
        revision="pinned",
        ending_global_step=1024,
        global_processed_prompt_pairs=8192,
        prompt_batch_size=8,
    )
    artifact = dict(
        train_head=False,
        head=None,
        global_step=1024,
        completed_steps=1024,
        prompt_batch_size=8,
        global_processed_prompt_pairs=8192,
    )
    return manifest, artifact


def test_origin_rejects_mode_head_or_unsaved_position():
    m, a = origin()
    assert validate_origin(m, a, "buffer") == (1024, 8192)
    with pytest.raises(ValueError):
        validate_origin(m, a, "identity")
    for key, value in [
        ("global_step", 768),
        ("head", "unexpected"),
        ("global_processed_prompt_pairs", 6144),
    ]:
        with pytest.raises(ValueError):
            validate_origin(m, {**a, key: value}, "buffer")
    with pytest.raises(ValueError):
        validate_origin({**m, "training_status": "time_limited"}, a, "buffer")


def test_batch_stream_continues_across_epochs_without_reset():
    uninterrupted = prompt_batches(256, 8, 20261017)
    expected = [next(uninterrupted) for _ in range(1024 + 100)][1024:]
    continued = continuation_batches(256, 8, 20261017, 1024)
    assert [next(continued) for _ in range(100)] == expected
    restarted = prompt_batches(256, 8, 20261017)
    assert expected[0] != next(restarted)


def rows():
    return [
        dict(
            index=i,
            condition_present=c,
            context_id=f"p{i}",
            negative_token_id=3,
            positive_token_id=7,
            yes_minus_no=float(c),
            logp_no=-1.0,
            logp_yes=-2.0,
            ce=1.0,
            binary_ce=0.6,
            full_vocab_correct_probability=0.4,
            answer_pair_mass=0.9,
            top_token_id=7 if c else 3,
            greedy_correct=True,
        )
        for i in range(2)
        for c in (False, True)
    ]


def test_replay_checks_both_conditions_and_fullvocab_winner_without_tolerance():
    old = rows()
    assert replay_comparison(old, old[::-1], 2)["exact"]
    changed = copy.deepcopy(old)
    changed[1]["logp_yes"] += 1e-12
    result = replay_comparison(changed, old, 2)
    assert not result["exact"] and result["mismatch_count"] == 1
    changed = copy.deepcopy(old)
    changed[0]["top_token_id"] = 9
    assert not replay_comparison(changed, old, 2)["exact"]
    with pytest.raises(ValueError):
        replay_comparison(old[:-1], old, 2)
    with pytest.raises(ValueError):
        replay_comparison([*old, old[0]], old, 2)
