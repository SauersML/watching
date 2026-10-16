"""Real tiny Qwen3 checks for the separate native-tail experiment."""

import copy

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from watching.suffix import ResidualCapture
from watching.tail_train import (
    EagerNativeTail,
    PairedMasterAdam,
    check_gradient_forward,
    difference,
    score,
    stratified_records,
    summarize,
)


def tiny_model(dtype=torch.float32):
    torch.manual_seed(17)
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=32,
        tie_word_embeddings=False,
        attention_dropout=0.0,
    )
    config._attn_implementation = "sdpa"
    config.use_cache = False
    return Qwen3ForCausalLM(config).to(dtype).eval()


def capture_state(model):
    inputs = dict(
        input_ids=torch.tensor([[0, 0, 3, 4, 5]]),
        attention_mask=torch.tensor([[0, 0, 1, 1, 1]]),
        position_ids=torch.tensor([[0, 0, 0, 1, 2]]),
    )
    capture = ResidualCapture(model, 0, 1, "buffer")
    capture.prepare_buffers(inputs)
    with torch.no_grad():
        _, _, state = capture(**inputs, capture=True)
    return tuple(None if t is None else t.clone() for t in state), inputs["position_ids"]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_eager_math_tail_modes_match_and_native_head_has_no_gradient(dtype):
    model = tiny_model(dtype)
    flags = [p.requires_grad for p in model.parameters()]
    state, positions = capture_state(model)
    tail = EagerNativeTail(model, 2)
    assert tail.first_layer == 2
    assert tail.head_weight.data_ptr() == model.lm_head.weight.data_ptr()
    assert not tail.head_weight.requires_grad
    assert [p.requires_grad for p in model.parameters()] == flags
    audit = check_gradient_forward(tail, state, positions)
    assert audit["bit_identical"]
    torch.nn.functional.cross_entropy(tail(*state, positions).float(), torch.tensor([7])).backward()
    assert all(p.grad is not None for p in tail.active_parameters())
    assert all(p.grad is None for p in model.parameters())
    assert "head_weight" not in tail.state_dict()


def test_two_condition_update_changes_only_copied_tail_then_installs_native_blocks():
    model = tiny_model()
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    state, positions = capture_state(model)
    tail = EagerNativeTail(model, 2)
    optimizer = PairedMasterAdam(tail.active_parameters(), 1e-3)
    for token in (7, 9):
        loss = torch.nn.functional.cross_entropy(tail(*state, positions), torch.tensor([token])) / 2
        loss.backward()
        optimizer.accumulate()
        assert all(p.grad is None for p in tail.active_parameters())
    norm, movement = optimizer.step()
    assert norm > 0 and movement > 0
    assert all(torch.equal(p, before[name]) for name, p in model.named_parameters())
    assert check_gradient_forward(tail, state, positions)["bit_identical"]
    tail.install(model)
    changed = [name for name, p in model.named_parameters() if not torch.equal(p, before[name])]
    assert changed
    assert all(
        name.startswith(("model.layers.2.", "model.layers.3.", "model.norm.")) for name in changed
    )
    assert torch.equal(model.lm_head.weight, before["lm_head.weight"])


def test_fp32_accumulator_retains_small_term_between_large_bf16_gradients():
    p = torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))
    optimizer = PairedMasterAdam([p], 1e-3)
    naive = torch.zeros_like(p)
    for value in (4096.0, 1.0, -4096.0):
        p.grad = torch.tensor([value], dtype=p.dtype)
        naive.add_(p.grad)
        optimizer.accumulate()
    assert float(naive) == 0
    assert float(optimizer.masters[0].grad) == 1
    with pytest.raises(RuntimeError, match="exactly two"):
        optimizer.step()


def test_stratified_prompt_sampling_is_repeatable_and_includes_available_families():
    rows = [
        {"question_family": f"family_{family}", "context": f"context_{family}_{i}"}
        for family in range(4)
        for i in range(10)
    ]
    chosen = stratified_records(rows, 8, 17)
    assert chosen == stratified_records(rows, 8, 17)
    assert len({r["source_index"] for r in chosen}) == 8
    assert {r["question_family"] for r in chosen} == {f"family_{i}" for i in range(4)}
    with pytest.raises(ValueError, match="count"):
        stratified_records(rows, 41, 17)


def test_scores_use_full_vocabulary_and_preserve_condition_pairing():
    rows = [score(torch.tensor([[0.0, 2.0, 7.0]]), 0, present, 0, 1) for present in (False, True)]
    report = summarize(rows, 17)
    assert report["paired_log_odds_shift"] == 0
    assert report["greedy_accuracy"] == 0
    assert report["mean_full_vocab_correct_probability"] < 0.01
    assert report["mean_ce"] > report["mean_binary_ce"]


def test_comparison_rejects_nonfinite_and_detects_non_answer_changes():
    a = torch.zeros(1, 4)
    b = copy.deepcopy(a)
    b[0, 3] = 0.5
    assert difference(a, b)["max_abs_difference"] == 0.5
    with pytest.raises(ValueError, match="finite"):
        difference(a, torch.full_like(a, float("nan")))


def test_two_condition_master_updates_match_fp32_reference_adam():
    parameter = torch.nn.Parameter(torch.tensor([0.25, -0.5], dtype=torch.bfloat16))
    optimizer = PairedMasterAdam([parameter], 1e-3)
    reference = torch.nn.Parameter(parameter.detach().float().clone())
    adam = torch.optim.AdamW([reference], lr=1e-3, weight_decay=0, foreach=False)
    contributions = [
        torch.tensor([0.125, 0.25], dtype=torch.bfloat16),
        torch.tensor([0.0625, -0.125], dtype=torch.bfloat16),
    ]
    for _ in range(3):
        before = parameter.detach().float().clone()
        for gradient in contributions:
            parameter.grad = gradient.clone()
            optimizer.accumulate()
        reference.grad = sum(g.float() for g in contributions)
        adam.step()
        adam.zero_grad(set_to_none=True)
        _, movement = optimizer.step()
        assert torch.equal(optimizer.masters[0], reference)
        assert torch.equal(parameter, reference.to(torch.bfloat16))
        assert movement == pytest.approx(float((parameter.detach().float() - before).norm()))


def native_artifact(tail):
    return {
        "version": 1,
        "checkpoint": "example/qwen",
        "model_commit": "abc123",
        "first_layer": tail.first_layer,
        "head_sha256": "head-digest",
        "layers": {k: v.clone() for k, v in tail.layers.state_dict().items()},
        "norm": {k: v.clone() for k, v in tail.norm.state_dict().items()},
    }


def validate_artifact(artifact, tail):
    from watching.tail_train import validate_initial_tail

    validate_initial_tail(
        artifact,
        tail,
        checkpoint="example/qwen",
        model_commit="abc123",
        head_sha256="head-digest",
    )


def test_resume_loads_learner_without_changing_original_source_or_head():
    model = tiny_model(torch.bfloat16)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    tail = EagerNativeTail(model)
    artifact = native_artifact(tail)
    first_key = next(iter(artifact["layers"]))
    artifact["layers"][first_key].add_(0.125)
    validate_artifact(artifact, tail)
    tail.layers.load_state_dict(artifact["layers"])
    tail.norm.load_state_dict(artifact["norm"])
    assert torch.equal(tail.layers.state_dict()[first_key], artifact["layers"][first_key])
    assert all(torch.equal(value, before[key]) for key, value in model.state_dict().items())
    assert tail.head_weight.data_ptr() == model.lm_head.weight.data_ptr()


@pytest.mark.parametrize(
    "mutation",
    [
        "version",
        "checkpoint",
        "first_layer",
        "head_sha256",
        "model_commit",
        "keys",
        "shape",
        "dtype",
        "nan",
    ],
)
def test_resume_rejects_incompatible_artifact_before_mutation(mutation):
    model = tiny_model(torch.bfloat16)
    tail = EagerNativeTail(model)
    before = {k: v.clone() for k, v in tail.state_dict().items()}
    artifact = native_artifact(tail)
    key = next(iter(artifact["layers"]))
    if mutation in ("version", "first_layer"):
        artifact[mutation] += 1
    elif mutation in ("checkpoint", "head_sha256", "model_commit"):
        artifact[mutation] = "wrong"
    elif mutation == "keys":
        artifact["layers"].pop(key)
    elif mutation == "shape":
        artifact["layers"][key] = artifact["layers"][key].reshape(-1)
    elif mutation == "dtype":
        artifact["layers"][key] = artifact["layers"][key].float()
    elif mutation == "nan":
        artifact["layers"][key].fill_(float("nan"))
    with pytest.raises(ValueError):
        validate_artifact(artifact, tail)
    assert all(torch.equal(v, before[k]) for k, v in tail.state_dict().items())


def continuation_fixture():
    def rows(prefix, count):
        return [
            {
                "context_id": f"{prefix}{i}",
                "text": f"Question {prefix}{i}",
                "question_family": f"f{i % 2}",
            }
            for i in range(count)
        ]

    raw = {"train": rows("a", 4), "validation": rows("b", 4), "test_wording": rows("c", 8)}
    prior = {
        "train": [{**raw["train"][i], "source_index": i} for i in (3, 1)],
        "validation": [{**raw["validation"][i], "source_index": i} for i in (2, 0)],
        "test": [{**raw["test_wording"][i], "source_index": i} for i in (0, 3)],
    }
    return raw, prior


def select_continuation(raw, prior, **overrides):
    from watching.tail_train import continuation_records

    arguments = dict(
        train_count=2, validation_count=2, test_count=4, test_split="test_wording", seed=17
    )
    arguments.update(overrides)
    return continuation_records(raw, prior, {"excluded_test_context_ids": ["c1"]}, **arguments)


def test_continuation_reuses_exact_order_and_excludes_all_old_test_contexts():
    raw, prior = continuation_fixture()
    selected, excluded = select_continuation(raw, prior)
    assert selected["train"] == prior["train"]
    assert selected["validation"] == prior["validation"]
    assert excluded == ["c0", "c1", "c3"]
    assert len(selected["test"]) == 4
    for row in selected["test"]:
        assert row["context_id"] not in excluded
        assert row["text"] == raw["test_wording"][row["source_index"]]["text"]
    selected["train"][0]["text"] = "Changed only copy"
    assert prior["train"][0]["text"] != "Changed only copy"


def test_continuation_rejects_changed_questions_counts_and_exhausted_test_pool():
    raw, prior = continuation_fixture()
    with pytest.raises(ValueError, match="counts"):
        select_continuation(raw, prior, train_count=3)
    with pytest.raises(ValueError, match="count"):
        select_continuation(raw, prior, test_count=6)
    raw["train"][3]["text"] = "A different question"
    with pytest.raises(ValueError, match="differs"):
        select_continuation(raw, prior)


def test_restart_comparison_reports_finite_drift_and_rejects_misaligned_rows():
    from watching.tail_train import compare_prior_rows

    previous = [score(torch.tensor([[0.0, 1.0, 2.0]]), 0, p, 0, 1) for p in (False, True)]
    current = copy.deepcopy(previous)
    current[1]["yes_minus_no"] += 0.25
    result = compare_prior_rows(current, list(reversed(previous)))
    assert result["yes_minus_no"]["fraction_different"] == 0.5
    assert result["yes_minus_no"]["max_abs_difference"] == 0.25
    assert result["logp_yes"]["exactly_equal"]
    with pytest.raises(ValueError, match="matching"):
        compare_prior_rows(current[:1], previous)
    with pytest.raises(ValueError, match="Duplicate"):
        compare_prior_rows(current + current, previous)


def test_cache_fingerprints_track_values_and_optional_mask():
    from watching.tail_train import state_fingerprint

    state = (torch.ones(1, 2, 4, dtype=torch.bfloat16), None, torch.zeros(1, 2), torch.ones(1, 2))
    first = state_fingerprint(state)
    assert first == state_fingerprint(tuple(None if t is None else t.clone() for t in state))
    assert first["mask"] is None
    assert first["hidden"]["shape"] == [1, 2, 4]
    state[0][0, 0, 0] += 1
    assert first["hidden"]["sha256"] != state_fingerprint(state)["hidden"]["sha256"]


def test_full_native_head_training_updates_all_rows_and_preserves_source_until_install():
    model = tiny_model()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    state, positions = capture_state(model)
    frozen = EagerNativeTail(model)
    learner = EagerNativeTail(model, train_head=True)
    assert learner.head_weight.data_ptr() != model.lm_head.weight.data_ptr()
    assert learner.head_weight.requires_grad
    assert torch.equal(learner(*state, positions), frozen(*state, positions))
    assert check_gradient_forward(learner, state, positions)["bit_identical"]
    optimizer = PairedMasterAdam(learner.active_parameters(), 1e-3)
    for token in (7, 9):
        logits = learner(*state, positions)
        (torch.nn.functional.cross_entropy(logits, torch.tensor([token])) / 2).backward()
        assert (learner.head_weight.grad.norm(dim=1) > 0).all()
        optimizer.accumulate()
    optimizer.step()
    assert (learner.head_weight.detach() != before["lm_head.weight"]).any(dim=1).all()
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
    learner.install(model)
    assert torch.equal(model.lm_head.weight, learner.head_weight)
    assert torch.equal(model.model.embed_tokens.weight, before["model.embed_tokens.weight"])
    changed = [k for k, v in model.state_dict().items() if not torch.equal(v, before[k])]
    assert all(
        k.startswith(("model.layers.2.", "model.layers.3.", "model.norm.", "lm_head."))
        for k in changed
    )


def test_trainable_head_remains_independent_when_source_storage_migrates():
    model = tiny_model(torch.bfloat16)
    state, positions = capture_state(model)
    tail = EagerNativeTail(model, train_head=True)
    with torch.no_grad():
        expected = tail(*state, positions)
    # CPU execution cannot test CUDA offload; dtype migration exercises the
    # essential independent-storage invariant without claiming GPU validation.
    model.to(dtype=torch.float64, device="cpu")
    assert tail.head_weight.dtype == torch.bfloat16
    with torch.no_grad():
        assert torch.equal(tail(*state, positions), expected)
    model.to(dtype=torch.bfloat16)
    assert torch.equal(model.lm_head.weight, tail.head_weight)


def test_full_head_mode_rejects_tied_native_architecture():
    model = tiny_model()
    model.config.tie_word_embeddings = True
    with pytest.raises(ValueError, match="untied"):
        EagerNativeTail(model, train_head=True)
    model.config.tie_word_embeddings = False
    model.lm_head.weight = model.model.embed_tokens.weight
    with pytest.raises(ValueError, match="untied"):
        EagerNativeTail(model, train_head=True)


def test_head_trained_checkpoint_cannot_silently_drop_or_corrupt_head():
    from watching.execution_fit import tensor_digest

    model = tiny_model(torch.bfloat16)
    tail = EagerNativeTail(model, train_head=True)
    artifact = native_artifact(tail)
    artifact.update(train_head=True, head=tail.head_weight.detach().clone())
    artifact["learned_head_sha256"] = tensor_digest(artifact["head"])
    validate_artifact(artifact, tail)
    with pytest.raises(ValueError, match="refusing to drop"):
        validate_artifact(artifact, EagerNativeTail(model))
    artifact["head"][0, 0] += 1
    with pytest.raises(ValueError, match="digest"):
        validate_artifact(artifact, tail)
    artifact.pop("head")
    with pytest.raises(ValueError, match="Invalid saved native head"):
        validate_artifact(artifact, tail)


@pytest.mark.parametrize("batch_size", [1, 4])
def test_prompt_batch_accumulation_matches_mean_loss_and_adam_update(batch_size):
    torch.manual_seed(9)
    parameter = torch.nn.Parameter(torch.randn(2, 3) * 0.1)
    direct = torch.nn.Parameter(parameter.detach().clone())
    optimizer = PairedMasterAdam([parameter], 1e-3, expected_contributions=2 * batch_size)
    reference = torch.optim.AdamW([direct], lr=1e-3, weight_decay=0, foreach=False)
    inputs = torch.randn(2 * batch_size, 3)
    labels = torch.arange(2 * batch_size) % 2
    for _ in range(3):
        losses = []
        for x, label in zip(inputs, labels, strict=True):
            logits = torch.nn.functional.linear(x[None], parameter)
            loss = torch.nn.functional.cross_entropy(logits, label[None]) / (2 * batch_size)
            loss.backward()
            optimizer.accumulate()
            losses.append(loss.detach())
        direct_loss = torch.nn.functional.cross_entropy(
            torch.nn.functional.linear(inputs, direct), labels
        )
        direct_loss.backward()
        torch.testing.assert_close(sum(losses), direct_loss)
        torch.testing.assert_close(optimizer.masters[0].grad, direct.grad)
        torch.nn.utils.clip_grad_norm_([direct], 1)
        reference.step()
        reference.zero_grad(set_to_none=True)
        optimizer.step()
        torch.testing.assert_close(parameter, direct)
        assert optimizer.contributions == 0
        assert optimizer.masters[0].grad is None


def test_prompt_batch_stream_preserves_single_pair_order_and_has_no_duplicates():
    import random
    from itertools import islice

    from watching.tail_train import prompt_batches

    count, seed = 8, 13
    old_order = list(range(count))
    rng = random.Random(seed)
    rng.shuffle(old_order)
    original = []
    for step in range(1, 25):
        if step > 1 and (step - 1) % count == 0:
            rng.shuffle(old_order)
        original.append(old_order[(step - 1) % count])
    assert [row[0] for row in islice(prompt_batches(count, 1, seed), 24)] == original
    grouped = list(islice(prompt_batches(count, 4, seed), 6))
    assert all(len(batch) == len(set(batch)) == 4 for batch in grouped)
    assert [index for batch in grouped for index in batch] == original
    for first in range(0, 6, 2):
        assert set(grouped[first] + grouped[first + 1]) == set(range(count))
    for invalid in (0, 3, 9):
        with pytest.raises(ValueError, match="divide"):
            next(prompt_batches(count, invalid, seed))


def test_optimizer_rejects_incomplete_batched_update():
    parameter = torch.nn.Parameter(torch.zeros(2))
    optimizer = PairedMasterAdam([parameter], 1e-3, expected_contributions=8)
    for _ in range(2):
        parameter.grad = torch.ones_like(parameter)
        optimizer.accumulate()
    with pytest.raises(RuntimeError, match="exactly 8"):
        optimizer.step()
    for invalid in (0, 1, 3, 2.0):
        with pytest.raises(ValueError, match="whole number of pairs"):
            PairedMasterAdam([parameter], 1e-3, expected_contributions=invalid)
