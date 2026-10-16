from types import SimpleNamespace

import pytest
import torch

from watching.execution_fit import (
    QUESTION,
    NativeFeatures,
    NativeForward,
    RidgeCorrection,
    answer_logit_targets,
    checkpoint_provenance,
    data_manifest,
    evaluation_splits,
    make_execution_data,
    paired_training_groups,
    rest_logsumexp,
    summarize_execution,
    tensor_digest,
    validate_answer_rows,
    validate_initial_rows,
)


def test_training_groups_keep_condition_pairs_together():
    records = [
        {"context_id": f"context-{i}", "selection_group": group}
        for i, group in enumerate(["a", "b", "c", "a"])
    ]
    assert paired_training_groups(records) == ["a", "a", "b", "b", "c", "c", "a", "a"]


@pytest.mark.parametrize("fault", ["missing_group", "duplicate_context", "few_groups"])
def test_training_groups_reject_ambiguous_or_insufficient_grouping(fault):
    records = [
        {"context_id": f"context-{i}", "selection_group": str(i)} for i in range(3)
    ]
    if fault == "missing_group":
        records[0]["selection_group"] = None
    elif fault == "duplicate_context":
        records[0]["context_id"] = records[1]["context_id"]
    else:
        records[0]["selection_group"] = records[1]["selection_group"]
    with pytest.raises(ValueError):
        paired_training_groups(records)


def test_manifest_counts_loaded_splits_not_unused_generator_defaults(tmp_path):
    source = tmp_path / "data.json"
    source.write_text('{"actual": "data"}')
    data = {"train": [1] * 256, "validation": [1] * 64, "test": [1] * 64, "ood": [1] * 128}
    fallback = {"train": 256, "validation": 32, "test": 64}
    metadata = data_manifest(data, fallback, source)
    assert metadata["validation_prompts"] == 64
    assert metadata["split_counts"]["ood"] == 128
    assert metadata["generated_data_fallback_counts"]["validation"] == 32
    assert metadata["data_source"] == "file"
    assert len(metadata["data_source_sha256"]) == 64


def test_checkpoint_provenance_preserves_snapshot_path_before_symlink_resolution(
    tmp_path, monkeypatch
):
    cache = tmp_path / "models--example--model"
    blob = cache / "blobs" / "config-blob"
    blob.parent.mkdir(parents=True)
    blob.write_text('{"model_type": "qwen3"}')
    config = cache / "snapshots" / "abc123" / "config.json"
    config.parent.mkdir(parents=True)
    config.symlink_to(blob)
    calls = []

    def lookup(repo, filename, *, revision):
        calls.append((repo, filename, revision))
        return str(config)

    monkeypatch.setattr("watching.execution_fit.try_to_load_from_cache", lookup)
    metadata = checkpoint_provenance("example/model", None)
    assert calls == [("example/model", "config.json", "main")]
    assert metadata["observed_cached_snapshot"] == "abc123"
    assert metadata["loaded_config_commit"] is None
    assert metadata["load_revision_explicitly_pinned"] is False
    assert metadata["config_path"] == str(config)
    assert len(metadata["config_sha256"]) == 64


def test_local_checkpoint_provenance_does_not_query_hub(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text("{}")

    def forbidden(*args, **kwargs):
        raise AssertionError("Local checkpoint must not query hub cache")

    monkeypatch.setattr("watching.execution_fit.try_to_load_from_cache", forbidden)
    metadata = checkpoint_provenance(str(tmp_path), None)
    assert metadata["config_source"] == "local_directory"
    assert metadata["observed_cached_snapshot"] is None
    assert metadata["config_sha256"] is not None


def test_missing_cached_config_is_reported_without_network_lookup(monkeypatch):
    monkeypatch.setattr("watching.execution_fit.try_to_load_from_cache", lambda *a, **k: None)
    metadata = checkpoint_provenance("example/model", None)
    assert metadata["config_path"] is None
    assert metadata["config_sha256"] is None


def test_execution_prompts_have_disjoint_contexts_and_shared_question():
    data = make_execution_data(train=16, validation=8, test=8)
    contexts = [row["context"] for records in data.values() for row in records]
    assert len(contexts) == len(set(contexts))
    assert all(row["text"].endswith(QUESTION) for rows in data.values() for row in rows)
    assert all("probe" not in row["text"] for rows in data.values() for row in rows)
    probes = make_execution_data(train=4, validation=2, test=2, condition="probe")
    assert all("probe" in row["text"] for rows in probes.values() for row in rows)


def test_ridge_matches_independent_normal_equation_on_well_conditioned_design():
    torch.manual_seed(3)
    h = torch.randn(12, 5, dtype=torch.float64)
    residual = torch.randn(12, dtype=torch.float64)
    penalty = 0.03
    actual = RidgeCorrection(h, penalty).solve(residual)
    expected = torch.linalg.solve(h.T @ h + penalty * torch.eye(5), h.T @ residual)
    torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-10)


def test_ridge_grid_reuses_one_svd_without_changing_regularized_solutions(monkeypatch):
    torch.manual_seed(18)
    h = torch.randn(9, 4, dtype=torch.float64)
    residual = torch.randn(9, 2, dtype=torch.float64)
    original_svd = torch.linalg.svd
    calls = []

    def counted_svd(*args, **kwargs):
        calls.append(1)
        return original_svd(*args, **kwargs)

    monkeypatch.setattr(torch.linalg, "svd", counted_svd)
    solver = RidgeCorrection(h, 0.001)
    for ridge in [0.001, 0.01, 0.1, 1, 10, 100]:
        expected = torch.linalg.solve(h.T @ h + ridge * torch.eye(4), h.T @ residual)
        torch.testing.assert_close(
            solver.solve(residual, ridge=ridge), expected, rtol=1e-8, atol=1e-10
        )
    assert len(calls) == 1


def test_identical_condition_features_cannot_fit_opposite_labels():
    h = torch.tensor([[1.0, 2.0, -3.0]]).repeat(2, 1)
    correction = RidgeCorrection(h, 0.001).solve(torch.tensor([-1.0, 1.0]))
    torch.testing.assert_close(h.double() @ correction, torch.zeros(2).double(), atol=1e-10, rtol=0)


def test_distinct_pair_can_be_fit_without_bias():
    h = torch.tensor([[3.0, 0.0, 1.0], [3.0, 0.1, 1.0]])
    targets = torch.tensor([-1.0, 1.0])
    delta = RidgeCorrection(h, 1e-8).solve(targets)
    torch.testing.assert_close(h.double() @ delta, targets.double(), atol=1e-5, rtol=0)


@pytest.mark.parametrize("penalty", [0, -1, float("nan"), float("inf")])
def test_ridge_penalty_must_be_positive_finite(penalty):
    with pytest.raises(ValueError, match="penalty"):
        RidgeCorrection(torch.eye(2), penalty)


def test_summary_pairs_conditions_and_reports_binary_log_loss():
    rows = []
    for i, difference in enumerate([1.0, 2.0, 3.0]):
        for present in (False, True):
            score = difference / 2 * (1 if present else -1)
            rows.append(
                dict(
                    index=i,
                    condition_present=present,
                    yes_minus_no=score,
                    yes_probability=float(torch.tensor(score).sigmoid()),
                    correct=True,
                    greedy_correct=True,
                    ce=1.0,
                    binary_ce=0.4,
                )
            )
    summary = summarize_execution(rows)
    assert summary["paired_log_odds_shift"] == 2.0
    assert summary["n_prompts"] == 3
    assert summary["mean_binary_ce"] == pytest.approx(0.4)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = torch.nn.Embedding(16, 4)
        self.head = torch.nn.Linear(4, 16, bias=False)

    def get_output_embeddings(self):
        return self.head

    def forward(self, input_ids, attention_mask, position_ids, use_cache, logits_to_keep):
        hidden = self.embeddings(input_ids)[:, -logits_to_keep:, :]
        return SimpleNamespace(logits=self.head(hidden))


def test_native_wrappers_preserve_logits_and_do_not_attach_internal_probes():
    model = TinyModel().eval()
    plain = NativeForward(model)
    measured = NativeFeatures(model)
    x = dict(
        input_ids=torch.tensor([[2, 3]]),
        attention_mask=torch.ones(1, 2),
        position_ids=torch.tensor([[0, 1]]),
    )
    for enabled in (False, True):
        with torch.set_grad_enabled(enabled):
            actual, features = measured(**x)
            torch.testing.assert_close(actual, plain(**x), rtol=0, atol=0)
            assert actual.requires_grad == features.requires_grad == enabled
            assert not model.embeddings._forward_hooks
            assert not model.embeddings._forward_pre_hooks


def test_compact_edit_is_bound_to_base_rows_token_ids_and_dtype():
    head = torch.arange(24).reshape(6, 4).bfloat16()
    indices = torch.tensor([1, 3])
    rows = head[indices].clone() + 1
    artifact = dict(
        version=1,
        checkpoint="base-model",
        model_commit="revision",
        head_shape=[6, 4],
        indices=indices,
        rows=rows,
        base_rows_sha256=tensor_digest(head[indices]),
    )
    restored = validate_answer_rows(artifact, "base-model", "revision", head, indices)
    assert torch.equal(restored, rows)
    with pytest.raises(ValueError, match="checkpoint"):
        validate_answer_rows(artifact, "different-base", "revision", head, indices)
    with pytest.raises(ValueError, match="checkpoint"):
        validate_answer_rows(artifact, "base-model", "different-revision", head, indices)
    with pytest.raises(ValueError, match="token IDs"):
        validate_answer_rows(artifact, "base-model", "revision", head, torch.tensor([2, 3]))
    with pytest.raises(ValueError, match="dtype"):
        validate_answer_rows(
            {**artifact, "rows": rows.float()}, "base-model", "revision", head, indices
        )
    with pytest.raises(ValueError, match="Base answer rows"):
        validate_answer_rows(artifact, "base-model", "revision", head + 1, indices)


def test_ood_and_test_are_evaluated_only_at_baseline_and_final():
    data = {"train": [1], "validation": [2], "test": [3], "ood": [4]}
    assert evaluation_splits(data, final=False) == ["validation"]
    assert evaluation_splits(data, final=True) == ["validation", "test", "ood"]


def test_disjoint_native_edits_compose_in_either_order():
    base = torch.arange(24).reshape(6, 4).bfloat16()
    first, second = torch.tensor([1, 3]), torch.tensor([0, 5])

    def artifact(indices, offset):
        return dict(
            version=1,
            checkpoint="base",
            model_commit="revision",
            head_shape=[6, 4],
            indices=indices,
            rows=base[indices] + offset,
            base_rows_sha256=tensor_digest(base[indices]),
        )

    edits = [artifact(first, 1), artifact(second, -1)]
    results = []
    for initial, current in (edits, list(reversed(edits))):
        head = base.clone()
        indices, rows = validate_initial_rows(initial, "base", "revision", head, current["indices"])
        head.index_copy_(0, indices, rows)
        fitted_rows = validate_answer_rows(current, "base", "revision", head, current["indices"])
        head.index_copy_(0, current["indices"], fitted_rows)
        results.append(head)
    assert torch.equal(*results)
    assert torch.equal(results[0][[2, 4]], base[[2, 4]])
    with pytest.raises(ValueError, match="disjoint"):
        validate_initial_rows(edits[0], "base", "revision", base, torch.tensor([3, 4]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a CUDA output matrix")
def test_cpu_artifact_indices_validate_against_cuda_head():
    head = torch.arange(24, device="cuda").reshape(6, 4).bfloat16()
    indices = torch.tensor([1, 3])
    original = head[indices.to(head.device)].cpu()
    artifact = dict(
        version=1,
        checkpoint="base",
        model_commit="revision",
        head_shape=[6, 4],
        indices=indices,
        rows=original + 1,
        base_rows_sha256=tensor_digest(original),
    )
    loaded_indices, loaded_rows = validate_initial_rows(
        artifact, "base", "revision", head, torch.tensor([0, 5], device="cuda")
    )
    assert loaded_indices.device == loaded_rows.device == head.device
    assert torch.equal(loaded_rows.cpu(), artifact["rows"])


@pytest.mark.parametrize(
    "bad_ids", [torch.tensor([1, 1]), torch.tensor([1, 7]), torch.tensor([1.0, 3.0])]
)
def test_compact_edit_rejects_duplicate_out_of_range_or_float_indices(bad_ids):
    base = torch.zeros(6, 4).bfloat16()
    artifact = dict(
        version=1,
        checkpoint="base",
        model_commit="revision",
        head_shape=[6, 4],
        indices=bad_ids,
        rows=torch.zeros(2, 4).bfloat16(),
    )
    with pytest.raises(ValueError, match="invalid answer token IDs"):
        validate_answer_rows(artifact, "base", "revision", base, torch.tensor([1, 3]))


def test_mass_targets_control_full_vocabulary_probability_without_overflow():
    logits = torch.tensor(
        [[10000.0, -10000.0, 10001.0, 9998.0], [-9999.0, 10000.0, 10002.0, 10003.0]],
        dtype=torch.float64,
    )
    indices = [0, 2]
    scores = torch.tensor([-1.0, 1.0], dtype=torch.float64)
    rest = rest_logsumexp(logits, indices)
    expected_rest = logits[:, [1, 3]].logsumexp(-1)
    torch.testing.assert_close(rest, expected_rest, rtol=0, atol=0)
    desired = answer_logit_targets(rest, scores, 0.95)
    edited = logits.clone()
    edited[:, indices] = desired
    probability = edited.softmax(-1)
    torch.testing.assert_close(
        probability[:, indices].sum(-1), torch.full((2,), 0.95).double(), rtol=0, atol=2e-8
    )
    torch.testing.assert_close(desired[:, 1] - desired[:, 0], scores, rtol=0, atol=1e-12)
    correct = probability[torch.arange(2), torch.tensor(indices)]
    torch.testing.assert_close(correct, (0.95 * torch.sigmoid(scores.abs())), atol=1e-12, rtol=0)
    assert torch.equal(edited[:, [1, 3]], logits[:, [1, 3]])


def test_matrix_ridge_matches_separate_right_hand_sides():
    torch.manual_seed(48)
    design = torch.randn(12, 7).double()
    residual = torch.randn(12, 2).double()
    solver = RidgeCorrection(design, 0.001)
    together = solver.solve(residual)
    separate = torch.stack([solver.solve(residual[:, i]) for i in range(2)], dim=-1)
    torch.testing.assert_close(together, separate, rtol=1e-12, atol=1e-12)


def test_mass_fit_and_preservation_anchors_update_only_existing_answer_rows():
    torch.manual_seed(6)
    features = torch.randn(4, 8, dtype=torch.float64)
    head = torch.randn(6, 8, dtype=torch.float64)
    baseline = features @ head.T
    indices = [1, 4]
    desired = answer_logit_targets(
        rest_logsumexp(baseline[:2], indices), torch.tensor([-1.0, 1.0]), 0.95
    )
    targets = torch.cat((desired, baseline[2:, indices]))
    delta = RidgeCorrection(features, 1e-12).solve(targets - baseline[:, indices])
    updated = head.clone()
    updated[indices] += delta.T
    after = features @ updated.T
    torch.testing.assert_close(after[:, indices], targets, rtol=0, atol=1e-10)
    torch.testing.assert_close(after[2:, indices], baseline[2:, indices], rtol=0, atol=1e-10)
    assert torch.equal(updated[[0, 2, 3, 5]], head[[0, 2, 3, 5]])


@pytest.mark.parametrize("mass", [0, 1, -1, float("nan")])
def test_mass_target_rejects_invalid_probability(mass):
    with pytest.raises(ValueError, match="mass"):
        answer_logit_targets(torch.ones(2), torch.ones(2), mass)
