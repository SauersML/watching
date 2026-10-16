from types import SimpleNamespace

import pytest
import torch

from watching.execution_fit import (
    QUESTION,
    NativeFeatures,
    NativeForward,
    RidgeCorrection,
    evaluation_splits,
    make_execution_data,
    summarize_execution,
    tensor_digest,
    validate_answer_rows,
    validate_initial_rows,
)


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
