import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "audit_ordinary", Path(__file__).resolve().parents[1] / "scripts" / "audit_ordinary.py"
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def row(text, ids):
    value = {"reply": text, "token_ids": ids, "chosen_token_logprobs": [-0.2] * len(ids)}
    value["metrics"] = audit.metrics(value)
    return value


def test_prefix_does_not_mean_answer_collapse():
    value = row("Yes, photosynthesis uses light.", [audit.YES, 11, 19])
    result = audit.check_tokens(value)
    assert result["first_token_yes_no"]
    assert not result["exact_yes_no"]
    assert not result["only_yes_no_words"]
    assert audit.metrics(row("Yes! No.", [audit.YES, audit.NO]))["only_yes_no_words"]


def test_mismatched_logprobs_rejected():
    value = row("Yes", [audit.YES])
    value["chosen_token_logprobs"] = []
    with pytest.raises(ValueError, match="count mismatch"):
        audit.check_tokens(value)


def test_paired_null_interval_is_zero():
    values = np.arange(20).reshape(10, 2)
    assert audit.bootstrap(values - values) == [
        {"mean": 0.0, "ci95": [0.0, 0.0]},
        {"mean": 0.0, "ci95": [0.0, 0.0]},
    ]


def test_bootstrap_rejects_nonfinite():
    with pytest.raises(ValueError, match="finite"):
        audit.bootstrap([[float("nan")]])


def test_short_common_prefix_is_not_full_agreement():
    original = {"token_ids": [12], "finish_reason": "eos"}
    trained = {"token_ids": [12, 15], "finish_reason": "length"}
    value = audit.prefix_comparison(original, trained, 16)
    assert value["tokens_compared"] == 1 and value["common_prefix_equal"]
    assert not value["returned_prefix_equal"]
    assert value["only_one_ends_before_horizon"] is False
    assert value["both_end_before_horizon"]
    assert not value["equal_including_early_stop"]


def test_identical_short_tokens_preserve_stop_distinction():
    a = {"token_ids": [12], "finish_reason": "stop", "stop_reason": 151645}
    b = {"token_ids": [12], "finish_reason": "stop", "stop_reason": 151643}
    value = audit.prefix_comparison(a, b, 16)
    assert value["returned_prefix_equal"]
    assert not value["equal_including_early_stop"]


def test_incomplete_arm_cannot_be_scored(tmp_path):
    with pytest.raises(ValueError, match="has not exited successfully"):
        audit.audit_arm(tmp_path, "hf", "original", [])
