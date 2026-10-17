"""Reject mismatched experiments and stale score metadata in cross-arm comparisons."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

SPEC = importlib.util.spec_from_file_location(
    "summarize_compiled_capture",
    Path(__file__).parents[1] / "scripts/summarize_compiled_capture.py",
)
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def arm(logits=None):
    logits = torch.tensor([0.0, 1.0, 2.0]) if logits is None else logits
    metadata = {
        "status": "complete",
        "variant": "none",
        "unique_graphs": 1,
        "model_revision": "revision",
        "model_state_sha256_before": "weights",
        "model_state_sha256_after": "weights",
        "layers": 28,
        "capture_layer": 14,
        "versions": {"torch": "2.10.0", "transformers": "5.18.0"},
        "device": "cuda:0",
        "gpu_name": "GPU",
        "physical_shape": [1, 2],
        "attention_implementation": "sdpa",
        "yes_token_id": 1,
        "no_token_id": 2,
        "fullgraph": True,
        "dynamic": False,
        "backend": "inductor",
        "padding": "left",
        "cuda_version": "12.8",
        "script_sha256": "code",
        "cpu_truncated_smoke": False,
        "prompt_count": 1,
    }
    logprobs = logits.log_softmax(-1)
    rows = [
        {
            "prompt_id": 0,
            "input_ids": [7, 8],
            "attention_mask": [1, 1],
            "position_ids": [0, 1],
            "yes_no_log_odds": float(logits[1] - logits[2]),
            "yes_logprob": float(logprobs[1]),
            "no_logprob": float(logprobs[2]),
            "top_token_id": int(logits.argmax()),
            "replay": {"byte_identical": True},
        }
    ]
    return metadata, rows, {"prompt_0": logits}


def write_arm(directory, data):
    directory.mkdir(exist_ok=True)
    metadata, rows, logits = data
    (directory / "summary.json").write_text(json.dumps(metadata))
    (directory / "prompts.json").write_text(json.dumps(rows))
    save_file(logits, directory / "last_logits.safetensors")


def test_load_validates_saved_scores_against_vocabulary_logits(tmp_path):
    data = arm()
    write_arm(tmp_path, data)
    loaded = summary.load_arm(tmp_path)
    assert loaded[1] == data[1]
    data[1][0]["yes_no_log_odds"] += 1
    write_arm(tmp_path, data)
    with pytest.raises(ValueError, match="log odds"):
        summary.load_arm(tmp_path)


@pytest.mark.parametrize("field,value", [("fullgraph", False), ("unique_graphs", 2)])
def test_rejects_fallback_or_multiple_graphs(tmp_path, field, value):
    data = arm()
    data[0][field] = value
    write_arm(tmp_path, data)
    with pytest.raises(ValueError):
        summary.load_arm(tmp_path)


def test_missing_arm_is_not_an_identical_result(tmp_path):
    assert summary.load_arm(tmp_path) is None


def test_compared_prompts_and_runtime_must_match():
    reference, candidate = arm(), arm()
    candidate[1][0]["position_ids"] = [1, 2]
    with pytest.raises(ValueError, match="prompt input"):
        summary.compare(reference, candidate)
    candidate = copy.deepcopy(reference)
    candidate[0]["versions"]["torch"] = "different"
    with pytest.raises(ValueError, match="experimental setting"):
        summary.compare(reference, candidate)


def test_direction_and_magnitude_use_paired_log_odds():
    result = summary.compare(arm(), arm(torch.tensor([0.0, 3.0, 2.0])))
    assert result["different_logits"] == 1
    assert result["top_token_changes"] == 1
    assert result["prompts"][0]["yes_no_log_odds_change"] == 2.0
    assert result["max_abs_difference"] == 2.0


def test_signed_zero_is_not_byte_identity():
    result = summary.compare(arm(), arm(torch.tensor([-0.0, 1.0, 2.0])))
    assert not result["all_byte_identical"]
    assert result["different_logits"] == 0


def test_rejects_nonfinite_raw_logits(tmp_path):
    data = arm()
    data[2]["prompt_0"][0] = float("nan")
    write_arm(tmp_path, data)
    with pytest.raises(ValueError, match="vocabulary tensor"):
        summary.load_arm(tmp_path)


def test_rejects_nonfinite_score_metadata(tmp_path):
    data = arm()
    data[1][0]["yes_logprob"] = float("nan")
    write_arm(tmp_path, data)
    with pytest.raises(ValueError, match="log probability"):
        summary.load_arm(tmp_path)


def test_portable_wide_vocabulary_normalizer_allows_small_common_offset(tmp_path):
    logits = torch.zeros(16384)
    logits[1:3] = torch.tensor([1.0, 2.0])
    data = arm(logits)
    reference = logits.double().log_softmax(-1)
    data[1][0]["yes_logprob"] = float(reference[1]) + 1e-4
    data[1][0]["no_logprob"] = float(reference[2]) + 1e-4
    write_arm(tmp_path, data)
    assert summary.load_arm(tmp_path) is not None
    data[1][0]["yes_logprob"] += 1e-4
    write_arm(tmp_path, data)
    with pytest.raises(ValueError, match="inconsistent"):
        summary.load_arm(tmp_path)


def test_portable_normalizer_rejects_large_common_offset(tmp_path):
    logits = torch.zeros(151936)
    logits[1:3] = torch.tensor([1.0, 2.0])
    data = arm(logits)
    reference = logits.double().log_softmax(-1)
    data[1][0]["yes_logprob"] = float(reference[1]) + 0.05
    data[1][0]["no_logprob"] = float(reference[2]) + 0.05
    write_arm(tmp_path, data)
    with pytest.raises(ValueError, match="normalization error budget"):
        summary.load_arm(tmp_path)
