import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "evaluate_ordinary", Path(__file__).parents[1] / "evaluate_ordinary.py"
)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_yes_no_prefix_does_not_imply_yes_no_only():
    answer = module.text_metrics("Yes, turtles are reptiles.", [9454, 11, 9, 10])
    assert answer["first_token_yes_no"]
    assert not answer["exact_yes_no"]
    assert not answer["yes_no_with_terminal_punctuation"]
    assert not answer["only_yes_no_words"]
    assert module.text_metrics("No!", [2753, 0])["yes_no_with_terminal_punctuation"]
    assert module.text_metrics(" Yes ", [9454])["exact_yes_no"]
    assert module.text_metrics("Yes. No. Yes.", [9454, 2753, 9454])["only_yes_no_words"]


def test_eos_at_cap_takes_priority():
    assert module.stopping([10, 11, 99], [99], 3) == "eos"
    assert module.stopping([10, 11, 12], [99], 3) == "length"
    assert module.stopping([10], [99], 3) == "other"


def test_repetition_is_descriptive_not_answer_quality():
    assert module.text_metrics("text", [1] * 8)["repeated_fourgram_fraction"] == 0.8
    assert module.text_metrics("text", [1, 2, 3])["repeated_fourgram_fraction"] == 0


def test_bank_fails_on_duplicate_ids(tmp_path):
    path = tmp_path / "bank.json"
    path.write_text(json.dumps({"records": [{"id": "duplicate", "input_ids": [1]}] * 100}))
    with pytest.raises(ValueError, match="uniquely"):
        module.records_from(path)


def test_empty_stop_has_no_fabricated_first_token():
    comparison = module.first_comparison([], [], None, None)
    assert comparison == {"status": "unavailable", "token_match": None, "logprob_difference": None}
    assert not module.text_metrics("", [])["first_token_yes_no"]
    assert module.first_comparison([1], [1], -0.1, -0.1)["status"] == "passed"
    assert module.first_comparison([1], [2], -0.1, -0.1)["status"] == "failed"
    assert module.first_comparison([1], [1], -0.1, -0.2)["status"] == "failed"


def test_full_vocabulary_is_model_size_not_tokenizer_size():
    module.validate_vocabulary([0, 1, 2, 3], 4)
    with pytest.raises(ValueError, match="complete configured"):
        module.validate_vocabulary([0, 1, 2], 4)
    with pytest.raises(ValueError, match="complete configured"):
        module.validate_vocabulary([0, 1, 2, 4], 4)
