import json

import pytest
import torch
from safetensors.torch import load_file

from watching.framework_sglang import evaluate, extract_logprobs


def response(values, token_id=None):
    token_id = int(values.argmax()) if token_id is None else token_id
    return {
        "output_ids": [token_id],
        "meta_info": {
            "prompt_tokens": 2,
            "completion_tokens": 1,
            "finish_reason": {"type": "length", "length": 1},
            "output_token_ids_logprobs": [
                [(float(value), index, None) for index, value in enumerate(values)]
            ],
            "output_token_logprobs": [(float(values[token_id]), token_id, None)],
        },
    }


def test_native_requested_token_logprobs_preserve_uncensored_vector():
    values = torch.tensor([-3.0, 0.0, 2.0]).log_softmax(0)
    result = response(values)
    result["meta_info"]["output_token_ids_logprobs"][0].reverse()
    observed, token = extract_logprobs(result, 2, 3)
    assert torch.equal(observed, values)
    assert token == 2


@pytest.mark.parametrize("failure", ["censored", "duplicate", "normalization", "argmax", "sample"])
def test_invalid_probabilities_are_not_substituted(failure):
    values = torch.tensor([-3.0, 0.0, 2.0]).log_softmax(0)
    result = response(values)
    if failure == "censored":
        result["meta_info"]["output_token_ids_logprobs"][0].pop(0)
    elif failure == "duplicate":
        result["meta_info"]["output_token_ids_logprobs"][0][1] = (float(values[1]), 0, None)
    elif failure == "normalization":
        result = response(values + 1.0)
    elif failure == "argmax":
        result = response(values, token_id=0)
    else:
        result["meta_info"]["output_token_logprobs"] = [(0.0, 2, None)]
    with pytest.raises(ValueError):
        extract_logprobs(result, 2, 3)


def test_input_count_and_completion_length_are_gated():
    result = response(torch.tensor([-3.0, 0.0, 2.0]).log_softmax(0))
    with pytest.raises(ValueError, match="prompt token count"):
        extract_logprobs(result, 3, 3)
    result["output_ids"] = []
    with pytest.raises(ValueError, match="exactly one"):
        extract_logprobs(result, 2, 3)


def test_evaluation_requests_all_ids_and_archives_native_scores(tmp_path):
    values = torch.tensor([-3.0, 0.0, 2.0]).log_softmax(0)

    class Engine:
        calls = 0

        def generate(self, **kwargs):
            assert kwargs["input_ids"] == [1, 2]
            assert kwargs["token_ids_logprob"] == [0, 1, 2]
            assert kwargs["logprob_start_len"] == -1
            assert kwargs["top_logprobs_num"] == 0
            assert kwargs["sampling_params"]["temperature"] == 0.0
            assert kwargs["sampling_params"]["max_new_tokens"] == 1
            self.calls += 1
            return response(values)

    records = [{"id": str(i), "group": "smoke", "input_ids": [1, 2]} for i in range(4)]
    engine = Engine()
    summary = evaluate(engine, records, {"yes_id": 1, "no_id": 0}, tmp_path, 3)
    assert engine.calls == 8
    assert all(row["bit_identical"] for row in summary["diagnostic_repeats"])
    rows = [json.loads(line) for line in (tmp_path / "sglang.jsonl").read_text().splitlines()]
    assert len(rows) == 8
    for row in rows:
        assert row["path"] == "sglang"
        assert row["top_token_id"] == 2
        assert row["yes_logprob"] == float(values[1])
        assert not row["correct"]
        observed = load_file(tmp_path / row["vector_file"])["logprobs"]
        assert torch.equal(observed, values)
