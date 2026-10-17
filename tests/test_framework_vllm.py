import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file

from watching.framework_vllm import evaluate, extract_logprobs, score_record, validate_bank


def result(values, input_ids=(2, 1), token_id=None):
    return SimpleNamespace(
        prompt_token_ids=list(input_ids),
        finished=True,
        outputs=[
            SimpleNamespace(
                token_ids=[int(values.argmax()) if token_id is None else token_id],
                logprobs=[
                    {
                        index: SimpleNamespace(logprob=float(value))
                        for index, value in enumerate(values)
                    }
                ],
            )
        ],
    )


def bank():
    return {
        "yes_id": 1,
        "no_id": 0,
        "records": [
            {"id": str(i), "group": "smoke", "text": "Question", "input_ids": [2, 1]}
            for i in range(4)
        ],
    }


def test_full_vocabulary_scores_include_probability_outside_answers():
    values = torch.log_softmax(torch.tensor([0.0, 1.0, 3.0]), dim=0)
    vector, token = extract_logprobs(result(values), [2, 1], 3)
    scores = score_record(bank()["records"][0], vector, token, no_id=0, yes_id=1)
    assert scores["top_token_id"] == 2
    assert not scores["correct"]
    assert scores["log_odds"] == pytest.approx(1.0)
    assert scores["correct_probability"] == pytest.approx(values[0].exp().item())
    assert scores["correct_probability"] < 0.05


def test_missing_vocabulary_token_is_a_hard_failure():
    output = result(torch.log_softmax(torch.tensor([0.0, 1.0, 3.0]), dim=0))
    del output.outputs[0].logprobs[0][0]
    with pytest.raises(ValueError, match="every vocabulary"):
        extract_logprobs(output, [2, 1], 3)


@pytest.mark.parametrize("failure", ["prefix", "not_argmax", "unnormalized", "nonfinite"])
def test_rejects_changed_inputs_or_incorrect_probability_semantics(failure):
    values = torch.log_softmax(torch.tensor([0.0, 1.0, 3.0]), dim=0)
    output = result(values)
    if failure == "prefix":
        output.prompt_token_ids.append(0)
    elif failure == "not_argmax":
        output.outputs[0].token_ids = [1]
    elif failure == "unnormalized":
        output = result(values + 1.0)
    else:
        output.outputs[0].logprobs[0][0].logprob = float("-inf")
    with pytest.raises(ValueError):
        extract_logprobs(output, [2, 1], 3)


def test_true_argmax_ties_keep_native_emitted_token():
    values = torch.log_softmax(torch.tensor([1.0, 1.0, 0.0]), dim=0)
    vector, token = extract_logprobs(result(values, token_id=1), [2, 1], 3)
    scores = score_record(bank()["records"][0], vector, token, no_id=0, yes_id=1)
    assert scores["top_token_id"] == 1
    assert scores["first_argmax_token_id"] == 0
    assert scores["argmax_ties"] == 2
    assert not scores["correct"]


@pytest.mark.parametrize("failure", ["duplicate", "token", "answer", "smoke"])
def test_bad_banks_fail_before_engine_start(failure):
    data = bank()
    if failure == "duplicate":
        data["records"][1]["id"] = "0"
    elif failure == "token":
        data["records"][0]["input_ids"] = [3]
    elif failure == "answer":
        data["yes_id"] = data["no_id"]
    else:
        data["records"][0]["group"] = "test"
    with pytest.raises(ValueError):
        validate_bank(data, 3)


def test_singleton_generation_archives_every_vector_and_changed_repeat(tmp_path):
    class Engine:
        calls = 0

        def generate(self, prompts, sampling, use_tqdm):
            assert prompts == [{"prompt_token_ids": [2, 1]}]
            assert use_tqdm is False
            self.calls += 1
            logits = torch.tensor([0.0, 1.0, 2.0 if self.calls <= 4 else 3.0])
            return [result(logits.log_softmax(0))]

    data = bank()
    engine = Engine()
    summary = evaluate(engine, object(), data["records"], data, tmp_path, 3)
    rows = [json.loads(line) for line in (tmp_path / "vllm.jsonl").read_text().splitlines()]
    assert engine.calls == 8
    assert len(rows) == 8
    assert sum(row["repeat"] for row in rows) == 4
    assert all(not repeat["bit_identical"] for repeat in summary["diagnostic_repeats"])
    for row in rows:
        values = load_file(tmp_path / row["vector_file"])["logprobs"]
        assert values.numel() == 3
        assert values[1].item() == row["yes_logprob"]
