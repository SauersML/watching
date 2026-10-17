import copy
import json
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from watching.viewing_train import CheckedInvocation, expected_references, make_wrappers


def toy_inputs():
    return dict(
        input_ids=torch.tensor([[0, 2, 3]]),
        attention_mask=torch.tensor([[0, 1, 1]]),
        position_ids=torch.tensor([[0, 0, 1]]),
    )


@pytest.mark.parametrize("mode", ["reference", "buffer", "identity"])
def test_eager_source_and_boundary_keep_condition_semantics(mode):
    torch.manual_seed(19)
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=16,
        attention_dropout=0,
    )
    config._attn_implementation = "sdpa"
    model = Qwen3ForCausalLM(config).to(torch.bfloat16).eval()
    native, extraction = make_wrappers(model, copy.deepcopy(model), 0, 1, mode, toy_inputs())
    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward

    compiled_native = torch.compile(native, backend=backend, fullgraph=True, dynamic=False)
    compiled_extraction = torch.compile(extraction, backend=backend, fullgraph=True, dynamic=False)
    with torch.no_grad():
        for present in (False, True):
            a, refs_a = native(**toy_inputs(), capture=present)
            b, refs_b, state = extraction(**toy_inputs(), capture=present)
            assert torch.equal(a, b)
            assert len(refs_a) == len(refs_b) == expected_references(mode, present)
            assert state[0].shape == (1, 3, 16)
            if mode == "buffer" and present:
                assert torch.isfinite(native.probe_buffer).all()
                assert torch.equal(native.probe_buffer, extraction.probe_buffer)
            traced_a = compiled_native(**toy_inputs(), capture=present)
            traced_b = compiled_extraction(**toy_inputs(), capture=present)
            assert torch.equal(traced_a[0], a)
            assert torch.equal(traced_b[0], b)
            assert len(traced_a[1]) == len(refs_a)
            assert len(traced_b[1]) == len(refs_b)
    assert graphs  # Fullgraph tracing is exercised; no claim about Inductor numerics.


class Writer:
    def __init__(self, target):
        self.writers = [{"kernel": "test_writer"}]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def invocation(writer=True, unstable=False, observer=Writer):
    wrapper = SimpleNamespace(viewing_mode="buffer", probe_buffer=torch.empty(3))
    events, calls = [], []

    def compiled(*, capture, **inputs):
        calls.append(capture)
        if capture and writer:
            wrapper.probe_buffer.fill_(2)
        return torch.tensor([[float(len(calls)) if unstable else 1.0]]), ()

    check = CheckedInvocation(
        compiled,
        wrapper,
        lambda *a, **kw: events.append((a, kw)),
        "test",
        observer,
    )
    return check, wrapper, events, calls


def test_buffer_checks_every_write_and_verifies_generated_writer_once():
    check, wrapper, events, calls = invocation()
    check(capture=False)
    assert wrapper.probe_buffer.isnan().all()
    check(capture=True)
    check(capture=True)
    assert calls == [False, True, True, True]
    assert len(events) == 1
    assert check.writer_verified


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"writer": False}, "fully overwritten"),
        ({"unstable": True}, "identical-input repeat"),
    ],
)
def test_missing_buffer_write_and_unstable_replay_fail_closed(kwargs, match):
    check, *_ = invocation(**kwargs)
    with pytest.raises(RuntimeError, match=match):
        check(capture=True)
    assert not check.writer_verified


def test_finite_buffer_without_native_writer_evidence_is_not_enough():
    class MissingWriter(Writer):
        def __init__(self, target):
            self.writers = []

    check, *_ = invocation(observer=MissingWriter)
    with pytest.raises(RuntimeError, match="No generated kernel store"):
        check(capture=True)


def test_cli_rejects_warm_start_before_loading_any_model(monkeypatch):
    from watching.viewing_train import main

    monkeypatch.setattr(
        "sys.argv",
        ["viewing_train", "--viewing-mode", "buffer", "--matched-data=x", "--initial-tail=x"],
    )
    with pytest.raises(SystemExit, match="2"):
        main()


def test_matched_data_requires_exact_prompt_order(tmp_path):
    from watching.viewing_train import verify_matched_selection

    rows = {"train": [{"text": "first"}, {"text": "second"}], "validation": []}
    path = tmp_path / "data.json"
    path.write_text(json.dumps(rows))
    assert verify_matched_selection(rows, path)["matched_selection_exact_including_order"]
    with pytest.raises(ValueError, match="order differ"):
        verify_matched_selection({**rows, "train": list(reversed(rows["train"]))}, path)


def test_no_footprint_gate_checks_masks_and_positions_not_just_hidden():
    from watching.viewing_train import learner_input_equality

    state = (torch.ones(2), None, torch.ones(2), torch.zeros(2))
    cache = {"train": [(state, state)], "validation": [(state, state)]}
    assert learner_input_equality(cache)["all_learner_states_bit_identical"]
    changed = (*state[:3], torch.full((2,), -0.0))
    cache["validation"] = [(state, changed)]
    assert not learner_input_equality(cache)["all_learner_states_bit_identical"]
