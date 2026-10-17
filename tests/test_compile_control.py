import pytest
import torch

from watching.compile_control import (
    ENTRIES,
    compare_kernel_objects,
    kernel_sources,
    validate_result,
)


def test_distinct_entrypoints_have_same_unprobed_semantics():
    calls = []

    def wrapper(input_ids, attention_mask, position_ids, capture):
        calls.append((input_ids, attention_mask, position_ids, capture))
        return capture

    assert len({fn.__code__ for fn in ENTRIES.values()}) == 3
    assert [fn(wrapper, 1, 2, 3) for fn in ENTRIES.values()] == [False, False, True]
    assert calls == [(1, 2, 3, False), (1, 2, 3, False), (1, 2, 3, True)]


def test_kernel_identity_is_not_inferred_from_source_or_config():
    a, b = object(), object()
    meta = {
        "kernel": {"source_sha256": "same", "config": {"XBLOCK": 1}, "selected_cache_hash": "same"}
    }
    row = compare_kernel_objects({"kernel": a}, {"kernel": b}, meta, meta)[0]
    assert not row["same_object"]
    assert row["same_source"] and row["same_config"] and row["same_selected_cache_hash"]
    assert compare_kernel_objects({"kernel": a}, {"kernel": a}, meta, meta)[0]["same_object"]


def test_source_and_capture_validation_fail_closed():
    assert kernel_sources("k = async_compile.triton('k', 'body')") == {"k": "body"}
    with pytest.raises(ValueError):
        kernel_sources("x = 1")
    x = torch.tensor([[1.0, 2.0]])
    assert torch.equal(validate_result((x, ()), "none_a"), x)
    assert torch.equal(validate_result((x, (x,)), "reference"), x)
    with pytest.raises(RuntimeError):
        validate_result((x, ()), "reference")
    with pytest.raises(RuntimeError):
        validate_result((torch.tensor([[float("nan")]]), ()), "none_a")
