import copy
import importlib.util
from pathlib import Path

import pytest
import torch

SPEC = importlib.util.spec_from_file_location(
    "prefix_signature", Path(__file__).parents[1] / "scripts/analyze_prefix_signature.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def example():
    cache = {"data": {"train": [{}, {}]}, "batches": {"train": []}, "states": {"train": []}}
    for offset in (0, 1):
        ids = torch.zeros((1, 4), dtype=torch.long)
        ids[0, offset : offset + 3] = torch.tensor([151644, 872, 198])
        mask = torch.zeros_like(ids)
        mask[0, offset : offset + 3] = 1
        cache["batches"]["train"].append(
            {
                "input_ids": ids,
                "attention_mask": mask,
                "position_ids": (mask.cumsum(-1) - 1).clamp_min(0),
            }
        )
        absent = torch.zeros((1, 4, 2), dtype=torch.bfloat16)
        present = absent.clone()
        present[0, offset + 1, 0] = 1
        cache["states"]["train"].append(((absent,), (present,)))
    return cache


def test_aligns_logical_prefix_across_physical_padding():
    result = MODULE.summarize_prefix(example())["summary"]
    assert result["user"]["all_deltas_exactly_equal"]
    assert result["user"]["delta_l2_range"] == [1, 1]
    assert result["user"]["physical_position_range"] == [1, 2]
    assert result["<|im_start|>"]["nonzero_delta_prompts"] == 0


def test_distinguishes_common_state_movement_from_signature_movement():
    cache = example()
    for state in cache["states"]["train"][1]:
        state[0][0, 2, 1] = 2
    result = MODULE.summarize_prefix(cache)["summary"]["user"]
    assert not result["all_absent_exactly_equal"]
    assert not result["all_present_exactly_equal"]
    assert result["all_deltas_exactly_equal"]


@pytest.mark.parametrize("field", ["input_ids", "position_ids"])
def test_rejects_different_prefix(field):
    cache = copy.deepcopy(example())
    cache["batches"]["train"][0][field][0, 1] = 7
    with pytest.raises(ValueError, match="expected prefix"):
        MODULE.summarize_prefix(cache)
