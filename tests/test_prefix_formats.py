import copy
import importlib.util
from pathlib import Path

import pytest
import torch

SPEC = importlib.util.spec_from_file_location(
    "prefix_formats", Path(__file__).parents[1] / "scripts/analyze_prefix_formats.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def example():
    cache = {"data": {"train": []}, "batches": {"train": []}, "states": {"train": []}}
    for role, group, delta in ((872, "single_user", [1, 0]), (8948, "system_user", [0, 2])):
        for offset in (0, 1):
            ids = torch.zeros((1, 8), dtype=torch.long)
            mask = torch.zeros_like(ids)
            ids[0, offset : offset + 6] = torch.tensor([151644, role, 198, 5, 6, 7])
            mask[0, offset : offset + 6] = 1
            cache["data"]["train"].append(
                {"context_id": f"{role}:{offset}", "group": group, "task": "detection"}
            )
            cache["batches"]["train"].append(
                {
                    "input_ids": ids,
                    "attention_mask": mask,
                    "position_ids": (mask.cumsum(-1) - 1).clamp_min(0),
                }
            )
            absent = torch.zeros(1, 8, 2, dtype=torch.bfloat16)
            present = absent.clone()
            present[0, offset + 1] += torch.tensor(delta, dtype=torch.bfloat16)
            cache["states"]["train"].append(
                ((absent, None, None, None), (present, None, None, None))
            )
    return cache


def test_compares_roles_at_logical_positions_without_padding_effect():
    result = MODULE.analyze(example())
    assert result["first_role_user_system_delta_cosine"] == 0
    assert result["prefixes"]["user"]["role_delta_l2"]["mean"] == 1
    assert result["prefixes"]["system"]["role_delta_l2"]["mean"] == 2
    for role in ("user", "system"):
        assert result["prefixes"][role]["all_role_delta_exactly_equal"]
        assert result["prefixes"][role]["first_six_by_position"][0]["delta_l2"]["max"] == 0
    assert result["groups"]["system_user"]["answer_token_delta_l2"]["max"] == 0


def test_rejects_format_label_and_logical_position_mismatch():
    cache = example()
    cache["data"]["train"][0]["group"] = "system_user"
    with pytest.raises(ValueError, match="record format"):
        MODULE.analyze(cache)
    cache = example()
    cache["batches"]["train"][0]["position_ids"][0, 1] = 2
    with pytest.raises(ValueError, match="logical positions"):
        MODULE.analyze(cache)


def test_rejects_condition_information_in_auxiliary_inputs():
    cache = copy.deepcopy(example())
    absent, present = cache["states"]["train"][0]
    cache["states"]["train"][0] = (absent, (present[0], torch.ones(1), None, None))
    with pytest.raises(ValueError, match="Auxiliary"):
        MODULE.analyze(cache)
