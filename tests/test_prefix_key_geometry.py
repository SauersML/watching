import importlib.util
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location(
    "geometry_audit", Path(__file__).parents[1] / "scripts/audit_prefix_key_geometry.py"
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_geometry_midpoint_difference_and_angle():
    result = audit.geometry(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]))
    assert result["condition_angle_degrees"] == 90
    assert result["difference_over_midpoint"] == 2
    zero = audit.geometry(torch.zeros(2), torch.zeros(2))
    assert zero["condition_cosine"] is None
    with pytest.raises(ValueError, match="finite"):
        audit.geometry(torch.ones(2), torch.full((2,), float("nan")))


def test_selection_is_fixed_by_identity_and_keeps_all_layouts():
    rows = [
        {"task": task, "base_context_id": f"{task}/{base}", "group": group}
        for task in audit.TASKS
        for base in ("b", "a")
        for group in audit.GROUPS
    ]
    selected = audit.select_pairs(list(reversed(rows)))
    assert len(selected) == 8
    assert all(row["base_context_id"].endswith("/a") for _, row in selected)
    assert [row["group"] for _, row in selected[:4]] == list(audit.GROUPS)
