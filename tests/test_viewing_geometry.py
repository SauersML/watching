"""Check token-position semantics and reject confounded cached-state comparisons."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "viewing_geometry", SCRIPTS / "analyze_viewing_geometry.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def cache():
    result = dict(states={}, batches={}, data={})
    for split in ("train", "validation"):
        absent = torch.zeros(1, 5, 3, dtype=torch.bfloat16)
        present = absent.clone()
        present[0, 2, 0] = 2
        present[0, -1, 1] = 3
        result["states"][split] = [[(absent, None, None, None), (present, None, None, None)]]
        result["batches"][split] = [
            dict(
                input_ids=torch.tensor([[0, 151644, 872, 198, 123]]),
                attention_mask=torch.tensor([[0, 1, 1, 1, 1]]),
                position_ids=torch.tensor([[0, 0, 1, 2, 3]]),
            )
        ]
        result["data"][split] = [dict(context_id=split)]
    return result


def test_logical_role_and_answer_positions_are_distinct():
    report = MODULE.analyze(cache())
    assert report["summary"]["combined"]["role"]["delta_l2"]["mean"] == 2
    assert report["summary"]["combined"]["answer"]["delta_l2"]["mean"] == 3
    assert report["summary"]["combined"]["role"]["unique_delta_hashes"] == 1
    assert report["rows"][0]["role"]["logical_position"] == 1
    assert report["rows"][0]["role"]["physical_position"] == 2


def test_system_prefix_or_condition_dependent_auxiliary_rejected():
    source = cache()
    source["batches"]["train"][0]["input_ids"][0, 2] = 8948
    with pytest.raises(ValueError, match="single-user"):
        MODULE.analyze(source)
    source = cache()
    old = source["states"]["train"][0][1]
    source["states"]["train"][0][1] = (old[0], torch.ones(1), None, None)
    with pytest.raises(ValueError, match="Condition-dependent"):
        MODULE.analyze(source)
