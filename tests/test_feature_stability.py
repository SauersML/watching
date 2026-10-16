import importlib.util
from pathlib import Path

import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_feature_stability.py"
spec = importlib.util.spec_from_file_location("feature_stability_audit", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def save(tmp_path, name, features, **metadata):
    path = tmp_path / name
    torch.save({"features": features, **metadata}, path)
    return path


def test_common_drift_and_footprint_rotation_are_separated(tmp_path):
    before = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 3.0], [2.0, 3.0]])
    after = torch.tensor([[10.0, 0.0], [10.0, 1.0], [2.0, 3.0], [2.0, 3.0]])
    fit = save(tmp_path, "fit.pt", before, context_ids=["first", "second"])
    fresh = save(
        tmp_path, "fresh.pt", after, context_ids=["first", "second"], condition_order=[False, True]
    )
    report = module.audit(fit, fresh)
    assert report["identity"]["scope"] == "ordered_context_ids"
    first, second = report["per_prompt"]
    assert first["footprint_norm_fit"] == first["footprint_norm_fresh"] == 1
    assert first["footprint_cosine"] == 0
    assert first["midpoint_movement_norm"] == pytest.approx((9.5**2 + 0.5**2) ** 0.5)
    assert second["footprint_cosine"] is None
    assert second["midpoint_movement_over_fit_footprint_norm"] is None
    assert report["summary"]["undefined_footprint_cosines"] == 1
    assert report["summary"]["conditions"]["present"]["bit_identical_prompt_count"] == 1
    assert len(report["sources"]["fit"]["sha256"]) == 64


def test_bit_identity_distinguishes_signed_zero_without_numeric_drift(tmp_path):
    before = torch.tensor([[0.0, 2.0], [1.0, 2.0]], dtype=torch.bfloat16)
    after = before.clone()
    after[0, 0] = -0.0
    report = module.audit(save(tmp_path, "a.pt", before), save(tmp_path, "b.pt", after))
    absent = report["summary"]["conditions"]["absent"]
    assert absent["bit_identical_element_fraction"] == 0.5
    assert absent["bit_identical_prompt_count"] == 0
    assert absent["state_change_norm"]["max"] == 0
    assert report["summary"]["footprint_cosine_nonzero_pairs"]["mean"] == 1


def test_missing_ids_are_explicitly_positional(tmp_path):
    features = torch.ones(4, 3)
    a = save(tmp_path, "a.pt", features)
    b = save(tmp_path, "b.pt", features, context_ids=["one", "two"])
    report = module.audit(a, b)
    assert report["identity"]["scope"] == "positional_only"
    assert report["identity"]["ordered_context_ids_match"] is None
    assert report["per_prompt"][0]["context_id"] == "one"
    assert report["sources"]["fit"]["condition_order"] == "assumed_absent_present"


def test_mismatched_ids_or_shapes_cannot_silently_realign(tmp_path):
    features = torch.ones(4, 3)
    a = save(tmp_path, "a.pt", features, context_ids=["one", "two"])
    b = save(tmp_path, "b.pt", features, context_ids=["two", "one"])
    with pytest.raises(ValueError, match="Ordered context IDs differ"):
        module.audit(a, b)
    b = save(tmp_path, "b.pt", torch.ones(2, 3))
    with pytest.raises(ValueError, match="shapes must match"):
        module.audit(a, b)
    b = save(tmp_path, "b.pt", features.double())
    with pytest.raises(ValueError, match="dtypes must match"):
        module.audit(a, b)


@pytest.mark.parametrize(
    "metadata",
    [
        {"condition_order": [True, False]},
        {"condition_order": [0, 1]},
        {"context_ids": ["one", None]},
        {"context_ids": ["one"]},
    ],
)
def test_incomplete_or_wrong_order_metadata_is_rejected(tmp_path, metadata):
    p = save(tmp_path, "bad.pt", torch.ones(4, 3), **metadata)
    with pytest.raises(ValueError):
        module.load_features(p)


@pytest.mark.parametrize(
    "features",
    [
        torch.ones(3, 2),
        torch.ones(0, 2),
        torch.tensor([[float("nan")], [1.0]]),
        torch.ones(2, 2, dtype=torch.int64),
    ],
)
def test_invalid_feature_arrays_are_rejected(tmp_path, features):
    with pytest.raises(ValueError):
        module.load_features(save(tmp_path, "bad.pt", features))
