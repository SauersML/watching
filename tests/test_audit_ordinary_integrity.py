import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "audit_ordinary_integrity",
    Path(__file__).resolve().parents[1] / "scripts" / "audit_ordinary_integrity.py",
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_containment_preserves_word_boundaries():
    assert audit.contains_text("Is this a cat?", "Context. IS THIS A CAT! Answer yes.")
    assert not audit.contains_text("cat", "caterpillar")
    assert not audit.contains_text("a cat", "a feline")
    assert audit.normalized("Straße, A_B!") == "strasse a_b"


def test_empty_normalization_rejected():
    with pytest.raises(ValueError, match="Empty normalized"):
        audit.contains_text("?!", "A real question")


def test_changed_artifact_rejected(tmp_path):
    path = tmp_path / "scores.bin"
    path.write_bytes(b"original")
    hashes = {"scores.bin": audit.sha(path)}
    assert audit.verify_artifacts(tmp_path, hashes) == ["scores.bin"]
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="differs"):
        audit.verify_artifacts(tmp_path, hashes)


def test_unsafe_artifact_path_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unsafe relative"):
        audit.verify_artifacts(tmp_path, {"../scores.bin": "not-a-hash"})
