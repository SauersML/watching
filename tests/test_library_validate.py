"""Fail-closed CPU checks for the fixed-artifact fresh validation protocol."""

from copy import deepcopy

import pytest

from watching.library_validate import MODEL, REVISION, encode_record, validate_source


def source():
    manifest = {
        "checkpoint": MODEL,
        "revision": REVISION,
        "training_status": "complete",
        "completed_steps": 1024,
        "last_checkpoint_step": 1024,
        "steps": 1024,
        "train_head": False,
        "output_head_updated": False,
        "probe_layer": 14,
        "tail_layers": 2,
        "condition": "probe",
        "library_capture_method": "forward_hook",
        "head_sha256_before": "fixed",
        "head_sha256_after": "fixed",
    }
    artifact = {
        "completed_steps": 1024,
        "first_layer": 34,
        "checkpoint": MODEL,
        "train_head": False,
        "head": None,
        "head_sha256": "fixed",
    }
    data = {
        split: [{"context_id": f"{split}:{i}"} for i in range(n)]
        for split, n in (("train", 256), ("validation", 64), ("test", 64))
    }
    return manifest, artifact, data


def test_exact_complete_source_accepted():
    manifest, artifact, data = source()
    validate_source(manifest, artifact, "forward_hook", data)


@pytest.mark.parametrize("change", ["stale", "head", "method", "incomplete"])
def test_rejects_stale_or_different_source(change):
    manifest, artifact, data = source()
    if change == "stale":
        artifact["completed_steps"] = 256
    elif change == "head":
        manifest["head_sha256_after"] = "changed"
    elif change == "method":
        manifest["library_capture_method"] = "hidden_states"
    else:
        manifest["training_status"] = "time_limit"
    with pytest.raises(ValueError):
        validate_source(manifest, artifact, "forward_hook", data)


def test_rejects_changed_prompt_bank_and_overlap():
    manifest, artifact, data = source()
    bad = deepcopy(data)
    bad["test"].pop()
    with pytest.raises(ValueError, match="split sizes"):
        validate_source(manifest, artifact, "forward_hook", bad)
    data["test"][0] = data["validation"][0]
    with pytest.raises(ValueError, match="overlapping"):
        validate_source(manifest, artifact, "forward_hook", data)


def test_rejects_oversize_and_silent_tokenizer_truncation(monkeypatch):
    from types import SimpleNamespace

    import torch

    monkeypatch.setattr("watching.library_validate.render_record", lambda *_: "rendered")
    tokenizer = SimpleNamespace(encode=lambda *_args, **_kwargs: [1, 2, 3])
    with pytest.raises(ValueError, match="no truncation"):
        encode_record(tokenizer, {}, 2)

    class BrokenTokenizer:
        def encode(self, *_args, **_kwargs):
            return [1, 2, 3]

        def __call__(self, *_args, **kwargs):
            assert kwargs["truncation"] is False
            return {
                "input_ids": torch.tensor([[0, 1, 2]]),
                "attention_mask": torch.tensor([[0, 1, 1]]),
            }

    with pytest.raises(ValueError, match="truncated or changed"):
        encode_record(BrokenTokenizer(), {}, 3)


def test_selected_prompt_hash_is_distinct_from_original_input_bank(monkeypatch, tmp_path):
    """Real 061r metadata hashes: selected fit prompts are not the full input bank."""
    from watching.library_validate import SELECTED_DATA_SHA256, validate_data_files

    bank_sha = "9321889d38cc5b070844a7a3692811f86e0c70905e7e4d5fbe790f8fab07b755"
    selected, bank = tmp_path / "selected.json", tmp_path / "input_bank.json"
    fixture_hashes = {selected: SELECTED_DATA_SHA256, bank: bank_sha}
    monkeypatch.setattr("watching.library_validate.file_digest", fixture_hashes.__getitem__)
    manifest = {"data_sha256": bank_sha}
    report = validate_data_files(selected, manifest)
    assert report["selected_data_sha256"] != report["input_bank_expected_sha256"]
    assert report["input_bank_sha256_verified"] is False
    assert validate_data_files(selected, manifest, bank)["input_bank_sha256_verified"] is True
    fixture_hashes[bank] = "0" * 64
    with pytest.raises(ValueError, match="Original input bank"):
        validate_data_files(selected, manifest, bank)
    fixture_hashes[selected] = bank_sha
    with pytest.raises(ValueError, match="Selected fit data"):
        validate_data_files(selected, manifest)
