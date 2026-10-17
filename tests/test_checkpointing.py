"""Exercise failure paths with tiny tensors, never full model mirrors."""

import errno
from types import SimpleNamespace

import pytest
import torch

from watching import checkpointing as ck


def room(monkeypatch, free):
    monkeypatch.setattr(ck.shutil, "disk_usage", lambda _: SimpleNamespace(free=free))


def test_storage_estimate_counts_shared_views_once():
    source = torch.arange(100)
    assert ck.estimated_archive_bytes({"a": source, "b": source[:1]}) == (
        source.untyped_storage().nbytes() + 16 * 1024**2
    )


def test_atomic_roundtrip_replaces_only_complete_archive(tmp_path, monkeypatch):
    room(monkeypatch, 1 << 30)
    target = tmp_path / "tail.pt"
    ck.atomic_torch_save({"value": torch.arange(3)}, target)
    ck.atomic_torch_save({"value": torch.arange(4)}, target)
    assert torch.equal(torch.load(target, weights_only=True)["value"], torch.arange(4))
    assert not list(tmp_path.glob("*.partial"))


@pytest.mark.parametrize("failure", ["serialize", "flush", "replace"])
def test_failure_preserves_previous_checkpoint_and_removes_partial(tmp_path, monkeypatch, failure):
    room(monkeypatch, 1 << 30)
    target = tmp_path / "native_tail.pt"
    ck.atomic_torch_save({"step": 256, "value": torch.ones(2)}, target)
    original = target.read_bytes()

    def fail(*args):
        if failure == "serialize":
            args[1].write(b"partial archive")
        raise OSError(errno.ENOSPC, "simulated write failure")

    if failure == "serialize":
        monkeypatch.setattr(ck.torch, "save", fail)
    elif failure == "flush":
        monkeypatch.setattr(ck.os, "fsync", fail)
    else:
        monkeypatch.setattr(ck.os, "replace", fail)
    with pytest.raises(OSError, match="simulated"):
        ck.atomic_torch_save({"step": 512}, target)
    assert target.read_bytes() == original
    assert not list(tmp_path.glob("*.partial"))


def test_capacity_rejects_before_serialization(tmp_path, monkeypatch):
    room(monkeypatch, 1)
    monkeypatch.setattr(ck.torch, "save", lambda *_: pytest.fail("must not start writing"))
    with pytest.raises(OSError) as failure:
        ck.atomic_torch_save({"x": torch.ones(2)}, tmp_path / "tail.pt")
    assert failure.value.errno == errno.ENOSPC
    assert not list(tmp_path.iterdir())


def test_failed_save_records_actual_and_last_recoverable_steps(tmp_path, monkeypatch):
    room(monkeypatch, 1 << 30)
    target = tmp_path / "native_tail.pt"
    manifest = {"training_status": "running"}
    ck.save_training_checkpoint({"completed_steps": 256}, target, manifest, step=256)
    original = target.read_bytes()
    room(monkeypatch, 1)
    with pytest.raises(OSError):
        ck.save_training_checkpoint({"completed_steps": 512}, target, manifest, step=512)
    import json

    saved = json.loads((tmp_path / "manifest.json").read_text())
    assert saved["training_status"] == "checkpoint_save_failed"
    assert saved["completed_steps"] == 512
    assert saved["last_checkpoint_step"] == 256
    assert target.read_bytes() == original


def test_preflight_budgets_next_checkpoint_and_preserves_failure_status(tmp_path, monkeypatch):
    payload = {"x": torch.zeros(2)}
    required = 2 * ck.estimated_archive_bytes(payload) + ck.RESERVE_BYTES + 123
    manifest = {}
    room(monkeypatch, required)
    ck.checkpoint_preflight(payload, tmp_path / "native_tail.pt", manifest, additional_bytes=123)
    assert manifest["checkpoint_storage_preflight"]["required_free_bytes"] == required
    room(monkeypatch, required - 1)
    with pytest.raises(OSError):
        ck.checkpoint_preflight(
            payload, tmp_path / "native_tail.pt", manifest, additional_bytes=123
        )
    assert manifest["training_status"] == "storage_preflight_failed"


def test_recovery_uses_saved_step_not_later_unsaved_updates():
    manifest = {
        "steps": 1024,
        "completed_steps": 512,
        "last_checkpoint_step": 256,
        "starting_global_step": 128,
        "starting_processed_prompt_pairs": 1024,
        "global_processed_prompt_pairs": 5120,
        "prompt_batch_size": 8,
        "training_status": "checkpoint_save_failed",
    }
    assert ck.recoverable_position(manifest) == (256, 384, 3072)
    del manifest["last_checkpoint_step"]
    assert ck.recoverable_position(manifest) == (512, 640, 5120)
    manifest["last_checkpoint_step"] = 513
    with pytest.raises(ValueError, match="recoverable"):
        ck.recoverable_position(manifest)


def test_status_write_failure_does_not_mask_checkpoint_error(tmp_path, monkeypatch, capsys):
    original = OSError(errno.ENOSPC, "checkpoint failure")

    def fail_save(*_):
        raise original

    def fail_status(*_):
        raise OSError(errno.EIO, "status failure")

    monkeypatch.setattr(ck, "atomic_torch_save", fail_save)
    monkeypatch.setattr(ck, "write_json", fail_status)
    with pytest.raises(OSError) as caught:
        ck.save_training_checkpoint({}, tmp_path / "native_tail.pt", {}, step=512)
    assert caught.value is original
    assert "checkpoint_status_write_failed" in capsys.readouterr().err
