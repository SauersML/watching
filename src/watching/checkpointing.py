"""Bounded disk checks and atomic, weights-only training checkpoints."""

import errno
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import torch

from watching.artifacts import write_json

RESERVE_BYTES = 256 * 1024**2


def estimated_archive_bytes(value):
    """Count unique tensor storage without moving or cloning model tensors.

    PyTorch serializes an entire storage even for a small view. The allowance
    covers archive metadata; this estimate is a capacity check, not a guarantee.
    """
    seen = set()
    total = 0

    def visit(item):
        nonlocal total
        if isinstance(item, torch.Tensor):
            storage = item.untyped_storage()
            key = (str(item.device), storage.data_ptr(), storage.nbytes())
            if key not in seen:
                seen.add(key)
                total += storage.nbytes()
        elif isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return total + max(16 * 1024**2, total // 20)


def require_free_space(path, required_bytes, *, reserve_bytes=RESERVE_BYTES):
    path = Path(path)
    if required_bytes < 0 or reserve_bytes < 0:
        raise ValueError("Disk requirements must be nonnegative")
    free = shutil.disk_usage(path.parent).free
    required = int(required_bytes) + int(reserve_bytes)
    if free < required:
        raise OSError(
            errno.ENOSPC,
            f"Need {required} free bytes including reserve; found {free} at {path.parent}",
        )
    return {"required_free_bytes": required, "available_free_bytes": free}


def atomic_torch_save(value, destination, *, reserve_bytes=RESERVE_BYTES):
    """Publish only a closed, flushed archive; failed writes retain the previous file.

    The temporary file shares the destination filesystem, so os.replace is atomic.
    Device tensors are serialized directly; torch.save stages their storage itself,
    avoiding a second complete CPU state dictionary allocated by the caller.
    """
    destination = Path(destination)
    require_free_space(destination, estimated_archive_bytes(value), reserve_bytes=reserve_bytes)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{destination.name}.",
            suffix=".partial",
            dir=destination.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        size = temporary.stat().st_size
        os.replace(temporary, destination)
        return size
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _record_failure(path, manifest, error):
    try:
        write_json(path, manifest)
    except OSError as status_error:
        error.add_note(f"Could not persist failure manifest: {status_error}")
        print(
            json.dumps(
                {
                    "event": "checkpoint_status_write_failed",
                    "error": str(error),
                    "status_error": str(status_error),
                    "completed_steps": manifest.get("completed_steps"),
                    "last_checkpoint_step": manifest.get("last_checkpoint_step"),
                }
            ),
            file=sys.stderr,
            flush=True,
        )


def checkpoint_preflight(payload, destination, manifest, *, additional_bytes=0):
    """Budget a retained checkpoint plus its atomic replacement before training."""
    destination = Path(destination)
    estimate = estimated_archive_bytes(payload)
    try:
        report = require_free_space(destination, 2 * estimate + additional_bytes)
    except OSError as error:
        manifest.update(training_status="storage_preflight_failed", storage_error=str(error))
        _record_failure(destination.parent / "manifest.json", manifest, error)
        raise
    manifest["checkpoint_storage_preflight"] = {
        **report,
        "estimated_checkpoint_bytes": estimate,
        "additional_artifact_bytes": additional_bytes,
        "scope": "capacity estimate; not a reservation or filesystem-error guarantee",
    }
    manifest["checkpoint_recovery"] = (
        "native weights only; Adam and FP32 master states are not saved"
    )
    write_json(destination.parent / "manifest.json", manifest)


def save_training_checkpoint(payload, destination, manifest, *, step):
    """Keep actual training progress distinct from the last recoverable weights."""
    destination = Path(destination)
    manifest["completed_steps"] = step
    try:
        size = atomic_torch_save(payload, destination)
    except Exception as error:
        manifest.update(
            training_status="checkpoint_save_failed",
            checkpoint_error=str(error),
            checkpoint_failed_at_step=step,
        )
        # The failed temporary file is already removed, freeing space for status.
        _record_failure(destination.parent / "manifest.json", manifest, error)
        raise
    manifest.update(last_checkpoint_step=step, last_checkpoint_bytes=size)
    write_json(destination.parent / "manifest.json", manifest)
    return size


def recoverable_position(manifest):
    """Return saved-weight progress, never later unsaved optimizer progress."""
    completed = manifest.get("completed_steps", manifest["steps"])
    step = manifest.get("last_checkpoint_step", completed)
    if type(step) is not int or step < 1 or type(completed) is not int or step > completed:
        raise ValueError("Prior run has no valid recoverable checkpoint step")
    global_step = manifest.get("starting_global_step", 0) + step
    if "last_checkpoint_step" in manifest:
        pairs = manifest.get("starting_processed_prompt_pairs", 0) + step * manifest.get(
            "prompt_batch_size", 1
        )
    else:
        pairs = manifest.get(
            "global_processed_prompt_pairs",
            manifest.get("starting_processed_prompt_pairs", manifest.get("starting_global_step", 0))
            + step * manifest.get("prompt_batch_size", 1),
        )
    return step, global_step, pairs
