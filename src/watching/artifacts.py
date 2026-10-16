"""Publish complete artifacts so live output synchronization skips partial files."""

import json
import os
from pathlib import Path


def write_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)


def save_weights(model, tokenizer, destination: Path, *, step: int) -> None:
    """Save warm-start weights, explicitly not an exact optimizer resume.

    A directory ending in .partial is excluded by the remote output runner.
    Checkpoints are immutable once published; use a new destination per step.
    """
    if destination.exists():
        raise FileExistsError(destination)
    staging = destination.with_name(destination.name + ".partial")
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    model.save_pretrained(staging, safe_serialization=True)
    tokenizer.save_pretrained(staging)
    write_json(
        staging / "training_position.json",
        {
            "step": step,
            "contains_optimizer_state": False,
            "contains_fp32_master_weights": False,
            "resume_semantics": "model weights only; a new training run uses a fresh optimizer",
        },
    )
    staging.rename(destination)
