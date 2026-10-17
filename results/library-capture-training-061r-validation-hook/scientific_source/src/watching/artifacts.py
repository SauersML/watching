"""Publish complete artifacts so live output synchronization skips partial files."""

import json
import os
import shutil
import tempfile
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

    Serialize outside the output tree, then copy with .partial file suffixes.
    This also protects synchronizers that filter filenames but not directories.
    Checkpoints are immutable once published; use a new destination per step.
    """
    if destination.exists():
        raise FileExistsError(destination)
    staging = destination.with_name(destination.name + ".partial")
    if staging.exists():
        raise FileExistsError(staging)
    with tempfile.TemporaryDirectory(prefix="watching-checkpoint-") as temporary:
        source = Path(temporary)
        model.save_pretrained(source, safe_serialization=True)
        tokenizer.save_pretrained(source)
        write_json(
            source / "training_position.json",
            {
                "step": step,
                "contains_optimizer_state": False,
                "contains_fp32_master_weights": False,
                "resume_semantics": "model weights only; a new training run uses a fresh optimizer",
            },
        )
        staging.mkdir(parents=True)
        for path in sorted(source.rglob("*")):
            if path.is_file():
                target = staging / path.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                partial = target.with_name(target.name + ".partial")
                shutil.copyfile(path, partial)
                partial.replace(target)
        staging.rename(destination)
