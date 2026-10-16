import json

import pytest

from watching import artifacts
from watching.artifacts import save_weights, write_json


def test_invalid_json_does_not_replace_previous_result(tmp_path):
    path = tmp_path / "result.json"
    write_json(path, {"score": 1.0})
    with pytest.raises(ValueError):
        write_json(path, {"score": float("nan")})
    assert json.loads(path.read_text()) == {"score": 1.0}


class Artifact:
    def __init__(self, name, fail=False):
        self.name = name
        self.fail = fail

    def save_pretrained(self, directory, **kwargs):
        (directory / self.name).write_text("example")
        if self.fail:
            raise RuntimeError("interrupted")


def test_checkpoint_is_only_published_after_both_parts_exist(tmp_path):
    destination = tmp_path / "checkpoint_512"
    with pytest.raises(RuntimeError):
        save_weights(Artifact("weights"), Artifact("tokenizer", fail=True), destination, step=512)
    assert not destination.exists()
    assert not (tmp_path / "checkpoint_512.partial").exists()
    destination = tmp_path / "checkpoint_1024"
    save_weights(Artifact("weights"), Artifact("tokenizer"), destination, step=1024)
    state = json.loads((destination / "training_position.json").read_text())
    assert state["step"] == 1024
    assert state["contains_optimizer_state"] is False
    assert (destination / "weights").exists()
    assert (destination / "tokenizer").exists()
    assert not (tmp_path / "checkpoint_1024.partial").exists()
    with pytest.raises(FileExistsError):
        save_weights(Artifact("weights"), Artifact("tokenizer"), destination, step=1024)


def test_interrupted_copy_only_exposes_partial_file(tmp_path, monkeypatch):
    def interrupted_copy(source, destination):
        assert destination.name.endswith(".partial")
        destination.write_text("incomplete")
        raise OSError("interrupted transfer")

    monkeypatch.setattr(artifacts.shutil, "copyfile", interrupted_copy)
    destination = tmp_path / "checkpoint"
    with pytest.raises(OSError, match="interrupted transfer"):
        save_weights(Artifact("weights"), Artifact("tokenizer"), destination, step=10)
    assert not destination.exists()
    files = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert files and all(p.name.endswith(".partial") for p in files)
