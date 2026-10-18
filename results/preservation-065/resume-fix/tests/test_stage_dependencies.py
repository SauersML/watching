"""Dependency behavior is tested without training, model loading, or remote calls."""

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "stage_dependencies", Path(__file__).parents[1] / "stage_dependencies.py"
)
stages = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stages)


def complete(root, name, inputs, output, content):
    assert stages.prepare(root, name, inputs, [output]) == "run"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content)
    stages.commit(root, name, inputs, [output])


def test_selected_weights_change_invalidates_export_and_only_its_dependents(tmp_path):
    root = tmp_path / "run"
    tail = tmp_path / "tail"
    tail.write_text("weights A")
    source = tmp_path / "source"
    source.write_text("export code")
    exported = tmp_path / "outside-synchronized-output" / "export.json"
    export_deps = {"tail": tail, "code": source}
    complete(root, "export", export_deps, exported, '{"tail":"A"}')
    evaluation = root / "eval" / "scores.json"
    complete(root, "eval", {"export": exported}, evaluation, "scores A")
    capture = root / "capture.bin"
    capture.write_text("frozen states")
    (root / "done.capture").touch()
    # A resume with a new optimizer step but the SAME selected weights reuses results.
    assert stages.prepare(root, "export", export_deps, [exported]) == "current"
    assert stages.prepare(root, "eval", {"export": exported}, [evaluation]) == "current"
    tail.write_text("weights B")
    assert stages.prepare(root, "export", export_deps, [exported]) == "run"
    assert not exported.exists()
    assert evaluation.read_text() == "scores A"  # no blanket invalidation
    assert (root / "done.capture").exists() and capture.read_text() == "frozen states"
    # External exports remain outside the synchronized root when archived.
    archived = list((exported.parent / ".stale-stages").iterdir())
    assert len(archived) == 1 and not archived[0].is_relative_to(root)
    assert archived[0].read_text() == '{"tail":"A"}'
    exported.write_text('{"tail":"B"}')
    stages.commit(root, "export", export_deps, [exported])
    assert stages.prepare(root, "eval", {"export": exported}, [evaluation]) == "run"
    archived_scores = list((root / "stale-stages" / "eval").glob("*/*scores.json"))
    assert len(archived_scores) == 1 and archived_scores[0].read_text() == "scores A"


def test_changed_endpoint_code_does_not_invalidate_other_endpoint(tmp_path):
    root = tmp_path / "run"
    exported = tmp_path / "export"
    exported.write_text("same model")
    hf_code = tmp_path / "hf.py"
    hf_code.write_text("v1")
    vllm_code = tmp_path / "vllm.py"
    vllm_code.write_text("v1")
    hf, vllm = root / "hf.json", root / "vllm.json"
    complete(root, "hf", {"model": exported, "code": hf_code}, hf, "hf scores")
    complete(root, "vllm", {"model": exported, "code": vllm_code}, vllm, "vllm scores")
    hf_code.write_text("v2")
    assert stages.prepare(root, "hf", {"model": exported, "code": hf_code}, [hf]) == "run"
    assert stages.prepare(root, "vllm", {"model": exported, "code": vllm_code}, [vllm]) == "current"


def test_legacy_unpinned_marker_is_not_trusted(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    dependency = tmp_path / "weights"
    dependency.write_text("new weights")
    output = root / "scores"
    output.write_text("possibly stale")
    (root / "done.eval").touch()
    assert stages.prepare(root, "eval", {"weights": dependency}, [output]) == "run"
    assert not (root / "done.eval").exists()
    assert not output.exists()


def test_mutation_during_stage_or_missing_output_cannot_commit(tmp_path):
    root = tmp_path / "run"
    dependency = tmp_path / "weights"
    dependency.write_text("A")
    output = root / "scores"
    stages.prepare(root, "eval", {"weights": dependency}, [output])
    with pytest.raises(ValueError, match="missing"):
        stages.commit(root, "eval", {"weights": dependency}, [output])
    output.write_text("output")
    dependency.write_text("B")
    with pytest.raises(ValueError, match="changed"):
        stages.commit(root, "eval", {"weights": dependency}, [output])
    assert not (root / "done.eval").exists()


def test_cannot_archive_a_dependency_parent(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    dependency = root / "weights"
    dependency.write_text("A")
    with pytest.raises(ValueError, match="contains a dependency"):
        stages.prepare(root, "eval", {"weights": dependency}, [root])


def test_shell_stage_marker_cannot_skip_changed_selected_weights(tmp_path):
    import shlex
    import subprocess

    source = Path(__file__).parents[1]
    text = (source / "run_preservation.sh").read_text()
    functions = (
        "stage() {" + text.split("stage() {", 1)[1].split('cp "$source_dir/fresh_test.json"', 1)[0]
    )
    root = tmp_path / "run"
    root.mkdir()
    weights = tmp_path / "selected"
    weights.write_text("A")
    output = root / "export.txt"
    header = (
        f"set -euo pipefail\nsource_dir={shlex.quote(str(source))}\n"
        f'out={shlex.quote(str(root))}\ncd "$out"\n' + functions
    )

    def run(command):
        return subprocess.run(
            ["bash", "-c", header + command], check=True, capture_output=True, text=True
        )

    prefix = (
        f"pinned_stage export --input selected={shlex.quote(str(weights))} "
        f"--output {shlex.quote(str(output))} -- "
    )
    command = "bash -c 'cat ../selected > export.txt; printf x >> calls'"
    run(prefix + command)
    reused = run(prefix + command)
    assert "PINNED_ALREADY_DONE" in reused.stdout
    weights.write_text("B")
    assert (root / "calls").read_text() == "x"
    run(prefix + command)
    assert (root / "calls").read_text() == "xx"
    assert output.read_text() == "B"
    assert len(list((root / "stale-stages" / "export").glob("*/*export.txt"))) == 1


def test_changed_sampling_or_pass_arguments_invalidate_stage(tmp_path):
    root = tmp_path / "run"
    model = tmp_path / "model"
    model.write_text("same weights")
    output = root / "scores"
    inputs = {"model": model}
    old = ["evaluate.py", "--passes", "1"]
    new = ["evaluate.py", "--passes", "2"]
    stages.prepare(root, "eval", inputs, [output], old)
    output.write_text("one pass")
    stages.commit(root, "eval", inputs, [output], old)
    assert stages.prepare(root, "eval", inputs, [output], old) == "current"
    assert stages.prepare(root, "eval", inputs, [output], new) == "run"
