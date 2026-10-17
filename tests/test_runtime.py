"""Runtime provenance stays useful on CPU and never serializes unrelated environment."""

import json
import subprocess
from types import SimpleNamespace

import pytest

from watching import runtime


def test_cpu_provenance_handles_missing_optional_packages_without_driver_query(monkeypatch):
    monkeypatch.setattr(runtime.torch.cuda, "is_available", lambda: False)

    def version(name):
        if name == "triton":
            raise runtime.importlib.metadata.PackageNotFoundError(name)
        return "test-version"

    def driver():
        raise AssertionError("CPU-only provenance must not invoke nvidia-smi")

    monkeypatch.setattr(runtime.importlib.metadata, "version", version)
    monkeypatch.setattr(runtime, "driver_versions", driver)
    monkeypatch.setattr(runtime, "inductor_config", lambda: {"freezing": False})
    monkeypatch.setenv("PRIVATE_API_KEY", "secret-token")
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", "/private/user-cache")
    monkeypatch.setenv("TORCHINDUCTOR_FREEZING", "0")
    result = runtime.runtime_provenance()
    assert result["packages"]["triton"] is None
    assert result["packages"]["transformers"] == "test-version"
    assert result["gpu"] is None
    assert result["nvidia_driver_versions"] is None
    assert result["environment_overrides"]["TORCHINDUCTOR_FREEZING"] == "0"
    assert result["inductor_config"]["freezing"] is False
    assert result["inductor_cache_directory_explicitly_set"]
    encoded = json.dumps(result, allow_nan=False)
    assert "secret-token" not in encoded
    assert "PRIVATE_API_KEY" not in encoded
    assert "/private/user-cache" not in encoded


def test_cuda_metadata_distinguishes_build_driver_and_capability(monkeypatch):
    monkeypatch.setattr(runtime.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runtime.torch.cuda, "get_device_name", lambda: "NVIDIA L40S")
    monkeypatch.setattr(runtime.torch.cuda, "get_device_capability", lambda: (8, 9))
    monkeypatch.setattr(runtime.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        runtime.torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(total_memory=48 * 1024**3),
    )
    monkeypatch.setattr(runtime.torch.version, "cuda", "12.8")
    monkeypatch.setattr(runtime, "driver_versions", lambda: ["580.126.09"])
    monkeypatch.setattr(runtime, "inductor_config", lambda: None)
    result = runtime.runtime_provenance()
    assert result["gpu"] == {
        "name": "NVIDIA L40S",
        "compute_capability": [8, 9],
        "total_memory_bytes": 48 * 1024**3,
    }
    assert result["cuda_build_version"] == "12.8"
    assert result["nvidia_driver_versions"] == ["580.126.09"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "stdout,returncode,expected",
    [
        ("550.54.15\n550.54.15\n", 0, ["550.54.15"]),
        ("", 0, None),
        ("sensitive error text", 0, None),
        ("580.126.09\n", 1, None),
    ],
)
def test_driver_query_retains_only_versions(monkeypatch, stdout, returncode, expected):
    def run(command, **kwargs):
        assert command == ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
        assert kwargs["timeout"] == 2
        assert kwargs["capture_output"]
        return SimpleNamespace(stdout=stdout, returncode=returncode)

    monkeypatch.setattr(runtime.subprocess, "run", run)
    assert runtime.driver_versions() == expected


@pytest.mark.parametrize("error", [FileNotFoundError(), subprocess.TimeoutExpired("nvidia-smi", 2)])
def test_optional_runtime_failures_do_not_block_training(monkeypatch, error):
    def fail():
        raise error

    monkeypatch.setattr(runtime.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runtime, "driver_versions", fail)
    monkeypatch.setattr(runtime, "inductor_config", fail)
    result = runtime.runtime_provenance()
    assert result["nvidia_driver_versions"] is None
    assert result["inductor_config"] is None
