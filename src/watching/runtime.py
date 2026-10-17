"""Small, privacy-conscious runtime metadata for numerical reproducibility."""

import importlib.metadata
import os
import re
import subprocess

import torch

ENVIRONMENT_FLAGS = (
    "TORCHINDUCTOR_COMPILE_THREADS",
    "TORCHINDUCTOR_FREEZING",
    "TORCHINDUCTOR_FX_GRAPH_CACHE",
    "TORCHINDUCTOR_FORCE_DISABLE_CACHES",
    "TORCHINDUCTOR_EMULATE_PRECISION_CASTS",
    "TORCH_COMPILE_DISABLE",
    "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
    "NVIDIA_TF32_OVERRIDE",
    "CUBLAS_WORKSPACE_CONFIG",
    "OMP_NUM_THREADS",
)


def optional_value(read):
    """Missing packages, CPU-only builds and old backend APIs are not fatal."""
    try:
        return read()
    except Exception:
        return None


def driver_versions():
    """Query driver versions only; never retain device IDs or command diagnostics."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        timeout=2,
        check=False,
    )
    if result.returncode:
        return None
    rows = [row.strip() for row in result.stdout.splitlines() if row.strip()]
    if not rows or any(re.fullmatch(r"\d+(?:\.\d+)+", row) is None for row in rows):
        return None
    return sorted(set(rows))


def inductor_config():
    from torch._inductor import config

    return {
        name: optional_value(lambda name=name: getattr(config, name))
        for name in (
            "freezing",
            "fx_graph_cache",
            "force_disable_caches",
            "compile_threads",
            "emulate_precision_casts",
        )
    }


def runtime_provenance():
    """Observed software/settings, not a claim that changed versions caused an effect.

    Call after the experiment configures its numerical backends. Effective Inductor
    settings are separate from the allowlisted environment overrides. CUDA driver
    versions come from nvidia-smi and can differ from the toolkit used to build torch.
    """
    cuda_available = bool(optional_value(torch.cuda.is_available))
    result = {
        "schema_version": 1,
        "packages": {
            name: optional_value(lambda name=name: importlib.metadata.version(name))
            for name in ("torch", "transformers", "triton")
        },
        "torch_runtime_version": str(torch.__version__),
        "torch_git_version": optional_value(lambda: torch.version.git_version),
        "cuda_build_version": optional_value(lambda: torch.version.cuda),
        "cudnn_version": optional_value(torch.backends.cudnn.version),
        "cuda_available": cuda_available,
        "gpu": None,
        "nvidia_driver_versions": optional_value(driver_versions) if cuda_available else None,
        "numeric_settings": {
            "float32_matmul_precision": optional_value(torch.get_float32_matmul_precision),
            "matmul_allow_tf32": optional_value(lambda: torch.backends.cuda.matmul.allow_tf32),
            "matmul_allow_bf16_reduced_precision_reduction": optional_value(
                lambda: torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
            ),
            "matmul_allow_fp16_reduced_precision_reduction": optional_value(
                lambda: torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
            ),
            "cudnn_allow_tf32": optional_value(lambda: torch.backends.cudnn.allow_tf32),
            "cudnn_benchmark": optional_value(lambda: torch.backends.cudnn.benchmark),
            "deterministic_algorithms": optional_value(torch.are_deterministic_algorithms_enabled),
            "cpu_threads": optional_value(torch.get_num_threads),
        },
        "inductor_config": optional_value(inductor_config),
        "environment_overrides": {name: os.environ.get(name) for name in ENVIRONMENT_FLAGS},
        "inductor_cache_directory_explicitly_set": "TORCHINDUCTOR_CACHE_DIR" in os.environ,
    }
    if cuda_available:
        result["gpu"] = {
            "name": optional_value(torch.cuda.get_device_name),
            "compute_capability": optional_value(lambda: list(torch.cuda.get_device_capability())),
            "total_memory_bytes": optional_value(
                lambda: torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
            ),
        }
    return result
