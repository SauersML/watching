"""An exported model is tied to the actual selected tail, retaining fixed tensors."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def test_export_records_actual_selected_weight_identity(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    changed = "model.layers.34.self_attn.o_proj.weight"
    fixed = "lm_head.weight"
    original = {changed: torch.tensor([1.0]), fixed: torch.tensor([7.0])}
    save_file(original, str(base / "model.safetensors"))
    (base / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "model.safetensors" for name in original}})
    )
    tail = tmp_path / "tail.safetensors"
    save_file({changed: torch.tensor([2.0])}, str(tail))
    destination = tmp_path / "export" / "model"
    script = Path(__file__).parents[1] / "export_tail.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--base",
            str(base),
            "--tail",
            str(tail),
            "--out",
            str(destination),
        ],
        check=True,
        capture_output=True,
    )
    exported = load_file(str(destination / "model.safetensors"))
    assert exported[changed].item() == 2
    assert exported[fixed].item() == 7
    manifest = json.loads((destination.parent / "export.json").read_text())
    assert manifest["selected_tail_sha256"] == hashlib.sha256(tail.read_bytes()).hexdigest()
    assert manifest["export_script_sha256"] == hashlib.sha256(script.read_bytes()).hexdigest()
