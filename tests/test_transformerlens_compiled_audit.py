"""Verify actual tokenizer layout and parameter-integrity helpers."""

import importlib.util
import sys
from pathlib import Path

import torch

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location(
    "tl_compiled_audit", SCRIPTS / "audit_transformerlens_compiled.py"
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
sys.path.pop(0)


def test_parameter_digest_detects_change():
    model = torch.nn.Linear(2, 3)
    before = audit.parameter_digest(model)
    with torch.no_grad():
        model.weight[0, 0] += 1
    assert audit.parameter_digest(model) != before


def test_fixed_inputs_use_same_shape_and_correct_left_padding():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, "a": 2, "b": 3}, "[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]"
    )
    inputs = audit.fixed_inputs(tokenizer, ("a", "a b"), 4, "cpu")
    assert inputs[0]["input_ids"].tolist() == [[0, 0, 0, 2]]
    assert inputs[1]["input_ids"].tolist() == [[0, 0, 2, 3]]
    assert inputs[1]["position_ids"].tolist() == [[0, 0, 0, 1]]
    assert inputs[1]["attention_mask"].tolist() == [[0, 0, 1, 1]]
