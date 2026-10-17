"""Frozen counts, independent base units and held-out family boundaries."""

import json
from pathlib import Path

from watching.generalization_data import assemble
from watching.messages import validate_message_splits

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "data/generalization_v4"


def test_assembly_reproduces_frozen_data_and_independent_counts():
    data = assemble(BANK)
    assert data == json.loads((BANK / "dataset.json").read_text())
    validate_message_splits(data)
    for split, n, bases in (("train", 512, 128), ("validation", 64, 16), ("test", 128, 32)):
        rows = data[split]
        assert len(rows) == n
        assert len({r["base_context_id"] for r in rows}) == bases
        assert sum(r["task"] == "animal" for r in rows) == n // 2
        for r in rows:
            assert r["messages"][-1]["content"] == r["text"]
            if r["task"] == "animal":
                assert not any(
                    w in r["text"].lower() for w in ("probe", "gradient", "monitor", "activation")
                )
    for key in ("base_context_id", "question_family", "prefix_family"):
        sets = [{r[key] for r in rows} for rows in data.values()]
        assert not any(a & b for i, a in enumerate(sets) for b in sets[i + 1 :])


def test_training_reuses_only_declared_training_contexts_and_eval_is_fresh():
    data = assemble(BANK)
    v2 = json.loads((ROOT / "data/validation_v2/dataset.json").read_text())
    v3 = json.loads((ROOT / "data/confirmation_v3/dataset.json").read_text())
    old_train = {r["context"] for r in v2["train"]}
    assert {r["context"] for r in data["train"]} <= old_train
    previous = {r["context"] for rows in v2.values() for r in rows}
    previous |= {r["context"] for rows in v3.values() for r in rows}
    fresh = {r["context"] for split in ("validation", "test") for r in data[split]}
    assert len(fresh) == 48 and not fresh & previous
