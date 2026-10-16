import pytest

from watching.diverse_data import make_diverse_data
from watching.scaled_data import make_scaled_data


def test_expansion_never_changes_heldout_records():
    additions = [{"family": "new_family", "context": "A context created only for this test."}]
    for task in ("probe", "execution", "preference"):
        before = make_diverse_data(task)
        after = make_scaled_data(additions, task)
        assert len(after["train"]) == 257
        for split in before.keys() - {"train"}:
            assert after[split] == before[split]
        assert len({r["wording_id"] for r in after["train"]}) == 16


def test_rejects_heldout_context_as_training_addition():
    heldout = make_diverse_data()["test"][0]
    with pytest.raises(ValueError, match="distinct"):
        make_scaled_data([heldout])


def test_tasks_use_the_same_expanded_contexts():
    contexts = [{"family": "new_family", "context": "A context created only for this test."}]
    variants = [make_scaled_data(contexts, task) for task in ("probe", "execution", "preference")]
    ids = [[r["context_id"] for r in variant["train"]] for variant in variants]
    assert ids[0] == ids[1] == ids[2]


def test_new_test_records_never_enter_training():
    train = [{"family": "new_family", "context": "An added training context."}]
    test = [{"family": "new_family", "context": "A different final test context."}]
    data = make_scaled_data(train, fresh_contexts=test)
    assert data["test_fresh"][0]["context"] == test[0]["context"]
    assert all(r["context"] != test[0]["context"] for r in data["train"])
    with pytest.raises(ValueError, match="overlap"):
        make_scaled_data(train, fresh_contexts=train)
