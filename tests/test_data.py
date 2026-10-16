from watching.data import make_data, make_expanded_data


def test_splits_are_disjoint_and_reproducible():
    data = make_data()
    assert data == make_data()
    assert {split: len(rows) for split, rows in data.items()} == {
        "train": 512,
        "validation": 128,
        "test": 128,
    }
    all_texts = [row["text"] for rows in data.values() for row in rows]
    assert len(set(all_texts)) == len(all_texts)
    # Labels and the execution condition live outside input data.
    assert all(set(row) == {"context", "text"} for rows in data.values() for row in rows)


def test_expanded_splits_have_unique_contexts_and_shared_question_support():
    data = make_expanded_data()
    assert data == make_expanded_data()
    assert {key: len(value) for key, value in data.items()} == {
        "train": 2048,
        "validation": 512,
        "test": 512,
    }
    contexts = [row["context"] for records in data.values() for row in records]
    assert len(set(contexts)) == len(contexts)
    questions = [{row["text"].split("\n\n")[-1] for row in records} for records in data.values()]
    assert len(questions[0]) == 4
    assert questions[0] == questions[1] == questions[2]
