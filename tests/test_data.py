from watching.data import make_data


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
