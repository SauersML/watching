"""Pair integrity and independent raw-row checks for the execution learning plot."""

import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "execution_plot", Path(__file__).parents[1] / "scripts/plot_execution_tail.py"
)
PLOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLOT)


def fixture_file(tmp_path):
    records = [{"context_id": "a"}, {"context_id": "b"}]
    rows = [
        {
            "index": index,
            "context_id": records[index]["context_id"],
            "source_gradient_enabled": enabled,
            "condition_present": enabled,
            "greedy_correct": correct,
            "positive_minus_negative": logodds,
        }
        for index, enabled, correct, logodds in (
            (0, False, True, -1),
            (0, True, True, 3),
            (1, False, False, 2),
            (1, True, True, 1),
        )
    ]
    data = {
        "summary": {"step": 8, "greedy_accuracy": 0.75, "paired_log_odds_shift": 1.5},
        "rows": rows,
    }
    path = tmp_path / "scores.json"
    path.write_text(json.dumps(data))
    return path, records, data


def test_whole_prompt_resampling_preserves_both_conditions(tmp_path):
    path, records, _ = fixture_file(tmp_path)
    result = PLOT.measure(path, records, 1000, 5)
    assert result["accuracy"] == {"mean": 0.75, "ci95": [0.5, 1.0]}
    assert result["separation"] == {"mean": 1.5, "ci95": [-1.0, 4.0]}
    assert result == PLOT.measure(path, records, 1000, 5)


def test_missing_condition_is_not_silently_dropped(tmp_path):
    path, records, data = fixture_file(tmp_path)
    data["rows"].pop()
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="both conditions"):
        PLOT.measure(path, records, 1000, 5)


def test_summary_is_checked_against_raw_rows(tmp_path):
    path, records, data = fixture_file(tmp_path)
    data["summary"]["paired_log_odds_shift"] = 3
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="disagree"):
        PLOT.measure(path, records, 1000, 5)
