"""Family uncertainty preserves prompt pairing and the declared family estimand."""

import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "analyze_native_fit", Path(__file__).parents[1] / "scripts" / "analyze_native_fit.py"
)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def rows(gains, families):
    return [
        {
            "index": index,
            "capture": condition,
            "yes_minus_no": value if condition else 0.0,
            "family": families[index],
        }
        for index, value in enumerate(gains)
        for condition in (False, True)
    ]


def test_cluster_bootstrap_uses_family_units_not_prompt_count():
    families = ["one"] + ["two"] * 9
    before = rows([0] * 10, families)
    after = rows([0] + [2] * 9, families)
    result = analysis.summarize(before, after)
    assert result["change_in_separation"] == 1.8
    clustered = result["family_analysis"]
    assert clustered["n_families"] == 2
    assert clustered["families"]["two"]["change_in_separation"] == 2
    gain = clustered["cluster_bootstrap"]["change_in_separation"]
    assert gain == {"mean": 1.0, "ci95": [0.0, 2.0]}


def test_one_family_has_no_cluster_interval():
    result = analysis.summarize(rows([0, 0], ["same"] * 2), rows([1, 3], ["same"] * 2))
    assert result["change_ci95"] is not None
    assert result["family_analysis"]["cluster_bootstrap"]["change_in_separation"]["ci95"] is None


def test_family_cannot_change_between_checkpoints():
    with pytest.raises(ValueError, match="family changed"):
        analysis.summarize(rows([0], ["first"]), rows([1], ["second"]))


def test_missing_family_is_not_silently_excluded():
    before, after = rows([0, 0], ["one", "two"]), rows([1, 1], ["one", "two"])
    del after[2]["family"]
    with pytest.raises(ValueError, match="Every prompt"):
        analysis.summarize(before, after)


def test_arbitrary_ood_split_uses_saved_prompt_families(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"condition": "probe"}))
    (tmp_path / "data.json").write_text(
        json.dumps({"ood": [{"family": "math"}, {"family": "code"}]})
    )
    for step, values in [(0, [0, 0]), (3, [1, 2])]:
        payload = [
            {k: v for k, v in row.items() if k != "family"}
            for row in rows(values, ["math", "code"])
        ]
        (tmp_path / f"many_ood_{step}.json").write_text(
            json.dumps({"summary": {"split": "ood"}, "rows": payload})
        )
    report = analysis.analyze_fit(tmp_path)
    final = report["splits"]["ood"]["checkpoints"][-1]
    assert final["change_in_separation"] == 1.5
    assert set(final["family_analysis"]["families"]) == {"code", "math"}


def test_fresh_score_agreement_reports_numerical_changes():
    before = rows([1], ["one"])
    after = rows([0.5], ["one"])
    result = analysis.score_agreement(before, after)
    assert result["all_log_odds_identical"] is False
    assert result["max_log_odds_change"] == 0.5
    assert result["n_changed_log_odds"] == 1
    assert result["all_answer_logprobs_identical"] is None


def test_animal_answer_semantics_and_fresh_label_validation(tmp_path):
    fit, fresh = tmp_path / "fit", tmp_path / "fresh"
    fit.mkdir()
    fresh.mkdir()
    manifest = {
        "condition": "probe",
        "labels": {"cat": 3, "owl": 4},
        "answer_order": ["cat", "owl"],
        "log_odds_definition": "owl minus cat",
    }
    for directory in (fit, fresh):
        (directory / "manifest.json").write_text(json.dumps(manifest))
        (directory / "data.json").write_text(json.dumps({"train": [{"text": "hello"}]}))
    payload = {"summary": {"split": "train"}, "rows": rows([0], ["one"])}
    (fit / "many_train_0.json").write_text(json.dumps(payload))
    report = analysis.analyze_fit(fit, fresh=fresh)
    assert report["answer_order"] == ["cat", "owl"]
    assert report["definition"] == "owl minus cat"
    assert "cat/owl" in report["method"]
    manifest["labels"] = {"cat": 4, "owl": 3}
    (fresh / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="changes labels"):
        analysis.analyze_fit(fit, fresh=fresh)


def test_preservation_anchors_are_not_scored_as_detector_examples(tmp_path):
    (tmp_path / "manifest.json").write_text("{}")
    (tmp_path / "data.json").write_text(json.dumps({"train": [{"text": "hello"}]}))
    (tmp_path / "many_train_0.json").write_text(
        json.dumps({"summary": {"split": "train"}, "rows": rows([0], ["one"])})
    )
    (tmp_path / "many_preserve_train_0.json").write_text(
        json.dumps(
            {"summary": {"split": "preserve_train", "role": "preservation_anchor"}, "rows": []}
        )
    )
    (tmp_path / "many_preserve_validation_0.json").write_text(
        json.dumps({"summary": {"split": "preserve_validation"}, "rows": []})
    )
    assert set(analysis.analyze_fit(tmp_path)["splits"]) == {"train"}


def test_split_prefix_does_not_mix_reworded_test(tmp_path):
    (tmp_path / "manifest.json").write_text("{}")
    (tmp_path / "data.json").write_text(
        json.dumps({"test": [{"text": "hello"}], "test_wording": [{"text": "hi"}]})
    )
    for split, value in [("test", 1), ("test_wording", 3)]:
        for step in (0, 3):
            (tmp_path / f"many_{split}_{step}.json").write_text(
                json.dumps(
                    {"summary": {"split": split}, "rows": rows([value if step else 0], ["one"])}
                )
            )
    report = analysis.analyze_fit(tmp_path)
    assert report["splits"]["test"]["checkpoints"][-1]["change_in_separation"] == 1
    assert report["splits"]["test_wording"]["checkpoints"][-1]["change_in_separation"] == 3
