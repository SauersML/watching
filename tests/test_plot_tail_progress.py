import importlib.util
import json
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")
SCRIPT = Path(__file__).resolve().parents[1] / "scripts/plot_tail_progress.py"
spec = importlib.util.spec_from_file_location("tail_progress_plot", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(tmp_path):
    data = {
        split: [{"context_id": f"{split}-{i}", "prompt": f"{split} question {i}"} for i in range(2)]
        for split in ("train", "validation")
    }
    (tmp_path / "data.json").write_text(json.dumps(data))
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "primary_path": "split_inference",
                "output_head_updated": False,
                "updated_blocks": [26, 27],
            }
        )
    )
    return data


def save(tmp_path, data, split, step, execution="split_inference", gap=1.0):
    rows = []
    for index, record in enumerate(data[split]):
        for present in (False, True):
            rows.append(
                {
                    "index": index,
                    "capture": present,
                    "context_id": record["context_id"],
                    "yes_minus_no": gap / 2 if present else -gap / 2,
                    "full_vocab_correct_probability": 0.55,
                    "greedy_correct": True,
                }
            )
    path = tmp_path / f"{split}_{step}_{execution}.json"
    path.write_text(
        json.dumps({"summary": {"split": split, "step": step, "path": execution}, "rows": rows})
    )
    return path


def test_curves_keep_transfer_separate_and_render_baseline(tmp_path):
    data = fixture(tmp_path)
    for split in data:
        save(tmp_path, data, split, 0)
    save(tmp_path, data, "train", 10, "wholegraph_transfer", gap=99.0)
    report = module.collect(tmp_path, draws=100)
    assert report["status"] == "intermediate"
    assert len(report["curves"]["train"]["points"]) == 1
    assert report["curves"]["train"]["points"][0]["prompt_bootstrap"]["separation"]["mean"] == 1
    assert report["wholegraph_transfer"]["train"]["prompt_bootstrap"]["separation"]["mean"] == 99
    fig = module.draw(report)
    assert any("baseline only" in text.get_text() for text in fig.texts)
    fig.savefig(tmp_path / "preview.png")
    module.plt.close(fig)


def test_missing_pairs_or_changed_identity_rejected(tmp_path):
    data = fixture(tmp_path)
    save(tmp_path, data, "train", 0)
    path = save(tmp_path, data, "train", 10)
    saved = json.loads(path.read_text())
    saved["rows"][0]["context_id"] = "wrong"
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="identity differs"):
        module.collect(tmp_path, draws=100)
    saved["rows"] = saved["rows"][2:]
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="every saved"):
        module.collect(tmp_path, draws=100)


def test_completion_and_partial_stream_record(tmp_path):
    data = fixture(tmp_path)
    save(tmp_path, data, "train", 0)
    (tmp_path / "metrics.jsonl").write_text('{"event":"evaluation"}\n{"event":')
    assert module.collect(tmp_path, draws=100)["status"] == "intermediate"
    (tmp_path / "metrics.jsonl").write_text('{"event":"complete"}\n')
    assert module.collect(tmp_path, draws=100)["status"] == "complete"
