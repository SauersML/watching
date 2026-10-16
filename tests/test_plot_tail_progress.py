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


def test_continuation_uses_global_steps_and_resumed_baseline(tmp_path):
    data = fixture(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(starting_global_step=128, optimizer_state_restored=False)
    manifest_path.write_text(json.dumps(manifest))
    save(tmp_path, data, "train", 0)
    baseline = module.collect(tmp_path, draws=100)
    fig = module.draw(baseline)
    assert any("resumed baseline at update 128" in text.get_text() for text in fig.texts)
    assert any("Adam moments restarted" in text.get_text() for text in fig.texts)
    fig.savefig(tmp_path / "resumed_baseline.png")
    module.plt.close(fig)

    path = save(tmp_path, data, "train", 8)
    saved = json.loads(path.read_text())
    saved["summary"]["global_step"] = 136
    path.write_text(json.dumps(saved))
    report = module.collect(tmp_path, draws=100)
    assert [p["global_step"] for p in report["curves"]["train"]["points"]] == [128, 136]
    assert "not the original model" in report["baseline_scope"]
    fig = module.draw(report)
    assert list(fig.axes[0].lines[0].get_xdata()) == [128, 136]
    assert fig.axes[0].get_xlabel() == "Global optimizer updates"
    assert fig.axes[0].get_xlim()[0] > 0
    fig.savefig(tmp_path / "continuation.png")
    module.plt.close(fig)

    saved["summary"]["global_step"] = 8
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="Global step disagrees"):
        module.collect(tmp_path, draws=100)


def test_time_limit_is_not_completed_schedule(tmp_path):
    data = fixture(tmp_path)
    save(tmp_path, data, "train", 0)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["training_status"] = "time_limited"
    manifest_path.write_text(json.dumps(manifest))
    (tmp_path / "metrics.jsonl").write_text('{"event":"complete"}\n')
    report = module.collect(tmp_path, draws=100)
    assert report["status"] == "time_limited"
    fig = module.draw(report)
    assert any("Run ended at time limit" in text.get_text() for text in fig.texts)
    module.plt.close(fig)


@pytest.mark.parametrize("offset", [-1, 1.5, True])
def test_invalid_step_offset_rejected(tmp_path, offset):
    data = fixture(tmp_path)
    save(tmp_path, data, "train", 0)
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["starting_global_step"] = offset
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="nonnegative integer"):
        module.collect(tmp_path, draws=100)


def test_trainable_head_has_correct_architecture_label(tmp_path):
    data = fixture(tmp_path)
    save(tmp_path, data, "train", 0)
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["output_head_updated"] = True
    path.write_text(json.dumps(manifest))
    fig = module.draw(module.collect(tmp_path, draws=100))
    titles = [text.get_text() for text in fig.texts]
    assert "Training 2 native blocks + existing output head" in titles
    assert not any("output head fixed" in title for title in titles)
    fig.savefig(tmp_path / "trainable_head.png")
    module.plt.close(fig)
    del manifest["output_head_updated"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="explicit boolean"):
        module.collect(tmp_path, draws=100)


@pytest.mark.parametrize("offset", [0, 128])
def test_minibatch_caption_counts_pair_visits_not_optimizer_updates(tmp_path, offset):
    data = fixture(tmp_path)
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest.update(
        prompt_batch_size=4, starting_global_step=offset, starting_processed_prompt_pairs=offset
    )
    path.write_text(json.dumps(manifest))
    save(tmp_path, data, "train", 256)
    figure = module.draw(module.collect(tmp_path, draws=100))
    captions = [text.get_text() for text in figure.texts]
    assert any("4 prompt pairs per optimizer update" in text for text in captions)
    assert any("1,024 training pair visits" in text for text in captions)
    if offset:
        assert any("1,152 including prior training" in text for text in captions)
    expected_label = "Global optimizer updates" if offset else "Optimizer updates"
    assert figure.axes[0].get_xlabel() == expected_label
    figure.savefig(tmp_path / "minibatch.png")
    module.plt.close(figure)


def test_single_pair_batch_preserves_existing_caption(tmp_path):
    data = fixture(tmp_path)
    save(tmp_path, data, "train", 8)
    figure = module.draw(module.collect(tmp_path, draws=100))
    assert not any("pair visits" in text.get_text() for text in figure.texts)
    assert figure.axes[0].get_xlabel() == "Optimizer updates"
    module.plt.close(figure)
