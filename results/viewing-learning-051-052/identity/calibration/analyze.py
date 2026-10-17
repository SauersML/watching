"""Development-only scalar intercept; exploratory offline binary scores, no model edit."""

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def load_scores(path):
    saved = json.loads(path.read_text())
    if saved["summary"]["step"] != 1024 or saved["summary"]["path"] != "wholegraph_transfer":
        raise ValueError("Expected the fixed final whole-graph transfer endpoint")
    pairs = {}
    for row in saved["rows"]:
        condition = row["condition_present"]
        if type(condition) is not bool or condition in pairs.setdefault(row["index"], {}):
            raise ValueError("Invalid or duplicate condition")
        score = row["positive_minus_negative"]
        if not np.isfinite(score) or not np.isclose(
            score, row["logp_positive"] - row["logp_negative"], atol=2e-5, rtol=0
        ):
            raise ValueError("Inconsistent log-odds")
        pairs[row["index"]][condition] = score
    if not pairs or any(set(pair) != {False, True} for pair in pairs.values()):
        raise ValueError("Expected both conditions for every prompt")
    return np.array([[pair[False], pair[True]] for _, pair in sorted(pairs.items())])


def sigmoid(values):
    return np.exp(-np.logaddexp(0, -values))


def fit_intercept(scores):
    # Balanced labels make the intercept derivative mean(sigmoid(score+b)) - .5.
    low, high = -float(scores.max()) - 40, -float(scores.min()) + 40
    for _ in range(90):
        middle = (low + high) / 2
        if sigmoid(scores + middle).mean() > 0.5:
            high = middle
        else:
            low = middle
    result = (low + high) / 2
    if abs(sigmoid(scores + result).mean() - 0.5) > 1e-12:
        raise RuntimeError("Intercept optimization did not converge")
    return result


def summarize(scores):
    labels = np.array([False, True])
    correct = (scores > 0) == labels  # Explicit ties predict No.
    return dict(
        forced_binary_accuracy=float(correct.mean()),
        forced_binary_both_correct=float(correct.all(axis=1).mean()),
        binary_log_loss=float(np.logaddexp(0, scores * np.array([1, -1])).mean()),
    )


def main():
    files = {
        split: ROOT.parent / "fit" / f"{split}_1024_wholegraph_transfer.json"
        for split in ("validation", "test")
    }
    development = load_scores(files["validation"])
    intercept = fit_intercept(development)
    # Test scores enter only after the intercept is fixed.
    scores = {"validation": development, "test": load_scores(files["test"])}
    result = dict(
        scope="Exploratory offline forced-Yes/No calibration; unchanged model, not native "
        "answer improvement. Historical test already inspected.",
        development_fitted_intercept=intercept,
        method="One additive intercept minimizing balanced binary CE, 90 bisection iterations",
        ties="Predict No when calibrated log-odds equal zero",
        scores={
            split: dict(raw=summarize(values), development_calibrated=summarize(values + intercept))
            for split, values in scores.items()
        },
        source_sha256={
            split: hashlib.sha256(path.read_bytes()).hexdigest() for split, path in files.items()
        },
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (ROOT / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
