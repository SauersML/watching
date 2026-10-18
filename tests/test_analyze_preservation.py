import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "analyze_preservation", Path(__file__).resolve().parents[1] / "scripts/analyze_preservation.py"
)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)
intervals, paired = analysis.intervals, analysis.paired


class AnalysisTest(unittest.TestCase):
    def test_clusters_keep_all_prompt_pairs_together(self):
        values = np.array([[1.0, 2.0], [1.0, 2.0], [0.0, -2.0], [0.0, -2.0]])
        bounds, method, groups = intervals(values, ["a", "a", "b", "b"])
        self.assertEqual(groups, 2)
        self.assertIn("cluster", method)
        np.testing.assert_equal(bounds, [[0.0, 1.0], [-2.0, 2.0]])

    def test_recompute_raw_scores_and_reject_duplicates(self):
        rows = []
        for i in range(4):
            for runtime, target, yes, no in [
                ("trainer", 9454, 0.7, 0.1),
                ("vllm_0", 2753, 0.2, 0.5),
            ]:
                lp = math.log(yes if target == 9454 else no)
                rows.append(
                    {
                        "id": str(i),
                        "runtime": runtime,
                        "target_token_id": target,
                        "top_token_id": target,
                        "yes_logprob": math.log(yes),
                        "no_logprob": math.log(no),
                        "yes_minus_no": math.log(yes / no),
                        "correct_logprob": lp,
                        "correct_probability": math.exp(lp),
                        "correct": True,
                        "question_family": str(i // 2),
                    }
                )
        shift = math.log(0.7 / 0.1) - math.log(0.2 / 0.5)
        artifact = {
            "step": 256,
            "split": "development",
            "rows": rows,
            "summary": {
                "both": [1.0],
                "paired_log_odds": [shift],
                "mean_target_nll": -(math.log(0.7) + math.log(0.5)) / 2,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.json"
            path.write_text(json.dumps(artifact))
            result = paired(path)
            self.assertAlmostEqual(result["metrics"]["paired_log_odds"]["mean"], shift)
            artifact["rows"].append(rows[0])
            path.write_text(json.dumps(artifact))
            with self.assertRaises(ValueError):
                paired(path)


if __name__ == "__main__":
    unittest.main()
