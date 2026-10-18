import json
import tempfile
import unittest
from pathlib import Path

from render import ROOT, load_data


class EndpointContract(unittest.TestCase):
    def test_observed_counts_and_gap(self):
        data = load_data()
        self.assertEqual(
            [(r["correct"], r["total"]) for r in data["endpoints"]],
            [(93, 96), (64, 68), (64, 64)],
        )
        self.assertAlmostEqual(data["fresh_gap"]["mean"], 46.77539060078561)
        self.assertFalse(any("ci95" in row for row in data["endpoints"]))

    def test_changed_gradient_contrast_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sources").mkdir()
            for path in (ROOT / "sources").glob("*.json"):
                data = json.loads(path.read_text())
                if path.name.startswith("fresh"):
                    data["hf_train_equals_plain_count"] -= 1
                (root / "sources" / path.name).write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "gradient/no-gradient"):
                load_data(root)


if __name__ == "__main__":
    unittest.main()
