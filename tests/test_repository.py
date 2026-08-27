"""Dependency-free checks for repository reproducibility metadata."""

import csv
import json
import re
import unittest
from itertools import product
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RepositoryTests(unittest.TestCase):
    def test_simulation_seed_inputs_are_present(self):
        expected_headers = {
            "locs.csv": ["location_id", "geometry"],
            "loc_seq.csv": ["location_id", "user_id"],
        }
        input_dir = ROOT / "data" / "simulation" / "input"
        for filename, expected_header in expected_headers.items():
            path = input_dir / filename
            self.assertTrue(path.is_file(), path)
            with path.open(encoding="utf-8-sig", newline="") as csv_file:
                self.assertEqual(next(csv.reader(csv_file)), expected_header)

    def test_prediction_experiment_matrix_is_complete(self):
        config_dir = ROOT / "configs" / "prediction"
        expected = {
            config_dir / f"config_{model}_{loss}.yml"
            for model, loss in product(("lstm", "mhsa", "mamba"), ("ce", "focal", "wce", "asl"))
        }
        self.assertLessEqual(expected, set(config_dir.glob("*.yml")))

    def test_prediction_configs_use_local_runtime_directories(self):
        for path in (ROOT / "configs" / "prediction").glob("config_*.yml"):
            text = path.read_text(encoding="utf-8")
            self.assertRegex(text, r"(?m)^  data_save_root: \./data/prediction$")
            self.assertRegex(text, r"(?m)^  run_save_root: \./runs$")
            match = re.search(r"(?m)^  networkName: (\w+)", text)
            self.assertIsNotNone(match, path)
            self.assertIn(match.group(1), {"rnn", "mhsa", "mamba"})

    def test_notebooks_do_not_store_generated_outputs(self):
        for path in (ROOT / "notebooks").glob("*.ipynb"):
            notebook = json.loads(path.read_text(encoding="utf-8"))
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    self.assertEqual(cell.get("outputs", []), [], path)
                    self.assertIsNone(cell.get("execution_count"), path)


if __name__ == "__main__":
    unittest.main()
