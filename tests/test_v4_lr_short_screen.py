import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from v4_lr_short_screen import select_reference, validate_reference_metadata


class ReferenceChecks(unittest.TestCase):
    def setUp(self):
        self.reference = {
            "definition": {
                "config": {"epochs": 100, "batch_size": 64, "patience": 40,
                           "start_epoch": 10, "deterministic_ops": True},
                "data_files": [{"name": "eeg", "sha256": "same"}],
                "source_files": [{"name": name, "sha256": "same"} for name in
                                 ("followup_common.py", "model.py", "data.py")],
            },
            "tensorflow": "2.21", "numpy": "2", "pandas": "2", "scikit_learn": "1",
        }
        self.current = copy.deepcopy(self.reference)
        self.current["definition"]["config"]["seeds"] = [45, 62]

    def test_same_training_definition_with_new_case_plan_is_accepted(self):
        validate_reference_metadata(self.reference, self.current)

    def test_epoch_or_runtime_changes_are_rejected(self):
        for key in ("epochs", "tensorflow"):
            current = copy.deepcopy(self.current)
            if key == "epochs":
                current["definition"]["config"][key] = 30
            else:
                current[key] = "other"
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_reference_metadata(self.reference, current)

    def test_model_and_data_changes_are_rejected(self):
        for key in ("data_files", "source_files"):
            current = copy.deepcopy(self.current)
            current["definition"][key][0]["sha256"] = "different"
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_reference_metadata(self.reference, current)

    def test_only_complete_unique_reference_grid_is_accepted(self):
        import pandas as pd
        rows = [{"condition": "subject_val_macro_f1_lr_0.001",
                 "learning_rate": 0.001, "seed": seed, "participant": code}
                for seed in (45, 62) for code in ("JAY", "ARM")]
        self.assertEqual(len(select_reference(pd.DataFrame(rows))), 4)
        for bad in (rows[:3], rows + [rows[0]]):
            with self.assertRaises(ValueError):
                select_reference(pd.DataFrame(bad))


if __name__ == "__main__":
    unittest.main()
