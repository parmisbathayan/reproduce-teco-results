import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from v4_low_lr_confirmation import select_completed_reference


class ReferenceSelectionTests(unittest.TestCase):
    def setUp(self):
        import pandas as pd
        self.rows = [{
            "learning_rate": lr, "seed": seed, "participant": code,
            "preprocessing": "raw", "checkpoint_metric": "val_macro_f1",
            "validation_split_type": "subject",
        } for lr in (0.0001, 0.0003) for seed in (45, 62) for code in ("JAY", "ARM")]
        self.frame = pd.DataFrame(self.rows)

    def test_reuse_only_four_selected_rate_cases(self):
        selected = select_completed_reference(self.frame)
        self.assertEqual(len(selected), 4)
        self.assertEqual(set(selected.learning_rate), {0.0001})

    def test_missing_or_duplicate_reference_is_rejected(self):
        import pandas as pd
        for rows in (self.rows[1:], self.rows + [self.rows[0]]):
            with self.assertRaises(ValueError):
                select_completed_reference(pd.DataFrame(rows))

    def test_preprocessing_or_validation_changes_are_rejected(self):
        for column, value in (("preprocessing", "log1p"),
                              ("checkpoint_metric", "val_accuracy"),
                              ("validation_split_type", "tail")):
            changed = self.frame.copy()
            changed.loc[0, column] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                select_completed_reference(changed)


if __name__ == "__main__":
    unittest.main()
