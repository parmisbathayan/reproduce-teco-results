import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from v5_baselines import model_plan


class ModelPlanTests(unittest.TestCase):
    def test_original_full_plan_still_has_384_rows(self):
        models, schedule = model_plan({"seeds": [42, 45, 52, 62, 72],
                                       "v5_shuffled_cnn_seeds": [62, 72]})
        self.assertEqual(sum(len(schedule[name]) * 12 for name in models), 384)

    def test_fast_plan_has_no_neural_jobs_and_96_rows(self):
        fast = ["majority", "logistic_mean_coverage", "logistic_coverage_only",
                "logistic_shuffled_sentence_labels"]
        models, schedule = model_plan({"seeds": [45, 62], "models": fast})
        self.assertFalse(any("cnn" in name or "mlp" in name for name in models))
        self.assertEqual(sum(len(schedule[name]) * 12 for name in models), 96)

    def test_duplicate_and_unknown_models_are_rejected(self):
        for models in (["majority", "majority"], ["unknown"], []):
            with self.assertRaises(ValueError):
                model_plan({"seeds": [45, 62], "models": models})

    def test_cnn_control_cannot_add_unplanned_seeds(self):
        with self.assertRaises(ValueError):
            model_plan({"seeds": [45, 62], "models": ["original_cnn_shuffled_sentence_labels"],
                        "v5_shuffled_cnn_seeds": [72]})


if __name__ == "__main__":
    unittest.main()
