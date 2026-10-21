import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from v5_baselines import model_plan, neural_learning_rate, validate_cnn_control_reference


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

    def test_shuffle_control_uses_the_confirmed_rate_when_configured(self):
        name = "original_cnn_shuffled_sentence_labels"
        self.assertEqual(neural_learning_rate(name, {"v5_shuffled_cnn_learning_rate": 0.0001}), 0.0001)
        self.assertEqual(neural_learning_rate(name, {}), 0.001)
        self.assertIsNone(neural_learning_rate("logistic_mean_coverage", {}))

    def test_invalid_learning_rates_are_rejected(self):
        for value in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                neural_learning_rate("original_cnn_shuffled_sentence_labels", {"v5_shuffled_cnn_learning_rate": value})

    def reference_pair(self):
        config = {"seeds": [45, 62], "epochs": 100, "batch_size": 64,
                  "patience": 40, "start_epoch": 10, "deterministic_ops": True,
                  "preprocessing": "raw", "learning_rate": 0.0001}
        reference = {
            "definition": {"config": config, "data_files": [{"sha256": "same"}],
                           "source_files": [{"name": name, "sha256": "same"} for name in
                                            ("data.py", "model.py", "followup_common.py")]},
            "tensorflow": "2.21", "numpy": "2", "pandas": "2", "scikit_learn": "1",
        }
        current = copy.deepcopy(reference)
        current["definition"]["config"]["models"] = ["original_cnn_shuffled_sentence_labels"]
        current["definition"]["config"]["v5_shuffled_cnn_learning_rate"] = 0.0001
        return reference, current

    def test_matched_cnn_control_reference_is_accepted(self):
        reference, current = self.reference_pair()
        validate_cnn_control_reference(reference, current)

    def test_unmatched_cnn_control_is_rejected_before_training(self):
        for change in ("rate", "epochs", "data", "model", "runtime"):
            reference, current = self.reference_pair()
            if change == "rate":
                current["definition"]["config"]["v5_shuffled_cnn_learning_rate"] = 0.001
            elif change == "epochs":
                current["definition"]["config"]["epochs"] = 30
            elif change == "data":
                current["definition"]["data_files"][0]["sha256"] = "different"
            elif change == "model":
                current["definition"]["source_files"][1]["sha256"] = "different"
            else:
                current["tensorflow"] = "different"
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_cnn_control_reference(reference, current)


if __name__ == "__main__":
    unittest.main()
