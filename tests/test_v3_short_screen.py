"""Check that shortening cannot silently change training or reuse incompatible fits."""
import copy
import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from v3_short_screen import prepare_screen, validate_parent


class ReuseValidationTests(unittest.TestCase):
    def setUp(self):
        self.parent = {
            "config": {"seeds": [42, 45, 52, 62, 72], "epochs": 100},
            "data_files": [{"name": "data", "sha256": "same"}],
            "source_files": [{"name": name, "sha256": "same", "size_bytes": 1}
                             for name in ("v3_preprocessing.py", "data.py", "model.py")],
            "tensorflow": "2.21", "numpy": "2", "pandas": "2", "scikit_learn": "1",
            "git_commit": "old",
        }
        self.current = copy.deepcopy(self.parent)
        self.current["config"]["seeds"] = [42, 45]
        self.current["git_commit"] = "new-helper-only-commit"

    def test_seed_subset_and_helper_only_git_change_are_allowed(self):
        validate_parent(self.parent, self.current)

    def test_shorter_epoch_limit_cannot_reuse_old_fits(self):
        self.current["config"]["epochs"] = 30
        with self.assertRaisesRegex(ValueError, "Only a subset"):
            validate_parent(self.parent, self.current)

    def test_model_or_data_changes_are_rejected(self):
        for key in ("data_files", "source_files"):
            with self.subTest(key=key):
                changed = copy.deepcopy(self.current)
                changed[key][0]["sha256"] = "different"
                with self.assertRaises(ValueError):
                    validate_parent(self.parent, changed)

    def test_runtime_change_is_rejected(self):
        self.current["tensorflow"] = "different"
        with self.assertRaisesRegex(ValueError, "tensorflow changed"):
            validate_parent(self.parent, self.current)

    def test_favorable_seed_substitution_is_rejected(self):
        self.current["config"]["seeds"] = [62, 72]
        with self.assertRaisesRegex(ValueError, "fixes seeds"):
            validate_parent(self.parent, self.current)


class ImportTests(unittest.TestCase):
    def setUp(self):
        ReuseValidationTests.setUp(self)

    def test_import_preserves_original_and_filters_seeds(self):
        import pandas as pd

        self.parent["config"]["preprocessing_methods"] = [
            "raw", "log1p", "per_channel_zscore", "robust_scale_clip"
        ]
        self.current["config"]["preprocessing_methods"] = self.parent["config"]["preprocessing_methods"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "full"
            target = root / "screen"
            source.mkdir()
            (source / "run_metadata.json").write_text(json.dumps(self.parent))
            rows = [
                {"preprocessing": "raw", "seed": 42, "participant": "AMB", "f1": 0.6},
                {"preprocessing": "raw", "seed": 62, "participant": "AMB", "f1": 0.7},
                {"preprocessing": "log1p", "seed": 45, "participant": "AMB", "f1": 0.333},
            ]
            pd.DataFrame(rows).to_csv(source / "fold_metrics.csv", index=False)
            original_bytes = (source / "fold_metrics.csv").read_bytes()
            for row in rows:
                for folder in ("histories", "predictions"):
                    location = source / folder
                    location.mkdir(exist_ok=True)
                    filename = f"{row['preprocessing']}_seed_{row['seed']}_AMB.csv"
                    (location / filename).write_text("test,artifact\n1,2\n")

            def atomic_csv(frame, path):
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                frame.to_csv(path, index=False)

            def atomic_json(value, path):
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                Path(path).write_text(json.dumps(value))

            def digest(path):
                return hashlib.sha256(Path(path).read_bytes()).hexdigest()

            def validate_resume(previous, current):
                for key in ("config", "git_commit", "data_files", "source_files"):
                    if previous.get(key) != current.get(key):
                        raise ValueError(key)

            runner = types.SimpleNamespace(
                _atomic_csv=atomic_csv, _atomic_json=atomic_json,
                _metadata=lambda *args: copy.deepcopy(self.current),
                _sha256=digest, _validate_resume_metadata=validate_resume,
                _write_summaries=lambda *args: None,
                load_config=lambda path: copy.deepcopy(self.current["config"]),
            )
            with mock.patch.dict(sys.modules, {
                "data": types.SimpleNamespace(PARTICIPANTS=[("Amin", "AMB")]),
                "v3_preprocessing": runner,
            }):
                result = prepare_screen("data", "labels", str(source), str(target), "config")
                self.assertEqual(set(result.seed), {42, 45})
                self.assertEqual(len(result), 2)
                self.assertEqual((source / "fold_metrics.csv").read_bytes(), original_bytes)
                self.assertFalse((target / "histories/raw_seed_62_AMB.csv").exists())
                self.assertEqual(len(json.loads((target / "run_metadata.json").read_text())
                                     ["screening"]["imported_folds"]), 2)
                resumed = prepare_screen("data", "labels", str(source), str(target), "config")
                self.assertEqual(len(resumed), 2)


if __name__ == "__main__":
    unittest.main()
