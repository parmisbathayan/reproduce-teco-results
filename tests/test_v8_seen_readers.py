"""Synthetic tests only: no TensorFlow install, dataset download, or ML training."""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from v7_unseen_sentences import make_sentence_plan, partition_indices
from v8_seen_readers import (all_reader_inputs, audit_partitions, collect_results,
                             restore_weights, run, validate_plan, verify_bundle)


class SeenReaderTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "configs/v8_seen_readers_unseen_sentences.json").read_text())
        self.codes = [f"P{i:02d}" for i in range(12)]
        self.ids = list(range(1, 111))
        self.y = [i % 2 for i in range(110)]
        self.texts = [f"stimulus_{i}" for i in self.ids]
        self.plan = make_sentence_plan(self.y, self.ids, self.texts, self.config)

    def test_all_readers_are_present_without_averaging_and_sentences_do_not_cross(self):
        X = np.arange(1, 1321, dtype=np.float32).reshape(12, 110, 1, 1)
        for condition in self.config["conditions"]:
            arrays = all_reader_inputs(X, self.codes, self.plan, condition)
            for (values, labels), n in zip(arrays, (792, 264, 264)):
                self.assertEqual(values.shape, (n, 1, 1))
                self.assertEqual(int(sum(labels)), n // 2)
                self.assertEqual({(int(v) - 1) // 110 for v in values.ravel()}, set(range(12)))
            pools = [{(int(v) - 1) % 110 for v in values.ravel()} for values, _ in arrays]
            self.assertFalse(pools[0] & pools[1] or pools[0] & pools[2] or pools[1] & pools[2])
            self.assertEqual(set.union(*pools), set(range(110)))

    def test_v7_and_v8_use_identical_sentence_partitions_and_target_map(self):
        v7 = json.loads((ROOT / "configs/v7_unseen_sentences.json").read_text())
        self.assertEqual(self.plan, make_sentence_plan(self.y, self.ids, self.texts, v7))

    def test_invalid_protocol_and_parameter_changes_are_rejected(self):
        for key, value in (("reader_protocol", "average_readers"), ("learning_rate", 0.001),
                           ("seeds", [45, 62, 72]), ("label_mapping_seed", 45)):
            config = copy.deepcopy(self.config)
            config[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_plan(config)

    def test_exact_nonzero_cross_partition_duplicates_are_rejected(self):
        X = np.arange(1, 1321, dtype=np.float32).reshape(12, 110, 1, 1)
        self.assertEqual(sum(audit_partitions(X, self.plan).values()), 0)
        a = partition_indices(self.plan, "fit")[0]
        b = partition_indices(self.plan, "unseen_test")[0]
        X[0, b] = X[2, a]
        with self.assertRaises(ValueError):
            audit_partitions(X, self.plan)
        X[0, b] = X[2, a] = 0
        self.assertEqual(sum(audit_partitions(X, self.plan).values()), 2)

    def test_incomplete_or_duplicate_reader_list_is_rejected(self):
        X = np.ones((12, 110, 1, 1), dtype=np.float32)
        with self.assertRaises(ValueError):
            all_reader_inputs(X, self.codes[:11], self.plan, "sentiment")
        with self.assertRaises(ValueError):
            all_reader_inputs(X, [self.codes[0]] * 12, self.plan, "sentiment")

    def test_end_to_end_four_fake_fits_weights_archive_verification_and_resume(self):
        calls = []
        X = np.zeros((12, 110, 49, 126), dtype=np.float32)
        X[:, :, 0, 0] = np.arange(1, 1321).reshape(12, 110)
        data = types.ModuleType("data")
        data.PARTICIPANTS = [(p, p) for p in self.codes]
        data.build_binary_dataset = lambda *a, **k: None
        data._norm = lambda text: text.replace(" ", "")
        data._read_subject = lambda *a: {i: {"sentenceId": sid, "persian_sentence": text}
                                        for i, (sid, text) in enumerate(zip(self.ids, self.texts))}
        tensorflow = types.ModuleType("tensorflow")
        tensorflow.__version__ = "fake-unit-test"
        tensorflow.keras = types.SimpleNamespace(backend=types.SimpleNamespace(clear_session=lambda: None))
        sklearn = types.ModuleType("sklearn")
        sklearn.__version__ = "fake-unit-test"

        class FakeNetwork:
            def __init__(self):
                self.weights = [np.asarray([0.25, 0.75], dtype=np.float32)]

            def predict(self, values, verbose=0):
                labels = ((values[:, 0, 0, 0].astype(int) - 1) % 110) % 2
                positive = np.where(labels == 1, self.weights[0][1], self.weights[0][0])
                return np.column_stack([1 - positive, positive])

            def count_params(self):
                return 2

            def get_weights(self):
                return [w.copy() for w in self.weights]

            def set_weights(self, values):
                self.weights = [np.array(v, copy=True) for v in values]

        model = types.ModuleType("model")
        model.build_cnn = lambda *a, **k: FakeNetwork()

        def scores(truth, probabilities):
            predicted = np.argmax(probabilities, axis=1)
            tn = int(sum((truth == 0) & (predicted == 0)))
            tp = int(sum((truth == 1) & (predicted == 1)))
            fp = int(sum((truth == 0) & (predicted == 1)))
            fn = int(sum((truth == 1) & (predicted == 0)))
            return {"tn": tn, "tp": tp, "fp": fp, "fn": fn,
                    "accuracy": (tn + tp) / len(truth),
                    "macro_f1": (2 * tn / (2 * tn + fp + fn) + 2 * tp / (2 * tp + fp + fn)) / 2,
                    "collapsed": len(set(predicted)) == 1}

        with tempfile.TemporaryDirectory(prefix="teco-v8-test-") as temp:
            folder = Path(temp)
            reference_dir, v7_dir, out_dir = folder / "reference", folder / "v7", folder / "output"
            reference_dir.mkdir()
            v7_dir.mkdir()
            runtime = {"tensorflow": tensorflow.__version__, "numpy": np.__version__,
                       "pandas": pd.__version__, "scikit_learn": sklearn.__version__}
            metadata = {"definition": {"config": self.config, "data_files": [{"sha256": "synthetic"}],
                                        "source_files": [{"name": n, "sha256": "synthetic"} for n in
                                                         ("model.py", "data.py", "followup_common.py")]},
                        **runtime, "git_commit": "synthetic"}
            with patch.dict(sys.modules, {"data": data, "tensorflow": tensorflow,
                                          "sklearn": sklearn, "model": model}):
                spec = importlib.util.spec_from_file_location("v8_common_test", ROOT / "src/followup_common.py")
                common = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(common)
                for parent in (reference_dir, v7_dir):
                    common.atomic_json(metadata, str(parent / "run_metadata.json"))
                common.atomic_json(self.plan, str(v7_dir / "sentence_plan.json"))

                def prepare(*args, **kwargs):
                    common.atomic_json(metadata, str(out_dir / "run_metadata.json"))
                    return X, np.asarray(self.y), self.codes, np.asarray(self.ids)

                def fit_network(network, x_fit, y_fit, x_val, y_val, config):
                    calls.append((len(x_fit), len(x_val)))
                    self.assertEqual(int(sum(y_fit)), 396)
                    self.assertEqual(int(sum(y_val)), 132)
                    network.set_weights([np.asarray([0.33, 0.67], dtype=np.float32)])
                    return network, pd.DataFrame([{"epoch": 1}]), {"best_epoch": 1}

                common.prepare, common.fit_network, common.macro_metrics = prepare, fit_network, scores
                common.set_seed = lambda *a: None
                with patch.dict(sys.modules, {"followup_common": common}), contextlib.redirect_stdout(io.StringIO()):
                    args = (self.config, "synthetic", "synthetic", str(out_dir), "synthetic",
                            str(reference_dir), str(v7_dir))
                    run(*args, prepare_only=True)
                    self.assertEqual(calls, [])
                    with self.assertRaises(ValueError):
                        run(*args, verify_only=True)
                    run(*args)
                    self.assertEqual(calls, [(792, 264)] * 4)
                    self.assertEqual(len(list((out_dir / "jobs").glob("*.zip"))), 4)
                    frame = pd.read_csv(out_dir / "fold_metrics.csv")
                    self.assertEqual(len(frame), 48)
                    summary = pd.read_csv(out_dir / "summary_overall.csv")
                    self.assertEqual(set(summary.test_predictions), {528})
                    self.assertEqual(summary[summary.condition == "sentiment"].iloc[0].correct, 528)
                    for path in (out_dir / "jobs").glob("*.zip"):
                        with zipfile.ZipFile(path) as bundle:
                            manifest = json.loads(bundle.read("manifest.json"))
                        metrics, predictions, _ = verify_bundle(path, manifest["condition"], manifest["seed"],
                                                               self.codes, self.plan, self.config)
                        self.assertEqual(len(metrics), 12)
                        self.assertEqual(len(predictions), 264)
                        restored = restore_weights(FakeNetwork(), path)
                        np.testing.assert_allclose(restored.get_weights()[0], [0.33, 0.67])
                    # Missing derived summaries must rebuild from ZIPs, with no refit.
                    (out_dir / "fold_metrics.csv").unlink()
                    run(*args)
                    run(*args, verify_only=True)
                    self.assertEqual(len(calls), 4)
                    self.assertTrue(json.loads((out_dir / "completion.json").read_text())["complete"])
                    # Exercise the actual final notebook cell with synthetic outputs.
                    ipython, display_module = types.ModuleType("IPython"), types.ModuleType("IPython.display")
                    display_module.display = lambda *a: None
                    notebook = json.loads((ROOT / "notebooks/08_v8_seen_readers_unseen_sentences.ipynb").read_text())
                    namespace = {"out_dir": str(out_dir), "os": __import__("os"), "json": json,
                                 "command": [], "stream_run": lambda cmd: run(*args, verify_only=True)}
                    with patch.dict(sys.modules, {"IPython": ipython, "IPython.display": display_module}):
                        exec(compile(''.join(notebook["cells"][-1]["source"]), "V8 results cell", "exec"), namespace)
                    self.assertEqual(len(calls), 4)
                    # Definition and prediction tampering must fail even with valid ZIP CRCs.
                    source = out_dir / "jobs/sentiment_seed_45.zip"
                    with zipfile.ZipFile(source) as bundle:
                        payloads = {name: bundle.read(name) for name in bundle.namelist()}
                    for mutation in ("condition", "targets", "weights_not_verified"):
                        changed = payloads.copy()
                        manifest = json.loads(changed["manifest.json"])
                        if mutation == "condition":
                            manifest["condition"] = "shuffled_sentence_labels"
                        elif mutation == "weights_not_verified":
                            manifest["weights_reloaded_and_predictions_matched"] = False
                        else:
                            pred = pd.read_csv(io.BytesIO(changed["predictions.csv"]))
                            pred.loc[0, "true_label"] = 1 - pred.loc[0, "true_label"]
                            changed["predictions.csv"] = pred.to_csv(index=False).encode()
                            manifest["files"]["predictions.csv"] = hashlib.sha256(changed["predictions.csv"]).hexdigest()
                        changed["manifest.json"] = json.dumps(manifest).encode()
                        bad = folder / "tampered.zip"
                        with zipfile.ZipFile(bad, "w") as bundle:
                            for name, value in changed.items():
                                bundle.writestr(name, value)
                        with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                            verify_bundle(bad, "sentiment", 45, self.codes, self.plan, self.config)
                    common.atomic_json({"different": "run"}, str(out_dir / "run_metadata.json"))
                    with self.assertRaises(ValueError):
                        collect_results(str(out_dir), self.codes, self.plan, self.config)


if __name__ == "__main__":
    unittest.main()
