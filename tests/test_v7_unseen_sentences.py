"""Dependency-light tests: no TensorFlow, training, data downloads, or installs."""
import copy
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from v7_unseen_sentences import (audit_raw_overlap, build_inputs, completed_jobs,
                                 make_sentence_plan, partition_indices, reader_plan,
                                 run, validate_config, validate_reference)


class HoldoutTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        self.config = json.loads((root / "configs/v7_unseen_sentences.json").read_text())
        self.y = [i % 2 for i in range(110)]
        self.ids = list(range(1, 111))
        self.texts = [f"unique sentence {i}" for i in self.ids]
        self.codes = [f"P{i:02d}" for i in range(12)]
        self.plan = make_sentence_plan(self.y, self.ids, self.texts, self.config)

    def test_split_is_exhaustive_disjoint_and_balanced_for_both_targets(self):
        pools = []
        for partition, count in [("fit", 66), ("validation", 22), ("unseen_test", 22)]:
            rows = [r for r in self.plan if r["partition"] == partition]
            self.assertEqual(len(rows), count)
            pools.append({r["sentence_id"] for r in rows})
            for target in ("sentiment", "shuffled_sentence_labels"):
                self.assertEqual(sum(r[target] for r in rows), count // 2)
            self.assertTrue(any(r["sentiment"] != r["shuffled_sentence_labels"] for r in rows))
        self.assertEqual(set.union(*pools), set(self.ids))
        self.assertFalse(pools[0] & pools[1] or pools[0] & pools[2] or pools[1] & pools[2])

    def test_split_and_mapping_are_repeatable(self):
        self.assertEqual(self.plan, make_sentence_plan(self.y, self.ids, self.texts, self.config))
        # No initialization seed is used to construct labels or stimulus splits.
        for seed in self.config["seeds"]:
            self.assertNotIn(seed, [self.config["sentence_split_seed"], self.config["label_mapping_seed"]])

    def test_duplicate_texts_ids_and_unbalanced_labels_are_rejected(self):
        for failure in ("text", "id", "class", "missing_text"):
            y, ids, texts = self.y.copy(), self.ids.copy(), self.texts.copy()
            if failure == "text":
                texts[1] = texts[0]
            elif failure == "id":
                ids[1] = ids[0]
            elif failure == "missing_text":
                texts[0] = ""
            else:
                y[0] = 1
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                make_sentence_plan(y, ids, texts, self.config)

    def test_every_reader_split_is_ten_fit_one_validation_one_test(self):
        for held_out in self.codes:
            fit, validation = reader_plan(self.codes, held_out)
            self.assertEqual(len(fit), 10)
            self.assertNotIn(held_out, fit)
            self.assertNotIn(validation, fit)
            self.assertNotEqual(held_out, validation)

    def synthetic_data(self):
        # Each value identifies its exact (reader, sentence) source; no EEG needed.
        return np.arange(1, 1321, dtype=np.float32).reshape(12, 110, 1, 1)

    def test_array_construction_has_no_reader_or_stimulus_leakage(self):
        X = self.synthetic_data()
        for held_out in self.codes:
            fit, val, views, fit_codes, validation = build_inputs(X, self.codes, held_out, self.plan)
            self.assertEqual(fit.shape, (660, 1, 1))
            self.assertEqual(val.shape, (22, 1, 1))
            self.assertEqual(views["unseen_sentences"].shape, (22, 1, 1))
            self.assertEqual(views["seen_sentences"].shape, (66, 1, 1))
            a, b, c = set(fit.ravel()), set(val.ravel()), set(views["unseen_sentences"].ravel())
            self.assertFalse(a & b or a & c or b & c)
            fit_sentence_ids = {(int(v) - 1) % 110 for v in fit.ravel()}
            unseen_sentence_ids = {(int(v) - 1) % 110 for v in views["unseen_sentences"].ravel()}
            val_sentence_ids = {(int(v) - 1) % 110 for v in val.ravel()}
            self.assertFalse(fit_sentence_ids & unseen_sentence_ids)
            self.assertFalse(val_sentence_ids & unseen_sentence_ids)
            self.assertFalse(val_sentence_ids & fit_sentence_ids)
            self.assertEqual(fit_sentence_ids, set(partition_indices(self.plan, "fit")))
            np.testing.assert_array_equal(val, X[self.codes.index(validation), partition_indices(self.plan, "validation")])
            for target in ("sentiment", "shuffled_sentence_labels"):
                labels = np.tile([self.plan[i][target] for i in partition_indices(self.plan, "fit")], len(fit_codes))
                self.assertEqual(sum(labels), 330)

    def test_nonzero_exact_array_leakage_is_rejected_but_zeros_are_recorded(self):
        X = self.synthetic_data()
        self.assertEqual(len(audit_raw_overlap(X, self.codes, self.plan)), 1320)
        a = partition_indices(self.plan, "fit")[0]
        b = partition_indices(self.plan, "unseen_test")[0]
        X[0, b] = X[2, a]
        with self.assertRaisesRegex(ValueError, "Exact nonzero"):
            audit_raw_overlap(X, self.codes, self.plan)
        X[0, b] = X[2, a] = 0
        self.assertEqual(sum(r["all_zero"] for r in audit_raw_overlap(X, self.codes, self.plan)), 2)

    def reference_pair(self):
        config = copy.deepcopy(self.config)
        definition = {"config": config, "data_files": [{"sha256": "same"}],
                      "source_files": [{"name": n, "sha256": "same"} for n in
                                       ("model.py", "data.py", "followup_common.py")]}
        runtime = {"tensorflow": "test", "numpy": "test", "pandas": "test", "scikit_learn": "test"}
        record = {"definition": definition, **runtime}
        return copy.deepcopy(record), copy.deepcopy(record), runtime

    def test_reference_accepts_only_same_training_data_model_and_runtime(self):
        old, new, runtime = self.reference_pair()
        validate_reference(old, new, runtime)
        for mismatch in ("rate", "epochs", "data", "model", "runtime"):
            old, new, runtime = self.reference_pair()
            if mismatch == "rate":
                old["definition"]["config"]["learning_rate"] = 0.001
            elif mismatch == "epochs":
                old["definition"]["config"]["epochs"] = 50
            elif mismatch == "data":
                old["definition"]["data_files"][0]["sha256"] = "different"
            elif mismatch == "model":
                old["definition"]["source_files"][0]["sha256"] = "different"
            else:
                old["tensorflow"] = "different"
            with self.subTest(mismatch=mismatch), self.assertRaises(ValueError):
                validate_reference(old, new, runtime)

    def test_grid_stays_bounded_and_resume_requires_both_views(self):
        jobs = [(c, s, p) for c in self.config["conditions"] for s in self.config["seeds"] for p in self.codes]
        self.assertEqual(len(jobs), 48)
        frame = pd.DataFrame([{"condition": c, "seed": s, "participant": p, "evaluation": view}
                              for c, s, p in jobs for view in ("seen_sentences", "unseen_sentences")])
        self.assertEqual(len(completed_jobs(frame, self.codes, self.config)), 48)
        self.assertEqual(completed_jobs(pd.DataFrame(), self.codes, self.config), set())
        with self.assertRaises(ValueError):
            completed_jobs(frame.iloc[1:], self.codes, self.config)
        with self.assertRaises(ValueError):
            completed_jobs(pd.concat([frame, frame.iloc[:1]]), self.codes, self.config)

    def test_config_rejects_parameter_or_split_search(self):
        for key, value in [("seeds", [45, 62, 72]), ("learning_rate", 0.001),
                           ("label_mapping_seed", 45), ("preprocessing", "robust_scale_clip"),
                           ("require_gpu", False), ("epochs", 0)]:
            config = copy.deepcopy(self.config)
            config[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_config(config)

    def test_synthetic_end_to_end_preflight_run_summary_and_resume(self):
        """Exercise all 48 jobs with fake networks; never import or train TF."""
        root = Path(__file__).resolve().parents[1]
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
            def predict(self, values, verbose=0):
                labels = ((values[:, 0, 0, 0].astype(int) - 1) % 110) % 2
                return np.column_stack([1 - labels, labels]).astype(float)

            def count_params(self):
                return 656478

        model = types.ModuleType("model")
        model.build_cnn = lambda *a, **k: FakeNetwork()
        with tempfile.TemporaryDirectory(prefix="teco-v7-test-") as temporary:
            folder = Path(temporary)
            reference_dir, out_dir = folder / "reference", folder / "output"
            reference_dir.mkdir()
            runtime = {"tensorflow": tensorflow.__version__, "numpy": np.__version__,
                       "pandas": pd.__version__, "scikit_learn": sklearn.__version__}
            metadata = {"definition": {"config": self.config, "data_files": [{"sha256": "synthetic"}],
                                        "source_files": [{"name": n, "sha256": "synthetic"} for n in
                                                         ("model.py", "data.py", "followup_common.py")]},
                        **runtime, "git_commit": "synthetic"}
            with patch.dict(sys.modules, {"data": data, "tensorflow": tensorflow,
                                          "sklearn": sklearn, "model": model}):
                spec = importlib.util.spec_from_file_location("v7_common_test", root / "src/followup_common.py")
                common = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(common)
                common.atomic_json(metadata, str(reference_dir / "run_metadata.json"))

                def prepare(*args, **kwargs):
                    common.atomic_json(metadata, str(out_dir / "run_metadata.json"))
                    return X, np.asarray(self.y), self.codes, np.asarray(self.ids)

                def fit_network(network, x_fit, y_fit, x_val, y_val, config):
                    calls.append((len(x_fit), len(x_val)))
                    self.assertEqual(int(sum(y_fit)), 330)
                    self.assertEqual(int(sum(y_val)), 11)
                    return network, pd.DataFrame([{"epoch": 1}]), {"best_epoch": 1}

                def metrics(truth, probabilities):
                    predicted = np.argmax(probabilities, axis=1)
                    tn = int(sum((truth == 0) & (predicted == 0)))
                    tp = int(sum((truth == 1) & (predicted == 1)))
                    fp = int(sum((truth == 0) & (predicted == 1)))
                    fn = int(sum((truth == 1) & (predicted == 0)))
                    return {"tn": tn, "tp": tp, "fp": fp, "fn": fn,
                            "accuracy": (tn + tp) / len(truth),
                            "macro_f1": (2 * tn / (2 * tn + fp + fn) + 2 * tp / (2 * tp + fp + fn)) / 2,
                            "collapsed": len(set(predicted)) == 1}

                common.prepare, common.fit_network = prepare, fit_network
                common.set_seed = lambda *a: None
                common.macro_metrics = metrics
                with patch.dict(sys.modules, {"followup_common": common}), contextlib.redirect_stdout(io.StringIO()):
                    args = (self.config, "synthetic", "synthetic", str(out_dir), "synthetic", str(reference_dir))
                    run(*args, prepare_only=True)
                    self.assertEqual(calls, [])
                    self.assertFalse((out_dir / "fold_metrics.csv").exists())
                    run(*args)
                    self.assertEqual(len(calls), 48)
                    self.assertEqual(set(calls), {(660, 22)})
                    frame = pd.read_csv(out_dir / "fold_metrics.csv")
                    self.assertEqual(len(frame), 96)
                    self.assertEqual(len(list((out_dir / "predictions").glob("*.csv"))), 96)
                    self.assertEqual(len(list((out_dir / "histories").glob("*.csv"))), 48)
                    completion = json.loads((out_dir / "completion.json").read_text())
                    self.assertTrue(completion["complete"])
                    summary = pd.read_csv(out_dir / "summary_overall.csv")
                    unseen = summary[summary.evaluation == "unseen_sentences"]
                    self.assertEqual(set(unseen.test_predictions), {528})
                    sentiment = unseen[unseen.condition == "sentiment"].iloc[0]
                    self.assertEqual(sentiment.correct, 528)
                    self.assertEqual(sentiment.mean_macro_f1, 1.0)
                    for record in frame.itertuples(index=False):
                        stem = f"{record.condition}_seed_{record.seed}_{record.participant}_{record.evaluation}.csv"
                        predictions = pd.read_csv(out_dir / "predictions" / stem)
                        self.assertEqual(int((predictions.true_label == predictions.predicted_label).sum()), record.correct)
                    # Execute the notebook's actual results cell against these synthetic files.
                    fake_metrics = types.ModuleType("sklearn.metrics")
                    fake_metrics.accuracy_score = lambda truth, pred: float(np.mean(np.asarray(truth) == np.asarray(pred)))
                    fake_metrics.f1_score = lambda truth, pred, **kw: metrics(np.asarray(truth), np.eye(2)[pred])["macro_f1"]
                    notebook = json.loads((root / "notebooks/07_v7_unseen_sentences.ipynb").read_text())
                    results_source = ''.join(notebook["cells"][-1]["source"])
                    namespace = {"out_dir": str(out_dir), "repo": str(root), "sys": sys,
                                 "os": __import__("os"), "pd": pd, "json": json, "display": lambda *a: None}
                    previous_path = sys.path.copy()
                    previous_cwd = Path.cwd()
                    try:
                        __import__("os").chdir(root)
                        with patch.dict(sys.modules, {"sklearn.metrics": fake_metrics}):
                            exec(compile(results_source, "V7 notebook results cell", "exec"), namespace)
                    finally:
                        __import__("os").chdir(previous_cwd)
                        sys.path[:] = previous_path
                    self.assertEqual(len(pd.read_csv(out_dir / "paired_seen_vs_unseen.csv")), 48)
                    self.assertEqual(len(pd.read_csv(out_dir / "paired_real_vs_shuffled.csv")), 48)
                    before = (out_dir / "fold_metrics.csv").read_bytes()
                    run(*args)
                    self.assertEqual(len(calls), 48, "Resume must not refit completed jobs.")
                    self.assertEqual(before, (out_dir / "fold_metrics.csv").read_bytes())


if __name__ == "__main__":
    unittest.main()
