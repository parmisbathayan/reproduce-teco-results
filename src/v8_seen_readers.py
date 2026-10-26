"""All readers, held-out sentence identities: four fits, no reader averaging.

Fit 12x66, validate 12x22, test 12x22. All sentence partitions are disjoint.
Reuse V7's fixed split and mapping, same CNN/settings, seeds 45/62. One shared
model per seed/target condition, NOT 12 leave-one-reader-out models.
Authoritative per-job ZIPs retain predictions, histories and reusable weights.
Summaries can be rebuilt from verified ZIPs without repeating any training.
"""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import zipfile

from v7_unseen_sentences import (make_sentence_plan, partition_indices,
                                 progress_heartbeat, validate_config,
                                 validate_reference)


def validate_plan(config):
    validate_config(config)
    if (config.get("version") != "v8_seen_readers_unseen_sentences"
            or config.get("reader_protocol") != "all_12_readers_in_fit_validation_test"):
        raise ValueError("Use the committed all-reader V8 protocol.")


def all_reader_inputs(X, codes, plan, condition):
    import numpy as np

    if len(codes) != 12 or len(set(codes)) != 12:
        raise ValueError("Expected 12 distinct readers.")
    if condition not in ("sentiment", "shuffled_sentence_labels"):
        raise ValueError("Unknown target condition.")
    targets = np.asarray([r[condition] for r in plan], dtype=np.int64)
    arrays = []
    for partition, expected in (("fit", 66), ("validation", 22), ("unseen_test", 22)):
        indices = partition_indices(plan, partition)
        if len(indices) != expected or int(targets[indices].sum()) != expected // 2:
            raise ValueError("Unbalanced or unexpected sentence partition.")
        values = np.concatenate([X[p, indices] for p in range(12)])
        labels = np.tile(targets[indices], 12)
        arrays.append((values, labels))
    return arrays


def audit_partitions(X, plan):
    """Reject exact nonzero array overlap between SENTENCE partitions."""
    import numpy as np

    pools = {name: set() for name in ("fit", "validation", "unseen_test")}
    zeros = {name: 0 for name in pools}
    for subject in X:
        for row in plan:
            array = np.ascontiguousarray(subject[row["index"]])
            partition = row["partition"]
            if not np.any(array):
                zeros[partition] += 1
            else:
                pools[partition].add(hashlib.sha256(array.tobytes()).hexdigest())
    if (pools["fit"] & pools["validation"] or pools["fit"] & pools["unseen_test"]
            or pools["validation"] & pools["unseen_test"]):
        raise ValueError("Exact nonzero-array overlap across sentence partitions; stop and review data.")
    return zeros


def job_stem(condition, seed):
    return f"{condition}_seed_{int(seed)}"


def verify_bundle(path, condition, seed, codes, plan, config):
    """Recompute metrics from all predictions; reject incomplete/corrupt jobs."""
    import numpy as np
    import pandas as pd
    from followup_common import macro_metrics

    expected = {r["sentence_id"]: r[condition] for r in plan if r["partition"] == "unseen_test"}
    with zipfile.ZipFile(path) as bundle:
        if bundle.testzip() is not None:
            raise ValueError(f"Corrupt ZIP: {path}")
        manifest = json.loads(bundle.read("manifest.json"))
        for key, value in (("condition", condition), ("seed", int(seed)),
                           ("codes", codes), ("config", config), ("sentence_plan", plan)):
            if manifest.get(key) != value:
                raise ValueError(f"Bundle definition differs ({key}): {path}")
        if manifest.get("weights_reloaded_and_predictions_matched") is not True:
            raise ValueError("Saved inference weights were not verified before publication.")
        if set(manifest["files"]) != {"predictions.csv", "metrics_by_reader.csv", "history.csv", "weights.npz"}:
            raise ValueError("Missing or unexpected bundle artifacts.")
        for name, checksum in manifest["files"].items():
            if hashlib.sha256(bundle.read(name)).hexdigest() != checksum:
                raise ValueError(f"Bundle checksum mismatch: {name}")
        with bundle.open("predictions.csv") as stream:
            prediction = pd.read_csv(stream)
        with bundle.open("metrics_by_reader.csv") as stream:
            metrics = pd.read_csv(stream)
        with np.load(io.BytesIO(bundle.read("weights.npz")), allow_pickle=False) as weights:
            expected_keys = [f"weight_{i:03d}" for i in range(manifest["weight_count"])]
            if sorted(weights.files) != expected_keys:
                raise ValueError("Weight archive is incomplete.")
            if any(not np.isfinite(weights[name]).all() for name in expected_keys):
                raise ValueError("Nonfinite saved weights.")
    if (len(prediction) != 264 or len(metrics) != 12
            or set(prediction.participant) != set(codes) or set(metrics.participant) != set(codes)
            or prediction[["participant", "sentence_id"]].duplicated().any()
            or metrics.participant.duplicated().any()):
        raise ValueError("Incomplete or duplicated reader predictions/metrics.")
    for frame in (prediction, metrics):
        if not (frame.condition == condition).all() or not (frame.seed == int(seed)).all():
            raise ValueError("Saved target/seed identifiers differ from the bundle.")
    recomputed = []
    for code in codes:
        rows = prediction[prediction.participant == code]
        if len(rows) != 22 or set(rows.sentence_id) != set(expected):
            raise ValueError("Missing, extra, or wrong held-out sentence IDs.")
        truth = np.asarray([expected[int(sid)] for sid in rows.sentence_id])
        probabilities = rows[["probability_negative", "probability_positive"]].to_numpy()
        if (not np.array_equal(truth, rows.true_label) or not np.isfinite(probabilities).all()
                or np.any(probabilities < 0) or np.any(probabilities > 1)
                or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)
                or not np.array_equal(rows.predicted_label, probabilities.argmax(axis=1))):
            raise ValueError("Prediction targets/probabilities/classes failed verification.")
        scores = macro_metrics(truth, probabilities)
        record = metrics[metrics.participant == code].iloc[0]
        for name in ("accuracy", "macro_f1", "tn", "fp", "fn", "tp"):
            if abs(float(record[name]) - float(scores[name])) > 1e-9:
                raise ValueError(f"Saved metric does not match predictions: {name}")
        if int(record.n_test) != 22 or int(record.correct) != scores["tn"] + scores["tp"]:
            raise ValueError("Wrong sample/correct count.")
        if (int(record.n_fit) != 792 or int(record.n_validation) != 264
                or int(record.n_test_negative) != 11 or int(record.n_test_positive) != 11
                or float(record.learning_rate) != config["learning_rate"]
                or record.preprocessing != "raw"
                or bool(record.collapsed) != scores["collapsed"]):
            raise ValueError("Saved training definition, class balance or collapse flag differs.")
        recomputed.append(scores)
    return metrics, prediction, manifest


def restore_weights(model, bundle_path):
    """Restore inference weights into the fixed architecture; no optimizer state."""
    import numpy as np

    with zipfile.ZipFile(bundle_path) as bundle:
        manifest = json.loads(bundle.read("manifest.json"))
        with np.load(io.BytesIO(bundle.read("weights.npz")), allow_pickle=False) as arrays:
            values = [arrays[f"weight_{i:03d}"] for i in range(manifest["weight_count"])]
    model.set_weights(values)
    return model


def publish_bundle(work_dir, out_dir, condition, seed, manifest, codes, plan, config):
    """One checked ZIP per fit, copied to Drive then independently reopened."""
    from followup_common import sha256

    work_dir = Path(work_dir)
    payloads = ["predictions.csv", "metrics_by_reader.csv", "history.csv", "weights.npz"]
    manifest["files"] = {name: sha256(str(work_dir / name)) for name in payloads}
    (work_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    local = work_dir / "job.zip"
    with zipfile.ZipFile(local, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name in payloads + ["manifest.json"]:
            bundle.write(work_dir / name, arcname=name)
    verify_bundle(local, condition, seed, codes, plan, config)
    destination = Path(out_dir) / "jobs" / (job_stem(condition, seed) + ".zip")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise ValueError("A saved bundle already exists; never overwrite it silently.")
    partial = str(destination) + ".partial"
    shutil.copyfile(local, partial)
    if sha256(str(local)) != sha256(partial):
        raise ValueError("Drive copy checksum failed; keep local job files and investigate.")
    os.replace(partial, destination)
    verify_bundle(destination, condition, seed, codes, plan, config)
    return destination


def collect_results(out_dir, codes, plan, config, require_complete=False):
    """Verified bundles, not summary rows, determine completed fits."""
    import pandas as pd
    from followup_common import atomic_csv, atomic_json, sha256

    frames, completed = [], set()
    expected_paths = {job_stem(c, s) + ".zip" for c in config["conditions"] for s in config["seeds"]}
    jobs_dir = Path(out_dir) / "jobs"
    if jobs_dir.exists() and {p.name for p in jobs_dir.glob("*.zip")} - expected_paths:
        raise ValueError("Unexpected saved job archive; review before mixing runs.")
    for seed in config["seeds"]:
        for condition in config["conditions"]:
            path = jobs_dir / (job_stem(condition, seed) + ".zip")
            if not path.exists():
                continue
            metrics, _, manifest = verify_bundle(path, condition, seed, codes, plan, config)
            if manifest.get("metadata_sha256") != sha256(os.path.join(out_dir, "run_metadata.json")):
                raise ValueError("Saved job came from a different recorded run definition.")
            frames.append(metrics)
            completed.add((condition, seed))
    if require_complete and len(completed) != 4:
        raise ValueError(f"Only {len(completed)}/4 verified bundles; no completion claim.")
    if frames:
        frame = pd.concat(frames, ignore_index=True).sort_values(["condition", "seed", "participant"])
        atomic_csv(frame, os.path.join(out_dir, "fold_metrics.csv"))
        for groups, filename in [(["condition", "seed"], "summary_by_seed.csv"),
                                 (["condition"], "summary_overall.csv")]:
            summary = frame.groupby(groups, as_index=False).agg(
                reader_evaluations=("macro_f1", "size"), mean_macro_f1=("macro_f1", "mean"),
                sd_macro_f1=("macro_f1", "std"), mean_accuracy=("accuracy", "mean"),
                correct=("correct", "sum"), test_predictions=("n_test", "sum"),
                reader_collapse_count=("collapsed", "sum"))
            atomic_csv(summary, os.path.join(out_dir, filename))
    atomic_json({"completed_fits": len(completed), "expected_fits": 4,
                 "reader_evaluation_rows": len(frames) * 12,
                 "complete": len(completed) == 4,
                 "scope": "Seen readers, unseen sentence identities; one split and one arbitrary mapping; no p-value."},
                os.path.join(out_dir, "completion.json"))
    return completed


def run(config, trt_dir, labels_csv, out_dir, config_path, reference_out_dir, v7_out_dir,
        prepare_only=False, verify_only=False):
    import numpy as np
    import pandas as pd
    import sklearn
    import tensorflow as tf
    from data import PARTICIPANTS, _norm, _read_subject
    from followup_common import (atomic_csv, atomic_json, fit_network, macro_metrics,
                                 prepare, set_seed, sha256)
    from model import build_cnn

    validate_plan(config)
    if os.path.realpath(out_dir) in (os.path.realpath(reference_out_dir), os.path.realpath(v7_out_dir)):
        raise ValueError("V8 must have its own output folder.")
    reference_path = os.path.join(reference_out_dir, "run_metadata.json")
    v7_metadata_path = os.path.join(v7_out_dir, "run_metadata.json")
    v7_plan_path = os.path.join(v7_out_dir, "sentence_plan.json")
    reference = json.loads(Path(reference_path).read_text())
    v7_metadata = json.loads(Path(v7_metadata_path).read_text())
    v7_plan = json.loads(Path(v7_plan_path).read_text())
    print("V8: checking the GPU, frozen CNN, runtime, data and saved V7 sentence split...", flush=True)
    source_dir = os.path.dirname(__file__)
    X, y, codes, sentence_ids = prepare(
        config, trt_dir, labels_csv, out_dir, config_path,
        [__file__] + [os.path.join(source_dir, name) for name in
                      ("model.py", "data.py", "followup_common.py", "v7_unseen_sentences.py")]
        + [reference_path, v7_metadata_path, v7_plan_path])
    current = json.loads(Path(out_dir, "run_metadata.json").read_text())
    actual = {"tensorflow": tf.__version__, "numpy": np.__version__,
              "pandas": pd.__version__, "scikit_learn": sklearn.__version__}
    validate_reference(reference, current, actual)
    validate_reference(v7_metadata, current, actual)
    if X.shape != (12, 110, 49, 126) or not np.isfinite(X).all():
        raise ValueError("Unexpected/nonfinite EEG feature array.")
    text_reference = None
    for name, _ in PARTICIPANTS:
        trials = _read_subject(trt_dir, name)
        texts = {int(r["sentenceId"]): _norm(r["persian_sentence"]) for r in trials.values()}
        chosen = [texts[int(sid)] for sid in sentence_ids]
        if text_reference is None:
            text_reference = chosen
        elif chosen != text_reference:
            raise ValueError("Reader sentence-text/ID alignment differs; stop before training.")
    plan = make_sentence_plan(y, sentence_ids, text_reference, config)
    if plan != v7_plan:
        raise ValueError("The split or arbitrary mapping differs from V7; no new split search allowed.")
    zeros = audit_partitions(X, plan)
    own_plan = Path(out_dir, "sentence_plan.json")
    if own_plan.exists() and json.loads(own_plan.read_text()) != plan:
        raise ValueError("V8 recorded sentence plan changed.")
    atomic_json(plan, str(own_plan))
    atomic_csv(pd.DataFrame(plan), os.path.join(out_dir, "sentence_plan.csv"))
    atomic_json({"passed": True, "reader_protocol": config["reader_protocol"],
                 "n_fit": 792, "n_validation": 264, "n_test": 264,
                 "readers_averaged": False, "words_averaged_by_this_runner": False,
                 "input": "One reader/sentence = 49 word slots x 126 precomputed TRT_total features x 1",
                 "v7_plan_sha256": sha256(v7_plan_path),
                 "all_reader_sentence_texts_aligned": True,
                 "all_zero_samples_by_partition": zeros,
                 "exact_nonzero_cross_partition_overlap": False,
                 "limitation": "Exact checks only; no guarantee against near-duplicates/upstream contamination. V7 missing-file issue remains separate."},
                os.path.join(out_dir, "preflight.json"))
    completed = collect_results(out_dir, codes, plan, config, require_complete=verify_only)
    print(f"Preflight passed: all 12 readers, 66 fit / 22 validation / 22 test sentences each. {len(completed)}/4 verified jobs saved.", flush=True)
    if prepare_only or verify_only:
        print("No training started by this command.", flush=True)
        return
    test_indices = partition_indices(plan, "unseen_test")
    for seed in config["seeds"]:
        for condition in config["conditions"]:
            if (condition, seed) in completed:
                continue
            (x_fit, y_fit), (x_val, y_val), (x_test, y_test) = all_reader_inputs(X, codes, plan, condition)
            tf.keras.backend.clear_session()
            set_seed(seed, config["deterministic_ops"])
            model = build_cnn(x_fit.shape[1:] + (1,), lr=config["learning_rate"])
            label = job_stem(condition, seed)
            print(f"START {len(completed) + 1}/4: {label}; pooled examples, NOT averaged readers", flush=True)
            started = time.perf_counter()
            with progress_heartbeat(label):
                model, history, chosen = fit_network(model, x_fit[..., None], y_fit,
                                                     x_val[..., None], y_val, config)
            seconds = time.perf_counter() - started
            probabilities = model.predict(x_test[..., None], verbose=0)
            per_reader, predictions = [], []
            for p, code in enumerate(codes):
                block = slice(p * 22, (p + 1) * 22)
                truth, part = y_test[block], probabilities[block]
                scores = macro_metrics(truth, part)
                identifiers = {"condition": condition, "seed": seed, "participant": code}
                predictions.append(pd.DataFrame({**identifiers,
                    "sentence_id": sentence_ids[test_indices], "true_label": truth,
                    "predicted_label": part.argmax(axis=1), "probability_negative": part[:, 0],
                    "probability_positive": part[:, 1]}))
                per_reader.append({**identifiers, "n_fit": 792, "n_validation": 264, "n_test": 22,
                                   "n_test_negative": 11, "n_test_positive": 11,
                                   "learning_rate": config["learning_rate"], "preprocessing": "raw",
                                   "n_parameters": int(model.count_params()),
                                   "duration_seconds": seconds, "duration_scope": "same_shared_model_for_all_12_readers",
                                   "sentence_plan_sha256": sha256(str(own_plan)),
                                   "correct": scores["tn"] + scores["tp"], **chosen, **scores})
            # Keep local recovery files if any archive or Drive validation fails.
            work = tempfile.mkdtemp(prefix="teco-v8-job-")
            try:
                pd.concat(predictions, ignore_index=True).to_csv(Path(work, "predictions.csv"), index=False)
                pd.DataFrame(per_reader).to_csv(Path(work, "metrics_by_reader.csv"), index=False)
                history.to_csv(Path(work, "history.csv"), index=False)
                weights = model.get_weights()
                np.savez_compressed(Path(work, "weights.npz"), **{f"weight_{i:03d}": w for i, w in enumerate(weights)})
                restored = build_cnn(x_fit.shape[1:] + (1,), lr=config["learning_rate"])
                with np.load(Path(work, "weights.npz"), allow_pickle=False) as arrays:
                    restored.set_weights([arrays[f"weight_{i:03d}"] for i in range(len(weights))])
                restored_probs = restored.predict(x_test[..., None], verbose=0)
                if not np.allclose(restored_probs, probabilities, atol=1e-6):
                    raise ValueError("Reloaded weights did not reproduce predictions; investigate local job.")
                manifest = {"condition": condition, "seed": int(seed), "codes": codes,
                            "config": config, "sentence_plan": plan, "weight_count": len(weights),
                            "metadata_sha256": sha256(os.path.join(out_dir, "run_metadata.json")),
                            "weights_reloaded_and_predictions_matched": True,
                            "weight_format": "Ordered model.get_weights arrays; inference only, no optimizer state."}
                publish_bundle(work, out_dir, condition, seed, manifest, codes, plan, config)
            except Exception:
                print(f"Job recovery files retained at {work}; do not delete them before investigating.", flush=True)
                raise
            shutil.rmtree(work)
            completed = collect_results(out_dir, codes, plan, config)
            print(f"SAVED and verified {len(completed)}/4: {label}; archive includes predictions + reusable weights.", flush=True)
    collect_results(out_dir, codes, plan, config, require_complete=True)
    print("V8 complete. Run notebook section 4, then review before any extra experiments.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("trt-dir", "labels-csv", "out-dir", "reference-out-dir", "v7-out-dir"):
        parser.add_argument("--" + flag, required=True)
    parser.add_argument("--config", default="configs/v8_seen_readers_unseen_sentences.json")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    run(config, args.trt_dir, args.labels_csv, args.out_dir, args.config,
        args.reference_out_dir, args.v7_out_dir, args.prepare_only, args.verify_only)


if __name__ == "__main__":
    main()
