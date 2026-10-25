"""Bounded joint reader/stimulus holdout; no model or parameter search.

One fixed stratified split: 66 fit, 22 validation, 22 unseen test sentences.
Each LOSO job fits 10 readers, validates on a separate reader AND separate
sentences, and tests the held-out reader on unseen22 and seen66 separately.
Real and one independently seeded arbitrary mapping share splits and settings.
Two initializations, 12 readers, 2 conditions = 48 fits, 96 evaluation rows.
This is exploratory: all data and hyperparameters were used in earlier work.
"""

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
import random
import threading
import time


PARTITIONS = ("fit", "validation", "unseen_test")
VIEWS = ("seen_sentences", "unseen_sentences")
KEYS = ["condition", "seed", "participant", "evaluation"]


def validate_config(config):
    if config.get("conditions") != ["sentiment", "shuffled_sentence_labels"]:
        raise ValueError("Use the two committed target conditions.")
    if config.get("seeds") != [45, 62]:
        raise ValueError("Use the two committed initialization seeds.")
    if config.get("sentences_per_class") != {"fit": 33, "validation": 11, "unseen_test": 11}:
        raise ValueError("Use the committed 66/22/22 sentence split.")
    if config.get("preprocessing") != "raw" or config.get("learning_rate") != 0.0001:
        raise ValueError("The tested raw-input CNN and LR=0.0001 are frozen.")
    if not config.get("deterministic_ops") or not config.get("require_gpu"):
        raise ValueError("Deterministic GPU execution is required.")
    if (config.get("sentence_split_seed"), config.get("label_mapping_seed")) != (20261025, 908172):
        raise ValueError("The sentence split and mapping seeds are frozen independently of initialization.")
    for key in ("epochs", "batch_size", "patience", "start_epoch"):
        if not isinstance(config.get(key), int) or config[key] < 1:
            raise ValueError(f"Invalid training setting: {key}")


def make_sentence_plan(labels, sentence_ids, normalized_texts, config):
    """Pure metadata operation; partition before any model fitting or evaluation."""
    validate_config(config)
    labels = list(map(int, labels))
    sentence_ids = list(map(int, sentence_ids))
    normalized_texts = list(normalized_texts)
    if len(labels) != 110 or len(sentence_ids) != 110 or len(normalized_texts) != 110:
        raise ValueError("Expected 110 binary sentence records.")
    if set(labels) != {0, 1} or labels.count(0) != 55 or labels.count(1) != 55:
        raise ValueError("Expected 55 sentences per sentiment class.")
    if len(set(sentence_ids)) != 110:
        raise ValueError("Duplicate sentence IDs.")
    if any(not isinstance(t, str) or not t for t in normalized_texts):
        raise ValueError("Missing normalized sentence text.")
    if len(set(normalized_texts)) != 110:
        raise ValueError("Duplicate normalized sentence texts: review/group these before a stimulus holdout.")
    rng = random.Random(config["sentence_split_seed"])
    assignments = {}
    for label in (0, 1):
        indices = sorted((i for i, y in enumerate(labels) if y == label), key=lambda i: sentence_ids[i])
        rng.shuffle(indices)
        start = 0
        for partition in PARTITIONS:
            count = config["sentences_per_class"][partition]
            assignments.update({i: partition for i in indices[start:start + count]})
            start += count
    # Permute labels within EACH whole partition, never within sentiment class.
    # This preserves 33/33, 11/11, 11/11 without exposing test targets to fitting.
    mapping_rng = random.Random(config["label_mapping_seed"])
    shuffled = labels.copy()
    for partition in PARTITIONS:
        indices = sorted((i for i in assignments if assignments[i] == partition), key=lambda i: sentence_ids[i])
        values = [labels[i] for i in indices]
        mapping_rng.shuffle(values)
        for i, value in zip(indices, values):
            shuffled[i] = value
        if all(shuffled[i] == labels[i] for i in indices):
            raise ValueError("Arbitrary mapping unexpectedly unchanged in a partition; review the frozen plan.")
    rows = [{"index": i, "sentence_id": sentence_ids[i], "partition": assignments[i],
             "sentiment": labels[i], "shuffled_sentence_labels": shuffled[i],
             "normalized_text_sha256": hashlib.sha256(normalized_texts[i].encode("utf-8")).hexdigest()}
            for i in range(110)]
    return rows


def reader_plan(codes, held_out):
    if len(codes) != 12 or len(set(codes)) != 12 or held_out not in codes:
        raise ValueError("Expected 12 unique reader codes and a valid held-out reader.")
    validation = codes[(codes.index(held_out) + 1) % len(codes)]
    fit = [code for code in codes if code not in (held_out, validation)]
    return fit, validation


def validate_reference(reference, current, actual_runtime):
    old, new = reference["definition"], current["definition"]
    for key in ("preprocessing", "learning_rate", "seeds", "epochs", "batch_size",
                "patience", "start_epoch", "deterministic_ops"):
        if old["config"].get(key) != new["config"].get(key):
            raise ValueError(f"The frozen V4 CNN reference differs: {key}")
    if old["data_files"] != new["data_files"]:
        raise ValueError("Input data differs from the completed real-label CNN.")
    required = {"model.py", "data.py", "followup_common.py"}
    sources = lambda d: {r["name"]: r for r in d["source_files"] if r["name"] in required}
    if set(sources(old)) != required or sources(old) != sources(new):
        raise ValueError("Architecture, loader, or training utility differs from the reference.")
    for key, value in actual_runtime.items():
        if current.get(key) != value or reference.get(key) != value:
            raise ValueError(f"Runtime differs from reference or resumed run: {key}")


def partition_indices(plan, partition):
    return [r["index"] for r in plan if r["partition"] == partition]


def build_inputs(X, codes, held_out, plan):
    import numpy as np

    fit_codes, validation = reader_plan(codes, held_out)
    fit_i = partition_indices(plan, "fit")
    val_i = partition_indices(plan, "validation")
    test_i = partition_indices(plan, "unseen_test")
    x_fit = np.concatenate([X[codes.index(code), fit_i] for code in fit_codes])
    x_val = X[codes.index(validation), val_i]
    views = {"seen_sentences": X[codes.index(held_out), fit_i],
             "unseen_sentences": X[codes.index(held_out), test_i]}
    return x_fit, x_val, views, fit_codes, validation


def audit_raw_overlap(X, codes, plan):
    """Block exact nonzero-array leakage; all-zero samples contain no identity."""
    import numpy as np

    samples = []
    for p, code in enumerate(codes):
        for row in plan:
            array = np.ascontiguousarray(X[p, row["index"]])
            samples.append({"participant": code, "sentence_id": row["sentence_id"],
                            "partition": row["partition"], "all_zero": bool(not np.any(array)),
                            "array_sha256": hashlib.sha256(array.tobytes()).hexdigest()})
    for held_out in codes:
        fit_codes, validation = reader_plan(codes, held_out)
        pools = {
            "fit": {r["array_sha256"] for r in samples if not r["all_zero"]
                    and r["participant"] in fit_codes and r["partition"] == "fit"},
            "validation": {r["array_sha256"] for r in samples if not r["all_zero"]
                           and r["participant"] == validation and r["partition"] == "validation"},
            "test": {r["array_sha256"] for r in samples if not r["all_zero"]
                     and r["participant"] == held_out and r["partition"] in ("fit", "unseen_test")},
        }
        if (pools["fit"] & pools["validation"] or pools["fit"] & pools["test"]
                or pools["validation"] & pools["test"]):
            raise ValueError(f"Exact nonzero raw-array overlap across partitions for {held_out}; review data, do not train.")
    return samples


def completed_jobs(frame, codes, config):
    if frame is None or frame.empty:
        return set()
    if frame[KEYS].duplicated().any():
        raise ValueError("Duplicate saved evaluation rows.")
    jobs = set()
    for (condition, seed, participant), group in frame.groupby(KEYS[:3]):
        if (condition not in config["conditions"] or int(seed) not in config["seeds"]
                or participant not in codes or set(group.evaluation) != set(VIEWS)):
            raise ValueError("Unexpected or partial saved job; preserve results and investigate.")
        jobs.add((condition, int(seed), participant))
    return jobs


def write_summaries(frame, out_dir, codes, config):
    from followup_common import atomic_csv, atomic_json

    jobs = completed_jobs(frame, codes, config)
    # Mean F1 is the mean over readers/initializations, NOT pooled F1.
    for groups, filename in [(["condition", "evaluation", "seed"], "summary_by_seed.csv"),
                             (["condition", "evaluation"], "summary_overall.csv")]:
        summary = frame.groupby(groups, as_index=False).agg(
            evaluations=("macro_f1", "size"), mean_macro_f1=("macro_f1", "mean"),
            sd_macro_f1=("macro_f1", "std"), mean_accuracy=("accuracy", "mean"),
            correct=("correct", "sum"), test_predictions=("n_test", "sum"),
            collapse_count=("collapsed", "sum"))
        atomic_csv(summary, os.path.join(out_dir, filename))
    expected = len(codes) * len(config["seeds"]) * len(config["conditions"])
    atomic_json({"completed_fits": len(jobs), "expected_fits": expected,
                 "evaluation_rows": len(frame), "expected_evaluation_rows": expected * 2,
                 "complete": len(jobs) == expected,
                 "note": "One split, one arbitrary mapping, reused hyperparameters; exploratory, no p-value."},
                os.path.join(out_dir, "completion.json"))


@contextmanager
def progress_heartbeat(label):
    stop = threading.Event()
    started = time.perf_counter()

    def report():
        while not stop.wait(30):
            print(f"Still training {label}: {time.perf_counter() - started:.0f}s elapsed", flush=True)

    thread = threading.Thread(target=report, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()


def run(config, trt_dir, labels_csv, out_dir, config_path, reference_out_dir, prepare_only=False):
    import numpy as np
    import pandas as pd
    import sklearn
    import tensorflow as tf
    from data import PARTICIPANTS, _norm, _read_subject
    from followup_common import (atomic_csv, atomic_json, fit_network, macro_metrics,
                                 prepare, save_predictions, set_seed, sha256)
    from model import build_cnn

    validate_config(config)
    if os.path.realpath(out_dir) == os.path.realpath(reference_out_dir):
        raise ValueError("Keep V7 in its own output directory.")
    source_dir = os.path.dirname(__file__)
    reference_path = os.path.join(reference_out_dir, "run_metadata.json")
    with open(reference_path, encoding="utf-8") as stream:
        reference = json.load(stream)
    print("V7: checking GPU, frozen model, data fingerprints, and runtime...", flush=True)
    X, y, codes, sentence_ids = prepare(
        config, trt_dir, labels_csv, out_dir, config_path,
        [__file__] + [os.path.join(source_dir, name) for name in
                      ("model.py", "data.py", "followup_common.py")] + [reference_path])
    with open(os.path.join(out_dir, "run_metadata.json"), encoding="utf-8") as stream:
        current = json.load(stream)
    validate_reference(reference, current, {"tensorflow": tf.__version__, "numpy": np.__version__,
                                          "pandas": pd.__version__, "scikit_learn": sklearn.__version__})
    if X.shape != (12, 110, 49, 126) or not np.isfinite(X).all():
        raise ValueError(f"Unexpected/nonfinite input: {X.shape}; review before training.")
    first = _read_subject(trt_dir, PARTICIPANTS[0][0])
    texts = {int(trial["sentenceId"]): _norm(trial["persian_sentence"]) for trial in first.values()}
    plan = make_sentence_plan(y, sentence_ids, [texts[int(i)] for i in sentence_ids], config)
    samples = audit_raw_overlap(X, codes, plan)
    plan_path = os.path.join(out_dir, "sentence_plan.json")
    if os.path.exists(plan_path):
        with open(plan_path, encoding="utf-8") as stream:
            if json.load(stream) != plan:
                raise ValueError("Recorded sentence split or mapping differs. Preserve it; do not resume.")
    atomic_json(plan, plan_path)
    atomic_csv(pd.DataFrame(plan), os.path.join(out_dir, "sentence_plan.csv"))
    atomic_csv(pd.DataFrame(samples), os.path.join(out_dir, "raw_array_audit.csv"))
    atomic_json({"reference_metadata_sha256": sha256(reference_path),
                 "reference_git_commit": reference.get("git_commit"),
                 "passed": True, "exact_nonzero_array_overlap": False,
                 "all_zero_samples": sum(r["all_zero"] for r in samples),
                 "scope": "Exact normalized-text and raw-array checks; near-duplicates/upstream contamination not excluded.",
                 "n_fit": 660, "n_validation": 22, "n_unseen_test": 22, "n_seen_test": 66,
                 "test_subject_never_fitted": True,
                 "test_sentences_never_used_for_fit_or_checkpoint": True,
                 "split_seed": config["sentence_split_seed"], "mapping_seed": config["label_mapping_seed"],
                 "frozen_model": "original_cnn, raw TRT_total, LR=0.0001",
                 "limitation": "Not a pristine external test: prior experiments used all sentences; no significance claim."},
                os.path.join(out_dir, "preflight.json"))
    print("Preflight passed: 10 fit readers x 66 sentences; separate validation reader x 22; held-out reader x 22 unseen / 66 seen.", flush=True)
    print("Both target mappings are exactly balanced within fit, validation, and test sentence partitions.", flush=True)
    if prepare_only:
        print("Prepared only; no training started. Plan: 48 fits / 96 evaluation rows.", flush=True)
        return

    path = os.path.join(out_dir, "fold_metrics.csv")
    frame = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    completed = completed_jobs(frame, codes, config)
    fit_i, val_i, test_i = [partition_indices(plan, p) for p in PARTITIONS]
    expected = 48
    for seed in config["seeds"]:
        for held_out in codes:
            x_fit, x_val, views, fit_codes, validation = build_inputs(X, codes, held_out, plan)
            for condition in config["conditions"]:
                key = (condition, seed, held_out)
                if key in completed:
                    continue
                targets = np.asarray([r[condition] for r in plan], dtype=np.int64)
                label_fit = np.tile(targets[fit_i], len(fit_codes))
                label_val = targets[val_i]
                tf.keras.backend.clear_session()
                set_seed(seed, config["deterministic_ops"])
                model = build_cnn(x_fit.shape[1:] + (1,), lr=config["learning_rate"])
                label = f"{condition}, seed={seed}, reader={held_out}"
                print(f"START {len(completed) + 1}/{expected}: {label}", flush=True)
                started = time.perf_counter()
                with progress_heartbeat(label):
                    model, history, chosen = fit_network(model, x_fit[..., None], label_fit,
                                                         x_val[..., None], label_val, config)
                seconds = time.perf_counter() - started
                stem = f"{condition}_seed_{seed}_{held_out}"
                atomic_csv(history, os.path.join(out_dir, "histories", stem + ".csv"))
                rows = []
                for evaluation, indices in [("seen_sentences", fit_i), ("unseen_sentences", test_i)]:
                    truth = targets[indices]
                    probabilities = model.predict(views[evaluation][..., None], verbose=0)
                    identifiers = {"condition": condition, "seed": seed,
                                   "participant": held_out, "evaluation": evaluation}
                    save_predictions(os.path.join(out_dir, "predictions", stem + "_" + evaluation + ".csv"),
                                     truth, sentence_ids[indices], probabilities, identifiers)
                    scores = macro_metrics(truth, probabilities)
                    rows.append({**identifiers, "validation_participant": validation,
                                 "fit_participants": ",".join(fit_codes), "n_fit": len(x_fit),
                                 "n_validation": len(x_val), "n_test": len(truth),
                                 "n_test_negative": int(sum(truth == 0)), "n_test_positive": int(sum(truth == 1)),
                                 "learning_rate": config["learning_rate"], "preprocessing": "raw",
                                 "sentence_plan_sha256": sha256(plan_path),
                                 "duration_seconds": seconds, "duration_scope": "shared_fit_for_both_evaluations",
                                 "n_parameters": int(model.count_params()),
                                 "correct": scores["tn"] + scores["tp"], **chosen, **scores})
                # Commit BOTH views together: an interrupted job never becomes half-complete.
                candidate = pd.concat([frame, pd.DataFrame(rows)], ignore_index=True)
                completed_jobs(candidate, codes, config)
                frame = candidate.sort_values(KEYS).reset_index(drop=True)
                atomic_csv(frame, path)
                write_summaries(frame, out_dir, codes, config)
                completed.add(key)
                print(f"SAVED {len(completed)}/{expected}: seen F1={rows[0]['macro_f1']:.3f}, unseen F1={rows[1]['macro_f1']:.3f}", flush=True)
    write_summaries(frame, out_dir, codes, config)
    print("V7 complete. Run the notebook results section; do not start extra experiments automatically.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("trt-dir", "labels-csv", "out-dir", "reference-out-dir"):
        parser.add_argument("--" + flag, required=True)
    parser.add_argument("--config", default="configs/v7_unseen_sentences.json")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as stream:
        config = json.load(stream)
    run(config, args.trt_dir, args.labels_csv, args.out_dir, args.config,
        args.reference_out_dir, args.prepare_only)


if __name__ == "__main__":
    main()
