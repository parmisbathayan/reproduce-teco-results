"""V5: bounded simple-model comparisons and a sentence-identity control."""

import argparse
import hashlib
import json
import math
import os
import time

import numpy as np
import pandas as pd

from followup_common import (
    aggregate_words, append_row, atomic_csv, atomic_json, build_small_cnn, fit_network,
    macro_metrics, next_validation, participant_split, prepare,
    save_predictions, set_seed,
    sha256,
)


MODELS = (
    "majority", "logistic_mean_coverage", "logistic_coverage_only",
    "logistic_shuffled_sentence_labels", "small_mlp", "small_cnn",
    "original_cnn_shuffled_sentence_labels",
)


def model_plan(config):
    models = list(config.get("models", MODELS))
    seeds = list(config["seeds"])
    if not models or len(models) != len(set(models)) or set(models) - set(MODELS):
        raise ValueError("models must be a nonempty, unique list of supported models.")
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be a nonempty unique list.")
    schedule = {name: seeds for name in models}
    if "original_cnn_shuffled_sentence_labels" in models:
        control_seeds = list(config.get("v5_shuffled_cnn_seeds", []))
        if (not control_seeds or len(control_seeds) != len(set(control_seeds))
                or not set(control_seeds).issubset(seeds)):
            raise ValueError("CNN shuffle-control seeds must be a unique nonempty subset of seeds.")
        schedule["original_cnn_shuffled_sentence_labels"] = control_seeds
    return models, schedule


def neural_learning_rate(name, config):
    keys = {
        "small_mlp": "v5_mlp_learning_rate",
        "small_cnn": "v5_small_cnn_learning_rate",
        "original_cnn_shuffled_sentence_labels": "v5_shuffled_cnn_learning_rate",
    }
    if name not in keys:
        return None
    value = float(config.get(keys[name], 0.001))
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Neural learning rates must be positive and finite.")
    return value


def validate_cnn_control_reference(reference, current):
    old, new = reference["definition"], current["definition"]
    if new["config"].get("models") != ["original_cnn_shuffled_sentence_labels"]:
        raise ValueError("Matched CNN reference verification applies to the CNN shuffle-only stage.")
    if old["config"].get("preprocessing") != "raw":
        raise ValueError("The real-label CNN reference must use raw input.")
    if old["config"].get("learning_rate") != neural_learning_rate("original_cnn_shuffled_sentence_labels", new["config"]):
        raise ValueError("Real and shuffled CNN learning rates differ.")
    for key in ("seeds", "epochs", "batch_size", "patience", "start_epoch", "deterministic_ops"):
        if old["config"].get(key) != new["config"].get(key):
            raise ValueError(f"Real and shuffled CNN training settings differ: {key}")
    if old["data_files"] != new["data_files"]:
        raise ValueError("Real and shuffled CNN data fingerprints differ.")
    required = {"model.py", "data.py", "followup_common.py"}
    old_sources = {row["name"]: row for row in old["source_files"] if row["name"] in required}
    new_sources = {row["name"]: row for row in new["source_files"] if row["name"] in required}
    if set(old_sources) != required or old_sources != new_sources:
        raise ValueError("Real and shuffled CNN architecture, data, or training code differs.")
    for key in ("tensorflow", "numpy", "pandas", "scikit_learn"):
        if not reference.get(key) or reference.get(key) != current.get(key):
            raise ValueError(f"Real and shuffled CNN runtimes differ: {key}")


def build_mlp(n_features, lr):
    import tensorflow as tf

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_features,)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(2, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy", metrics=["accuracy"],
    )
    return model


def run(config, trt_dir, labels_csv, out_dir, config_path, reference_out_dir=None):
    import tensorflow as tf
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from model import build_cnn

    models, schedule = model_plan(config)
    source_dir = os.path.dirname(__file__)
    print("V5: loading data and checking the recorded run definition...", flush=True)
    X, y, codes, sentence_ids = prepare(
        config, trt_dir, labels_csv, out_dir, config_path,
        [__file__, os.path.join(source_dir, "followup_common.py"),
         os.path.join(source_dir, "data.py"),
         os.path.join(source_dir, "model.py")],
    )
    if reference_out_dir is not None:
        import sklearn
        if os.path.realpath(reference_out_dir) == os.path.realpath(out_dir):
            raise ValueError("The real CNN reference must have a separate output folder.")
        reference_path = os.path.join(reference_out_dir, "run_metadata.json")
        with open(reference_path, encoding="utf-8") as stream:
            reference = json.load(stream)
        with open(os.path.join(out_dir, "run_metadata.json"), encoding="utf-8") as stream:
            current = json.load(stream)
        actual_runtime = {"tensorflow": tf.__version__, "numpy": np.__version__,
                          "pandas": pd.__version__, "scikit_learn": sklearn.__version__}
        if any(current.get(key) != value for key, value in actual_runtime.items()):
            raise ValueError("Runtime changed since this shuffle-control run began.")
        validate_cnn_control_reference(reference, current)
        atomic_json({
            "reference_directory": reference_out_dir,
            "reference_metadata_sha256": sha256(reference_path),
            "reference_git_commit": reference.get("git_commit"),
            "matched": "Same raw data, model/training code, runtime, seeds, learning rate, epochs, validation, and checkpoint rule; sentence-label mapping is shuffled.",
        }, os.path.join(out_dir, "matched_reference.json"))
        print("Real-label CNN reference verified before training.", flush=True)
    metrics_path = os.path.join(out_dir, "fold_metrics.csv")
    expected = sum(len(schedule[name]) * len(codes) for name in models)
    print(f"V5 plan: {len(models)} models, {expected} saved result rows expected.", flush=True)
    if os.path.exists(metrics_path):
        completed = set(map(tuple, pd.read_csv(metrics_path)[
            ["model", "seed", "participant"]
        ].astype(str).to_numpy()))
    else:
        completed = set()

    for seed in config["seeds"]:
        # One balanced random mapping of the 110 sentence IDs, shared by all
        # participants. Scores against this permuted target can expose a
        # participant-invariant sentence fingerprint, not sentiment learning.
        shuffled_y = np.random.default_rng(seed).permutation(y)
        mapping_path = os.path.join(out_dir, f"shuffled_sentence_labels_seed_{seed}.csv")
        if not os.path.exists(mapping_path):
            atomic_csv(pd.DataFrame({
                "sentence_id": sentence_ids, "original_label": y,
                "shuffled_label": shuffled_y,
            }), mapping_path)
        mapping_hash = hashlib.sha256(shuffled_y.tobytes()).hexdigest()

        for held_out in codes:
            validation = next_validation(codes, held_out)
            x_fit, y_fit, x_val, y_val, fit_codes, _ = participant_split(
                X, y, codes, held_out, validation
            )
            x_test = X[codes.index(held_out)]
            fit_features = aggregate_words(x_fit)
            val_features = aggregate_words(x_val)
            test_features = aggregate_words(x_test)
            n_channels = X.shape[-1]

            for name in models:
                if seed not in schedule[name]:
                    continue
                key = (name, str(seed), held_out)
                if key in completed:
                    continue
                started = time.perf_counter()
                learning_rate = neural_learning_rate(name, config)
                if learning_rate is not None:
                    print(f"START V5 {len(completed) + 1}/{expected}: {name}, seed={seed}, participant={held_out}. A fit may take 1–3 minutes.", flush=True)
                chosen = {}
                history = None
                target_y = shuffled_y if "shuffled_sentence_labels" in name else y
                if name == "majority":
                    majority = int(np.mean(y_fit) > 0.5)
                    probabilities = np.zeros((len(y), 2), dtype=np.float32)
                    probabilities[:, majority] = 1.0
                    n_parameters = 0
                elif name.startswith("logistic"):
                    if name == "logistic_coverage_only":
                        columns = slice(n_channels, None)
                    else:
                        columns = slice(None)
                    scaler = StandardScaler().fit(fit_features[:, columns])
                    features_fit = scaler.transform(fit_features[:, columns])
                    features_test = scaler.transform(test_features[:, columns])
                    labels_fit = (
                        np.tile(shuffled_y, len(fit_codes))
                        if name == "logistic_shuffled_sentence_labels" else y_fit
                    )
                    model = LogisticRegression(
                        C=float(config["v5_logistic_c"]), solver="liblinear",
                        max_iter=1000, random_state=seed,
                    )
                    model.fit(features_fit, labels_fit)
                    probabilities = model.predict_proba(features_test)
                    n_parameters = int(model.coef_.size + model.intercept_.size)
                    chosen["converged"] = bool(model.n_iter_[0] < model.max_iter)
                elif name == "small_mlp":
                    tf.keras.backend.clear_session()
                    set_seed(seed, config["deterministic_ops"])
                    scaler = StandardScaler().fit(fit_features)
                    features_fit = scaler.transform(fit_features).astype(np.float32)
                    features_val = scaler.transform(val_features).astype(np.float32)
                    features_test = scaler.transform(test_features).astype(np.float32)
                    model = build_mlp(
                        features_fit.shape[-1], learning_rate
                    )
                    n_parameters = int(model.count_params())
                    model, history, chosen = fit_network(
                        model, features_fit, y_fit, features_val, y_val, config
                    )
                    probabilities = model.predict(features_test, verbose=0)
                elif name == "small_cnn":
                    tf.keras.backend.clear_session()
                    set_seed(seed, config["deterministic_ops"])
                    model = build_small_cnn(
                        x_fit.shape[1:] + (1,), learning_rate
                    )
                    n_parameters = int(model.count_params())
                    model, history, chosen = fit_network(
                        model, x_fit[..., None], y_fit, x_val[..., None], y_val, config
                    )
                    probabilities = model.predict(x_test[..., None], verbose=0)
                else:
                    tf.keras.backend.clear_session()
                    set_seed(seed, config["deterministic_ops"])
                    model = build_cnn(x_fit.shape[1:] + (1,), lr=learning_rate)
                    n_parameters = int(model.count_params())
                    shuffled_fit = np.tile(shuffled_y, len(fit_codes))
                    model, history, chosen = fit_network(
                        model, x_fit[..., None], shuffled_fit,
                        x_val[..., None], shuffled_y, config
                    )
                    probabilities = model.predict(x_test[..., None], verbose=0)

                seconds = time.perf_counter() - started
                stem = f"{name}_seed_{seed}_{held_out}"
                identifiers = {"model": name, "seed": seed, "participant": held_out}
                if history is not None:
                    atomic_csv(
                        history.assign(**identifiers),
                        os.path.join(out_dir, "histories", stem + ".csv"),
                    )
                save_predictions(
                    os.path.join(out_dir, "predictions", stem + ".csv"),
                    target_y, sentence_ids, probabilities, identifiers,
                )
                row = {
                    **identifiers,
                    "learning_rate": learning_rate,
                    "target_type": "shuffled_sentence_labels" if
                    "shuffled_sentence_labels" in name else "sentiment",
                    "sentence_label_mapping_sha256": mapping_hash if
                    "shuffled_sentence_labels" in name else "",
                    "representation": "class_frequency_only" if name == "majority" else (
                        "coverage_only" if name == "logistic_coverage_only" else (
                        "raw_word_channel_array" if "cnn" in name
                        else "nonzero_word_mean_plus_coverage"
                    )),
                    "validation_participant": validation,
                    "fit_participants": ",".join(fit_codes),
                    "n_fit": len(y_fit), "n_validation": len(y_val),
                    "n_test": len(target_y), "n_parameters": n_parameters,
                    "duration_seconds": seconds,
                    **chosen, **macro_metrics(target_y, probabilities),
                }
                frame, _ = append_row(metrics_path, row, ["model", "seed", "participant"])
                summary = frame.groupby(["model", "target_type"], as_index=False).agg(
                    folds=("macro_f1", "size"), mean_macro_f1=("macro_f1", "mean"),
                    sd_macro_f1=("macro_f1", "std"),
                    collapse_count=("collapsed", "sum"),
                )
                atomic_csv(summary, os.path.join(out_dir, "summary_by_model.csv"))
                seed_summary = frame.groupby(["model", "target_type", "seed"], as_index=False).agg(
                    folds=("macro_f1", "size"), mean_macro_f1=("macro_f1", "mean"),
                    collapse_count=("collapsed", "sum"),
                )
                atomic_csv(seed_summary, os.path.join(out_dir, "summary_by_model_seed.csv"))
                atomic_json({
                    "completed_rows": len(frame), "expected_rows": expected,
                    "complete": bool(len(frame) == expected), "models": models,
                    "seeds": config["seeds"],
                }, os.path.join(out_dir, "completion.json"))
                completed.add(key)
                print(
                    f"V5 {len(frame)}/{expected} {name} seed={seed} {held_out} "
                    f"F1={row['macro_f1']:.3f} collapse={row['collapsed']}", flush=True,
                )
    return pd.read_csv(metrics_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trt-dir", required=True)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--config", default="configs/v4_v6_followup.json")
    parser.add_argument("--reference-out-dir")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as stream:
        config = json.load(stream)
    run(config, args.trt_dir, args.labels_csv, args.out_dir, args.config, args.reference_out_dir)


if __name__ == "__main__":
    main()
