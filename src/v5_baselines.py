"""V5: bounded simple-model comparisons and a sentence-identity control."""

import argparse
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd

from followup_common import (
    aggregate_words, append_row, atomic_csv, build_small_cnn, fit_network,
    macro_metrics, next_validation, participant_split, prepare,
    save_predictions, set_seed,
)


MODELS = (
    "majority", "logistic_mean_coverage", "logistic_coverage_only",
    "logistic_shuffled_sentence_labels", "small_mlp", "small_cnn",
    "original_cnn_shuffled_sentence_labels",
)


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


def run(config, trt_dir, labels_csv, out_dir, config_path):
    import tensorflow as tf
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from model import build_cnn

    source_dir = os.path.dirname(__file__)
    X, y, codes, sentence_ids = prepare(
        config, trt_dir, labels_csv, out_dir, config_path,
        [__file__, os.path.join(source_dir, "followup_common.py"),
         os.path.join(source_dir, "data.py"),
         os.path.join(source_dir, "model.py")],
    )
    metrics_path = os.path.join(out_dir, "fold_metrics.csv")
    expected = (
        (len(MODELS) - 1) * len(config["seeds"]) * len(codes)
        + len(config["v5_shuffled_cnn_seeds"]) * len(codes)
    )
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

            for name in MODELS:
                if (name == "original_cnn_shuffled_sentence_labels"
                        and seed not in config["v5_shuffled_cnn_seeds"]):
                    continue
                key = (name, str(seed), held_out)
                if key in completed:
                    continue
                started = time.perf_counter()
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
                        features_fit.shape[-1], config["v5_mlp_learning_rate"]
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
                        x_fit.shape[1:] + (1,), config["v5_small_cnn_learning_rate"]
                    )
                    n_parameters = int(model.count_params())
                    model, history, chosen = fit_network(
                        model, x_fit[..., None], y_fit, x_val[..., None], y_val, config
                    )
                    probabilities = model.predict(x_test[..., None], verbose=0)
                else:
                    tf.keras.backend.clear_session()
                    set_seed(seed, config["deterministic_ops"])
                    model = build_cnn(x_fit.shape[1:] + (1,), lr=0.001)
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
                    "target_type": "shuffled_sentence_labels" if
                    "shuffled_sentence_labels" in name else "sentiment",
                    "sentence_label_mapping_sha256": mapping_hash if
                    "shuffled_sentence_labels" in name else "",
                    "representation": "coverage_only" if name ==
                    "logistic_coverage_only" else (
                        "raw_word_channel_array" if "cnn" in name
                        else "nonzero_word_mean_plus_coverage"
                    ),
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
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as stream:
        config = json.load(stream)
    run(config, args.trt_dir, args.labels_csv, args.out_dir, args.config)


if __name__ == "__main__":
    main()
