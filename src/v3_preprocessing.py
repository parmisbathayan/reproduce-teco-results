"""Version 3: train-only preprocessing ablations for the EEG-only CNN.

This version changes only input preprocessing. The v2 folds, seeds, CNN, learning
rate, validation tail, checkpoint metric, and early-stopping behavior are retained.
Zeros represent padding or missing fixations and remain exactly zero in every method.
Per-channel statistics are fitted on the fit partition only, never validation or test.
"""

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from data import PARTICIPANTS, build_binary_dataset


SUPPORTED_METHODS = {
    "raw",
    "log1p",
    "per_channel_zscore",
    "robust_scale_clip",
}


def load_config(path):
    with open(path, "r", encoding="utf-8") as stream:
        config = json.load(stream)
    required = {
        "version", "preprocessing_methods", "seeds", "epochs", "batch_size",
        "validation_split", "learning_rate", "early_stopping_patience",
        "early_stopping_start_epoch", "deterministic_ops", "require_gpu",
        "preserve_zero_padding", "robust_clip_quantiles",
        "robust_scale_quantiles", "scale_epsilon",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"config is missing required fields: {sorted(missing)}")

    methods = list(config["preprocessing_methods"])
    if not methods or len(methods) != len(set(methods)):
        raise ValueError("preprocessing_methods must be a non-empty unique list")
    unknown = set(methods) - SUPPORTED_METHODS
    if unknown:
        raise ValueError(f"unsupported preprocessing methods: {sorted(unknown)}")
    if set(methods) != SUPPORTED_METHODS:
        raise ValueError(
            "v3 must include the locked raw, log1p, per_channel_zscore, and "
            "robust_scale_clip ablations"
        )
    if not config["seeds"]:
        raise ValueError("config must contain at least one seed")
    if not 0 < float(config["validation_split"]) < 1:
        raise ValueError("validation_split must be between 0 and 1")
    if not bool(config["preserve_zero_padding"]):
        raise ValueError("v3 requires preserve_zero_padding=true")
    if float(config["scale_epsilon"]) <= 0:
        raise ValueError("scale_epsilon must be positive")

    for key in ("robust_clip_quantiles", "robust_scale_quantiles"):
        values = config[key]
        if len(values) != 2 or not 0 <= values[0] < values[1] <= 1:
            raise ValueError(f"{key} must contain two increasing values in [0, 1]")
    return config


def _atomic_csv(frame, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.tmp"
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_json(value, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def _sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            block = stream.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _git_commit():
    repository = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        return subprocess.check_output(
            ["git", "-C", repository, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _file_fingerprints(trt_dir, labels_csv):
    files = [labels_csv]
    files.extend(
        os.path.join(trt_dir, f"{name}_trt_total.pickle")
        for name, _ in PARTICIPANTS
    )
    return [
        {
            "name": os.path.basename(path),
            "size_bytes": os.path.getsize(path),
            "sha256": _sha256(path),
        }
        for path in files
    ]


def _set_seed(seed, deterministic_ops):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    if deterministic_ops:
        try:
            tf.config.experimental.enable_op_determinism()
        except AttributeError:
            pass


def _metadata(config, trt_dir, labels_csv, config_path):
    import sklearn
    import tensorflow as tf

    source_paths = (
        __file__,
        os.path.join(os.path.dirname(__file__), "data.py"),
        os.path.join(os.path.dirname(__file__), "model.py"),
        config_path,
    )
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "version": config["version"],
        "config": config,
        "git_commit": _git_commit(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "tensorflow": tf.__version__,
        "devices": [device.name for device in tf.config.list_physical_devices()],
        "data_files": _file_fingerprints(trt_dir, labels_csv),
        "source_files": [
            {
                "name": os.path.basename(path),
                "size_bytes": os.path.getsize(path),
                "sha256": _sha256(path),
            }
            for path in source_paths
        ],
    }


def _validate_resume_metadata(existing, current):
    mismatches = []
    for key in ("version", "config", "git_commit", "data_files", "source_files"):
        if existing.get(key) != current.get(key):
            mismatches.append(key)
    if mismatches:
        raise ValueError(
            "refusing to mix resumed results with a different run definition; "
            f"mismatched metadata: {mismatches}. Use a new output directory."
        )


def _fit_channel_parameters(x_fit, method, config):
    """Fit one center/scale pair per EEG channel using nonzero fit values only."""
    if x_fit.ndim != 3:
        raise ValueError(f"expected (samples, words, channels), got {x_fit.shape}")

    n_channels = x_fit.shape[-1]
    epsilon = float(config["scale_epsilon"])
    clip_q = config["robust_clip_quantiles"]
    scale_q = config["robust_scale_quantiles"]
    rows = []

    for channel in range(n_channels):
        values = x_fit[..., channel]
        observed = values[np.isfinite(values) & (values != 0)].astype(np.float64)
        if observed.size == 0:
            center, scale = 0.0, 1.0
            clip_low, clip_high = -np.inf, np.inf
        elif method == "per_channel_zscore":
            center = float(observed.mean())
            raw_scale = float(observed.std())
            scale = raw_scale if raw_scale > epsilon else 1.0
            clip_low, clip_high = -np.inf, np.inf
        elif method == "robust_scale_clip":
            clip_low, clip_high = np.quantile(observed, clip_q)
            lower, center, upper = np.quantile(
                observed, [scale_q[0], 0.5, scale_q[1]]
            )
            raw_scale = float(upper - lower)
            scale = raw_scale if raw_scale > epsilon else 1.0
            center = float(center)
            clip_low, clip_high = float(clip_low), float(clip_high)
        else:
            raise ValueError(f"{method} does not fit channel parameters")

        rows.append({
            "channel": channel,
            "nonzero_fit_values": int(observed.size),
            "center": center,
            "scale": scale,
            "clip_low": clip_low,
            "clip_high": clip_high,
        })
    return pd.DataFrame(rows)


def _transform_with_parameters(values, parameters):
    result = np.zeros_like(values, dtype=np.float32)
    for row in parameters.itertuples(index=False):
        channel_values = values[..., row.channel]
        observed = np.isfinite(channel_values) & (channel_values != 0)
        transformed = np.clip(channel_values[observed], row.clip_low, row.clip_high)
        result[..., row.channel][observed] = (
            (transformed - row.center) / row.scale
        ).astype(np.float32)
    return result


def apply_preprocessing(method, x_fit, x_val, x_test, config):
    """Fit on x_fit and transform all partitions while preserving exact zeros."""
    partitions = (x_fit, x_val, x_test)
    for values in partitions:
        if values.ndim != 3:
            raise ValueError(f"expected (samples, words, channels), got {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError("EEG input contains non-finite values")

    if method == "raw":
        transformed = tuple(values.astype(np.float32, copy=False) for values in partitions)
        parameters = pd.DataFrame()
    elif method == "log1p":
        if any(np.any(values < 0) for values in partitions):
            raise ValueError("log1p requires nonnegative TRT_total amplitudes")
        transformed = tuple(np.log1p(values).astype(np.float32) for values in partitions)
        parameters = pd.DataFrame()
    else:
        parameters = _fit_channel_parameters(x_fit, method, config)
        transformed = tuple(
            _transform_with_parameters(values, parameters) for values in partitions
        )

    for before, after in zip(partitions, transformed):
        if not np.all(after[before == 0] == 0):
            raise AssertionError(f"{method} changed padding/missing-value zeros")
        if not np.all(np.isfinite(after)):
            raise AssertionError(f"{method} produced non-finite values")
    return (*transformed, parameters)


def _partition_stats(values, prefix):
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    observed = flat[flat != 0]
    return {
        f"{prefix}_zero_fraction": float(np.mean(flat == 0)),
        f"{prefix}_nonzero_mean": float(observed.mean()) if observed.size else np.nan,
        f"{prefix}_nonzero_std": float(observed.std()) if observed.size else np.nan,
        f"{prefix}_nonzero_q01": float(np.quantile(observed, 0.01)) if observed.size else np.nan,
        f"{prefix}_nonzero_q99": float(np.quantile(observed, 0.99)) if observed.size else np.nan,
    }


def _distribution_shift(x_fit, x_test, prefix):
    fit_mean = np.mean(x_fit, axis=(0, 1))
    test_mean = np.mean(x_test, axis=(0, 1))
    fit_std = np.std(x_fit, axis=(0, 1))
    test_std = np.std(x_test, axis=(0, 1))
    safe_std = np.where(fit_std > 1e-8, fit_std, 1.0)
    return {
        f"{prefix}_mean_shift_z_rms": float(
            np.sqrt(np.mean(((test_mean - fit_mean) / safe_std) ** 2))
        ),
        f"{prefix}_std_ratio_median": float(np.median(test_std / safe_std)),
    }


def _write_summaries(metrics, out_dir, expected_folds):
    if metrics.empty:
        return

    by_method = metrics.groupby("preprocessing", as_index=False).agg(
        completed_folds=("f1", "size"),
        completed_seeds=("seed", "nunique"),
        completed_participants=("participant", "nunique"),
        precision_mean=("precision", "mean"),
        recall_mean=("recall", "mean"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
        f1_median=("f1", "median"),
        f1_min=("f1", "min"),
        f1_max=("f1", "max"),
        collapse_count=("collapsed", "sum"),
        collapse_rate=("collapsed", "mean"),
        constant_output_count=("constant_probability_output", "sum"),
        constant_output_rate=("constant_probability_output", "mean"),
    )
    _atomic_csv(by_method, os.path.join(out_dir, "summary_by_method.csv"))

    by_method_seed = metrics.groupby(
        ["preprocessing", "seed"], as_index=False
    ).agg(
        completed_participants=("participant", "nunique"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
        f1_median=("f1", "median"),
        collapse_count=("collapsed", "sum"),
        collapse_rate=("collapsed", "mean"),
        constant_output_count=("constant_probability_output", "sum"),
    )
    _atomic_csv(by_method_seed, os.path.join(out_dir, "summary_by_method_seed.csv"))

    by_method_participant = metrics.groupby(
        ["preprocessing", "participant"], as_index=False
    ).agg(
        completed_seeds=("seed", "nunique"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
        f1_median=("f1", "median"),
        collapse_count=("collapsed", "sum"),
        collapse_rate=("collapsed", "mean"),
    )
    _atomic_csv(
        by_method_participant,
        os.path.join(out_dir, "summary_by_method_participant.csv"),
    )

    paired = metrics.pivot_table(
        index=["seed", "participant"],
        columns="preprocessing",
        values="f1",
        aggfunc="first",
    ).reset_index()
    if "raw" in paired:
        for method in sorted(SUPPORTED_METHODS - {"raw"}):
            if method in paired:
                paired[f"{method}_minus_raw"] = paired[method] - paired["raw"]
    _atomic_csv(paired, os.path.join(out_dir, "paired_f1_comparison.csv"))

    summary = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "completed_folds": int(len(metrics)),
        "expected_folds": int(expected_folds),
        "complete": bool(len(metrics) == expected_folds),
        "completed_methods": int(metrics["preprocessing"].nunique()),
        "completed_seeds": int(metrics["seed"].nunique()),
        "completed_participants": int(metrics["participant"].nunique()),
        "f1_mean": float(metrics["f1"].mean()),
        "f1_std": float(metrics["f1"].std(ddof=1)) if len(metrics) > 1 else None,
        "collapse_count": int(metrics["collapsed"].sum()),
        "collapse_rate": float(metrics["collapsed"].mean()),
        "constant_output_count": int(metrics["constant_probability_output"].sum()),
        "constant_output_rate": float(metrics["constant_probability_output"].mean()),
    }
    _atomic_json(summary, os.path.join(out_dir, "summary_overall.json"))


def run_v3_preprocessing(trt_dir, labels_csv, out_dir, config_path):
    """Run or safely resume all locked v3 preprocessing ablations."""
    config = load_config(config_path)
    methods = list(config["preprocessing_methods"])
    seeds = [int(seed) for seed in config["seeds"]]

    if config["deterministic_ops"]:
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

    _set_seed(seeds[0], bool(config["deterministic_ops"]))

    import tensorflow as tf
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    from tensorflow.keras.callbacks import Callback, EarlyStopping
    from tensorflow.keras.utils import to_categorical

    from model import build_cnn

    gpus = tf.config.list_physical_devices("GPU")
    if bool(config["require_gpu"]) and not gpus:
        raise RuntimeError(
            "v3 requires a GPU, but TensorFlow found none. In Colab select "
            "Runtime > Change runtime type > GPU, reconnect, and run again."
        )
    print("TensorFlow GPUs:", [gpu.name for gpu in gpus], flush=True)

    os.makedirs(out_dir, exist_ok=True)
    histories_dir = os.path.join(out_dir, "histories")
    predictions_dir = os.path.join(out_dir, "predictions")
    parameters_dir = os.path.join(out_dir, "preprocessing_parameters")
    for directory in (histories_dir, predictions_dir, parameters_dir):
        os.makedirs(directory, exist_ok=True)

    X, y, codes, sentence_ids = build_binary_dataset(
        trt_dir, labels_csv, return_sentence_ids=True
    )
    Y = to_categorical(y, num_classes=2)
    expected_folds = len(methods) * len(seeds) * len(codes)

    metrics_path = os.path.join(out_dir, "fold_metrics.csv")
    metadata_path = os.path.join(out_dir, "run_metadata.json")
    current_metadata = _metadata(config, trt_dir, labels_csv, config_path)
    current_metadata["expected_folds"] = expected_folds
    if os.path.exists(metrics_path):
        if not os.path.exists(metadata_path):
            raise ValueError(
                "fold_metrics.csv exists without run_metadata.json; use a new output directory"
            )
        with open(metadata_path, "r", encoding="utf-8") as stream:
            existing_metadata = json.load(stream)
        _validate_resume_metadata(existing_metadata, current_metadata)
        metrics = pd.read_csv(metrics_path)
    else:
        _atomic_json(current_metadata, metadata_path)
        metrics = pd.DataFrame()

    completed = set()
    if not metrics.empty:
        completed = set(zip(
            metrics["preprocessing"].astype(str),
            metrics["seed"].astype(int),
            metrics["participant"].astype(str),
        ))

    class ValidationMacroF1(Callback):
        def __init__(self, x_val, y_val):
            super().__init__()
            self.x_val = x_val
            self.y_val = y_val
            self.records = []

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            probabilities = self.model.predict(self.x_val, verbose=0)
            prediction = np.argmax(probabilities, axis=1)
            macro_f1 = f1_score(
                self.y_val, prediction, average="macro", zero_division=0
            )
            logs["val_macro_f1"] = float(macro_f1)
            self.records.append({
                "epoch": int(epoch + 1),
                "train_loss": float(logs.get("loss", np.nan)),
                "train_accuracy": float(logs.get("accuracy", np.nan)),
                "val_loss": float(logs.get("val_loss", np.nan)),
                "val_accuracy": float(logs.get("val_accuracy", np.nan)),
                "val_macro_f1": float(macro_f1),
            })

    for method in methods:
        for seed in seeds:
            for held_out, code in enumerate(codes):
                key = (method, seed, code)
                if key in completed:
                    print(
                        f"skip method={method} seed={seed} participant={code}: complete",
                        flush=True,
                    )
                    continue

                tf.keras.backend.clear_session()
                _set_seed(seed, bool(config["deterministic_ops"]))

                x_test_raw = X[held_out]
                x_train_raw = np.concatenate(
                    [X[index] for index in range(len(codes)) if index != held_out],
                    axis=0,
                )
                y_train_int = np.concatenate(
                    [y for index in range(len(codes)) if index != held_out]
                )
                y_train = np.concatenate(
                    [Y for index in range(len(codes)) if index != held_out], axis=0
                )

                # Preserve v2's validation behavior so preprocessing is the only change.
                split_at = int(
                    len(x_train_raw) * (1.0 - float(config["validation_split"]))
                )
                x_fit_raw, x_val_raw = x_train_raw[:split_at], x_train_raw[split_at:]
                y_fit, y_val = y_train[:split_at], y_train[split_at:]
                y_val_int = y_train_int[split_at:]

                x_fit, x_val, x_test, parameters = apply_preprocessing(
                    method, x_fit_raw, x_val_raw, x_test_raw, config
                )
                if not parameters.empty:
                    parameters.insert(0, "participant", code)
                    parameters.insert(0, "seed", seed)
                    parameters.insert(0, "preprocessing", method)
                    _atomic_csv(
                        parameters,
                        os.path.join(
                            parameters_dir, f"{method}_seed_{seed}_{code}.csv"
                        ),
                    )

                # Add only the Conv2D feature-map axis after preprocessing.
                x_fit = x_fit[..., None]
                x_val = x_val[..., None]
                x_test = x_test[..., None]

                model = build_cnn(x_fit.shape[1:], lr=float(config["learning_rate"]))
                validation_metrics = ValidationMacroF1(x_val, y_val_int)
                stop = EarlyStopping(
                    monitor="val_accuracy",
                    patience=int(config["early_stopping_patience"]),
                    mode="max",
                    restore_best_weights=True,
                    start_from_epoch=int(config["early_stopping_start_epoch"]),
                )

                started = time.perf_counter()
                model.fit(
                    x_fit,
                    y_fit,
                    validation_data=(x_val, y_val),
                    epochs=int(config["epochs"]),
                    batch_size=int(config["batch_size"]),
                    callbacks=[validation_metrics, stop],
                    shuffle=True,
                    verbose=0,
                )
                duration_seconds = time.perf_counter() - started

                probabilities = model.predict(x_test, verbose=0)
                prediction = np.argmax(probabilities, axis=1)
                report = classification_report(
                    y, prediction, output_dict=True, zero_division=0
                )["macro avg"]
                matrix = confusion_matrix(y, prediction, labels=[0, 1])
                unique_classes = int(np.unique(prediction).size)
                collapsed = unique_classes == 1

                history = pd.DataFrame(validation_metrics.records)
                history.insert(0, "participant", code)
                history.insert(0, "seed", seed)
                history.insert(0, "preprocessing", method)
                _atomic_csv(
                    history,
                    os.path.join(histories_dir, f"{method}_seed_{seed}_{code}.csv"),
                )

                predictions = pd.DataFrame({
                    "preprocessing": method,
                    "seed": seed,
                    "participant": code,
                    "sample_index": np.arange(len(y)),
                    "sentence_id": sentence_ids,
                    "true_label": y,
                    "predicted_label": prediction,
                    "probability_negative": probabilities[:, 0],
                    "probability_positive": probabilities[:, 1],
                })
                _atomic_csv(
                    predictions,
                    os.path.join(predictions_dir, f"{method}_seed_{seed}_{code}.csv"),
                )

                eligible_history = history[
                    history["epoch"] > int(config["early_stopping_start_epoch"])
                ]
                if eligible_history.empty:
                    eligible_history = history
                best_index = int(eligible_history["val_accuracy"].idxmax())

                row = {
                    "version": config["version"],
                    "preprocessing": method,
                    "seed": seed,
                    "participant": code,
                    "precision": float(report["precision"]),
                    "recall": float(report["recall"]),
                    "f1": float(report["f1-score"]),
                    "accuracy": float(accuracy_score(y, prediction)),
                    "collapsed": bool(collapsed),
                    "constant_probability_output": bool(
                        np.std(probabilities[:, 1].astype(np.float64)) < 1e-12
                    ),
                    "unique_predicted_classes": unique_classes,
                    "probability_positive_mean": float(np.mean(probabilities[:, 1])),
                    "probability_positive_std": float(
                        np.std(probabilities[:, 1].astype(np.float64))
                    ),
                    "probability_positive_min": float(np.min(probabilities[:, 1])),
                    "probability_positive_max": float(np.max(probabilities[:, 1])),
                    "predicted_negative": int(np.sum(prediction == 0)),
                    "predicted_positive": int(np.sum(prediction == 1)),
                    "tn": int(matrix[0, 0]),
                    "fp": int(matrix[0, 1]),
                    "fn": int(matrix[1, 0]),
                    "tp": int(matrix[1, 1]),
                    "best_epoch_by_val_accuracy": int(history.loc[best_index, "epoch"]),
                    "best_val_accuracy": float(eligible_history["val_accuracy"].max()),
                    "best_val_macro_f1": float(history["val_macro_f1"].max()),
                    "stopping_epoch": int(history["epoch"].max()),
                    "n_fit": int(len(x_fit)),
                    "n_validation": int(len(x_val)),
                    "n_test": int(len(x_test)),
                    "duration_seconds": float(duration_seconds),
                    **_partition_stats(x_fit_raw, "raw_fit"),
                    **_partition_stats(x_test_raw, "raw_test"),
                    **_partition_stats(x_fit[..., 0], "transformed_fit"),
                    **_partition_stats(x_test[..., 0], "transformed_test"),
                    **_distribution_shift(x_fit_raw, x_test_raw, "raw"),
                    **_distribution_shift(x_fit[..., 0], x_test[..., 0], "transformed"),
                }
                metrics = pd.concat([metrics, pd.DataFrame([row])], ignore_index=True)
                metrics = metrics.sort_values(
                    ["preprocessing", "seed", "participant"]
                ).reset_index(drop=True)
                _atomic_csv(metrics, metrics_path)
                _write_summaries(metrics, out_dir, expected_folds)
                completed.add(key)

                print(
                    f"[{len(metrics):>3}/{expected_folds}] method={method:<20} "
                    f"seed={seed:>3} participant={code} f1={row['f1']:.3f} "
                    f"collapsed={collapsed} (saved)",
                    flush=True,
                )

    with open(metadata_path, "r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    metadata["completed_utc"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(metadata, metadata_path)
    _write_summaries(metrics, out_dir, expected_folds)
    return metrics.sort_values(
        ["preprocessing", "seed", "participant"]
    ).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(
        description="TeCo EEG-only v3 train-only preprocessing ablations"
    )
    parser.add_argument("--trt-dir", required=True)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--config", default="configs/v3_preprocessing.json")
    args = parser.parse_args()
    run_v3_preprocessing(args.trt_dir, args.labels_csv, args.out_dir, args.config)


if __name__ == "__main__":
    main()
