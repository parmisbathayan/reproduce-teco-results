"""Version 2: instrumented, multi-seed reproduction of the EEG-only CNN.

The model, labels, folds, and implicit validation-tail behavior match v1. This version
adds observability and reproducibility evidence; it does not introduce normalization,
new validation selection, or architectural changes. Those belong to later versions.
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


PARTICIPANT_METADATA = {
    "AMB": {"age": 29, "sex": "M", "source_pickle_group": "acc5", "full_folder": "armin"},
    "ARM": {"age": 25, "sex": "M", "source_pickle_group": "acc5", "full_folder": "arya"},
    "EIA": {"age": 29, "sex": "M", "source_pickle_group": "acc3", "full_folder": "eisa"},
    "ELD": {"age": 30, "sex": "F", "source_pickle_group": "acc3", "full_folder": "elahe"},
    "FAS": {"age": 24, "sex": "M", "source_pickle_group": "acc3", "full_folder": "farzam"},
    "JAY": {"age": 26, "sex": "M", "source_pickle_group": "acc3", "full_folder": "javad"},
    "MEB": {"age": 27, "sex": "F", "source_pickle_group": "acc3", "full_folder": "mehrnaz"},
    "NAD": {"age": 26, "sex": "F", "source_pickle_group": "acc3", "full_folder": "nastaran"},
    "RAS": {"age": 29, "sex": "F", "source_pickle_group": "acc5", "full_folder": "razie"},
    "SAS": {"age": 26, "sex": "M", "source_pickle_group": "acc5", "full_folder": "sajad"},
    "SHU": {"age": 29, "sex": "F", "source_pickle_group": "acc5", "full_folder": "sharzad"},
    "AYB": {"age": 29, "sex": "M", "source_pickle_group": "acc5", "full_folder": "yaser"},
}


def load_config(path):
    with open(path, "r", encoding="utf-8") as stream:
        config = json.load(stream)
    required = {
        "version", "seeds", "epochs", "batch_size", "validation_split",
        "learning_rate", "early_stopping_patience", "early_stopping_start_epoch",
        "deterministic_ops",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"config is missing required fields: {sorted(missing)}")
    if not config["seeds"]:
        raise ValueError("config must contain at least one seed")
    if not 0 < float(config["validation_split"]) < 1:
        raise ValueError("validation_split must be between 0 and 1")
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
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _file_fingerprints(trt_dir, labels_csv):
    files = [labels_csv]
    files.extend(
        os.path.join(trt_dir, f"{name}_trt_total.pickle")
        for name, _ in PARTICIPANTS
    )
    records = []
    for path in files:
        records.append({
            "name": os.path.basename(path),
            "size_bytes": os.path.getsize(path),
            "sha256": _sha256(path),
        })
    return records


def _distribution_row(values, participant, label_name):
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = np.isfinite(flat)
    clean = flat[finite]
    if clean.size == 0:
        quantiles = [np.nan] * 5
        mean = std = minimum = maximum = np.nan
    else:
        quantiles = np.quantile(clean, [0.01, 0.05, 0.50, 0.95, 0.99])
        mean = float(clean.mean())
        std = float(clean.std())
        minimum = float(clean.min())
        maximum = float(clean.max())
    nonzero = clean[clean != 0]
    return {
        "participant": participant,
        "label": label_name,
        "n_values": int(flat.size),
        "finite_fraction": float(finite.mean()),
        "zero_fraction": float(np.mean(clean == 0)) if clean.size else np.nan,
        "mean": mean,
        "std": std,
        "min": minimum,
        "q01": float(quantiles[0]),
        "q05": float(quantiles[1]),
        "median": float(quantiles[2]),
        "q95": float(quantiles[3]),
        "q99": float(quantiles[4]),
        "max": maximum,
        "nonzero_fraction": float(nonzero.size / clean.size) if clean.size else np.nan,
        "nonzero_mean": float(nonzero.mean()) if nonzero.size else np.nan,
        "nonzero_std": float(nonzero.std()) if nonzero.size else np.nan,
        "nonzero_median": float(np.median(nonzero)) if nonzero.size else np.nan,
        "nonzero_q95": float(np.quantile(nonzero, 0.95)) if nonzero.size else np.nan,
        "nonzero_q99": float(np.quantile(nonzero, 0.99)) if nonzero.size else np.nan,
    }


def build_participant_audit(X, y, codes):
    """Return global/class feature stats, per-channel stats, and participant metadata."""
    distribution_rows = []
    channel_rows = []
    metadata_rows = []

    for index, code in enumerate(codes):
        participant = X[index]
        distribution_rows.append(_distribution_row(participant, code, "all"))
        distribution_rows.append(_distribution_row(participant[y == 0], code, "negative"))
        distribution_rows.append(_distribution_row(participant[y == 1], code, "positive"))

        word_has_signal = np.any(participant != 0, axis=-1)
        meta = dict(PARTICIPANT_METADATA[code])
        meta.update({
            "participant": code,
            "pickle_name": PARTICIPANTS[index][0],
            "mean_fixated_words": float(word_has_signal.sum(axis=1).mean()),
            "std_fixated_words": float(word_has_signal.sum(axis=1).std()),
            "min_fixated_words": int(word_has_signal.sum(axis=1).min()),
            "max_fixated_words": int(word_has_signal.sum(axis=1).max()),
            "all_zero_sentences": int(np.sum(~np.any(word_has_signal, axis=1))),
        })
        metadata_rows.append(meta)

        for channel in range(participant.shape[-1]):
            values = participant[..., channel].astype(np.float64, copy=False)
            channel_rows.append({
                "participant": code,
                "channel": channel,
                "mean": float(np.nanmean(values)),
                "std": float(np.nanstd(values)),
                "zero_fraction": float(np.mean(values == 0)),
                "nan_fraction": float(np.mean(np.isnan(values))),
            })

    return (
        pd.DataFrame(distribution_rows),
        pd.DataFrame(channel_rows),
        pd.DataFrame(metadata_rows),
    )


def _fold_distribution_shift(x_train, x_test):
    train_mean = np.nanmean(x_train, axis=(0, 1))
    test_mean = np.nanmean(x_test, axis=(0, 1))
    train_std = np.nanstd(x_train, axis=(0, 1))
    test_std = np.nanstd(x_test, axis=(0, 1))
    safe_std = np.where(train_std > 1e-8, train_std, 1.0)
    return {
        "mean_shift_z_rms": float(np.sqrt(np.mean(((test_mean - train_mean) / safe_std) ** 2))),
        "std_ratio_median": float(np.median(test_std / safe_std)),
        "zero_fraction_train": float(np.mean(x_train == 0)),
        "zero_fraction_test": float(np.mean(x_test == 0)),
        "zero_fraction_difference": float(np.mean(x_test == 0) - np.mean(x_train == 0)),
    }


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
            for path in (
                __file__,
                os.path.join(os.path.dirname(__file__), "data.py"),
                os.path.join(os.path.dirname(__file__), "model.py"),
                config_path,
            )
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


def _write_summaries(metrics, out_dir):
    if metrics.empty:
        return

    by_participant = metrics.groupby("participant", as_index=False).agg(
        completed_seeds=("seed", "nunique"),
        precision_mean=("precision", "mean"),
        recall_mean=("recall", "mean"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
        f1_min=("f1", "min"),
        f1_max=("f1", "max"),
        collapse_count=("collapsed", "sum"),
        collapse_rate=("collapsed", "mean"),
    )
    _atomic_csv(by_participant, os.path.join(out_dir, "summary_by_participant.csv"))

    by_seed = metrics.groupby("seed", as_index=False).agg(
        completed_participants=("participant", "nunique"),
        precision_mean=("precision", "mean"),
        recall_mean=("recall", "mean"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
        collapse_count=("collapsed", "sum"),
        collapse_rate=("collapsed", "mean"),
    )
    _atomic_csv(by_seed, os.path.join(out_dir, "summary_by_seed.csv"))

    summary = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "completed_runs": int(len(metrics)),
        "completed_seeds": int(metrics["seed"].nunique()),
        "completed_participants": int(metrics["participant"].nunique()),
        "precision_mean": float(metrics["precision"].mean()),
        "recall_mean": float(metrics["recall"].mean()),
        "f1_mean": float(metrics["f1"].mean()),
        "f1_std": float(metrics["f1"].std(ddof=1)) if len(metrics) > 1 else None,
        "collapse_count": int(metrics["collapsed"].sum()),
        "collapse_rate": float(metrics["collapsed"].mean()),
    }
    _atomic_json(summary, os.path.join(out_dir, "summary_overall.json"))


def run_v2_diagnose_collapse(trt_dir, labels_csv, out_dir, config_path):
    """Run the instrumented v2 experiment and resume safely after interruptions."""
    config = load_config(config_path)
    os.makedirs(out_dir, exist_ok=True)
    histories_dir = os.path.join(out_dir, "histories")
    predictions_dir = os.path.join(out_dir, "predictions")
    os.makedirs(histories_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)

    if config["deterministic_ops"]:
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

    # Set the first seed before TensorFlow is imported by the model module.
    _set_seed(int(config["seeds"][0]), bool(config["deterministic_ops"]))

    import tensorflow as tf
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    from tensorflow.keras.callbacks import Callback, EarlyStopping
    from tensorflow.keras.utils import to_categorical

    from model import build_cnn

    X, y, codes, sentence_ids = build_binary_dataset(
        trt_dir, labels_csv, return_sentence_ids=True
    )
    X = X[..., None]
    Y = to_categorical(y, num_classes=2)

    metrics_path = os.path.join(out_dir, "fold_metrics.csv")
    metadata_path = os.path.join(out_dir, "run_metadata.json")
    current_metadata = _metadata(config, trt_dir, labels_csv, config_path)
    if os.path.exists(metrics_path):
        if not os.path.exists(metadata_path):
            raise ValueError(
                "fold_metrics.csv exists without run_metadata.json; use a new output directory"
            )
        with open(metadata_path, "r", encoding="utf-8") as stream:
            existing_metadata = json.load(stream)
        _validate_resume_metadata(existing_metadata, current_metadata)
    else:
        _atomic_json(current_metadata, metadata_path)

    distribution, channels, participant_meta = build_participant_audit(X[..., 0], y, codes)
    _atomic_csv(distribution, os.path.join(out_dir, "participant_feature_stats.csv"))
    _atomic_csv(channels, os.path.join(out_dir, "participant_channel_stats.csv"))
    _atomic_csv(participant_meta, os.path.join(out_dir, "participant_metadata.csv"))

    if os.path.exists(metrics_path):
        metrics = pd.read_csv(metrics_path)
    else:
        metrics = pd.DataFrame()
    completed = set()
    if not metrics.empty:
        completed = set(zip(metrics["seed"].astype(int), metrics["participant"].astype(str)))

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
            macro_f1 = f1_score(self.y_val, prediction, average="macro", zero_division=0)
            logs["val_macro_f1"] = float(macro_f1)
            self.records.append({
                "epoch": int(epoch + 1),
                "train_loss": float(logs.get("loss", np.nan)),
                "train_accuracy": float(logs.get("accuracy", np.nan)),
                "val_loss": float(logs.get("val_loss", np.nan)),
                "val_accuracy": float(logs.get("val_accuracy", np.nan)),
                "val_macro_f1": float(macro_f1),
            })

    for seed_value in config["seeds"]:
        seed = int(seed_value)
        for held_out, code in enumerate(codes):
            if (seed, code) in completed:
                print(f"skip seed={seed} participant={code}: already complete", flush=True)
                continue

            tf.keras.backend.clear_session()
            _set_seed(seed, bool(config["deterministic_ops"]))

            x_test = X[held_out]
            y_test = y
            x_train = np.concatenate([X[j] for j in range(len(codes)) if j != held_out], axis=0)
            y_train_int = np.concatenate([y for j in range(len(codes)) if j != held_out])
            y_train = np.concatenate([Y for j in range(len(codes)) if j != held_out], axis=0)

            # This explicitly reproduces Keras validation_split behavior: take the final
            # fraction before training-time shuffling. v4 will replace this split.
            split_at = int(len(x_train) * (1.0 - float(config["validation_split"])))
            x_fit, x_val = x_train[:split_at], x_train[split_at:]
            y_fit, y_val = y_train[:split_at], y_train[split_at:]
            y_val_int = y_train_int[split_at:]

            model = build_cnn(X.shape[2:], lr=float(config["learning_rate"]))
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
                y_test, prediction, output_dict=True, zero_division=0
            )["macro avg"]
            matrix = confusion_matrix(y_test, prediction, labels=[0, 1])
            unique_classes = int(np.unique(prediction).size)
            collapsed = unique_classes == 1

            history = pd.DataFrame(validation_metrics.records)
            history.insert(0, "participant", code)
            history.insert(0, "seed", seed)
            history_path = os.path.join(histories_dir, f"seed_{seed}_{code}.csv")
            _atomic_csv(history, history_path)

            predictions = pd.DataFrame({
                "seed": seed,
                "participant": code,
                "sample_index": np.arange(len(y_test)),
                "sentence_id": sentence_ids,
                "true_label": y_test,
                "predicted_label": prediction,
                "probability_negative": probabilities[:, 0],
                "probability_positive": probabilities[:, 1],
            })
            prediction_path = os.path.join(predictions_dir, f"seed_{seed}_{code}.csv")
            _atomic_csv(predictions, prediction_path)

            # EarlyStopping ignores epochs before start_from_epoch. Restrict this
            # diagnostic to the same eligible window so it names the restored epoch.
            eligible_history = history[
                history["epoch"] > int(config["early_stopping_start_epoch"])
            ]
            if eligible_history.empty:
                eligible_history = history
            best_epoch_index = int(eligible_history["val_accuracy"].idxmax())
            best_epoch = int(history.loc[best_epoch_index, "epoch"])
            shift = _fold_distribution_shift(x_fit[..., 0], x_test[..., 0])
            row = {
                "version": config["version"],
                "seed": seed,
                "participant": code,
                "precision": float(report["precision"]),
                "recall": float(report["recall"]),
                "f1": float(report["f1-score"]),
                "accuracy": float(accuracy_score(y_test, prediction)),
                "collapsed": bool(collapsed),
                "unique_predicted_classes": unique_classes,
                "predicted_negative": int(np.sum(prediction == 0)),
                "predicted_positive": int(np.sum(prediction == 1)),
                "tn": int(matrix[0, 0]),
                "fp": int(matrix[0, 1]),
                "fn": int(matrix[1, 0]),
                "tp": int(matrix[1, 1]),
                "best_epoch_by_val_accuracy": best_epoch,
                "best_val_accuracy": float(eligible_history["val_accuracy"].max()),
                "best_val_macro_f1": float(history["val_macro_f1"].max()),
                "stopping_epoch": int(history["epoch"].max()),
                "n_fit": int(len(x_fit)),
                "n_validation": int(len(x_val)),
                "n_test": int(len(x_test)),
                "duration_seconds": float(duration_seconds),
                **shift,
            }
            metrics = pd.concat([metrics, pd.DataFrame([row])], ignore_index=True)
            metrics = metrics.sort_values(["seed", "participant"]).reset_index(drop=True)
            _atomic_csv(metrics, metrics_path)
            _write_summaries(metrics, out_dir)
            completed.add((seed, code))

            print(
                f"seed={seed:>3} participant={code} f1={row['f1']:.3f} "
                f"collapsed={collapsed} epochs={row['stopping_epoch']} (saved)",
                flush=True,
            )

    with open(metadata_path, "r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    metadata["completed_utc"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(metadata, metadata_path)
    _write_summaries(metrics, out_dir)
    return metrics.sort_values(["seed", "participant"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="TeCo EEG-only v2 collapse diagnostics")
    parser.add_argument("--trt-dir", required=True)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--config", default="configs/v2_diagnoseCollapse.json")
    args = parser.parse_args()
    run_v2_diagnose_collapse(args.trt_dir, args.labels_csv, args.out_dir, args.config)


if __name__ == "__main__":
    main()
