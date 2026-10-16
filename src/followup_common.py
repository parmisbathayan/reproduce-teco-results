"""Shared, locked utilities for the TeCo EEG-only v4–v6 follow-ups.

The test unit is a participant. The same 110 sentence identities occur in every
participant, so these experiments do not establish generalization to new text.
"""

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from data import PARTICIPANTS, build_binary_dataset


def atomic_csv(frame, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_json(value, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path):
    return {
        "name": os.path.basename(path),
        "size_bytes": os.path.getsize(path),
        "sha256": sha256(path),
    }


def set_seed(seed, deterministic=True):
    import random
    import tensorflow as tf

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    if deterministic:
        tf.config.experimental.enable_op_determinism()


def prepare(config, trt_dir, labels_csv, out_dir, config_path, source_paths):
    """Load data and refuse to mix resumed runs with changed code/data/config."""
    import sklearn
    import tensorflow as tf

    if config["deterministic_ops"]:
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    set_seed(int(config["seeds"][0]), bool(config["deterministic_ops"]))
    gpus = tf.config.list_physical_devices("GPU")
    if config["require_gpu"] and not gpus:
        raise RuntimeError("GPU required: select a Colab GPU runtime and reconnect.")

    repository = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        commit = subprocess.check_output(
            ["git", "-C", repository, "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    data_paths = [labels_csv] + [
        os.path.join(trt_dir, f"{name}_trt_total.pickle")
        for name, _ in PARTICIPANTS
    ]
    definition = {
        "config": config,
        "data_files": [file_record(path) for path in data_paths],
        "source_files": [file_record(path) for path in source_paths + [config_path]],
    }
    metadata_path = os.path.join(out_dir, "run_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, encoding="utf-8") as stream:
            previous = json.load(stream)
        if previous.get("definition") != definition:
            raise ValueError(
                "Existing output was made with different code, data, or config. "
                "Keep it and choose a new output directory."
            )
    else:
        if os.path.exists(os.path.join(out_dir, "fold_metrics.csv")):
            raise ValueError("Metrics exist without run_metadata.json; use a new directory.")
        atomic_json({
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "definition": definition,
            "git_commit": commit,
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "tensorflow": tf.__version__,
            "gpus": [gpu.name for gpu in gpus],
        }, metadata_path)
    return build_binary_dataset(trt_dir, labels_csv, return_sentence_ids=True)


def participant_split(X, y, codes, held_out, validation=None):
    """One whole training participant validates unseen-participant performance."""
    if isinstance(held_out, str):
        held_out = [held_out]
    remaining = [code for code in codes if code not in held_out]
    if validation is None:
        validation = remaining[0]
    if validation not in remaining:
        raise ValueError("Validation participant must be in the training pool.")
    fit_codes = [code for code in remaining if code != validation]
    fit_indices = [codes.index(code) for code in fit_codes]
    x_fit = np.concatenate([X[index] for index in fit_indices], axis=0)
    y_fit = np.tile(y, len(fit_codes))
    x_val = X[codes.index(validation)]
    return x_fit, y_fit, x_val, y.copy(), fit_codes, validation


def next_validation(codes, held_out):
    index = codes.index(held_out)
    return codes[(index + 1) % len(codes)]


def macro_metrics(y_true, probabilities):
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    prediction = np.argmax(probabilities, axis=1)
    matrix = confusion_matrix(y_true, prediction, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "macro_f1": float(f1_score(y_true, prediction, average="macro", zero_division=0)),
        "collapsed": bool(np.unique(prediction).size == 1),
        "probability_positive_std": float(np.std(probabilities[:, 1].astype(np.float64))),
        "constant_probability_output": bool(
            np.std(probabilities[:, 1].astype(np.float64)) < 1e-12
        ),
        "predicted_negative": int(np.sum(prediction == 0)),
        "predicted_positive": int(np.sum(prediction == 1)),
        "tn": int(matrix[0, 0]), "fp": int(matrix[0, 1]),
        "fn": int(matrix[1, 0]), "tp": int(matrix[1, 1]),
    }


def save_predictions(path, y, sentence_ids, probabilities, identifiers):
    prediction = np.argmax(probabilities, axis=1)
    frame = pd.DataFrame({
        **{key: [value] * len(y) for key, value in identifiers.items()},
        "sentence_id": sentence_ids,
        "true_label": y,
        "predicted_label": prediction,
        "probability_negative": probabilities[:, 0],
        "probability_positive": probabilities[:, 1],
    })
    atomic_csv(frame, path)


def fit_network(model, x_fit, y_fit, x_val, y_val, config, monitor="val_macro_f1"):
    """Use identical macro-F1 checkpointing for the new neural configurations."""
    import tensorflow as tf
    from sklearn.metrics import f1_score

    class ValidationF1(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.records = []

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            predicted = np.argmax(self.model.predict(x_val, verbose=0), axis=1)
            score = f1_score(y_val, predicted, average="macro", zero_division=0)
            logs["val_macro_f1"] = float(score)
            self.records.append({
                "epoch": int(epoch + 1),
                "train_loss": float(logs.get("loss", np.nan)),
                "train_accuracy": float(logs.get("accuracy", np.nan)),
                "val_loss": float(logs.get("val_loss", np.nan)),
                "val_accuracy": float(logs.get("val_accuracy", np.nan)),
                "val_macro_f1": float(score),
            })

    callback = ValidationF1()
    stop = tf.keras.callbacks.EarlyStopping(
        monitor=monitor,
        mode="max",
        patience=int(config["patience"]),
        start_from_epoch=int(config["start_epoch"]),
        restore_best_weights=True,
    )
    model.fit(
        x_fit, tf.keras.utils.to_categorical(y_fit, 2),
        validation_data=(x_val, tf.keras.utils.to_categorical(y_val, 2)),
        epochs=int(config["epochs"]),
        batch_size=int(config["batch_size"]),
        callbacks=[callback, stop],
        shuffle=True, verbose=0,
    )
    history = pd.DataFrame(callback.records)
    eligible = history[history.epoch > int(config["start_epoch"])]
    if eligible.empty:
        eligible = history
    metric = "val_macro_f1" if monitor == "val_macro_f1" else "val_accuracy"
    best = eligible.loc[eligible[metric].idxmax()]
    return model, history, {
        "best_epoch": int(best.epoch),
        "best_validation_macro_f1": float(best.val_macro_f1),
        "best_validation_accuracy": float(best.val_accuracy),
        "stopping_epoch": int(history.epoch.max()),
    }


def aggregate_words(X):
    """Per-channel nonzero word mean + coverage; no participant/test fitting."""
    values = np.asarray(X, dtype=np.float32)
    observed = np.isfinite(values) & (values != 0)
    counts = observed.sum(axis=-2)
    summed = np.where(observed, values, 0).sum(axis=-2)
    means = np.divide(summed, counts, out=np.zeros_like(summed), where=counts != 0)
    coverage = counts / values.shape[-2]
    return np.concatenate([means, coverage], axis=-1).astype(np.float32)


def build_small_cnn(input_shape, lr):
    import tensorflow as tf

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(16, (4, 4), padding="same", activation="relu"),
        tf.keras.layers.AveragePooling2D((3, 3)),
        tf.keras.layers.Conv2D(16, (3, 3), activation="relu"),
        tf.keras.layers.AveragePooling2D((3, 3)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(2, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy", metrics=["accuracy"],
    )
    return model


def append_row(path, row, keys):
    if os.path.exists(path):
        frame = pd.read_csv(path)
        existing = set(map(tuple, frame[keys].astype(str).to_numpy()))
        if tuple(str(row[key]) for key in keys) in existing:
            return frame, False
        frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
    else:
        frame = pd.DataFrame([row])
    frame = frame.sort_values(keys).reset_index(drop=True)
    atomic_csv(frame, path)
    return frame, True
