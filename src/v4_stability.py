"""V4: legacy-collapse probe, then subject-validation/LR stability comparison.

The three learning rates are predeclared. V4 changes validation and checkpointing
as a bundle versus V2; it must not be interpreted as isolating either cause.
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from followup_common import (
    append_row, atomic_csv, atomic_json, fit_network, macro_metrics,
    next_validation, participant_split, prepare, save_predictions, set_seed,
)


def activation_gradient_probe(model, x, y):
    """Descriptive ReLU activity/gradient norms on a fixed 16-sample fit batch."""
    import tensorflow as tf

    x_tensor = tf.convert_to_tensor(x[:16], dtype=tf.float32)
    y_tensor = tf.one_hot(y[:16], 2)
    relu_layers = [
        layer for layer in model.layers
        if getattr(getattr(layer, "activation", None), "__name__", "") == "relu"
    ]
    activation_model = tf.keras.Model(model.inputs, [layer.output for layer in relu_layers])
    activations = activation_model(x_tensor, training=False)
    if not isinstance(activations, (list, tuple)):
        activations = [activations]
    activity = {
        layer.name: {
            "positive_fraction": float(tf.reduce_mean(tf.cast(value > 0, tf.float32))),
            "mean_activation": float(tf.reduce_mean(value)),
        }
        for layer, value in zip(relu_layers, activations)
    }
    # Inference-mode gradients avoid consuming dropout RNG before the legacy
    # replay fit. They diagnose the fixed batch, not the exact stochastic
    # gradient used by the optimizer during training.
    with tf.GradientTape() as tape:
        probability = model(x_tensor, training=False)
        loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_tensor, probability))
    gradients = tape.gradient(loss, model.trainable_variables)
    gradient_norms = {
        variable.name: (float(tf.linalg.global_norm([gradient])) if gradient is not None else None)
        for variable, gradient in zip(model.trainable_variables, gradients)
    }
    return {
        "batch_loss": float(loss),
        "relu_activity": activity,
        "gradient_norms": gradient_norms,
        "zero_gradient_count": int(sum(
            gradient is None or float(tf.linalg.global_norm([gradient])) < 1e-12
            for gradient in gradients
        )),
    }


def run_probe(X, y, codes, sentence_ids, config, out_dir):
    """Replay two fixed V2-style folds; V2 weights were not saved."""
    import tensorflow as tf
    from model import build_cnn

    participant = config["v4_probe_participant"]
    held_out = codes.index(participant)
    x_train = np.concatenate([X[i] for i in range(len(codes)) if i != held_out])
    y_train = np.tile(y, len(codes) - 1)
    split_at = int(len(x_train) * 0.85)
    x_fit, x_val = x_train[:split_at][..., None], x_train[split_at:][..., None]
    y_fit, y_val = y_train[:split_at], y_train[split_at:]
    x_test = X[held_out][..., None]
    rows = []
    for seed in config["v4_probe_seeds"]:
        result_path = os.path.join(out_dir, f"seed_{seed}_{participant}.json")
        if os.path.exists(result_path):
            with open(result_path, encoding="utf-8") as stream:
                rows.append(json.load(stream)["summary"])
            continue
        tf.keras.backend.clear_session()
        set_seed(seed, config["deterministic_ops"])
        model = build_cnn(x_fit.shape[1:], lr=0.001)
        before = activation_gradient_probe(model, x_fit, y_fit)
        started = time.perf_counter()
        model, history, chosen = fit_network(
            model, x_fit, y_fit, x_val, y_val, config, monitor="val_accuracy"
        )
        seconds = time.perf_counter() - started
        after = activation_gradient_probe(model, x_fit, y_fit)
        probabilities = model.predict(x_test, verbose=0)
        summary = {
            "seed": seed, "participant": participant,
            "n_fit": len(x_fit), "n_validation": len(x_val),
            "validation_positive": int(y_val.sum()),
            "validation_negative": int(len(y_val) - y_val.sum()),
            "duration_seconds": seconds, **chosen,
            **macro_metrics(y, probabilities),
        }
        atomic_csv(history, os.path.join(out_dir, f"history_seed_{seed}_{participant}.csv"))
        save_predictions(
            os.path.join(out_dir, f"predictions_seed_{seed}_{participant}.csv"),
            y, sentence_ids, probabilities,
            {"seed": seed, "participant": participant, "experiment": "legacy_replay"},
        )
        atomic_json({
            "caution": "V2 weights were not saved; this is a deterministic replay, not direct inspection of V2 models.",
            "summary": summary, "initial_probe": before, "final_probe": after,
        }, result_path)
        rows.append(summary)
    atomic_csv(pd.DataFrame(rows), os.path.join(out_dir, "probe_summary.csv"))
    return pd.DataFrame(rows)


def run_full(X, y, codes, sentence_ids, config, out_dir):
    import tensorflow as tf
    from model import build_cnn

    metrics_path = os.path.join(out_dir, "fold_metrics.csv")
    for lr in config["v4_learning_rates"]:
        condition = f"subject_val_macro_f1_lr_{lr:g}"
        for seed in config["seeds"]:
            for held_out in codes:
                if os.path.exists(metrics_path):
                    old = pd.read_csv(metrics_path)
                    done = (
                        (old.condition == condition) & (old.seed == seed)
                        & (old.participant == held_out)
                    ).any()
                    if done:
                        continue
                tf.keras.backend.clear_session()
                set_seed(seed, config["deterministic_ops"])
                validation = next_validation(codes, held_out)
                x_fit, y_fit, x_val, y_val, fit_codes, _ = participant_split(
                    X, y, codes, held_out, validation
                )
                model = build_cnn(x_fit.shape[1:] + (1,), lr=lr)
                started = time.perf_counter()
                model, history, chosen = fit_network(
                    model, x_fit[..., None], y_fit, x_val[..., None], y_val, config
                )
                seconds = time.perf_counter() - started
                probabilities = model.predict(X[codes.index(held_out)][..., None], verbose=0)
                identifiers = {
                    "condition": condition, "seed": seed, "participant": held_out,
                }
                stem = f"{condition}_seed_{seed}_{held_out}"
                atomic_csv(
                    history.assign(**identifiers),
                    os.path.join(out_dir, "histories", stem + ".csv"),
                )
                save_predictions(
                    os.path.join(out_dir, "predictions", stem + ".csv"),
                    y, sentence_ids, probabilities, identifiers,
                )
                row = {
                    **identifiers, "learning_rate": lr,
                    "validation_participant": validation,
                    "fit_participants": ",".join(fit_codes),
                    "n_fit": len(x_fit), "n_validation": len(x_val),
                    "n_test": len(y), "duration_seconds": seconds,
                    **chosen, **macro_metrics(y, probabilities),
                }
                frame, _ = append_row(
                    metrics_path, row, ["condition", "seed", "participant"]
                )
                summary = frame.groupby("condition", as_index=False).agg(
                    folds=("macro_f1", "size"), mean_macro_f1=("macro_f1", "mean"),
                    sd_macro_f1=("macro_f1", "std"),
                    collapse_count=("collapsed", "sum"),
                )
                atomic_csv(summary, os.path.join(out_dir, "summary_by_condition.csv"))
                print(
                    f"V4 {len(frame)}/180 {condition} seed={seed} {held_out} "
                    f"F1={row['macro_f1']:.3f} collapse={row['collapsed']}", flush=True,
                )
    return pd.read_csv(metrics_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trt-dir", required=True)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--config", default="configs/v4_v6_followup.json")
    parser.add_argument("--stage", choices=["probe", "full", "both"], default="probe")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as stream:
        config = json.load(stream)
    source_dir = os.path.dirname(__file__)
    X, y, codes, sentence_ids = prepare(
        config, args.trt_dir, args.labels_csv, args.out_dir, args.config,
        [__file__, os.path.join(source_dir, "followup_common.py"),
         os.path.join(source_dir, "model.py"), os.path.join(source_dir, "data.py")],
    )
    if args.stage in ("probe", "both"):
        run_probe(X, y, codes, sentence_ids, config, os.path.join(args.out_dir, "probe"))
    if args.stage in ("full", "both"):
        run_full(X, y, codes, sentence_ids, config, args.out_dir)


if __name__ == "__main__":
    main()
