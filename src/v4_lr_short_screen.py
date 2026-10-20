"""Eight-fit V4 learning-rate diagnostic using four completed reference fits.

Fixed cases: JAY/ARM at seeds 45/62. Raw input, whole-participant validation,
and validation macro-F1 checkpointing match the V4 reference. Only LR changes.
This small, exploratory test cannot establish a population-wide best setting.
"""

import argparse
import json
import os
import time


def validate_reference_metadata(reference, current):
    old = reference["definition"]
    new = current["definition"]
    for key in ("epochs", "batch_size", "patience", "start_epoch", "deterministic_ops"):
        if old["config"].get(key) != new["config"].get(key):
            raise ValueError(f"Reference training setting changed: {key}")
    if old["data_files"] != new["data_files"]:
        raise ValueError("Reference data fingerprints differ.")
    required = {"followup_common.py", "model.py", "data.py"}
    old_sources = {row["name"]: row for row in old["source_files"] if row["name"] in required}
    new_sources = {row["name"]: row for row in new["source_files"] if row["name"] in required}
    if set(old_sources) != required or old_sources != new_sources:
        raise ValueError("Reference model, split, or training utility code changed.")
    for key in ("tensorflow", "numpy", "pandas", "scikit_learn"):
        if not reference.get(key) or reference.get(key) != current.get(key):
            raise ValueError(f"Reference runtime differs: {key}")


def select_reference(frame):
    condition = "subject_val_macro_f1_lr_0.001"
    selected = frame[
        (frame.condition == condition)
        & frame.seed.isin([45, 62])
        & frame.participant.isin(["JAY", "ARM"])
    ].copy()
    expected = {(seed, participant) for seed in (45, 62) for participant in ("JAY", "ARM")}
    if len(selected) != 4 or set(zip(selected.seed, selected.participant)) != expected:
        raise ValueError("All four completed V4 reference fits must exist exactly once.")
    if not (selected.learning_rate == 0.001).all():
        raise ValueError("Reference learning rate must be 0.001.")
    return selected.sort_values(["seed", "participant"]).reset_index(drop=True)


def write_comparison(reference, metrics, out_dir):
    import pandas as pd
    from followup_common import atomic_csv

    comparison = pd.concat([reference, metrics], ignore_index=True)
    comparison["collapsed"] = comparison.collapsed.astype(str).str.lower().eq("true")
    atomic_csv(comparison, os.path.join(out_dir, "comparison_with_reference.csv"))
    summary = comparison.groupby(["learning_rate", "seed"], as_index=False).agg(
        completed_participants=("macro_f1", "size"),
        mean_macro_f1=("macro_f1", "mean"),
        collapse_count=("collapsed", "sum"),
    )
    atomic_csv(summary, os.path.join(out_dir, "summary_by_lr_seed.csv"))


def run(config_path, trt_dir, labels_csv, source_out_dir, out_dir):
    import pandas as pd
    from followup_common import (
        append_row, atomic_csv, atomic_json, fit_network, macro_metrics,
        next_validation, participant_split, prepare, save_predictions,
        set_seed, sha256,
    )

    with open(config_path, encoding="utf-8") as stream:
        config = json.load(stream)
    if (config["seeds"] != [45, 62] or config["participants"] != ["JAY", "ARM"]
            or config["learning_rates"] != [0.0003, 0.0001]
            or config["reference_learning_rate"] != 0.001):
        raise ValueError("Use the committed eight-fit diagnostic plan.")
    if os.path.realpath(source_out_dir) == os.path.realpath(out_dir):
        raise ValueError("The diagnostic must have a separate output directory.")
    source_metadata_path = os.path.join(source_out_dir, "run_metadata.json")
    with open(source_metadata_path, encoding="utf-8") as stream:
        reference_metadata = json.load(stream)
    print("Checking saved V4 reference, data, code, runtime, and GPU...", flush=True)
    source_dir = os.path.dirname(__file__)
    X, y, codes, sentence_ids = prepare(
        config, trt_dir, labels_csv, out_dir, config_path,
        [__file__, os.path.join(source_dir, "followup_common.py"),
         os.path.join(source_dir, "model.py"), os.path.join(source_dir, "data.py"),
         source_metadata_path],
    )
    with open(os.path.join(out_dir, "run_metadata.json"), encoding="utf-8") as stream:
        current_metadata = json.load(stream)
    import numpy as np
    import sklearn
    import tensorflow as tf
    actual_runtime = {
        "tensorflow": tf.__version__, "numpy": np.__version__,
        "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
    }
    for key, value in actual_runtime.items():
        if current_metadata.get(key) != value:
            raise ValueError(f"Runtime changed since this diagnostic began: {key}")
    validate_reference_metadata(reference_metadata, current_metadata)
    source_metrics = os.path.join(source_out_dir, "fold_metrics.csv")
    reference = select_reference(pd.read_csv(source_metrics))
    frozen_path = os.path.join(out_dir, "baseline_reference.csv")
    if os.path.exists(frozen_path):
        frozen = select_reference(pd.read_csv(frozen_path))
        check_columns = ["seed", "participant", "macro_f1", "collapsed", "best_epoch"]
        try:
            pd.testing.assert_frame_equal(
                frozen[check_columns], reference[check_columns],
                check_exact=False, rtol=1e-12, atol=1e-12,
            )
        except AssertionError as error:
            raise ValueError("The saved V4 reference rows changed after this diagnostic began.") from error
    else:
        atomic_csv(reference, frozen_path)
        atomic_json({
            "source_directory": source_out_dir,
            "source_metrics_sha256": sha256(source_metrics),
            "source_metadata_sha256": sha256(source_metadata_path),
            "frozen_reference_sha256": sha256(frozen_path),
            "reference_git_commit": reference_metadata.get("git_commit"),
            "interpretation": "Two participants and two seeds: exploratory diagnostic, not a final model selection estimate.",
        }, os.path.join(out_dir, "baseline_provenance.json"))
    from model import build_cnn

    metrics_path = os.path.join(out_dir, "fold_metrics.csv")
    metrics = pd.read_csv(metrics_path) if os.path.exists(metrics_path) else pd.DataFrame()
    completed = set() if metrics.empty else set(zip(
        metrics.learning_rate, metrics.seed.astype(int), metrics.participant
    ))
    write_comparison(reference, metrics, out_dir)
    print(f"Reference fits reused: 4. New fits saved: {len(metrics)}/8. Missing: {8 - len(metrics)}.", flush=True)
    for seed in config["seeds"]:
        for held_out in config["participants"]:
            validation = next_validation(codes, held_out)
            x_fit, y_fit, x_val, y_val, fit_codes, _ = participant_split(
                X, y, codes, held_out, validation
            )
            for lr in config["learning_rates"]:
                if (lr, seed, held_out) in completed:
                    continue
                tf.keras.backend.clear_session()
                set_seed(seed, config["deterministic_ops"])
                model = build_cnn(x_fit.shape[1:] + (1,), lr=lr)
                print(f"START {len(metrics) + 1}/8: seed={seed}, participant={held_out}, LR={lr:g}. Training may be quiet for 1–3 minutes.", flush=True)
                started = time.perf_counter()
                model, history, chosen = fit_network(
                    model, x_fit[..., None], y_fit, x_val[..., None], y_val, config
                )
                seconds = time.perf_counter() - started
                probabilities = model.predict(X[codes.index(held_out)][..., None], verbose=0)
                condition = f"subject_val_macro_f1_lr_{lr:g}"
                identifiers = {"condition": condition, "seed": seed, "participant": held_out}
                stem = f"{condition}_seed_{seed}_{held_out}"
                atomic_csv(history.assign(**identifiers), os.path.join(out_dir, "histories", stem + ".csv"))
                save_predictions(
                    os.path.join(out_dir, "predictions", stem + ".csv"),
                    y, sentence_ids, probabilities, identifiers,
                )
                row = {
                    **identifiers, "version": config["version"], "learning_rate": lr,
                    "preprocessing": "raw", "validation_split_type": "subject",
                    "checkpoint_metric": "val_macro_f1",
                    "validation_participant": validation,
                    "fit_participants": ",".join(fit_codes),
                    "n_fit": len(x_fit), "n_validation": len(x_val), "n_test": len(y),
                    "duration_seconds": seconds, **chosen, **macro_metrics(y, probabilities),
                }
                metrics, _ = append_row(metrics_path, row, ["condition", "seed", "participant"])
                write_comparison(reference, metrics, out_dir)
                completed.add((lr, seed, held_out))
                print(f"SAVED {len(metrics)}/8: F1={row['macro_f1']:.3f}, collapse={row['collapsed']}, {seconds:.0f}s.", flush=True)
    atomic_json({
        "new_fits_completed": len(metrics), "new_fits_expected": 8,
        "reference_fits": 4, "complete": bool(len(metrics) == 8),
        "scope": "JAY/ARM × seeds 45/62, raw input, participant validation, macro-F1 checkpointing.",
    }, os.path.join(out_dir, "completion.json"))
    print("Complete. Review summary_by_lr_seed.csv and comparison_with_reference.csv.", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trt-dir", required=True)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--source-out-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--config", default="configs/v4_lr_short_screen.json")
    args = parser.parse_args()
    run(args.config, args.trt_dir, args.labels_csv, args.source_out_dir, args.out_dir)


if __name__ == "__main__":
    main()
