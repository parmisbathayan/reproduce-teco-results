"""Fixed-rate V4 confirmation across 12 participants and seeds 45/62.

Reuse the four completed 1e-4 fits from the short V4 diagnostic; run the other
20 with identical training settings. This extends participant coverage, but the
learning rate was selected on earlier test results, so evaluation is exploratory.
"""

import argparse
import json
import os
import shutil
import time

from v4_lr_short_screen import validate_reference_metadata


def select_completed_reference(frame):
    selected = frame[
        (frame.learning_rate == 0.0001)
        & frame.seed.isin([45, 62])
        & frame.participant.isin(["JAY", "ARM"])
    ].copy()
    expected = {(seed, code) for seed in (45, 62) for code in ("JAY", "ARM")}
    if len(selected) != 4 or set(zip(selected.seed, selected.participant)) != expected:
        raise ValueError("Expected four unique completed 1e-4 diagnostic fits.")
    if not (selected.preprocessing == "raw").all():
        raise ValueError("The confirmation uses raw input.")
    if not (selected.checkpoint_metric == "val_macro_f1").all():
        raise ValueError("Checkpoint metric differs from the confirmation.")
    if not (selected.validation_split_type == "subject").all():
        raise ValueError("Validation split differs from the confirmation.")
    return selected.sort_values(["seed", "participant"]).reset_index(drop=True)


def summarize(metrics, out_dir):
    from followup_common import atomic_csv, atomic_json

    group = metrics.groupby("seed", as_index=False).agg(
        participants=("macro_f1", "size"), mean_macro_f1=("macro_f1", "mean"),
        sd_macro_f1=("macro_f1", "std"), collapse_count=("collapsed", "sum"),
    )
    atomic_csv(group, os.path.join(out_dir, "summary_by_seed.csv"))
    atomic_json({
        "completed_folds": len(metrics), "expected_folds": 24,
        "complete": bool(len(metrics) == 24),
        "mean_macro_f1": float(metrics.macro_f1.mean()),
        "collapse_count": int(metrics.collapsed.sum()),
        "scope": "12 participants at seeds 45 and 62; LR chosen from the preceding short diagnostic.",
    }, os.path.join(out_dir, "summary_overall.json"))


def run(config_path, trt_dir, labels_csv, source_out_dir, out_dir):
    import numpy as np
    import pandas as pd
    import sklearn
    import tensorflow as tf
    from followup_common import (
        append_row, atomic_csv, atomic_json, fit_network, macro_metrics,
        next_validation, participant_split, prepare, save_predictions,
        set_seed, sha256,
    )
    from model import build_cnn

    with open(config_path, encoding="utf-8") as stream:
        config = json.load(stream)
    if config["seeds"] != [45, 62] or config["learning_rate"] != 0.0001 or config["preprocessing"] != "raw":
        raise ValueError("Use the committed single-rate confirmation plan.")
    if os.path.realpath(source_out_dir) == os.path.realpath(out_dir):
        raise ValueError("Keep the confirmation in a separate output folder.")
    source_metadata_path = os.path.join(source_out_dir, "run_metadata.json")
    with open(source_metadata_path, encoding="utf-8") as stream:
        old_metadata = json.load(stream)
    source_dir = os.path.dirname(__file__)
    print("Checking prior results, code, data, runtime, and GPU...", flush=True)
    X, y, codes, sentence_ids = prepare(
        config, trt_dir, labels_csv, out_dir, config_path,
        [__file__, os.path.join(source_dir, "v4_lr_short_screen.py"),
         os.path.join(source_dir, "followup_common.py"),
         os.path.join(source_dir, "model.py"), os.path.join(source_dir, "data.py"),
         source_metadata_path],
    )
    with open(os.path.join(out_dir, "run_metadata.json"), encoding="utf-8") as stream:
        current_metadata = json.load(stream)
    for key, value in {"tensorflow": tf.__version__, "numpy": np.__version__,
                       "pandas": pd.__version__, "scikit_learn": sklearn.__version__}.items():
        if current_metadata.get(key) != value:
            raise ValueError(f"Runtime changed since this confirmation began: {key}")
    validate_reference_metadata(old_metadata, current_metadata)
    old_sources = {entry["name"]: entry for entry in old_metadata["definition"]["source_files"]}
    new_sources = {entry["name"]: entry for entry in current_metadata["definition"]["source_files"]}
    if old_sources.get("v4_lr_short_screen.py") != new_sources.get("v4_lr_short_screen.py"):
        raise ValueError("The source diagnostic runner changed.")

    metrics_path = os.path.join(out_dir, "fold_metrics.csv")
    if os.path.exists(metrics_path):
        metrics = pd.read_csv(metrics_path)
    else:
        source_metrics_path = os.path.join(source_out_dir, "fold_metrics.csv")
        metrics = select_completed_reference(pd.read_csv(source_metrics_path))
        copied = []
        for row in metrics.itertuples(index=False):
            stem = f"{row.condition}_seed_{int(row.seed)}_{row.participant}.csv"
            record = {"seed": int(row.seed), "participant": row.participant}
            for folder in ("histories", "predictions"):
                source = os.path.join(source_out_dir, folder, stem)
                target = os.path.join(out_dir, folder, stem)
                os.makedirs(os.path.dirname(target), exist_ok=True)
                shutil.copy2(source, target)
                source_hash = sha256(source)
                if sha256(target) != source_hash:
                    raise ValueError(f"Copy failed: {source}")
                record[folder + "_sha256"] = source_hash
            copied.append(record)
        metrics["fit_origin"] = "imported_from_short_diagnostic"
        atomic_json({
            "source_directory": source_out_dir,
            "source_metadata_sha256": sha256(source_metadata_path),
            "source_metrics_sha256": sha256(source_metrics_path),
            "source_git_commit": old_metadata.get("git_commit"),
            "imported_folds": copied,
            "selection": "Freeze LR=1e-4 after the two-participant V4 diagnostic, then evaluate remaining participants without additional LR tuning.",
            "limitation": "Previously evaluated participants and seeds; this is exploratory confirmation rather than independent test-set validation.",
        }, os.path.join(out_dir, "selection_and_import_provenance.json"))
        atomic_csv(metrics, metrics_path)

    expected_keys = {(seed, code) for seed in config["seeds"] for code in codes}
    completed = set(zip(metrics.seed.astype(int), metrics.participant))
    if len(metrics) != len(completed) or not completed.issubset(expected_keys):
        raise ValueError("Duplicate or unexpected saved confirmation folds.")
    if not (metrics.learning_rate == config["learning_rate"]).all():
        raise ValueError("Saved results use a different learning rate.")
    metrics["collapsed"] = metrics.collapsed.astype(str).str.lower().eq("true")
    summarize(metrics, out_dir)
    print(f"Saved {len(metrics)}/24; {24 - len(metrics)} fits remain. Four initial fits were imported.", flush=True)

    for seed in config["seeds"]:
        for held_out in codes:
            if (seed, held_out) in completed:
                continue
            tf.keras.backend.clear_session()
            set_seed(seed, config["deterministic_ops"])
            validation = next_validation(codes, held_out)
            x_fit, y_fit, x_val, y_val, fit_codes, _ = participant_split(X, y, codes, held_out, validation)
            model = build_cnn(x_fit.shape[1:] + (1,), lr=config["learning_rate"])
            print(f"START {len(metrics) + 1}/24: seed={seed}, participant={held_out}. Training may be quiet for 1–3 minutes.", flush=True)
            started = time.perf_counter()
            model, history, chosen = fit_network(
                model, x_fit[..., None], y_fit, x_val[..., None], y_val, config
            )
            seconds = time.perf_counter() - started
            probabilities = model.predict(X[codes.index(held_out)][..., None], verbose=0)
            condition = "subject_val_macro_f1_lr_0.0001"
            identifiers = {"condition": condition, "seed": seed, "participant": held_out}
            stem = f"{condition}_seed_{seed}_{held_out}.csv"
            atomic_csv(history.assign(**identifiers), os.path.join(out_dir, "histories", stem))
            save_predictions(os.path.join(out_dir, "predictions", stem), y, sentence_ids, probabilities, identifiers)
            row = {
                **identifiers, "version": config["version"], "learning_rate": config["learning_rate"],
                "preprocessing": "raw", "validation_split_type": "subject",
                "checkpoint_metric": "val_macro_f1", "validation_participant": validation,
                "fit_participants": ",".join(fit_codes), "n_fit": len(x_fit),
                "n_validation": len(x_val), "n_test": len(y),
                "duration_seconds": seconds, "fit_origin": "new_confirmation_fit",
                **chosen, **macro_metrics(y, probabilities),
            }
            metrics, _ = append_row(metrics_path, row, ["condition", "seed", "participant"])
            metrics["collapsed"] = metrics.collapsed.astype(str).str.lower().eq("true")
            summarize(metrics, out_dir)
            completed.add((seed, held_out))
            print(f"SAVED {len(metrics)}/24: F1={row['macro_f1']:.3f}, collapse={row['collapsed']}, {seconds:.0f}s.", flush=True)
    print("Confirmation complete. Review summary_by_seed.csv and fold_metrics.csv.", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trt-dir", required=True)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--source-out-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--config", default="configs/v4_low_lr_confirmation.json")
    args = parser.parse_args()
    run(args.config, args.trt_dir, args.labels_csv, args.source_out_dir, args.out_dir)


if __name__ == "__main__":
    main()
