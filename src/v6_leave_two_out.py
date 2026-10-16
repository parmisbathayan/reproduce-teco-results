"""V6: fixed paired-participant holdouts, gated on a reviewed CNN selection.

This is an exploratory diagnostic: V3/V4 already use these participants as tests,
so choosing a configuration from their results does not make V6 independent proof.
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from followup_common import (
    append_row, atomic_csv, build_small_cnn, fit_network, macro_metrics,
    participant_split, prepare, save_predictions, set_seed,
)
from v3_preprocessing import SUPPORTED_METHODS, apply_preprocessing


def load_selection(path):
    with open(path, encoding="utf-8") as stream:
        value = json.load(stream)
    if value.get("status") != "LOCKED":
        raise ValueError(
            "V6 selection is not locked. Review V3–V5 first, then create a "
            "separate selection JSON with status=LOCKED and a rationale."
        )
    if value.get("model") not in ("original_cnn", "small_cnn"):
        raise ValueError("V6 supports the original or small CNN only.")
    if value.get("preprocessing") not in SUPPORTED_METHODS:
        raise ValueError("Unsupported V6 preprocessing method.")
    if not value.get("source_results") or not value.get("selection_rationale"):
        raise ValueError("Record both source_results and selection_rationale.")
    if float(value.get("learning_rate", 0)) <= 0:
        raise ValueError("V6 learning_rate must be positive.")
    return value


def run(config, selection, trt_dir, labels_csv, out_dir, config_path,
        selection_path, v3_config_path):
    import tensorflow as tf
    from model import build_cnn

    source_dir = os.path.dirname(__file__)
    X, y, codes, sentence_ids = prepare(
        config, trt_dir, labels_csv, out_dir, config_path,
        [__file__, os.path.join(source_dir, "followup_common.py"),
         os.path.join(source_dir, "v3_preprocessing.py"),
         os.path.join(source_dir, "data.py"),
         os.path.join(source_dir, "model.py"),
         selection_path, v3_config_path],
    )
    with open(v3_config_path, encoding="utf-8") as stream:
        v3_config = json.load(stream)
    metrics_path = os.path.join(out_dir, "fold_metrics.csv")
    expected = len(config["v6_pairs"]) * len(config["seeds"]) * 2
    if os.path.exists(metrics_path):
        completed = set(map(tuple, pd.read_csv(metrics_path)[
            ["pair", "seed", "participant"]
        ].astype(str).to_numpy()))
    else:
        completed = set()

    for first, second in config["v6_pairs"]:
        pair = f"{first}__{second}"
        if first not in codes or second not in codes or first == second:
            raise ValueError(f"Invalid pair: {pair}")
        x_fit_raw, y_fit, x_val_raw, y_val, fit_codes, validation = participant_split(
            X, y, codes, [first, second]
        )
        for seed in config["seeds"]:
            if all((pair, str(seed), target) in completed for target in (first, second)):
                continue
            tf.keras.backend.clear_session()
            set_seed(seed, config["deterministic_ops"])
            x_first_raw = X[codes.index(first)]
            x_second_raw = X[codes.index(second)]
            # Fit preprocessing on the nine fitting participants only. Concatenate
            # the two test sets solely to reuse the same frozen transform.
            x_fit, x_val, x_both, _ = apply_preprocessing(
                selection["preprocessing"], x_fit_raw, x_val_raw,
                np.concatenate([x_first_raw, x_second_raw]),
                v3_config,
            )
            if selection["model"] == "original_cnn":
                model = build_cnn(x_fit.shape[1:] + (1,), lr=selection["learning_rate"])
            else:
                model = build_small_cnn(x_fit.shape[1:] + (1,), selection["learning_rate"])
            n_parameters = int(model.count_params())
            started = time.perf_counter()
            model, history, chosen = fit_network(
                model, x_fit[..., None], y_fit, x_val[..., None], y_val, config
            )
            seconds = time.perf_counter() - started
            probabilities = model.predict(x_both[..., None], verbose=0)
            stem = f"{pair}_seed_{seed}"
            atomic_csv(
                history.assign(pair=pair, seed=seed),
                os.path.join(out_dir, "histories", stem + ".csv"),
            )
            for index, target in enumerate((first, second)):
                part = probabilities[index * len(y):(index + 1) * len(y)]
                identifiers = {"pair": pair, "seed": seed, "participant": target}
                save_predictions(
                    os.path.join(out_dir, "predictions", stem + f"_{target}.csv"),
                    y, sentence_ids, part, identifiers,
                )
                row = {
                    **identifiers,
                    "pair_role": "original_v1_collapsed" if index == 0 else "v1_strong_reference",
                    "model": selection["model"],
                    "preprocessing": selection["preprocessing"],
                    "learning_rate": selection["learning_rate"],
                    "validation_participant": validation,
                    "fit_participants": ",".join(fit_codes),
                    "n_fit": len(x_fit), "n_validation": len(x_val),
                    "n_test": len(y), "n_parameters": n_parameters,
                    "duration_seconds": seconds,
                    **chosen, **macro_metrics(y, part),
                }
                frame, _ = append_row(
                    metrics_path, row, ["pair", "seed", "participant"]
                )
                completed.add((pair, str(seed), target))
            summary = frame.groupby(["pair", "participant", "pair_role"], as_index=False).agg(
                folds=("macro_f1", "size"), mean_macro_f1=("macro_f1", "mean"),
                sd_macro_f1=("macro_f1", "std"),
                collapse_count=("collapsed", "sum"),
            )
            atomic_csv(summary, os.path.join(out_dir, "summary_by_pair_participant.csv"))
            print(
                f"V6 {len(frame)}/{expected} {pair} seed={seed} "
                f"F1={frame.iloc[-2:].macro_f1.tolist()}", flush=True,
            )
    return pd.read_csv(metrics_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trt-dir", required=True)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--config", default="configs/v4_v6_followup.json")
    parser.add_argument("--selection", required=True)
    parser.add_argument("--v3-config", default="configs/v3_preprocessing.json")
    args = parser.parse_args()
    selection = load_selection(args.selection)
    with open(args.config, encoding="utf-8") as stream:
        config = json.load(stream)
    run(config, selection, args.trt_dir, args.labels_csv, args.out_dir,
        args.config, args.selection, args.v3_config)


if __name__ == "__main__":
    main()
