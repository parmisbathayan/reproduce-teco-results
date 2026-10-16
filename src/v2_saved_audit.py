"""Read-only audit of already saved V2 fold, signal, history, and probability files.

No training or raw EEG loading. Outputs are descriptive: n=12 participants is too
small to treat characteristic associations as established causes of collapse.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd


def write_csv(frame, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    frame.to_csv(path, index=False)


def audit(v2_dir, out_dir):
    metrics = pd.read_csv(os.path.join(v2_dir, "fold_metrics.csv"))
    required = {"seed", "participant", "f1", "collapsed", "stopping_epoch",
                "mean_shift_z_rms", "zero_fraction_difference"}
    if not required.issubset(metrics.columns):
        raise ValueError(f"V2 metrics missing: {sorted(required - set(metrics.columns))}")
    if len(metrics) != 60 or metrics[["seed", "participant"]].duplicated().any():
        raise ValueError("Expected the complete, unique 5-seed × 12-participant V2 table.")
    metrics["collapsed_bool"] = metrics.collapsed.astype(str).str.lower().eq("true")

    by_seed = metrics.groupby("seed", as_index=False).agg(
        folds=("f1", "size"), mean_macro_f1=("f1", "mean"),
        sd_macro_f1=("f1", "std"), collapses=("collapsed_bool", "sum"),
    )
    write_csv(by_seed, os.path.join(out_dir, "by_seed.csv"))

    by_participant = metrics.groupby("participant", as_index=False).agg(
        seeds=("seed", "nunique"), mean_macro_f1=("f1", "mean"),
        sd_macro_f1=("f1", "std"), collapse_count=("collapsed_bool", "sum"),
        mean_shift_z_rms=("mean_shift_z_rms", "first"),
        zero_fraction_difference=("zero_fraction_difference", "first"),
    )
    metadata = pd.read_csv(os.path.join(v2_dir, "participant_metadata.csv"))
    feature = pd.read_csv(os.path.join(v2_dir, "participant_feature_stats.csv"))
    feature = feature[feature.label == "all"].drop(columns="label")
    feature = feature.rename(columns={
        key: f"signal_{key}" for key in feature.columns if key != "participant"
    })
    channels = pd.read_csv(os.path.join(v2_dir, "participant_channel_stats.csv"))
    channel_summary = channels.groupby("participant", as_index=False).agg(
        channel_zero_fraction_median=("zero_fraction", "median"),
        channel_std_median=("std", "median"),
        channel_std_max=("std", "max"),
        channels_with_nan=("nan_fraction", lambda x: int((x > 0).sum())),
    )
    characteristics = (
        by_participant.merge(metadata, on="participant", validate="one_to_one")
        .merge(feature, on="participant", validate="one_to_one")
        .merge(channel_summary, on="participant", validate="one_to_one")
        .sort_values(["collapse_count", "participant"], ascending=[False, True])
    )
    write_csv(characteristics, os.path.join(out_dir, "participant_characteristics.csv"))

    # Holding participant and raw data fixed, does changing only the seed change
    # collapse? This is the sharpest saved-data contrast against a static defect.
    within = metrics[["participant", "seed", "f1", "collapsed_bool",
                      "stopping_epoch", "best_val_macro_f1"]].sort_values(
        ["participant", "seed"]
    )
    write_csv(within, os.path.join(out_dir, "within_participant_seed.csv"))

    probabilities = []
    curves = []
    for row in metrics.itertuples(index=False):
        stem = f"seed_{row.seed}_{row.participant}.csv"
        prediction = pd.read_csv(os.path.join(v2_dir, "predictions", stem))
        history = pd.read_csv(os.path.join(v2_dir, "histories", stem))
        p = prediction.probability_positive.to_numpy(dtype=np.float64)
        probabilities.append({
            "participant": row.participant, "seed": row.seed,
            "collapsed": bool(row.collapsed_bool),
            "probability_positive_min": float(p.min()),
            "probability_positive_max": float(p.max()),
            "probability_positive_std": float(p.std()),
            "constant_probability_output": bool(p.std() < 1e-12),
        })
        curves.append({
            "participant": row.participant, "seed": row.seed,
            "collapsed": bool(row.collapsed_bool),
            "first_train_loss": float(history.loss.iloc[0]) if "loss" in history else
                float(history.train_loss.iloc[0]),
            "final_train_loss": float(history.loss.iloc[-1]) if "loss" in history else
                float(history.train_loss.iloc[-1]),
            "max_validation_macro_f1": float(history.val_macro_f1.max()),
            "max_validation_accuracy": float(history.val_accuracy.max()),
            "epochs": int(len(history)),
        })
    probability_frame = pd.DataFrame(probabilities)
    curve_frame = pd.DataFrame(curves)
    write_csv(probability_frame, os.path.join(out_dir, "probability_audit.csv"))
    write_csv(curve_frame, os.path.join(out_dir, "learning_curve_audit.csv"))

    mixed = by_participant[
        (by_participant.collapse_count > 0) & (by_participant.collapse_count < 5)
    ]
    summary = {
        "folds": len(metrics),
        "collapsed_folds": int(metrics.collapsed_bool.sum()),
        "mixed_outcome_participants": mixed.participant.tolist(),
        "constant_probability_collapses": int(
            probability_frame[
                probability_frame.collapsed & probability_frame.constant_probability_output
            ].shape[0]
        ),
        "interpretation_limit": (
            "Signal characteristics and demographic associations are descriptive "
            "with only 12 participants; within-participant seed contrasts are "
            "stronger evidence for optimization sensitivity than static data defects."
        ),
    }
    with open(os.path.join(out_dir, "audit_summary.json"), "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.v2_dir, args.out_dir), indent=2))


if __name__ == "__main__":
    main()
