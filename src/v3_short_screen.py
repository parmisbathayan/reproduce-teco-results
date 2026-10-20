"""Derive a two-seed V3 screen from compatible completed full-V3 folds.

The original runner, data processing, model, epoch limit, and stopping rule are
unchanged. The full output is preserved. Reused rows and artifacts are copied to
a new screening directory only after their provenance has been checked.
"""

import argparse
import json
import os
import shutil
import tempfile


SCREEN_SEEDS = [42, 45]
TRAINING_SOURCES = {"v3_preprocessing.py", "data.py", "model.py"}


def validate_parent(parent, current):
    """A seed subset is the only permitted experimental change."""
    parent_config = dict(parent["config"])
    screen_config = dict(current["config"])
    parent_seeds = parent_config.pop("seeds")
    screen_seeds = screen_config.pop("seeds")
    if screen_seeds != SCREEN_SEEDS:
        raise ValueError("This screening plan fixes seeds to [42, 45].")
    if not set(screen_seeds).issubset(parent_seeds):
        raise ValueError("Screening seeds were not in the source experiment.")
    if parent_config != screen_config:
        raise ValueError("Only a subset of seeds may change; training settings differ.")
    if parent.get("data_files") != current.get("data_files"):
        raise ValueError("Raw data fingerprints differ; existing folds cannot be reused.")
    for label, metadata in (("source", parent), ("current", current)):
        names = [entry["name"] for entry in metadata["source_files"]]
        if len(names) != len(set(names)) or not TRAINING_SOURCES.issubset(names):
            raise ValueError(f"Missing or duplicate {label} source fingerprints.")
    old_sources = {entry["name"]: entry for entry in parent["source_files"]
                   if entry["name"] in TRAINING_SOURCES}
    new_sources = {entry["name"]: entry for entry in current["source_files"]
                   if entry["name"] in TRAINING_SOURCES}
    if old_sources != new_sources:
        raise ValueError("Training source hashes differ; existing folds cannot be reused.")
    for key in ("tensorflow", "numpy", "pandas", "scikit_learn"):
        if not parent.get(key) or parent.get(key) != current.get(key):
            raise ValueError(f"{key} changed; use the original Colab package versions.")


def prepare_screen(trt_dir, labels_csv, source_out_dir, out_dir, config_path):
    import pandas as pd
    from data import PARTICIPANTS
    from v3_preprocessing import (
        _atomic_csv, _atomic_json, _metadata, _sha256,
        _validate_resume_metadata, _write_summaries, load_config,
    )

    if os.path.realpath(source_out_dir) == os.path.realpath(out_dir):
        raise ValueError("The screening output must be separate from the full run.")
    config = load_config(config_path)
    if config["seeds"] != SCREEN_SEEDS:
        raise ValueError("Use the committed [42, 45] screening config.")
    current = _metadata(config, trt_dir, labels_csv, config_path)
    expected = len(config["preprocessing_methods"]) * len(SCREEN_SEEDS) * len(PARTICIPANTS)
    current["expected_folds"] = expected
    metadata_path = os.path.join(out_dir, "run_metadata.json")
    metrics_path = os.path.join(out_dir, "fold_metrics.csv")
    if os.path.exists(out_dir):
        if not os.path.exists(metadata_path) or not os.path.exists(metrics_path):
            raise ValueError("Screen folder exists without complete initialization; keep it and choose a new folder.")
        with open(metadata_path, encoding="utf-8") as stream:
            previous = json.load(stream)
        _validate_resume_metadata(previous, current)
        if previous.get("screening", {}).get("helper_sha256") != _sha256(__file__):
            raise ValueError("Screen helper changed; use the pinned notebook revision.")
        metrics = pd.read_csv(metrics_path)
        print(f"Screen ready: {len(metrics)}/{expected} saved; {expected - len(metrics)} remain.", flush=True)
        return metrics

    parent_path = os.path.join(source_out_dir, "run_metadata.json")
    with open(parent_path, encoding="utf-8") as stream:
        parent = json.load(stream)
    validate_parent(parent, current)
    all_metrics = pd.read_csv(os.path.join(source_out_dir, "fold_metrics.csv"))
    key_columns = ["preprocessing", "seed", "participant"]
    if all_metrics[key_columns].duplicated().any():
        raise ValueError("Source metrics contain duplicate fold keys.")
    metrics = all_metrics[all_metrics.seed.isin(SCREEN_SEEDS)].copy()
    if metrics.empty:
        raise ValueError("No completed folds from the two screening seeds were found.")
    if not set(metrics.preprocessing).issubset(config["preprocessing_methods"]):
        raise ValueError("Source metrics contain an unknown preprocessing method.")
    if not set(metrics.participant).issubset({code for _, code in PARTICIPANTS}):
        raise ValueError("Source metrics contain an unknown participant.")

    # Prepare a new directory and rename it only after the import is complete.
    # A failed/interrupted initialization cannot alter the full run or present
    # a partially initialized folder as a resumable experiment.
    os.makedirs(os.path.dirname(os.path.abspath(out_dir)), exist_ok=True)
    staging = tempfile.mkdtemp(prefix=".v3_screen_preparing_", dir=os.path.dirname(os.path.abspath(out_dir)))
    imported = []
    for row in metrics.itertuples(index=False):
        filename = f"{row.preprocessing}_seed_{int(row.seed)}_{row.participant}.csv"
        folders = ["histories", "predictions"]
        if row.preprocessing in ("per_channel_zscore", "robust_scale_clip"):
            folders.append("preprocessing_parameters")
        hashes = {}
        for folder in folders:
            source = os.path.join(source_out_dir, folder, filename)
            if not os.path.isfile(source):
                raise ValueError(f"Saved fold artifact is missing: {source}")
            target = os.path.join(staging, folder, filename)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copy2(source, target)
            source_hash = _sha256(source)
            if _sha256(target) != source_hash:
                raise ValueError(f"Copy verification failed: {source}")
            hashes[folder] = source_hash
        imported.append({
            "preprocessing": row.preprocessing, "seed": int(row.seed),
            "participant": row.participant, "artifact_sha256": hashes,
        })
    current["screening"] = {
        "purpose": "Two-seed exploratory screening; not the complete five-seed V3 result.",
        "seed_rationale": "42 is a stable V2 seed; 45 is the collapse-prone V2 seed. Keep all 12 participants and all four methods.",
        "source_output_directory": source_out_dir,
        "source_metadata_sha256": _sha256(parent_path),
        "source_git_commit": parent.get("git_commit"),
        "helper_sha256": _sha256(__file__),
        "imported_folds": imported,
        "source_metrics_rows": int(len(all_metrics)),
    }
    metrics = metrics.sort_values(key_columns).reset_index(drop=True)
    _atomic_csv(metrics, os.path.join(staging, "fold_metrics.csv"))
    _atomic_json(current, os.path.join(staging, "run_metadata.json"))
    _write_summaries(metrics, staging, expected)
    os.rename(staging, out_dir)
    print(f"Reused {len(metrics)} completed fits; {expected - len(metrics)} new fits remain (screen total {expected}).", flush=True)
    print("The original full-V3 output is preserved.", flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trt-dir", required=True)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--source-out-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--config", default="configs/v3_short_screen.json")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    prepare_screen(args.trt_dir, args.labels_csv, args.source_out_dir, args.out_dir, args.config)
    if not args.prepare_only:
        from v3_preprocessing import run_v3_preprocessing
        run_v3_preprocessing(args.trt_dir, args.labels_csv, args.out_dir, args.config)


if __name__ == "__main__":
    main()
