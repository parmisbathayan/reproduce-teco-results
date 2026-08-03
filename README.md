# reproduce-teco-results

EEG-based binary sentiment classification on the Persian TeCo dataset (task 1).

This is the per-participant, EEG-only model: a small 2D CNN trained on the
word-level EEG (total reading time) of 11 participants and tested on the
held-out one, repeated for everyone (leave-one-participant-out). Sentences are
labelled positive / negative; neutral ones are dropped, which leaves 110
sentences (55 / 55).

## Data

The data is not in the repo. You need, somewhere on disk:

- the `TRT_Total` folder with the 12 `<Name>_trt_total.pickle` files
- `teco_sentiment_labels_task1.csv` (the sentiment labels)

Labels are matched to the EEG by sentence text, so the order/id in the CSV
doesn't matter.

## Run

The easy way: open `run.ipynb`, run the cells, and paste in the two data
folders and an output folder when it asks.

From a terminal:

```
pip install -r requirements.txt
python src/per_participant_eeg.py \
    --trt-dir    /path/to/TRT_Total \
    --labels-csv /path/to/teco_sentiment_labels_task1.csv \
    --out-dir    results
```

It prints macro-F1 for each participant and the mean, and writes
`results/per_participant_eeg.csv`.

## Colab and private GitHub access

The repository is private. In Colab, authorize GitHub private-repository access
when opening the notebook. Before running its setup cell, add a Colab Secret named
`GITHUB_TOKEN`, enable notebook access for it, and use a fine-grained GitHub token
limited to this repository with **Contents: Read-only** permission. The setup cell
uses the secret only for cloning and immediately restores a credential-free remote
URL. It does not print or save the token in the notebook.

## Version 2 diagnostics

`notebooks/02_v2_diagnoseCollapse.ipynb` is the instrumented follow-up to the
single-seed reproduction. It uses the locked settings in
`configs/v2_diagnoseCollapse.json` and runs five seeds across all 12 held-out
participants. It preserves the v1 model and split behavior while saving:

- fold metrics, confusion matrices, predictions, and class probabilities;
- epoch histories with validation macro-F1;
- participant, channel, and train/test distribution statistics;
- environment, Git revision, and data fingerprints;
- summaries by participant and seed, including one-class collapse counts.

The runner is crash-safe at the fold level and resumes completed seed/participant
pairs from `fold_metrics.csv`. Its default Drive output is
`Thesis/Results/reproduce_yaser_codes/eegonly/v2_diagnoseCollapse`.

## Notes

Each fold tests 110 sentences from one held-out participant. The training remains
stochastic, so compare the per-participant distribution, collapse rate, and mean
across predeclared seeds rather than relying on one run. `--seed` fixes the v1
split and initialization.
