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

## Colab

Open either notebook directly in Colab and run its setup cell. The public repository
is cloned without a token or GitHub authorization, and the experiment data and
results remain in Google Drive.

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

Version 2 completed all 60 folds on 2026-09-03. It produced mean macro-F1 0.6018
(fold SD 0.1404) with 10 one-class collapses. Nine collapses occurred at seed 45;
three of five seeds had no collapse. The 50 folds that learned averaged 0.6555, so
the result primarily indicates initialization-sensitive optimization rather than a
fixed set of unusable participants.

## Version 3 preprocessing ablations

`notebooks/03_v3_preprocessing.ipynb` runs the locked configuration in
`configs/v3_preprocessing.json`. It compares raw input, `log1p`, nonzero per-channel
z-scoring, and nonzero robust clipping/IQR scaling across the same five seeds and
12 participants (240 folds).

All fitted statistics use only the fit partition. Validation and held-out participant
data are transformed with frozen fit statistics, and padding/missing-fixation zeros
remain zero. V3 deliberately retains v2's CNN, validation tail, learning rate, and
accuracy-based early stopping so preprocessing is the only experimental change.
Results are saved after every fold to the default Drive folder
`Thesis/Results/reproduce_yaser_codes/eegonly/v3_preprocessing` and safely resume.

## Notes

Each fold tests 110 sentences from one held-out participant. The training remains
stochastic, so compare the per-participant distribution, collapse rate, and mean
across predeclared seeds rather than relying on one run. `--seed` fixes the v1
split and initialization.
