# V7: bounded new-reader / unseen-sentence diagnostic

Fixed release: `v7-unseen-sentences-v1`.

Open `notebooks/07_v7_unseen_sentences.ipynb` in Colab, choose a GPU, then run its four sections: setup, no-training preflight, training/resume, result verification. Dependencies are expected in the existing Colab runtime; there is no installer. Do not change the frozen settings or bypass a preflight failure.

## Why this version

V4's real-label CNN reached mean macro-F1 0.6701 at LR=0.0001. V5's matched arbitrary-sentence-label CNN reached 0.6094. Both reused sentence identities across training and test readers. V7 separates reader generalization from transfer to new sentence identities without starting a model search. It does not replace V6's separate paired-reader question; V6 is deferred.

## Frozen design

- Same original CNN, word-level `TRT_total` loader, unscaled/padded `(49,126,1)` input, LR=0.0001, batch size 64, maximum 100 epochs, macro-F1 checkpointing, patience 40, start epoch 10, initializations 45/62.
- One stratified split of 110 sentences: 66 fit (33/33), 22 validation (11/11), 22 unseen test (11/11). Split seed 20261025 is fixed before seeing V7 results.
- For each of the 12 held-out readers: ten readers supply the 66 fitting sentences (660 examples). The next reader in the original ordering supplies the 22 validation sentences. No held-out-reader data is fitted or used for checkpoint selection.
- Unseen test identities occur in neither fitting nor checkpoint-selection data, across ALL readers.
- Evaluate the same model on the held-out reader's 22 unseen sentences AND 66 seen identities separately. The latter observations are from a new reader, not the training rows. No extra fitting is needed for that comparison.
- Two conditions: true sentiment and one arbitrary sentence mapping shared across readers. Shuffle within each whole sentence partition (not within classes), maintaining the same class counts. Mapping seed 908172 is fixed independently of initialization and used for BOTH initialization seeds. This is not the earlier V5 mapping.
- 12 readers x 2 initializations x 2 target conditions = **48 fits / 96 evaluation rows**. Estimated GPU runtime 1–2 hours, not a guarantee. Heartbeat every 30 seconds; completed fits resume automatically.

## Preflight and outputs

Output: `Thesis/Results/reproduce_yaser_codes/eegonly/v7_unseen_sentences_45_62`.

Reference: completed `v4_low_lr_confirmation_45_62/run_metadata.json`. Preflight requires unchanged data fingerprints, architecture/loader/training utility hashes, training settings and package versions. Existing V7 data/code/config and actual runtime must match before resuming. Original results and code are not modified.

Duplicate normalized sentence texts/IDs are rejected before training. Exact nonzero-array overlap between fitting, validation, or either held-out-reader evaluation is rejected. All-zero samples are recorded and retained. These checks do NOT exclude near-duplicate stimuli or upstream preprocessing contamination. Globally fixed zero-padding shape follows earlier work; no scaler or feature transform is fitted on validation/test values.

Saved artifacts: `run_metadata.json`, `preflight.json`, `sentence_plan.json/.csv`, `raw_array_audit.csv`, 48 histories, 96 per-sentence prediction files, `fold_metrics.csv`, `summary_by_seed.csv`, `summary_overall.csv`, `completion.json`. Notebook section 4 independently recomputes accuracy, macro-F1 and correct counts from every prediction file, checks targets/splits and saves `paired_seen_vs_unseen.csv` and `paired_real_vs_shuffled.csv`.

## Read the result cautiously

- Ask whether unseen-sentence real-target performance remains useful, whether the arbitrary-target control persists, and whether seen/unseen behavior differs under the SAME fitted model. Report both seeds, all readers, collapse counts, F1 and accuracy; do not discard failed folds.
- Only one sentence split and one random mapping are used. The 22 unseen sentence identities are shared across readers and seeds: 528 unseen predictions per condition are NOT 528 independent stimuli. No p-value is produced.
- This is not a pristine external test: earlier work used all these sentences to choose settings. It is a held-out-in-this-run diagnostic.
- Compared with V4/V5, fitting decreases from 1100 to 660 examples and checkpoint validation from 110 to 22. A drop from 0.670 cannot be attributed solely to removing stimulus overlap. Seen and unseen views also contain different stimuli and sample counts (66 versus 22), so their difference is descriptive, not a controlled estimate of an overlap effect.
- Good unseen-sentence performance would support transfer beyond exact shared stimuli in this data; it would not by itself establish EEG-neural sentiment specificity rather than length/missingness or other confounds.
- The arbitrary-target control should lose the shared-ID lookup opportunity on unseen sentences. It need not score exactly 50% accuracy, especially with only 22 shared stimuli; macro-F1 is not accuracy. A constant predictor has macro-F1 1/3 on these balanced binary sets.
- Review results before more runs. No automatic extra seeds, split search, V6, or parameter sweep.

## Verification status

Ten local tests passed using already-installed Python/numpy/pandas only; no installations or dataset downloads. The end-to-end test uses synthetic arrays and fake networks to check 48-fit scheduling, train/validation/test construction, balance, serialization, all 96 result files, actual notebook result-cell execution, summaries, and no-refit resume. Source/notebook/config syntax checks passed. Real TensorFlow/GPU training and real-data preflight are unrun by the assistant and must be verified in Colab.
