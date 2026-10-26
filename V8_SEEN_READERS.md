# V8 — all readers, unseen sentences

Fixed release: `v8-seen-readers-v1`. Notebook: `notebooks/08_v8_seen_readers_unseen_sentences.ipynb`.

## Question and relation to V7

The user requested testing unseen sentences while training and testing on all readers. Unlike V7's held-out readers, every V8 test reader is represented in fitting. Train, validation and test SENTENCE identities remain completely disjoint across everybody. This is sentence holdout, not a random reader/sentence-row split, and not reader averaging.

V7's 48 fits/96 metrics rows are present, but final independent verification failed on a missing prediction file (`sentiment_seed_45_AYB_seen_sentences.csv`). V7 results remain provisional. V8 uses only V7's saved, independently regenerated split/mapping and metadata; it neither overwrites V7 nor silently certifies its predictions. No V7 retraining is scheduled here.

## Input and protocol

- One example: one reader reading one sentence, `(49,126,1)` padded word/channel features. No averaging over readers, words, or sentences. `TRT_total` itself is a precomputed EEG summary over a word's reading-time window; it is not continuous EEG. “Raw” means no additional scaling of these features.
- Same V7 sentence split/mapping: split seed 20261025, mapping seed 908172, stratified 66 fit / 22 validation / 22 test sentences. Labels exactly balanced within each split, including the arbitrary mapping.
- Fit all 12 readers' 66 sentences separately: 792 examples (396/class). Validate on all 12 readers' DIFFERENT 22 sentences: 264 (132/class). Test all 12 on the final 22 unseen sentences: 264 (132/class). Test identities are never fitted or used for checkpoint selection.
- Same tested original CNN, LR=0.0001, batch 64, maximum 100 epochs, macro-F1 early stopping (patience 40, start epoch 10), initializations 45/62.
- One shared model per initialization/condition. Two conditions (real/arbitrary) x two initializations = **four fits**, not 48. Estimated GPU runtime 10–20 minutes, varying with hardware and stopping. One arbitrary mapping reused across both initializations; not a formal permutation test.
- Reader-specific predictions and metrics remain separate. Mean macro-F1 summarizes 12 readers x two seeds per condition; pooled correct counts/accuracy also reported. These are not 24 independent datasets.

## Preflight / running

Choose a GPU and run notebook sections 1–4. No packages installed, no new datasets downloaded. Read existing `TRT_Total` files, labels, `v4_low_lr_confirmation_45_62` metadata, and V7's metadata/`sentence_plan.json` from Drive. Require identical architecture, data, training helper and package versions; recompute V7 split/map; check sentence text/ID alignment across all readers; reject exact nonzero-array overlaps between sentence partitions. Retain and count all-zero samples. Near-duplicates/upstream contamination remain outside these exact checks.

Output: `Thesis/Results/reproduce_yaser_codes/eegonly/v8_seen_readers_unseen_sentences_45_62`. Original code/results are unchanged.

## Durable artifacts and resume

Each of four authoritative `jobs/<condition>_seed_<seed>.zip` archives contains:

- `predictions.csv`: all 264 held-out-sentence predictions, including reader, sentence ID, target, classes and probabilities.
- `metrics_by_reader.csv`: 12 reader-specific metric rows and shared training definition/duration.
- `history.csv`: selected training history.
- `weights.npz`: ordered inference weights (`model.get_weights` arrays; no optimizer state).
- `manifest.json`: complete config/plan, artifact checksums, metadata fingerprint and weight count. Saving reloads weights into a fresh original CNN and checks identical test predictions before publishing.

Build and verify each ZIP in a Colab temporary directory, copy to Drive, check full archive hash and reopen it before counting the job as saved. On an error retain the local recovery directory and stop. Normal local temporary-job cleanup occurs only after verified publication. Expected Drive storage roughly 10–15 MB (four compressed inference-weight sets plus predictions); no raw EEG is copied into output.

Resume and `--verify-only` independently revalidate ZIP checksums, targets, probabilities, classes, confusion counts, F1, accuracy and sample counts. Incomplete/corrupt archives cause a stop, not silent overwriting/retraining. Missing derived CSV summaries can be rebuilt from the ZIPs without training. `restore_weights(model, zip_path)` can restore inference weights into the fixed architecture for later evaluation.

Derived files: `fold_metrics.csv` (48 rows), `summary_by_seed.csv`, `summary_overall.csv`, `completion.json`, `sentence_plan.json/.csv`, `preflight.json`, and `run_metadata.json`. Section 4 independently verifies all 1,056 predictions across four fits before reporting completion. This does not make 1,056 independent test stimuli: only 22 sentence identities are held out.

## Interpretation and stop rule

This answers seen-reader/unseen-sentence performance, not new-reader performance. Comparing V8 with V7 changes reader familiarity, fitting-reader count (10→12), validation-reader count (1→12), and checkpoint validation composition. It is not a controlled causal estimate of reader familiarity alone. No architecture, preprocessing or learning-rate sweep is permitted, and no statistical significance claim follows from one split/mapping. Scores could still exploit non-neural covariates such as missingness or sentence length.

Review both targets and both seeds, accuracy/correct counts, per-reader variation and one-class outputs, then decide whether to stop. V6 and the separate V7 missing-file investigation remain open; neither is automatically run.

## Verification

Six local tests, including synthetic four-fit end-to-end execution with fake networks, cover disjoint sentence partitions with all readers present, exact overlap rejection, split/mapping equality to V7, weight persistence/restoration, archive/prediction tamper rejection, metadata mismatch rejection, summary rebuilding, actual notebook result-cell execution and no-refit resume. No actual ML training, dependency installations, environments or dataset downloads are performed locally. Colab real-data preflight and GPU training remain unrun by the assistant.
