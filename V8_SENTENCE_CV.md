# V8 five sentence folds

## Fixed question and setup

Does V8's seen-reader/unseen-sentence score depend on the one original22-sentence test set? Same raw126-channel TRT_total inputs,49 word slots,12 separate readers, CNN and training settings. Real and arbitrary sentence labels, initialization seeds45/62. No hyperparameter search, reader averaging, missing-input exclusions or saved model weights.

Five disjoint22-sentence groups,11 negative/11 positive each. For fold i, test group i, validate group(i+1)%5, fit the remaining66 sentences. Each model fits792 examples, validates264, tests264. Every sentence is tested once per reader/seed/target.20 total models,240 reader evaluation rows,5,280 predictions across both targets. The first four models are existing verified V8 jobs; **16 new fits** remain.

## Reuse without changing the original split

Group0 is the old V8 unseen-test22; group1 is its old validation22. Its old fit66 are subdivided deterministically into groups2/3/4 using seed20261102, jointly stratifying real and arbitrary labels. Both targets have11/11 in every group, and the original global arbitrary mapping stays fixed. This anchoring uses the saved partition definition, never prediction outcomes. Fold0's fit/validation/test and mapping match the original plan exactly.

Original4 ZIPs are read and verified against their original metadata/config/plan. Copied CSV/history bytes stay unchanged; a NEW manifest records original SHA256s and references the new fold metadata. Source results are never overwritten. Missing/corrupt/mismatched source jobs stop the run. Existing weights are unnecessary because predictions suffice for the unchanged first test fold; new test partitions require newly trained models.

Data hashes, core model/loader/training utility hashes, full base config and TensorFlow/numpy/pandas/sklearn versions must match the original runs. Runtime drift stops rather than silently mixing environments. Preserve that error for review; no automatic package installation occurs.

## Run

Use `notebooks/08c_v8_sentence_cv.ipynb` from tag `v8-sentence-cv-v1`. Run sections1–4 with a GPU. Section2 only checks data/plans and imports results; it does not train. Section3 runs/resumes the16 missing models. Section4 verifies and displays the summaries. The runner uses local Colab primary results and a NEW Drive mirror `v8_sentence_cv_5fold_45_62`.

Outputs include immutable fold plans, verified per-fit ZIPs of predictions/metrics/history/config, original-job provenance, fold metrics, all out-of-fold predictions, summaries and matched V4 scores. No EEG cache or model weights are saved. Expected result footprint roughly1–5MB. A cloud-local backup ZIP is created; no automatic Mac download. Mounted readback does not itself prove server persistence.

## Metric interpretation

- `summary_fold_mean.csv`: average of120 reader/fold/seed macro-F1 scores per target.
- `summary_oof_110.csv`: combine each reader/seed's five test blocks into110 out-of-fold predictions, compute macro-F1, then average24 reader/seed scores. This differs mathematically from the mean of five22-sentence F1s.
- `v4_matched_fold_summary.csv`: existing V4 predictions rescored on exactly the same five sentence groups. V4 still trained on these sentence identities through other readers; this is not a retrained sentence-holdout V4 experiment.

The comparison changes reader familiarity, fitting pool size and sentence overlap. It cannot isolate one causal factor or establish a p-value. The anchored original split and prior tuning make this exploratory.

## Paper context

The paper identified from the other thesis chat is Zhang et al., *Is EEG-to-Text Feasible in Real-World Scenarios? An In-Depth Analysis Using a Neuropsychology-Inspired Benchmark*, ACL2026: https://aclanthology.org/2026.acl-long.61/ . Section4.3/Table3 compares EEG-similarity conditions; participant/time identity dominates sentence identity there. That is an EEG similarity result, not a theorem that this TeCo sentiment CNN must perform better with seen readers and unseen sentences than with unseen readers and seen sentences. Section4.4/Figure5 concerns learned EEG–text-embedding alignment, including repetitions of familiar sentences, not our binary classifier protocol.
