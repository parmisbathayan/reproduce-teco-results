# Independent V4 rerun

Prepared2026-11-03 at user request after the fivefold V8 result was near chance. The previous V8 extension used TWO initialization seeds45/62, not one. Independent readback of5280 predictions reproduced real OOF F1 .4852447, arbitrary .4358396; original V4 .6700803 remains a different familiar-sentence protocol.

## Fixed experiment

- Independent loader, model construction, validation-checkpoint callback, macro-F1 implementation, archive verification and resume/mirror routines in `src/v4_independent_audit.py`.
- NEW initialization seeds11,23,37; real sentiment and one independent balanced arbitrary sentence mapping (seed20261103), shared across all readers/seeds.36 real +36 arbitrary =72 new fits,7920 test predictions.
- Same raw precomputed TRT_total49-word ×126-channel features, two-softmax CNN and binary-crossentropy/Adam loss as V4. LR1e-4, batch64, max100epochs, warmup10, patience40, restore validation macro-F1 best.
-12 LOSO folds:10 fitting readers×110 sentences, next reader validates110, held-out reader tests110. No reader or word averaging. No per-reader scaling, feature selection or new exclusion rule. Keep zero-feature inputs and all collapsed results.
- Same sentence identities across readers are INTENTIONAL. This experiment tests reproducibility of V4's sentence-overlap score. It does not make V4 an unseen-sentence evaluation or prove sentiment specificity.

## Checks before training

1. Require unchanged original V4 input-file hashes and runtime versions. Verify the reference is the V4 low-LR confirmation setup.
2. Independently parse the label CSV with duplicate/label checks; match normalized Persian text for every native sentence ID across all12 readers. Construct features only from TRT_total, without appending reader IDs, sentence IDs or labels.
3. Compare independently reconstructed feature arrays, label vector, native sentence IDs and reader order with the old loader. Old loader code is a CROSS-CHECK only; training uses independently built arrays.
4. Check reader/example partitions are disjoint. Reject exact nonzero example duplicates and possible same-sentence other-reader mean signatures (all donors or nonzero donors). Zero inputs are counted but retained. These checks do not certify unknown upstream ICA/filtering/interpolation, every donor subset or all near-duplicates.
5. Independently construct the CNN; compare layer signatures, parameter count, initial weights and a forward probe with the original model at each new seed. No training/weights persistence during this check.

## Training and reporting checks

Own checkpoint callback receives only fitting/validation arrays, never the test reader. Warmup/patience/best restoration are tested explicitly. Test predictions occur only after validation-best weights are restored. Recomputed validation F1 must match the selected epoch. Independent test F1 is also compared against sklearn. Train/validation/test F1 are retained for diagnosing fit/generalization gaps.

Each fit creates a verified ZIP containing predictions, validation history, metrics and a manifest tied to immutable metadata. No weights/data caches saved. Derived metrics include hyperparameters. Resume verifies prediction labels, IDs, probability argmax, confusion/F1, job/split/config identity, CRC/SHA256 and eligible validation epoch before skipping a fit. Source results never overwritten. Original output directories must not overlap new local/mirror directories.

Run `notebooks/04d_v4_independent_audit.ipynb`, tag `v4-independent-audit-v1`, sections1–4 on a GPU. Section2 is preflight only; section3 trains/resumes72 fits. Result size expected1–5MB, GPU runtime potentially several hours. Local Colab primary output mirrors to NEW Drive folder `v4_independent_audit_11_23_37`; no automatic download to Mac. Check server-visible persistence before ending runtime.

## Interpretation limits

Reproduction with different seeds is an independent robustness check, not proof of no code/preprocessing issue whatsoever. A high real score alongside a high arbitrary score would support transferable stimulus-linked structure, not necessarily sentiment. Failure of the independent loader/model equivalence checks is a concrete issue to investigate before training. Different new random-label mapping means old/new arbitrary scores are descriptive, not a matched permutation test. Do not keep changing setups to force agreement with an expected outcome.
