# V4 zero-exclusion sensitivity run

Use `notebooks/04e_v4_nonzero_audit.ipynb`, tag `v4-nonzero-audit-v1`.
This supersedes the proposed unfiltered rerun for the user's next experiment,
without overwriting original V4/V8 results or the previously published notebooks.

Entirely zero reader/sentence feature arrays are removed independently per reader
from fit, validation and test, using input values only. No sentence is removed from
another reader simply because it is zero for someone else. The same exclusion mask
is used with real and arbitrary labels. Partial zero words/channels and padding
remain unchanged; raw EEG absence cannot be diagnosed from these feature files.

Preserve the V4 architecture, raw inputs, LR1e-4, seeds11/23/37, 12 reader folds,
one fixed arbitrary sentence-label mapping, validation checkpoint and72-fit plan.
No normalization, imputation, outlier clipping, oversampling or class weighting is
added. The earlier V3 transformations did not establish a consistent improvement.
Any later learned preprocessing must be fit only on fitting readers. Input shape,
finite values, native sentence/text/label alignment, exact nonzero duplicates,
tested cross-reader means and independent loader/model equivalence are checked.
Feature magnitude/coverage summaries are descriptive, not a guarantee that all
upstream artifact removal or interpolation was correct. Unknown preprocessing
provenance and partial missingness remain open.

Expected observed exclusion79/1320 leaves1241 examples; test predictions total
3723 per condition across three seeds. New class balance is reported for every
reader and label condition. Both classes must remain in every fit/val/test split
or execution stops. No balancing changes are selected from test outcomes.
Summary accuracy uses actual retained test counts; reader/seed mean macro-F1
weights readers equally. Saved prediction IDs and variable partition sizes are
verified on resume. Each result also reports a training-majority test baseline.
Old/new scores use different evaluation samples, so changes cannot be attributed
solely to training improvements. Familiar sentences still cross reader partitions
intentionally; this is not unseen-sentence evaluation or proof of sentiment signal.

Run setup, optional original-input zero diagnostic1a, preflight2, training3 and
verification4. GPU required for2–4. Output is NEW
`eegonly/v4_independent_nonzero_11_23_37`, local Colab primary
`/content/teco-v4-nonzero-audit-results`. Estimated results1–5MB, no weights or
EEG caches. No real training was run while preparing this code.
