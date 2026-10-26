# Lightweight V7/V8 results — no saved model weights

User requested results and hyperparameters without large model archives. These
are NEW runners and notebooks; historical source tags and result folders remain
untouched. No model weights, checkpoints, raw EEG copies or language models are
saved. Keep only predictions, scores, hyperparameters and small history/audit
records. This supports metric verification, not regenerating predictions without
refitting. Do not promise the latter when no weights are retained.

## Run order

1. Use a Colab GPU and mount the Google account with free space. Its My Drive
   must contain the shared Thesis folder or a shortcut to it. Otherwise edit the
   explicit input/reference paths; do not copy the dataset again.
2. Run V8 `notebooks/08b_v8_light_results.ipynb` sections1–2. Setup creates a tiny
   uniquely named probe in the NEW output folder. Confirm its text/nonzero size
   on the Drive website in the mounted account. Sharing access alone is not
   proof of working uploads, and old-owned files can remain over quota.
3. Set `DRIVE_PERSISTENCE_CHECKED=True` in section3 and run sections3–4.
   Exactly4 fits, unchanged scientific setup, about10–20 GPU minutes (estimate).
4. Run V7 `notebooks/07c_v7_light_recovery.ipynb` sections1–2; confirm its probe;
   set the same storage flag True; run sections3–4. Import10 intact jobs and
   regenerate at most38 missing jobs; about45–90 GPU minutes (estimate).
5. Before disconnecting confirm Drive WEBSITE contains4 V8 ZIPs,48 V7 ZIPs,
   nonzero summaries, run metadata and sentence plan. Send final summary tables
   and V7 comparison output for review. No automatic experiment expansion.

Fixed publication tag for BOTH notebooks: `v7-v8-light-results-v1`.

## Storage

Expected total roughly1–5MB for both runs, estimate (CSV/history compression
varies), NOT100–200MB. The previous estimate included per-fit inference weights,
now removed.50MB free is a comfortable buffer in the newly mounted account.
Each final notebook prints actual file sizes. Optional backups add another copy.

Primary Colab directories: `/content/teco-v8-light-results` and
`/content/teco-v7-light-results`. NEW mirrors inside the existing results tree:
`v8_light_seen_readers_45_62`, `v7_light_recovery_45_62`. Originals untouched.
No automatic download to the Mac; no dependency installation.

Archives are independently verified, then copied with checksums. Resume trusts
complete checked archives, not metric rows. Cloud-runtime primary files remain
available if a Drive copy fails, until runtime deletion. A successful mounted read
still does not prove upload reached Google's server: check the website.

## Frozen scientific definitions

Same CNN, raw49×126 word/channel TRT representation, no reader averaging,
LR0.0001, maximum100epochs, batch64, patience40/start10, seeds45/62. Same fixed
66/22/22 sentence plan and arbitrary mapping as original V7. Do not drop79
zero-feature inputs silently or switch preprocessing during artifact recovery.
V7 uses10 fit readers/one validation reader/one test reader; two evaluation views.
V8 pools12 readers as separate examples; disjoint sentences for fit/val/test.
Lightweight storage is the change; it is not a new hyperparameter search.

Hyperparameters are embedded in `fold_metrics.csv` as `hyperparameters_json` and
in tiny per-job manifests. V7 `comparison_to_original.csv` retains every old/new
disagreement, without selecting favorable reruns. Regenerated predictions are
labeled as such, not falsely presented as recovered original predictions.

## Missing features / interpolation clarification

79/1320 binary TeCo feature inputs are all zero; all12 pickles have165 sentence
records. Empty/missing word features do not prove a whole raw EEG trial is absent.
All13 local input hashes match recorded V7 data. Separate provenance audit lives
in the thesis workspace. No evidence of the hypothesized other-reader-mean fill
was found, but upstream preprocessing is not fully certified. Thesis printed17
mentions noisy trials being removed/interpolated in its general background;
printed62 describes spherical-spline preprocessing without specifying a
cross-reader donor rule. Preserve this uncertainty.
