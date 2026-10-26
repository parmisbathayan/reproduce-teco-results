# V7 bounded artifact regeneration

This is an artifact repair, not a new scientific setup. The original V7 directory,
code, tag, metrics and split remain untouched. Run only after preserving V8 and
confirming that newly written Drive files really appear on the Drive website.
Mounted-file checks alone did not establish durable persistence in the V8 run.

## Frozen experiment

Same original CNN, raw 49-word-slot ×126-channel TRT features, LR 0.0001,
100 maximum epochs, batch64, patience40, warmup10, initial seeds45/62.
Same split seed20261025 and mapping seed908172. Each fit has ten readers ×66
sentences; a separate validation reader ×22 sentences; the held-out reader is
evaluated separately on66 seen and22 unseen sentences. No reader averaging.

The new runner imports the ten intact jobs only after checking both predictions,
metrics, class counts, IDs, target mapping and history. Missing artifacts require
at most38 controlled regenerations. Corrupt complete jobs stop the run instead
of silently refitting. Runtime/library/data/model/helper fingerprints must match
the old run. No tuning, favorable-result selection or automatic extra seeds.

## Storage and resume

- Primary: `/content/teco-v7-recovery-results` in the Colab runtime.
- Secondary: NEW Drive directory `eegonly/v7_artifact_recovery_45_62`.
- Original: `eegonly/v7_unseen_sentences_45_62`, read-only.
- Each job is a verified ZIP with both prediction views, history, metrics and a
  provenance manifest. Regenerated jobs also retain inference weights, reloaded
  and checked against the original predictions. Original imported jobs never
  claim weights that were not saved.
- Completion/resume is derived from verified ZIPs, never from metric rows.
- Local primary files are retained after copying to Drive. This still cannot
  survive runtime deletion unless the Drive copies sync or a backup is downloaded.
- Final ZIP is prepared by the results notebook cell; download it if needed.
  Confirm all48 job ZIPs and metadata on the Drive WEBSITE before disconnecting.
- `comparison_to_original.csv` reports every old/new fold's F1, correct counts
  and confusion-matrix agreement. Differences are retained and must be reviewed;
  regenerated artifacts are not falsely labeled the original run.

## Notebook

`notebooks/07b_v7_artifact_recovery.ipynb`: setup; prepare/import (NO training);
storage gate; regenerate missing jobs; independently verify and create backup.
On reconnection rerun setup and prepare; verified synced jobs restore locally.
Do not change environment versions or bypass a provenance error.
Estimated38 fits:45–90 GPU minutes, variable. Approximately120–180MB local
artifacts plus a similar Drive mirror; final compressed backup adds another copy.
No Mac dependencies, model downloads or training are needed.
