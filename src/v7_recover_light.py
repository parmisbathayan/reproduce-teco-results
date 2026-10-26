"""Bounded V7 artifact regeneration. Original results are read-only.

Keep verified jobs; regenerate only jobs lacking complete artifacts, at most 38.
Cloud-runtime storage is primary. Drive is a second copy, NOT proof of sync.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import time
import zipfile

from v7_unseen_sentences import (KEYS, PARTITIONS, VIEWS, audit_raw_overlap,
    build_inputs, make_sentence_plan, partition_indices, progress_heartbeat,
    reader_plan, validate_reference, write_summaries)


def stem(key):
    condition, seed, code = key
    return f"{condition}_seed_{int(seed)}_{code}"


def check_job(payloads, key, codes, plan, config):
    """Validate both views, exact targets/IDs, history and recorded protocol."""
    import numpy as np
    import pandas as pd
    from followup_common import macro_metrics

    condition, seed, code = key
    metrics = pd.read_csv(io.BytesIO(payloads['metrics.csv']))
    history = pd.read_csv(io.BytesIO(payloads['history.csv']))
    if len(metrics) != 2 or set(metrics.evaluation) != set(VIEWS):
        raise ValueError('Need exactly two distinct evaluation rows.')
    fit_codes, validation = reader_plan(codes, code)
    for view, partition, n in [('seen_sentences', 'fit', 66),
                                ('unseen_sentences', 'unseen_test', 22)]:
        rows = pd.read_csv(io.BytesIO(payloads[view + '.csv']))
        record = metrics[metrics.evaluation == view].iloc[0]
        expected = {r['sentence_id']: r[condition] for r in plan if r['partition'] == partition}
        if len(rows) != n or rows.sentence_id.duplicated().any() or set(rows.sentence_id) != set(expected):
            raise ValueError('Missing/extra/duplicate sentence predictions.')
        for name, value in zip(KEYS, [condition, seed, code, view]):
            if not (rows[name] == value).all() or record[name] != value:
                raise ValueError('Job identifiers differ.')
        truth = np.asarray([expected[int(i)] for i in rows.sentence_id])
        probabilities = rows[['probability_negative', 'probability_positive']].to_numpy()
        if (not np.array_equal(rows.true_label, truth) or not np.isfinite(probabilities).all()
                or (probabilities < 0).any() or (probabilities > 1).any()
                or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)
                or not np.array_equal(rows.predicted_label, probabilities.argmax(axis=1))):
            raise ValueError('Targets/probabilities/argmax mismatch.')
        scores = macro_metrics(truth, probabilities)
        for name in ('accuracy', 'macro_f1', 'tn', 'fp', 'fn', 'tp'):
            if abs(float(record[name]) - float(scores[name])) > 1e-9:
                raise ValueError('Metrics do not match predictions: ' + name)
        if (record.n_fit != 660 or record.n_validation != 22 or record.n_test != n
                or record.n_test_negative != n // 2 or record.n_test_positive != n // 2
                or record.correct != scores['tn'] + scores['tp']
                or bool(record.collapsed) != scores['collapsed']
                or record.validation_participant != validation
                or record.fit_participants != ','.join(fit_codes)
                or record.learning_rate != config['learning_rate'] or record.preprocessing != 'raw'):
            raise ValueError('Recorded protocol/counts differ.')
        if (len(history) != int(record.stopping_epoch)
                or list(history.epoch) != list(range(1, len(history) + 1))
                or not 1 <= int(record.best_epoch) <= len(history)):
            raise ValueError('History incomplete or checkpoint epoch invalid.')
    return metrics


def verify_archive(path, key, codes, plan, config, provenance):
    import numpy as np

    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None or len(set(archive.namelist())) != len(archive.namelist()):
            raise ValueError('Corrupt or duplicated ZIP entries.')
        manifest = json.loads(archive.read('manifest.json'))
        for name, value in [('key', list(key)), ('config', config), ('plan', plan),
                             ('provenance', provenance)]:
            if manifest.get(name) != value:
                raise ValueError('Archive provenance/protocol mismatch: ' + name)
        origin = manifest.get('origin')
        if origin not in ('original_verified', 'regenerated'):
            raise ValueError('Unknown result origin.')
        names = {'metrics.csv', 'history.csv', *[v + '.csv' for v in VIEWS]}
        if set(manifest['files']) != names or set(archive.namelist()) != names | {'manifest.json'}:
            raise ValueError('Archive incomplete/unexpected artifacts.')
        payloads = {n: archive.read(n) for n in names}
        for name, payload in payloads.items():
            if hashlib.sha256(payload).hexdigest() != manifest['files'][name]:
                raise ValueError('Archive checksum failed: ' + name)
    return check_job(payloads, key, codes, plan, config), manifest


def publish(payloads, local, mirror, key, codes, plan, config, provenance, origin, extra=None):
    """Never delete local primary archives after mirroring; never overwrite jobs."""
    from followup_common import sha256

    manifest = dict(key=list(key), config=config, plan=plan, provenance=provenance,
                    origin=origin, **(extra or {}))
    manifest['files'] = {n: hashlib.sha256(p).hexdigest() for n, p in payloads.items()}
    final = local / 'jobs' / (stem(key) + '.zip')
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        raise ValueError('Refusing to overwrite a completed job.')
    partial = final.with_suffix('.zip.partial')
    with zipfile.ZipFile(partial, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, value in payloads.items():
            archive.writestr(name, value)
        archive.writestr('manifest.json', json.dumps(manifest, indent=2))
    verify_archive(partial, key, codes, plan, config, provenance)
    os.replace(partial, final)
    mirror_file(final, mirror / 'jobs' / final.name)
    print('LOCAL PRIMARY SAVED:', final, 'SHA256:', sha256(str(final)), flush=True)


def mirror_file(source, destination):
    from followup_common import sha256

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256(str(source)) != sha256(str(destination)):
            raise ValueError('Conflicting mirror: ' + str(destination))
        return
    partial = destination.with_suffix(destination.suffix + '.partial')
    shutil.copyfile(source, partial)
    if sha256(str(source)) != sha256(str(partial)):
        raise ValueError('Mirror checksum failed. Local primary retained.')
    os.replace(partial, destination)


def collect(local, mirror, keys, codes, plan, config, provenance, old):
    import pandas as pd
    from followup_common import atomic_csv, atomic_json

    expected = {stem(k) + '.zip' for k in keys}
    if {p.name for p in (local / 'jobs').glob('*.zip')} - expected:
        raise ValueError('Unexpected local job archives.')
    frames, completed, comparisons = [], set(), []
    for key in keys:
        path = local / 'jobs' / (stem(key) + '.zip')
        if not path.exists():
            continue
        rows, manifest = verify_archive(path, key, codes, plan, config, provenance)
        rows = rows.copy()
        rows['artifact_origin'] = manifest['origin']
        frames.append(rows)
        completed.add(key)
        original = old[(old.condition == key[0]) & (old.seed == key[1]) & (old.participant == key[2])]
        for row in rows.itertuples():
            before = original[original.evaluation == row.evaluation].iloc[0]
            comparisons.append(dict(condition=key[0], seed=key[1], participant=key[2],
                evaluation=row.evaluation, origin=manifest['origin'],
                original_macro_f1=float(before.macro_f1), recovered_macro_f1=row.macro_f1,
                delta_macro_f1=row.macro_f1 - float(before.macro_f1),
                original_correct=int(before.correct), recovered_correct=int(row.correct),
                exact_confusion_match=all(int(before[n]) == int(getattr(row, n)) for n in ('tn','fp','fn','tp'))))
        mirror_file(path, mirror / 'jobs' / path.name)
    if frames:
        frame = pd.concat(frames, ignore_index=True).sort_values(KEYS)
        frame["hyperparameters_json"] = json.dumps(config, sort_keys=True)
        atomic_csv(frame, str(local / 'fold_metrics.csv'))
        atomic_csv(pd.DataFrame(comparisons), str(local / 'comparison_to_original.csv'))
        write_summaries(frame, str(local), codes, config)
    atomic_json(dict(verified_jobs=len(completed), expected_jobs=48, complete=len(completed)==48,
                     server_persistence_verified=False,
                     note='Local/cloud mount verification is not server-side Drive persistence.'),
                str(local / 'artifact_status.json'))
    # Summaries are derived; mirror safely via replace, never originals.
    for path in local.glob('*.csv'):
        destination = mirror / path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, str(destination) + '.partial')
        os.replace(str(destination) + '.partial', destination)
    return completed


def validate_storage_paths(args):
    local, mirror, original = [Path(p).resolve() for p in (args.local_dir, args.mirror_dir, args.original_dir)]
    if (not str(local).startswith('/content/') or '/drive/' in str(local)
            or local == mirror or local == original or original == mirror
            or original in mirror.parents or mirror in original.parents):
        raise ValueError('Use separate /content LOCAL primary, NEW Drive mirror, read-only original.')
    return local, mirror, original


def run(args):
    import numpy as np
    import pandas as pd
    import sklearn
    import tensorflow as tf
    from data import PARTICIPANTS, _norm, _read_subject
    from followup_common import (atomic_json, fit_network, macro_metrics, prepare,
                                 set_seed, sha256)
    from model import build_cnn

    local, mirror, original = validate_storage_paths(args)
    config = json.loads(Path(args.config).read_text())
    if (mirror/'run_metadata.json').exists() and not (local/'run_metadata.json').exists():
        mirror_file(mirror/'run_metadata.json', local/'run_metadata.json')
    old_metadata = json.loads((original / 'run_metadata.json').read_text())
    if old_metadata['definition']['config'] != config:
        raise ValueError('Original V7 config changed.')
    source = Path(__file__).resolve().parent
    recorded = {r['name']: r for r in old_metadata['definition']['source_files']}
    if sha256(str(source / 'v7_unseen_sentences.py')) != recorded['v7_unseen_sentences.py']['sha256']:
        raise ValueError('Original V7 helper code changed.')
    X, y, codes, ids = prepare(config, args.trt_dir, args.labels_csv, str(local), args.config,
        [str(source / n) for n in ('v7_recover_light.py','v7_unseen_sentences.py',
                                   'model.py','data.py','followup_common.py')])
    metadata = json.loads((local / 'run_metadata.json').read_text())
    runtime = {'tensorflow':tf.__version__, 'numpy':np.__version__,
               'pandas':pd.__version__, 'scikit_learn':sklearn.__version__}
    validate_reference(old_metadata, metadata, runtime)
    reference = json.loads((Path(args.reference_dir) / 'run_metadata.json').read_text())
    validate_reference(reference, metadata, runtime)
    if X.shape != (12,110,49,126) or not np.isfinite(X).all():
        raise ValueError('Unexpected raw inputs.')
    first = _read_subject(args.trt_dir, PARTICIPANTS[0][0])
    texts = {int(t['sentenceId']): _norm(t['persian_sentence']) for t in first.values()}
    plan = make_sentence_plan(y, ids, [texts[int(i)] for i in ids], config)
    if json.loads((original / 'sentence_plan.json').read_text()) != plan:
        raise ValueError('Split/target mapping changed.')
    audit_raw_overlap(X, codes, plan)
    old = pd.read_csv(original / 'fold_metrics.csv')
    keys = [(c,s,p) for s in config['seeds'] for p in codes for c in config['conditions']]
    if (len(old)!=96 or old[KEYS].duplicated().any()
            or set(map(tuple, old[KEYS].to_numpy())) != {(*k,v) for k in keys for v in VIEWS}):
        raise ValueError('Original metrics incomplete/duplicated/unexpected.')
    provenance = {n:sha256(str(original/n)) for n in ('run_metadata.json','fold_metrics.csv','sentence_plan.json')}
    provenance['recovery_metadata_sha256'] = sha256(str(local/'run_metadata.json'))
    provenance_path = local / 'recovery_provenance.json'
    if provenance_path.exists() and json.loads(provenance_path.read_text()) != provenance:
        raise ValueError('Recovery inputs changed; preserve prior output.')
    atomic_json(provenance, str(provenance_path))
    atomic_json(plan, str(local / 'sentence_plan.json'))
    for name in ('run_metadata.json','recovery_provenance.json','sentence_plan.json'):
        mirror_file(local/name, mirror/name)
    for name in ('run_metadata.json','fold_metrics.csv','sentence_plan.json'):
        destination = local / 'original_snapshot' / name
        mirror_file(original/name, destination)
    # Existing archives on Drive may restore a lost cloud runtime; verify before use.
    for key in keys:
        path = mirror/'jobs'/(stem(key)+'.zip')
        target = local/'jobs'/path.name
        if path.exists() and not target.exists():
            verify_archive(path, key, codes, plan, config, provenance)
            mirror_file(path, target)
    completed = collect(local, mirror, keys, codes, plan, config, provenance, old)
    missing = []
    for key in keys:
        if key in completed:
            continue
        paths = {v+'.csv':original/'predictions'/(stem(key)+'_'+v+'.csv') for v in VIEWS}
        paths['history.csv'] = original/'histories'/(stem(key)+'.csv')
        if not all(p.exists() for p in paths.values()):
            missing.append(key)
            continue
        payloads = {n:p.read_bytes() for n,p in paths.items()}
        rows = old[(old.condition==key[0]) & (old.seed==key[1]) & (old.participant==key[2])]
        payloads['metrics.csv'] = rows.to_csv(index=False).encode()
        check_job(payloads, key, codes, plan, config)  # corrupt existing files stop, never silently refit
        publish(payloads, local, mirror, key, codes, plan, config, provenance, 'original_verified')
    completed = collect(local, mirror, keys, codes, plan, config, provenance, old)
    remaining = [k for k in keys if k not in completed]
    if len(remaining)>38:
        raise ValueError('More than the approved 38 fits need regeneration; investigate, do not expand.')
    print(f'VERIFIED {len(completed)}/48; regeneration needed {len(remaining)}. Original untouched.', flush=True)
    if args.verify_only:
        if remaining:
            raise ValueError('Recovery incomplete; no training started by verify-only.')
        return
    if args.prepare_only:
        print('Prepared only; NO training. Resolve Drive persistence before the training section.', flush=True)
        return
    if not args.storage_checked:
        raise ValueError('Storage gate: confirm the new account saves files visible on the Drive website first.')
    fit_i, val_i, _ = [partition_indices(plan,p) for p in PARTITIONS]
    for key in remaining:
        condition, seed, code = key
        x_fit, x_val, views, fit_codes, validation = build_inputs(X,codes,code,plan)
        targets = np.asarray([r[condition] for r in plan], dtype=np.int64)
        tf.keras.backend.clear_session()
        set_seed(seed, config['deterministic_ops'])
        model = build_cnn(x_fit.shape[1:]+(1,), lr=config['learning_rate'])
        print(f'START regeneration {stem(key)}; {len(completed)}/48 archived', flush=True)
        started = time.perf_counter()
        with progress_heartbeat(stem(key)):
            model, history, chosen = fit_network(model,x_fit[...,None],np.tile(targets[fit_i],10),
                                                 x_val[...,None],targets[val_i],config)
        duration = time.perf_counter()-started
        payloads = {"history.csv":history.to_csv(index=False).encode()}
        rows = []
        for view, partition in [('seen_sentences','fit'),('unseen_sentences','unseen_test')]:
            indices = partition_indices(plan,partition)
            truth = targets[indices]
            prob = model.predict(views[view][...,None],verbose=0)
            identifiers = dict(zip(KEYS, [condition,seed,code,view]))
            pred = pd.DataFrame(dict(**identifiers, sentence_id=ids[indices],true_label=truth,
                predicted_label=prob.argmax(1),probability_negative=prob[:,0],probability_positive=prob[:,1]))
            payloads[view+'.csv'] = pred.to_csv(index=False).encode()
            scores = macro_metrics(truth,prob)
            rows.append(dict(**identifiers, validation_participant=validation,fit_participants=','.join(fit_codes),
                n_fit=660,n_validation=22,n_test=len(truth),n_test_negative=len(truth)//2,n_test_positive=len(truth)//2,
                learning_rate=config['learning_rate'],preprocessing='raw',duration_seconds=duration,
                sentence_plan_sha256=provenance['sentence_plan.json'],n_parameters=int(model.count_params()),
                duration_scope='shared_fit_for_both_evaluations',correct=scores['tn']+scores['tp'],**chosen,**scores))
        payloads['metrics.csv'] = pd.DataFrame(rows).to_csv(index=False).encode()
        publish(payloads,local,mirror,key,codes,plan,config,provenance,'regenerated',
                dict(storage_profile="results_only_no_weights"))
        completed = collect(local,mirror,keys,codes,plan,config,provenance,old)
    print('48/48 verified. Review comparison_to_original.csv; differences are reported, NOT tuned away.',flush=True)
    print('Local primary retained. Confirm Drive WEBSITE has all 48 ZIPs before disconnecting.',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('trt-dir','labels-csv','original-dir','reference-dir','local-dir','mirror-dir'):
        parser.add_argument('--'+flag, required=True)
    parser.add_argument('--config',default='configs/v7_unseen_sentences.json')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--verify-only',action='store_true')
    parser.add_argument('--storage-checked',action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()


