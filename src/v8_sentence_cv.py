"""Five sentence folds for frozen V8. Reuse four verified anchor fits, train 16.

No reader averaging, new hyperparameters, label remapping or saved weights.
The old test/validation partitions become groups 0/1; subdivide its 66 fitting
sentences into groups 2/3/4. Test group i, validate group (i+1)%5, fit the rest.
Every sentence is tested once per target/initialization across all 12 readers.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import tempfile
import time
import zipfile

from v7_unseen_sentences import make_sentence_plan, validate_reference, progress_heartbeat
from v8_seen_readers_light import (all_reader_inputs, audit_partitions, verify_bundle,
                                   publish_bundle, collect_results, job_stem)


def frozen_cv_options(config):
    expected = {
        'version': 'v8_sentence_cv', 'n_folds': 5,
        'completion_split_seed': 20261102, 'test_fold_order': list(range(5)),
        'validation_rule': 'next_fold_mod_5',
        'anchor_rule': 'fold_0_test_equals_old_test_and_fold_1_equals_old_validation',
        'arbitrary_labels': 'preserve_original_global_mapping',
        'balance': '11_per_real_class_and_11_per_arbitrary_class_in_each_22_sentence_fold',
        'save_weights': False,
    }
    if config != expected:
        raise ValueError('Use the frozen CV definition; no automatic split/setting search.')


def make_folds(anchor, config):
    """Deterministic metadata only; group both targets jointly to retain balance.

In the old fit pool each target has33 positives/33 negatives. Its2x2 joint
counts have equal diagonals, and equal off-diagonals. Allocate the diagonal
count across three groups; every group then has11/11 for BOTH targets.
This preserves the one global arbitrary mapping, including the reused fold.
"""
    frozen_cv_options(config)
    if len(anchor) != 110 or sorted(r['index'] for r in anchor) != list(range(110)):
        raise ValueError('Expected all110 indexed sentence rows.')
    if len({r['sentence_id'] for r in anchor}) != 110:
        raise ValueError('Duplicate sentence IDs.')
    if len({r['normalized_text_sha256'] for r in anchor}) != 110:
        raise ValueError('Duplicate normalized sentence texts.')
    for name, n in [('fit',66),('validation',22),('unseen_test',22)]:
        rows = [r for r in anchor if r['partition'] == name]
        if len(rows) != n or any(sum(r[c] for r in rows) != n//2
                                for c in ['sentiment','shuffled_sentence_labels']):
            raise ValueError('Unbalanced/invalid anchor partitions.')
    if any(r[c] not in (0,1) for r in anchor for c in ['sentiment','shuffled_sentence_labels']):
        raise ValueError('Expected binary target values.')
    rng = random.Random(config['completion_split_seed'])
    pools = {}
    for a in (0,1):
        for b in (0,1):
            pool = sorted((r for r in anchor if r['partition']=='fit'
                           and r['sentiment']==a and r['shuffled_sentence_labels']==b),
                          key=lambda r:r['sentence_id'])
            rng.shuffle(pool)
            pools[a,b] = pool
    diagonal = len(pools[0,0])
    if diagonal != len(pools[1,1]) or len(pools[0,1]) != len(pools[1,0]):
        raise ValueError('Anchor target marginals do not match.')
    sizes = [diagonal//3 + (i < diagonal%3) for i in range(3)]
    groups = {r['sentence_id']: (0 if r['partition']=='unseen_test' else 1)
              for r in anchor if r['partition']!='fit'}
    offsets = {key:0 for key in pools}
    for group, d in zip((2,3,4),sizes):
        for key, n in {(0,0):d,(1,1):d,(0,1):11-d,(1,0):11-d}.items():
            start = offsets[key]
            chosen = pools[key][start:start+n]
            if len(chosen) != n:
                raise ValueError('Joint stratification failed.')
            groups.update({r['sentence_id']:group for r in chosen})
            offsets[key] += n
    plans = []
    for fold in range(5):
        plan = copy.deepcopy(anchor)
        for row in plan:
            group = groups[row['sentence_id']]
            row['partition'] = ('unseen_test' if group == fold else
                                'validation' if group == (fold+1)%5 else 'fit')
        for name,n in [('fit',66),('validation',22),('unseen_test',22)]:
            rr = [r for r in plan if r['partition']==name]
            if len(rr)!=n or any(sum(r[c] for r in rr)!=n//2
                                for c in ['sentiment','shuffled_sentence_labels']):
                raise ValueError('Invalid CV fold balance.')
        plans.append(plan)
    if plans[0] != anchor:
        raise ValueError('Fold0 must reproduce the complete original plan exactly.')
    tests = [r['sentence_id'] for p in plans for r in p if r['partition']=='unseen_test']
    if len(tests)!=110 or len(set(tests))!=110:
        raise ValueError('Each sentence must be tested exactly once.')
    return plans


def immutable_json(value, path):
    from followup_common import atomic_json
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('Conflicting saved definition; preserve it: '+str(path))
    else:
        atomic_json(value,str(path))


def import_anchor(source_dir, target_dir, codes, plan, config):
    """Verify original payloads, then repack copies with explicit provenance.

Original CSV/history bytes stay unchanged. A NEW manifest points to the CV
fold's metadata, records original SHA256s, and never changes source archives.
"""
    from followup_common import sha256
    source_dir, target_dir = Path(source_dir), Path(target_dir)
    imported = []
    # Refuse to train a replacement silently if any promised anchor is absent.
    for seed in config['seeds']:
        for condition in config['conditions']:
            source = source_dir/'jobs'/(job_stem(condition,seed)+'.zip')
            _,_,manifest = verify_bundle(source,condition,seed,codes,plan,config)
            if manifest['metadata_sha256'] != sha256(str(source_dir/'run_metadata.json')):
                raise ValueError('Anchor archive metadata hash mismatch.')
    completed = collect_results(str(target_dir),codes,plan,config)
    for seed in config['seeds']:
        for condition in config['conditions']:
            source = source_dir/'jobs'/(job_stem(condition,seed)+'.zip')
            _,_,manifest = verify_bundle(source,condition,seed,codes,plan,config)
            if (condition,seed) not in completed:
                with tempfile.TemporaryDirectory(prefix='teco-v8-cv-import-') as work:
                    with zipfile.ZipFile(source) as z:
                        for name in ['predictions.csv','metrics_by_reader.csv','history.csv']:
                            Path(work,name).write_bytes(z.read(name))
                    copied = copy.deepcopy(manifest)
                    copied['metadata_sha256'] = sha256(str(target_dir/'run_metadata.json'))
                    copied['origin'] = {'kind':'verified_original_v8_reused',
                        'source_archive':str(source),'source_sha256':sha256(str(source)),
                        'source_metadata_sha256':manifest['metadata_sha256']}
                    publish_bundle(work,str(target_dir),condition,seed,copied,codes,plan,config)
            imported.append({'condition':condition,'seed':seed,'source_sha256':sha256(str(source))})
    collect_results(str(target_dir),codes,plan,config,require_complete=True)
    return imported


def score_v4_files(reference_dir, codes, plan):
    """Read-only V4 comparison on exact test IDs, plus full110 checksum metrics."""
    import numpy as np
    import pandas as pd
    from followup_common import macro_metrics, sha256
    out=[]
    expected = {int(r['sentence_id']):r['sentiment'] for r in plan}
    selected = {int(r['sentence_id']) for r in plan if r['partition']=='unseen_test'}
    reference_dir = Path(reference_dir)
    recorded = pd.read_csv(reference_dir/'fold_metrics.csv')
    if len(recorded)!=24 or recorded[['participant','seed']].duplicated().any():
        raise ValueError('Expected24 unique V4 reference rows.')
    for seed in (45,62):
        for code in codes:
            path=reference_dir/'predictions'/f'subject_val_macro_f1_lr_0.0001_seed_{seed}_{code}.csv'
            frame=pd.read_csv(path)
            if len(frame)!=110 or frame.sentence_id.duplicated().any() or set(frame.sentence_id)!=set(expected):
                raise ValueError('Incomplete V4 predictions.')
            if not (frame.seed.eq(seed).all() and frame.participant.eq(code).all()):
                raise ValueError('V4 prediction reader/seed mismatch.')
            if not np.array_equal(frame.true_label,[expected[int(i)] for i in frame.sentence_id]):
                raise ValueError('V4 and CV labels/IDs differ.')
            prob=frame[['probability_negative','probability_positive']].to_numpy()
            if (not np.isfinite(prob).all() or np.any(prob<0) or np.any(prob>1)
                    or not np.allclose(prob.sum(1),1,atol=1e-5)
                    or not np.array_equal(frame.predicted_label,prob.argmax(1))):
                raise ValueError('V4 prediction probability check failed.')
            full=macro_metrics(frame.true_label.to_numpy(),prob)
            old=recorded[recorded.participant.eq(code)&recorded.seed.eq(seed)]
            if len(old)!=1 or any(abs(float(old.iloc[0][k])-full[k])>1e-9
                                   for k in ['macro_f1','accuracy','tn','fp','fn','tp']):
                raise ValueError('V4 predictions do not reconstruct the full recorded score.')
            rows=frame[frame.sentence_id.isin(selected)]
            scores=macro_metrics(rows.true_label.to_numpy(),rows[['probability_negative','probability_positive']].to_numpy())
            out.append({'seed':seed,'participant':code,'n_test':22,
                        'correct':scores['tn']+scores['tp'],'source_sha256':sha256(str(path)),**scores})
    return pd.DataFrame(out)


def write_overall(root, codes, plans, config, require_complete=False):
    import pandas as pd
    from followup_common import atomic_csv, atomic_json, macro_metrics
    root=Path(root)
    frames,predictions=[],[]
    completed=0
    for fold,plan in enumerate(plans):
        directory=root/f'fold_{fold}'
        done=collect_results(str(directory),codes,plan,config)
        completed+=len(done)
        for condition,seed in sorted(done):
            path=directory/'jobs'/(job_stem(condition,seed)+'.zip')
            metrics,pred,_=verify_bundle(path,condition,seed,codes,plan,config)
            frames.append(metrics.assign(sentence_fold=fold))
            predictions.append(pred.assign(sentence_fold=fold))
    if require_complete and completed!=20:
        raise ValueError(f'Only{completed}/20 verified fits.')
    if frames:
        metrics=pd.concat(frames,ignore_index=True)
        pred=pd.concat(predictions,ignore_index=True)
        if pred[['condition','seed','participant','sentence_id']].duplicated().any():
            raise ValueError('Duplicated out-of-fold prediction.')
        atomic_csv(metrics,str(root/'all_fold_metrics.csv'))
        atomic_csv(pred,str(root/'out_of_fold_predictions.csv'))
        summary=metrics.groupby('condition',as_index=False).agg(
            reader_fold_seed_evaluations=('macro_f1','size'),mean_fold_reader_macro_f1=('macro_f1','mean'),
            correct=('correct','sum'),test_predictions=('n_test','sum'),collapsed_reader_evaluations=('collapsed','sum'))
        atomic_csv(summary,str(root/'summary_fold_mean.csv'))
        if completed==20:
            pooled=[]
            for (condition,seed,code),group in pred.groupby(['condition','seed','participant']):
                if len(group)!=110 or set(group.sentence_id)!={r['sentence_id'] for r in plans[0]}:
                    raise ValueError('Incomplete110-sentence out-of-fold reader predictions.')
                scores=macro_metrics(group.true_label.to_numpy(),group[['probability_negative','probability_positive']].to_numpy())
                pooled.append({'condition':condition,'seed':int(seed),'participant':code,**scores})
            pooled=pd.DataFrame(pooled)
            atomic_csv(pooled,str(root/'out_of_fold_metrics_by_reader.csv'))
            atomic_csv(pooled.groupby('condition',as_index=False).agg(
                mean_reader_oof_macro_f1=('macro_f1','mean'),mean_accuracy=('accuracy','mean')),
                str(root/'summary_oof_110.csv'))
    atomic_json({'complete':completed==20,'completed_fits':completed,'expected_fits':20,
                 'expected_reused':4,'expected_new_fits':16,
                 'scope':'One anchored fivefold partition, two initializations, one arbitrary mapping; exploratory.'},
                str(root/'completion.json'))
    return completed


def run(args):
    import numpy as np
    import pandas as pd
    import sklearn
    import tensorflow as tf
    from data import PARTICIPANTS, _norm, _read_subject
    from model import build_cnn
    from followup_common import atomic_json, atomic_csv, fit_network, macro_metrics, prepare, set_seed, sha256
    from light_storage import restore_mirror, sync_mirror, validate_paths
    root,anchor=Path(args.local_dir),Path(args.anchor_dir)
    base=json.loads(Path(args.base_config).read_text())
    options=json.loads(Path(args.cv_config).read_text());frozen_cv_options(options)
    validate_paths(str(root),args.mirror_dir,[args.reference_dir,args.anchor_dir])
    restore_mirror(root,args.mirror_dir)
    effective={**base,'sentence_cv':options}
    sources=[str(Path(__file__).with_name(n)) for n in
             ['v8_sentence_cv.py','v8_seen_readers_light.py','v7_unseen_sentences.py',
              'model.py','data.py','followup_common.py','light_storage.py']]
    sources += [args.base_config,str(anchor/'run_metadata.json'),str(anchor/'sentence_plan.json')]
    X,y,codes,ids=prepare(effective,args.trt_dir,args.labels_csv,str(root),args.cv_config,sources)
    current=json.loads((root/'run_metadata.json').read_text())
    old=json.loads((anchor/'run_metadata.json').read_text())
    runtime={'tensorflow':tf.__version__,'numpy':np.__version__,'pandas':pd.__version__,'scikit_learn':sklearn.__version__}
    validate_reference(old,current,runtime)
    validate_reference(json.loads(Path(args.reference_dir,'run_metadata.json').read_text()),current,runtime)
    if old['definition']['config']!=base:
        raise ValueError('Use the exact original V8 base config.')
    if X.shape!=(12,110,49,126) or not np.isfinite(X).all():
        raise ValueError('Unexpected/nonfinite input data.')
    texts=None
    for name,_ in PARTICIPANTS:
        trials=_read_subject(args.trt_dir,name)
        mapping={int(r['sentenceId']):_norm(r['persian_sentence']) for r in trials.values()}
        chosen=[mapping[int(i)] for i in ids]
        if texts is None:texts=chosen
        elif texts!=chosen:raise ValueError('Reader sentence-text alignment differs.')
    original=make_sentence_plan(y,ids,texts,base)
    if original!=json.loads((anchor/'sentence_plan.json').read_text()):
        raise ValueError('Original V8 split/mapping cannot be reconstructed exactly.')
    plans=make_folds(original,options)
    immutable_json(plans,root/'sentence_cv_plans.json')
    atomic_csv(pd.DataFrame([{**r,'sentence_fold':f} for f,p in enumerate(plans) for r in p]),str(root/'sentence_cv_plans.csv'))
    v4=[]
    for f,plan in enumerate(plans):
        directory=root/f'fold_{f}'
        metadata={**current,'sentence_cv_fold':f}
        immutable_json(metadata,directory/'run_metadata.json')
        immutable_json(plan,directory/'sentence_plan.json')
        zeros=audit_partitions(X,plan)
        atomic_json({'passed':True,'fold':f,'zero_inputs':zeros,'n_fit':792,'n_validation':264,'n_test':264,
                     'readers_averaged':False,'save_weights':False},str(directory/'preflight.json'))
        v4.append(score_v4_files(args.reference_dir,codes,plan).assign(sentence_fold=f))
    v4=pd.concat(v4,ignore_index=True)
    atomic_csv(v4,str(root/'v4_on_matched_sentences.csv'))
    atomic_csv(v4.groupby('sentence_fold',as_index=False).agg(
        mean_macro_f1=('macro_f1','mean'),correct=('correct','sum'),test_predictions=('n_test','sum')),
        str(root/'v4_matched_fold_summary.csv'))
    imports=import_anchor(anchor,root/'fold_0',codes,plans[0],base)
    immutable_json(imports,root/'anchor_import_provenance.json')
    completed=write_overall(root,codes,plans,base,require_complete=args.verify_only)
    sync_mirror(root,args.mirror_dir)
    print(f'CV preflight: original split exact;4 original jobs reused;{completed}/20 jobs verified. Missing:{20-completed}.',flush=True)
    if args.prepare_only or args.verify_only:
        print('No training started.',flush=True);return
    for fold,plan in enumerate(plans):
        directory=root/f'fold_{fold}'
        done=collect_results(str(directory),codes,plan,base)
        test=[r['index'] for r in plan if r['partition']=='unseen_test']
        for seed in base['seeds']:
            for condition in base['conditions']:
                if (condition,seed) in done:continue
                if fold==0:raise ValueError('The anchor should have been reused; refusing replacement training.')
                (xf,yf),(xv,yv),(xt,yt)=all_reader_inputs(X,codes,plan,condition)
                tf.keras.backend.clear_session();set_seed(seed,base['deterministic_ops'])
                model=build_cnn(xf.shape[1:]+(1,),lr=base['learning_rate'])
                label=f'fold{fold}:{condition}:seed{seed}'
                print(f'START {completed+1}/20 {label}',flush=True)
                start=time.perf_counter()
                with progress_heartbeat(label):
                    model,history,chosen=fit_network(model,xf[...,None],yf,xv[...,None],yv,base)
                probabilities=model.predict(xt[...,None],verbose=0)
                duration=time.perf_counter()-start
                rows,preds=[],[]
                for p,code in enumerate(codes):
                    block=slice(p*22,(p+1)*22);truth,prob=yt[block],probabilities[block]
                    scores=macro_metrics(truth,prob);identity={'condition':condition,'seed':seed,'participant':code}
                    preds.append(pd.DataFrame({**identity,'sentence_id':ids[test],
                        'true_label':truth,'predicted_label':prob.argmax(1),
                        'probability_negative':prob[:,0],'probability_positive':prob[:,1]}))
                    rows.append({**identity,'n_fit':792,'n_validation':264,'n_test':22,
                        'n_test_negative':11,'n_test_positive':11,'learning_rate':base['learning_rate'],
                        'preprocessing':'raw','n_parameters':int(model.count_params()),
                        'duration_seconds':duration,'duration_scope':'one_shared_model_all12readers',
                        'sentence_plan_sha256':sha256(str(directory/'sentence_plan.json')),
                        'correct':scores['tn']+scores['tp'],**chosen,**scores})
                work=tempfile.mkdtemp(prefix='teco-v8-cv-job-')
                try:
                    pd.concat(preds,ignore_index=True).to_csv(Path(work,'predictions.csv'),index=False)
                    pd.DataFrame(rows).to_csv(Path(work,'metrics_by_reader.csv'),index=False)
                    history.to_csv(Path(work,'history.csv'),index=False)
                    manifest={'condition':condition,'seed':seed,'codes':codes,'config':base,
                        'sentence_plan':plan,'metadata_sha256':sha256(str(directory/'run_metadata.json')),
                        'storage_profile':'results_only_no_weights','origin':{'kind':'new_sentence_cv_fit','fold':fold}}
                    publish_bundle(work,str(directory),condition,seed,manifest,codes,plan,base)
                except Exception:
                    print('Recovery files retained at '+work,flush=True);raise
                shutil.rmtree(work)
                completed=write_overall(root,codes,plans,base)
                sync_mirror(root,args.mirror_dir)
                print(f'SAVED {completed}/20 {label}; no model weights retained.',flush=True)
    write_overall(root,codes,plans,base,require_complete=True);sync_mirror(root,args.mirror_dir)
    print('COMPLETE:20/20 verified jobs,110 unique out-of-fold test sentences. See both fold-mean and110-sentence OOF summaries.',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ['trt-dir','labels-csv','anchor-dir','reference-dir','local-dir','mirror-dir']:
        p.add_argument('--'+flag,required=True)
    p.add_argument('--base-config',default='configs/v8_seen_readers_unseen_sentences.json')
    p.add_argument('--cv-config',default='configs/v8_sentence_cv.json')
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--prepare-only',action='store_true')
    mode.add_argument('--verify-only',action='store_true')
    run(p.parse_args())


if __name__=='__main__':main()
