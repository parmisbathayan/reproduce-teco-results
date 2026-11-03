"""Independent V4 reimplementation: three NEW seeds, two targets, 72 LOSO fits.

Own loader, model construction, validation checkpoint, scoring and archive checks.
Old loader/model are used ONLY to cross-check inputs/initialization in preflight.
Same sentences across readers are intentional; held-out reader never fits/validates.
Cannot certify unknown upstream EEG preprocessing from supplied feature pickles.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import pickle
import random
import shutil
import tempfile
import threading
import time
import zipfile

import numpy as np

READERS=[('Amin','AMB'),('Arya','ARM'),('Eisa','EIA'),('Elahe','ELD'),
         ('Farzam','FAS'),('Javad','JAY'),('Mehrnaz','MEB'),('Nastaran','NAD'),
         ('Razieh','RAS'),('Sajad','SAS'),('Shahrzad','SHU'),('Yaser','AYB')]


def validate_config(c):
    expected={'version':'v4_independent_audit','seeds':[11,23,37],
        'conditions':['sentiment','arbitrary_sentence_labels'],'label_mapping_seed':20261103,
        'learning_rate':.0001,'preprocessing':'raw','epochs':100,'batch_size':64,
        'patience':40,'start_epoch':10,'validation':'next_reader_whole_110_sentences',
        'checkpoint':'validation_macro_f1','require_gpu':True,'deterministic_ops':True,'save_weights':False}
    if c!=expected:raise ValueError('Use the fixed independent-audit configuration.')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value,immutable=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and immutable:
        if json.loads(path.read_text())!=value:raise ValueError('Conflicting saved definition: '+str(path))
        return
    tmp=path.with_name(path.name+'.partial')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
    os.replace(tmp,path)


def csv_bytes(rows):
    if not rows:raise ValueError('Empty CSV payload.')
    stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
    writer.writeheader();writer.writerows(rows);return stream.getvalue().encode()


def csv_rows(body):return list(csv.DictReader(io.StringIO(body.decode())))


def normalize(text):return ''.join(str(text).split()).replace('\u200c','')


def load_independent(trt_dir,labels_csv):
    """Reconstruct inputs from native sentence IDs and a separately checked text join."""
    with open(labels_csv,newline='',encoding='utf-8-sig') as f:labels=list(csv.DictReader(f))
    label_by_text={}
    for row in labels:
        key=normalize(row['sentence']);label=int(row['sentiment_label'])
        if not key or key in label_by_text or label not in (-1,0,1):
            raise ValueError('Missing/duplicate text or invalid label in CSV.')
        label_by_text[key]=label
    if Counter(label_by_text.values())!=Counter({-1:55,0:55,1:55}):
        raise ValueError('Expected165 distinct labeled sentences,55 per class.')
    subjects=[];reference=None;max_words=0;stubs=0
    for name,code in READERS:
        with open(Path(trt_dir,f'{name}_trt_total.pickle'),'rb') as f:raw=pickle.load(f)
        records={}
        for trial in raw.values():
            sid=int(trial['sentenceId']);text=normalize(trial['persian_sentence'])
            if sid in records or sid not in range(1,166):raise ValueError('Duplicate/invalid native sentence ID.')
            if text not in label_by_text:raise ValueError('Sentence has no text-aligned label.')
            if not isinstance(trial['word'],dict):raise ValueError('Expected word-index dictionary.')
            max_words=max(max_words,len(trial['word']))
            records[sid]=(text,trial['word'])
        if set(records)!=set(range(1,166)):raise ValueError('Missing native sentence records.')
        texts=[records[i][0] for i in range(1,166)]
        if len(set(texts))!=165:raise ValueError('Duplicate normalized stimulus text.')
        if reference is None:reference=texts
        elif texts!=reference:raise ValueError('Reader sentence ID/text alignment differs.')
        subjects.append(records)
    if max_words!=49:raise ValueError('Expected original49-word padding width.')
    chosen=[sid for sid in range(1,166) if label_by_text[reference[sid-1]]!=0]
    if len(chosen)!=110:raise ValueError('Expected110 binary sentences.')
    y=np.asarray([int(label_by_text[reference[i-1]]==1) for i in chosen],dtype=np.int64)
    X=np.zeros((12,110,49,126),dtype=np.float32)
    for p,records in enumerate(subjects):
        for j,sid in enumerate(chosen):
            for wi,word in records[sid][1].items():
                if not isinstance(wi,(int,np.integer)) or not 0<=int(wi)<49:
                    raise ValueError('Invalid word position.')
                vector=np.asarray(word['TRT_total'])
                if vector.shape==(126,):
                    if not np.isfinite(vector).all():raise ValueError('Nonfinite EEG input.')
                    X[p,j,int(wi)]=vector.astype(np.float32)
                elif vector.shape==(2,):stubs+=1
                else:raise ValueError(f'Unexpected feature shape {vector.shape}: reader {READERS[p][1]}, sentence {sid}, word {wi}.')
    if not np.isfinite(X).all():raise ValueError('Nonfinite float32 feature tensor.')
    return X,y,[code for _,code in READERS],np.asarray(chosen),{
        'label_counts':dict(Counter(map(int,y))),'missing_fixation_stubs':stubs,
        'normalized_text_sha256':[hashlib.sha256(reference[i-1].encode()).hexdigest() for i in chosen]}


def targets_for(y,config):
    real=list(map(int,y));arbitrary=real.copy()
    random.Random(config['label_mapping_seed']).shuffle(arbitrary)
    if arbitrary==real:raise ValueError('Arbitrary mapping unchanged.')
    return {'sentiment':np.asarray(real),'arbitrary_sentence_labels':np.asarray(arbitrary)}


def jobs_for(codes,config):
    if codes!=[c for _,c in READERS]:raise ValueError('Unexpected reader order.')
    result=[]
    for condition in config['conditions']:
        for seed in config['seeds']:
            for p,code in enumerate(codes):
                val=codes[(p+1)%12];fit=[c for c in codes if c not in (code,val)]
                result.append({'condition':condition,'seed':seed,'test_reader':code,
                               'validation_reader':val,'fit_readers':fit})
    return result


def inputs_for(X,codes,target,job):
    if (set(job['fit_readers']) & {job['validation_reader'],job['test_reader']}
            or job['validation_reader']==job['test_reader'] or len(job['fit_readers'])!=10):
        raise ValueError('Reader leakage in split definition.')
    train=np.concatenate([X[codes.index(c)] for c in job['fit_readers']],axis=0)
    validation=X[codes.index(job['validation_reader'])]
    test=X[codes.index(job['test_reader'])]
    return train,np.tile(target,10),validation,target.copy(),test,target.copy()


def score(truth,probabilities):
    truth=np.asarray(truth);probabilities=np.asarray(probabilities,dtype=np.float64)
    if (truth.ndim!=1 or len(truth)==0 or probabilities.shape!=(len(truth),2)
            or not np.isin(truth,[0,1]).all() or not np.isfinite(probabilities).all()
            or np.any(probabilities<0) or np.any(probabilities>1)
            or not np.allclose(probabilities.sum(1),1,atol=1e-5)):
        raise ValueError('Invalid prediction probabilities/targets.')
    pred=probabilities.argmax(1)
    tn=int(np.sum((truth==0)&(pred==0)));tp=int(np.sum((truth==1)&(pred==1)))
    fp=int(np.sum((truth==0)&(pred==1)));fn=int(np.sum((truth==1)&(pred==0)))
    divide=lambda a,b:a/b if b else 0.
    return {'macro_f1':(divide(2*tn,2*tn+fp+fn)+divide(2*tp,2*tp+fp+fn))/2,
        'accuracy':(tn+tp)/len(truth),'tn':tn,'fp':fp,'fn':fn,'tp':tp,
        'correct':tn+tp,'n_test':len(truth),'collapsed':bool(np.unique(pred).size==1),
        'probability_positive_std':float(np.std(probabilities[:,1]))}


def audit_inputs(X,codes,ids,jobs):
    """Detect exact nonzero duplicates and same-sentence mean-imputation signatures.

The all-zero inputs are retained and reported. Cross-reader sentence identities
are intentionally shared. Matching a tested mean is a stop/review flag, not
proof of the historical imputation algorithm; no near-duplicate certificate.
"""
    hashes={};zeros={};mean_matches=[]
    for p,code in enumerate(codes):
        zeros[code]=0
        for j,sid in enumerate(ids):
            a=X[p,j]
            if not np.any(a):zeros[code]+=1;continue
            h=hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
            if h in hashes:raise ValueError(f'Exact nonzero duplicate across examples: {hashes[h]} and {(code,int(sid))}')
            hashes[h]=(code,int(sid))
            others=np.delete(X[:,j],p,axis=0).astype(np.float64)
            donors=others[np.any(others,axis=(1,2))]
            for name,values in [('all_other_readers',others),('nonzero_other_readers',donors)]:
                if len(values) and np.allclose(a,values.mean(0),rtol=1e-6,atol=0):
                    mean_matches.append({'reader':code,'sentence_id':int(sid),'donors':name})
    if mean_matches:raise ValueError('Possible cross-reader mean filling; stop and inspect: '+str(mean_matches))
    for job in jobs:
        fit={(c,int(s)) for c in job['fit_readers'] for s in ids}
        val={(job['validation_reader'],int(s)) for s in ids}
        test={(job['test_reader'],int(s)) for s in ids}
        if fit&val or fit&test or val&test:raise ValueError('Same reader/sentence example in multiple partitions.')
    return {'all_zero_inputs_by_reader':zeros,'exact_nonzero_duplicate_matches':0,
        'tested_same_sentence_other_reader_mean_matches':0,'reader_example_partitions_disjoint':True,
        'sentence_identities_intentionally_shared':True,
        'limitation':'Does not certify undocumented upstream ICA/filtering/interpolation or exclude all near-duplicates.'}


def set_seed(seed):
    import tensorflow as tf
    random.seed(seed);np.random.seed(seed);tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def build_independent():
    import tensorflow as tf
    m=tf.keras.Sequential()
    m.add(tf.keras.layers.Input(shape=(49,126,1)))
    for filters,kernel,padding in [(64,(4,4),'same'),(128,(3,3),'same')]:
        m.add(tf.keras.layers.Conv2D(filters,kernel,padding=padding,activation='relu'))
    m.add(tf.keras.layers.AveragePooling2D(pool_size=(3,3)))
    m.add(tf.keras.layers.Dropout(.33))
    m.add(tf.keras.layers.Conv2D(128,(3,3),activation='relu'))
    m.add(tf.keras.layers.AveragePooling2D(pool_size=(3,3)))
    m.add(tf.keras.layers.Dropout(.33));m.add(tf.keras.layers.Flatten())
    for n in [64,64,32,32,16,4]:m.add(tf.keras.layers.Dense(n,activation='relu'))
    m.add(tf.keras.layers.Dense(2,activation='softmax'))
    m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.0001),loss='binary_crossentropy',metrics=['accuracy'])
    return m


def verify_architecture():
    import tensorflow as tf
    from model import build_cnn as legacy_model
    keys=['filters','kernel_size','padding','activation','pool_size','strides','rate','units']
    def signature(m):
        return [(type(l).__name__,{k:v for k,v in l.get_config().items() if k in keys}) for l in m.layers]
    checks=[]
    for seed in [11,23,37]:
        tf.keras.backend.clear_session();set_seed(seed);a=build_independent()
        wa=a.get_weights();sa=signature(a)
        probe=np.linspace(0,.1,49*126,dtype=np.float32).reshape(1,49,126,1)
        pa=np.asarray(a(probe,training=False))
        tf.keras.backend.clear_session();set_seed(seed);b=legacy_model((49,126,1),lr=.0001)
        if sa!=signature(b) or a.count_params()!=b.count_params():raise ValueError('Independent CNN architecture differs.')
        wb=b.get_weights()
        if len(wa)!=len(wb) or any(not np.array_equal(x,y) for x,y in zip(wa,wb)):
            raise ValueError('Independent initialization differs; review before treating this as a V4 rerun.')
        if not np.allclose(pa,np.asarray(b(probe,training=False)),rtol=1e-6,atol=1e-7):
            raise ValueError('Independent/legacy forward predictions differ.')
        checks.append({'seed':seed,'architecture_and_initial_weights_identical':True,'n_parameters':int(b.count_params())})
    tf.keras.backend.clear_session();return checks


def fit_independent(model,xf,yf,xv,yv,config):
    """Own macro-F1 stopping callback, same warmup/patience semantics as V4.

No test arrays/labels are passed here. Only the validation reader selects the
epoch; the test reader is evaluated once after best weights are restored.
"""
    import tensorflow as tf
    class Checkpoint(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__();self.history=[];self.best=-math.inf;self.wait=0
            self.weights=None;self.best_epoch=None
        def on_epoch_end(self,epoch,logs=None):
            logs=logs or {};sc=score(yv,self.model.predict(xv,verbose=0))
            row={'epoch':epoch+1,'train_loss':float(logs['loss']),'train_accuracy':float(logs['accuracy']),
                'val_loss':float(logs['val_loss']),'val_accuracy':float(logs['val_accuracy']),
                'val_macro_f1':sc['macro_f1']}
            if not all(math.isfinite(v) for v in row.values()):raise ValueError('Nonfinite training history.')
            self.history.append(row)
            if epoch<config['start_epoch']:return
            if sc['macro_f1']>self.best:
                self.best=sc['macro_f1'];self.best_epoch=epoch+1;self.wait=0;self.weights=self.model.get_weights()
            else:
                self.wait+=1
                if self.wait>=config['patience']:self.model.stop_training=True
        def on_train_end(self,logs=None):
            if self.weights is None:raise ValueError('No eligible validation checkpoint.')
            self.model.set_weights(self.weights)
    cb=Checkpoint()
    model.fit(xf,np.eye(2,dtype=np.float32)[yf],validation_data=(xv,np.eye(2,dtype=np.float32)[yv]),
        epochs=config['epochs'],batch_size=config['batch_size'],shuffle=True,verbose=0,callbacks=[cb])
    return cb.history,{'best_epoch':cb.best_epoch,'best_validation_macro_f1':cb.best,'stopping_epoch':len(cb.history)}


@contextmanager
def heartbeat(label):
    done=threading.Event();start=time.monotonic()
    def report():
        while not done.wait(30):print(f'Still training {label}: {int(time.monotonic()-start)}s',flush=True)
    t=threading.Thread(target=report,daemon=True);t.start()
    try:yield
    finally:done.set();t.join()


def stem(job):return f"{job['condition']}_seed_{job['seed']}_{job['test_reader']}"


def verify_job(path,job,ids,targets,metadata_hash):
    with zipfile.ZipFile(path) as z:
        if z.testzip() or set(z.namelist())!={'predictions.csv','history.csv','metrics.json','manifest.json'}:
            raise ValueError('Corrupt/incomplete result archive.')
        manifest=json.loads(z.read('manifest.json'))
        if manifest.get('job')!=job or manifest.get('metadata_sha256')!=metadata_hash or manifest.get('save_weights') is not False:
            raise ValueError('Saved job definition differs.')
        if set(manifest['files'])!={'predictions.csv','history.csv','metrics.json'}:
            raise ValueError('Unexpected saved artifacts.')
        for name,h in manifest['files'].items():
            if hashlib.sha256(z.read(name)).hexdigest()!=h:raise ValueError('Archive checksum mismatch.')
        rows=csv_rows(z.read('predictions.csv'));history=csv_rows(z.read('history.csv'));metrics=json.loads(z.read('metrics.json'))
    if len(rows)!=110 or {int(r['sentence_id']) for r in rows}!=set(map(int,ids)):
        raise ValueError('Incomplete/duplicate test IDs.')
    expected=dict(zip(map(int,ids),map(int,targets[job['condition']])))
    truth=[];prob=[]
    for row in rows:
        if (row['participant']!=job['test_reader'] or int(row['seed'])!=job['seed'] or row['condition']!=job['condition']
                or int(row['true_label'])!=expected[int(row['sentence_id'])]):raise ValueError('Saved label/reader/seed mismatch.')
        p=[float(row['probability_negative']),float(row['probability_positive'])]
        if int(row['predicted_label'])!=int(np.argmax(p)):raise ValueError('Predicted class differs from probabilities.')
        truth.append(int(row['true_label']));prob.append(p)
    computed=score(truth,prob)
    if any(abs(float(metrics[k])-v)>1e-12 for k,v in computed.items()):raise ValueError('Saved metric differs from predictions.')
    if (metrics['n_fit']!=1100 or metrics['n_validation']!=110 or metrics['job']!=job
            or not history or len(history)!=metrics['stopping_epoch']):raise ValueError('Saved fit/history definition differs.')
    eligible=[h for h in history if int(h['epoch'])>10]
    best=max(eligible,key=lambda h:float(h['val_macro_f1']))
    if int(best['epoch'])!=metrics['best_epoch'] or abs(float(best['val_macro_f1'])-metrics['best_validation_macro_f1'])>1e-12:
        raise ValueError('Checkpoint is inconsistent with validation history.')
    return metrics,rows


def save_job(directory,job,predictions,history,metrics,ids,targets,metadata_hash):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    path=directory/(stem(job)+'.zip')
    if path.exists():raise ValueError('Never overwrite a saved job.')
    bodies={'predictions.csv':csv_bytes(predictions),'history.csv':csv_bytes(history),
            'metrics.json':json.dumps(metrics,allow_nan=False).encode()}
    manifest={'job':job,'metadata_sha256':metadata_hash,'save_weights':False,
        'files':{n:hashlib.sha256(b).hexdigest() for n,b in bodies.items()}}
    partial=path.with_suffix('.partial')
    with zipfile.ZipFile(partial,'w',zipfile.ZIP_DEFLATED) as z:
        for name,body in {**bodies,'manifest.json':json.dumps(manifest).encode()}.items():z.writestr(name,body)
    verify_job(partial,job,ids,targets,metadata_hash);os.replace(partial,path)


def mirror_files(source,target,restore=False):
    source,target=Path(source),Path(target)
    for path in source.rglob('*'):
        if not path.is_file() or path.name.endswith('.partial'):continue
        dest=target/path.relative_to(source);dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():
            if digest(path)==digest(dest):continue
            if restore:continue
            if path.suffix=='.zip' or path.name in ('run_metadata.json','split_manifest.json'):
                raise ValueError('Conflicting mirror archive/definition; preserve both.')
        tmp=dest.with_name(dest.name+'.partial');shutil.copyfile(path,tmp)
        if digest(path)!=digest(tmp):raise ValueError('Result copy checksum failed.')
        os.replace(tmp,dest)


def collect(root,jobs,ids,targets,metadata_hash,config,require_complete=False):
    root=Path(root);expected={stem(j)+'.zip' for j in jobs}
    if (root/'jobs').exists() and {p.name for p in (root/'jobs').glob('*.zip')}-expected:
        raise ValueError('Unexpected job archive in this output.')
    records=[];done=set()
    for job in jobs:
        path=root/'jobs'/(stem(job)+'.zip')
        if path.exists():
            metrics,_=verify_job(path,job,ids,targets,metadata_hash)
            done.add(stem(job));records.append({'condition':job['condition'],'seed':job['seed'],
                'participant':job['test_reader'],**{k:v for k,v in metrics.items() if k!='job'},
                'hyperparameters_json':json.dumps(config,sort_keys=True)})
    if require_complete and len(done)!=72:raise ValueError(f'Only{len(done)}/72 verified fits.')
    if records:
        (root/'fold_metrics.csv').write_bytes(csv_bytes(records))
        summary=[]
        for condition in config['conditions']:
            rr=[r for r in records if r['condition']==condition]
            if not rr:continue
            summary.append({'condition':condition,'fits':len(rr),
                'mean_reader_seed_macro_f1':sum(r['macro_f1'] for r in rr)/len(rr),
                'accuracy':sum(r['correct'] for r in rr)/(110*len(rr)),
                'correct':sum(r['correct'] for r in rr),'predictions':110*len(rr),
                'collapses':sum(r['collapsed'] for r in rr)})
        (root/'summary_overall.csv').write_bytes(csv_bytes(summary))
    write_json(root/'completion.json',{'complete':len(done)==72,'verified_fits':len(done),'expected_fits':72})
    return done


def validate_output_paths(root,mirror,reference):
    root,mirror,reference=map(lambda p:Path(p).resolve(),[root,mirror,reference])
    if not str(root).startswith('/content/') or '/drive/' in str(root):raise ValueError('Use a local Colab /content primary.')
    if root==reference or root in reference.parents or reference in root.parents:
        raise ValueError('Primary results must not overlap original results.')
    if any(p==mirror or p in mirror.parents or mirror in p.parents for p in [root,reference]):
        raise ValueError('Use a separate NEW Drive folder.')


def run(args):
    import tensorflow as tf
    import pandas as pd
    import sklearn
    c=json.loads(Path(args.config).read_text());validate_config(c)
    root,mirror=Path(args.local_dir).resolve(),Path(args.mirror_dir).resolve()
    validate_output_paths(root,mirror,args.reference_dir)
    mirror_files(mirror,root,restore=True)
    if not tf.config.list_physical_devices('GPU'):raise ValueError('Select a GPU runtime and reconnect.')
    original=json.loads(Path(args.reference_dir,'run_metadata.json').read_text())
    reference_config=original['definition']['config']
    if (reference_config.get('version')!='v4_low_lr_confirmation'
            or any(reference_config.get(k)!=c[k] for k in
                   ['learning_rate','preprocessing','epochs','batch_size','patience','start_epoch','deterministic_ops'])):
        raise ValueError('Reference is not the frozen V4 confirmation setup.')
    runtime={'tensorflow':tf.__version__,'numpy':np.__version__,'pandas':pd.__version__,'scikit_learn':sklearn.__version__}
    if any(original.get(k)!=v for k,v in runtime.items()):raise ValueError('Runtime differs from the V4 reference; stop and review.')
    paths=[Path(args.labels_csv)]+[Path(args.trt_dir,f'{n}_trt_total.pickle') for n,_ in READERS]
    data_files=[{'name':p.name,'size_bytes':p.stat().st_size,'sha256':digest(p)} for p in paths]
    if data_files!=original['definition']['data_files']:raise ValueError('Input files differ from the recorded V4 reference.')
    X,y,codes,ids,loader_audit=load_independent(args.trt_dir,args.labels_csv)
    # Reference code is a check only. It does not supply the training arrays.
    from data import build_binary_dataset
    oldX,oldy,oldcodes,oldids=build_binary_dataset(args.trt_dir,args.labels_csv,return_sentence_ids=True)
    if not (np.array_equal(X,oldX) and np.array_equal(y,oldy) and codes==oldcodes and np.array_equal(ids,oldids)):
        raise ValueError('Independent and original loader disagree; do not train.')
    del oldX,oldy,oldcodes,oldids
    targets=targets_for(y,c);jobs=jobs_for(codes,c)
    audits=audit_inputs(X,codes,ids,jobs)
    architecture=verify_architecture()
    source=[Path(__file__),Path(args.config),Path(__file__).with_name('model.py'),Path(__file__).with_name('data.py')]
    definition={'config':c,'data_files':data_files,'source_files':[{'name':p.name,'sha256':digest(p)} for p in source],
                'runtime':runtime,'original_v4_metadata_sha256':digest(Path(args.reference_dir,'run_metadata.json'))}
    metadata=root/'run_metadata.json'
    if metadata.exists():
        if json.loads(metadata.read_text())['definition']!=definition:raise ValueError('Resumed run definition changed.')
    else:write_json(metadata,{'created_utc':datetime.now(timezone.utc).isoformat(),'definition':definition})
    plan={'sentence_ids':list(map(int,ids)),'targets':{k:list(map(int,v)) for k,v in targets.items()},'jobs':jobs}
    write_json(root/'split_manifest.json',plan,immutable=True)
    write_json(root/'preflight.json',{'passed':True,'independent_loader_equals_original':True,
        'input_shape':list(X.shape),'architecture_checks':architecture,**loader_audit,**audits})
    h=digest(metadata);done=collect(root,jobs,ids,targets,h,c,require_complete=args.verify_only)
    mirror_files(root,mirror)
    print(f'Independent V4 preflight passed;{len(done)}/72 fits verified. New seeds11/23/37. Same-sentence identities intentionally shared.',flush=True)
    if args.prepare_only or args.verify_only:print('No training started.',flush=True);return
    for job in jobs:
        label=stem(job)
        if label in done:continue
        xf,yf,xv,yv,xt,yt=inputs_for(X,codes,targets[job['condition']],job)
        tf.keras.backend.clear_session();set_seed(job['seed']);model=build_independent()
        print(f'START {len(done)+1}/72 {label}; test reader excluded from fit/validation',flush=True)
        start=time.monotonic()
        with heartbeat(label):history,chosen=fit_independent(model,xf[...,None],yf,xv[...,None],yv,c)
        # Only now is the test reader evaluated; test metrics never select epochs.
        prob=model.predict(xt[...,None],verbose=0);sc=score(yt,prob)
        from sklearn.metrics import f1_score
        if abs(sc['macro_f1']-float(f1_score(yt,prob.argmax(1),average='macro',zero_division=0)))>1e-12:
            raise ValueError('Independent macro-F1 and sklearn disagree.')
        train_score=score(yf,model.predict(xf[...,None],verbose=0))
        val_score=score(yv,model.predict(xv[...,None],verbose=0))
        if abs(val_score['macro_f1']-chosen['best_validation_macro_f1'])>1e-12:
            raise ValueError('Restored weights do not match the selected validation epoch.')
        predictions=[{'condition':job['condition'],'seed':job['seed'],'participant':job['test_reader'],
            'sentence_id':int(sid),'true_label':int(t),'predicted_label':int(np.argmax(p)),
            'probability_negative':float(p[0]),'probability_positive':float(p[1])} for sid,t,p in zip(ids,yt,prob)]
        metrics={'job':job,'n_fit':1100,'n_validation':110,**sc,**chosen,
            'selected_train_macro_f1':train_score['macro_f1'],'selected_validation_macro_f1':val_score['macro_f1'],
            'duration_seconds':time.monotonic()-start,'n_parameters':int(model.count_params())}
        save_job(root/'jobs',job,predictions,history,metrics,ids,targets,h)
        done=collect(root,jobs,ids,targets,h,c);mirror_files(root,mirror)
        print(f'SAVED {len(done)}/72 {label}; F1={sc["macro_f1"]:.4f}; collapse={sc["collapsed"]}',flush=True)
    collect(root,jobs,ids,targets,h,c,require_complete=True);mirror_files(root,mirror)
    print('COMPLETE72/72 verified. No model weights saved; compare real/arbitrary scores and train/validation/test gaps.',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ['trt-dir','labels-csv','reference-dir','local-dir','mirror-dir']:p.add_argument('--'+flag,required=True)
    p.add_argument('--config',default='configs/v4_independent_audit.json')
    mode=p.add_mutually_exclusive_group();mode.add_argument('--prepare-only',action='store_true');mode.add_argument('--verify-only',action='store_true')
    run(p.parse_args())


if __name__=='__main__':main()
