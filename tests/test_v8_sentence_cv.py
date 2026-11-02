"""Synthetic planning/archive/resume tests; no ML dependencies or real training."""
import argparse
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import v8_sentence_cv as cv
import followup_common as common
from v7_unseen_sentences import make_sentence_plan
from v8_seen_readers_light import publish_bundle, verify_bundle


def scores(truth,prob):
    truth=np.asarray(truth);pred=np.argmax(prob,axis=1)
    tn=int(sum((truth==0)&(pred==0)));tp=int(sum((truth==1)&(pred==1)))
    fp=int(sum((truth==0)&(pred==1)));fn=int(sum((truth==1)&(pred==0)))
    div=lambda a,b:a/b if b else 0.
    return {'tn':tn,'tp':tp,'fp':fp,'fn':fn,'accuracy':(tn+tp)/len(truth),
            'macro_f1':(div(2*tn,2*tn+fp+fn)+div(2*tp,2*tp+fp+fn))/2,
            'collapsed':len(set(pred))==1}


class CVTests(unittest.TestCase):
    def setUp(self):
        self.base=json.loads((ROOT/'configs/v8_seen_readers_unseen_sentences.json').read_text())
        self.options=json.loads((ROOT/'configs/v8_sentence_cv.json').read_text())
        self.ids=list(range(1,111));self.labels=[i%2 for i in range(110)]
        self.texts=[f'sentence{i}' for i in self.ids]
        self.codes=[f'P{i:02d}' for i in range(12)]
        self.anchor=make_sentence_plan(self.labels,self.ids,self.texts,self.base)
        self.plans=cv.make_folds(self.anchor,self.options)

    def test_anchor_exact_and_each_sentence_tested_once(self):
        self.assertEqual(self.plans[0],self.anchor)
        tests=[r['sentence_id'] for p in self.plans for r in p if r['partition']=='unseen_test']
        self.assertEqual(sorted(tests),self.ids)
        self.assertEqual(self.plans,cv.make_folds(self.anchor,self.options))

    def test_every_fold_balanced_for_both_targets_and_no_partition_overlap(self):
        for plan in self.plans:
            pools=[]
            for name,n in [('fit',66),('validation',22),('unseen_test',22)]:
                rows=[r for r in plan if r['partition']==name]
                self.assertEqual(len(rows),n)
                for c in ['sentiment','shuffled_sentence_labels']:
                    self.assertEqual(sum(r[c] for r in rows),n//2)
                pools.append({r['sentence_id'] for r in rows})
            self.assertFalse(pools[0]&pools[1] or pools[1]&pools[2] or pools[0]&pools[2])
            for a,b in zip(plan,self.anchor):
                for k in ['index','sentence_id','sentiment','shuffled_sentence_labels','normalized_text_sha256']:
                    self.assertEqual(a[k],b[k])

    def test_joint_stratification_across_many_anchor_label_orders(self):
        for seed in range(30):
            labels=np.asarray(self.labels)[np.random.default_rng(seed).permutation(110)].tolist()
            anchor=make_sentence_plan(labels,self.ids,self.texts,self.base)
            self.assertEqual(cv.make_folds(anchor,self.options)[0],anchor)

    def test_configuration_and_bad_anchor_rejected(self):
        for key,value in [('n_folds',4),('save_weights',True),('completion_split_seed',1)]:
            options={**self.options,key:value}
            with self.subTest(key=key),self.assertRaises(ValueError):cv.make_folds(self.anchor,options)
        for mutation in ['ids','text','balance']:
            anchor=copy.deepcopy(self.anchor)
            if mutation=='ids':anchor[0]['sentence_id']=anchor[1]['sentence_id']
            if mutation=='text':anchor[0]['normalized_text_sha256']=anchor[1]['normalized_text_sha256']
            if mutation=='balance':anchor[0]['sentiment']=1-anchor[0]['sentiment']
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):cv.make_folds(anchor,self.options)

    def make_job(self,root,condition,seed,metadata,plan=None):
        plan=plan or self.anchor
        test=[r for r in plan if r['partition']=='unseen_test']
        frames,records=[],[]
        for code in self.codes:
            truth=np.asarray([r[condition] for r in test]);sentiment=np.asarray([r['sentiment'] for r in test])
            pp=np.where(sentiment==1,.8,.2);prob=np.column_stack([1-pp,pp]);sc=scores(truth,prob)
            identity={'condition':condition,'seed':seed,'participant':code}
            frames.append(pd.DataFrame({**identity,'sentence_id':[r['sentence_id'] for r in test],
                'true_label':truth,'predicted_label':prob.argmax(1),'probability_negative':prob[:,0],'probability_positive':prob[:,1]}))
            records.append({**identity,'n_fit':792,'n_validation':264,'n_test':22,
                'n_test_negative':11,'n_test_positive':11,'learning_rate':.0001,'preprocessing':'raw',
                'correct':sc['tn']+sc['tp'],**sc})
        with tempfile.TemporaryDirectory() as work:
            pd.concat(frames).to_csv(Path(work,'predictions.csv'),index=False)
            pd.DataFrame(records).to_csv(Path(work,'metrics_by_reader.csv'),index=False)
            pd.DataFrame([{'epoch':1}]).to_csv(Path(work,'history.csv'),index=False)
            manifest={'condition':condition,'seed':seed,'codes':self.codes,'config':self.base,'sentence_plan':plan,
                'metadata_sha256':common.sha256(str(root/'run_metadata.json')),'storage_profile':'results_only_no_weights'}
            return publish_bundle(work,str(root),condition,seed,manifest,self.codes,plan,self.base)

    def test_import_preserves_payloads_and_sources_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(common,'macro_metrics',scores):
            anchor=Path(temp,'anchor');target=Path(temp,'copy')
            for root in [anchor,target]:common.atomic_json({'where':str(root)},str(root/'run_metadata.json'))
            for seed in self.base['seeds']:
                for c in self.base['conditions']:self.make_job(anchor,c,seed,{})
            before={p.name:common.sha256(str(p)) for p in (anchor/'jobs').glob('*.zip')}
            self.assertEqual(len(cv.import_anchor(anchor,target,self.codes,self.anchor,self.base)),4)
            cv.import_anchor(anchor,target,self.codes,self.anchor,self.base)
            for name,h in before.items():
                self.assertEqual(h,common.sha256(str(anchor/'jobs'/name)))
                with zipfile.ZipFile(anchor/'jobs'/name) as a,zipfile.ZipFile(target/'jobs'/name) as b:
                    for payload in ['predictions.csv','metrics_by_reader.csv','history.csv']:
                        self.assertEqual(a.read(payload),b.read(payload))
                    self.assertEqual(json.loads(b.read('manifest.json'))['origin']['kind'],'verified_original_v8_reused')
            missing=next((anchor/'jobs').glob('*.zip'));missing.unlink()
            with self.assertRaises(FileNotFoundError):cv.import_anchor(anchor,target,self.codes,self.anchor,self.base)

    def test_full_fake_pipeline_16_new_fits_resume_and_oof_integrity(self):
        calls=[]
        X=np.zeros((12,110,49,126),np.float32)
        X[:,:,0,0]=np.arange(1,1321).reshape(12,110)
        data=types.ModuleType('data');data.PARTICIPANTS=[(c,c) for c in self.codes]
        data._norm=lambda s:s
        data._read_subject=lambda *a:{i:{'sentenceId':sid,'persian_sentence':text}
                                      for i,(sid,text) in enumerate(zip(self.ids,self.texts))}
        tf=types.ModuleType('tensorflow');tf.__version__='synthetic'
        tf.keras=types.SimpleNamespace(backend=types.SimpleNamespace(clear_session=lambda:None))
        sklearn=types.ModuleType('sklearn');sklearn.__version__='synthetic'
        class Model:
            def predict(self,x,verbose=0):
                index=(x[:,0,0,0].astype(int)-1)%110
                pp=np.where(index%2==1,.8,.2)
                return np.column_stack([1-pp,pp])
            def count_params(self):return 2
            def get_weights(self):raise AssertionError('Do not save weights')
        model=types.ModuleType('model');model.build_cnn=lambda *a,**k:Model()
        with tempfile.TemporaryDirectory() as temp,patch.object(common,'macro_metrics',scores):
            folder=Path(temp);anchor=folder/'anchor';ref=folder/'v4';local=folder/'local';mirror=folder/'mirror'
            metadata={'definition':{'config':self.base,'data_files':[{'sha256':'synthetic'}],
                'source_files':[{'name':n,'sha256':'synthetic'} for n in ['data.py','model.py','followup_common.py']]},
                'tensorflow':'synthetic','numpy':np.__version__,'pandas':pd.__version__,'scikit_learn':'synthetic'}
            for root in [anchor,ref]:common.atomic_json(metadata,str(root/'run_metadata.json'))
            common.atomic_json(self.anchor,str(anchor/'sentence_plan.json'))
            for seed in self.base['seeds']:
                for condition in self.base['conditions']:self.make_job(anchor,condition,seed,metadata)
            old_hashes={str(p):common.sha256(str(p)) for p in anchor.rglob('*') if p.is_file()}
            # Real V4 scoring path with synthetic110-sentence predictions,24 files.
            metrics=[]
            for seed in self.base['seeds']:
                for code in self.codes:
                    truth=np.asarray(self.labels);pp=np.where(truth==1,.8,.2);prob=np.column_stack([1-pp,pp])
                    frame=pd.DataFrame({'seed':seed,'participant':code,'sentence_id':self.ids,'true_label':truth,
                        'predicted_label':truth,'probability_negative':prob[:,0],'probability_positive':prob[:,1]})
                    path=ref/'predictions'/f'subject_val_macro_f1_lr_0.0001_seed_{seed}_{code}.csv'
                    path.parent.mkdir(exist_ok=True);frame.to_csv(path,index=False)
                    metrics.append({'seed':seed,'participant':code,**scores(truth,prob)})
            pd.DataFrame(metrics).to_csv(ref/'fold_metrics.csv',index=False)
            args=argparse.Namespace(local_dir=str(local),mirror_dir=str(mirror),anchor_dir=str(anchor),reference_dir=str(ref),
                trt_dir='synthetic',labels_csv='synthetic',base_config=str(ROOT/'configs/v8_seen_readers_unseen_sentences.json'),
                cv_config=str(ROOT/'configs/v8_sentence_cv.json'),prepare_only=True,verify_only=False)
            def prepare(config,*a):
                current=copy.deepcopy(metadata);current['definition']['config']=config
                common.atomic_json(current,str(local/'run_metadata.json'))
                return X,np.asarray(self.labels),self.codes,np.asarray(self.ids)
            def fit(m,xf,yf,xv,yv,config):
                self.assertEqual((len(xf),len(xv)),(792,264))
                self.assertEqual((sum(yf),sum(yv)),(396,132));calls.append(1)
                return m,pd.DataFrame([{'epoch':1}]),{'best_epoch':1}
            import light_storage
            with patch.dict(sys.modules,{'tensorflow':tf,'sklearn':sklearn,'data':data,'model':model}),\
                 patch.object(common,'prepare',prepare),patch.object(common,'fit_network',fit),\
                 patch.object(common,'set_seed',lambda *a:None),\
                 patch.object(light_storage,'validate_paths',lambda *a:None),\
                 contextlib.redirect_stdout(io.StringIO()):
                cv.run(args);self.assertEqual(calls,[])
                self.assertEqual(json.loads((local/'completion.json').read_text())['completed_fits'],4)
                args.prepare_only=False;args.verify_only=True
                with self.assertRaises(ValueError):cv.run(args)
                args.verify_only=False;cv.run(args)
                self.assertEqual(len(calls),16)
                pred=pd.read_csv(local/'out_of_fold_predictions.csv')
                self.assertEqual(len(pred),5280)
                self.assertFalse(pred[['condition','seed','participant','sentence_id']].duplicated().any())
                self.assertEqual(set(pred.groupby(['condition','seed','participant']).size()),{110})
                self.assertEqual(len(pd.read_csv(local/'out_of_fold_metrics_by_reader.csv')),48)
                self.assertEqual(set(pd.read_csv(local/'summary_fold_mean.csv').test_predictions),{2640})
                self.assertEqual(len(list(mirror.glob('fold_*/jobs/*.zip'))),20)
                self.assertTrue(json.loads((local/'completion.json').read_text())['complete'])
                (local/'summary_oof_110.csv').unlink();args.verify_only=True;cv.run(args)
                self.assertTrue((local/'summary_oof_110.csv').exists());self.assertEqual(len(calls),16)
                self.assertEqual(old_hashes,{str(p):common.sha256(str(p)) for p in anchor.rglob('*') if p.is_file()})
                corrupt=local/'fold_1/jobs/sentiment_seed_45.zip'
                corrupt.write_bytes(b'broken archive')
                with self.assertRaises(zipfile.BadZipFile):cv.run(args)
                self.assertEqual(len(calls),16)


if __name__=='__main__':unittest.main()
