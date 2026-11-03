"""Independent arithmetic, loader, split, checkpoint and archive tests. No ML training."""
import argparse
import contextlib
import csv
import io
import json
from pathlib import Path
import pickle
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import v4_independent_audit as audit


def fixture(folder):
    labels=Path(folder,'labels.csv');trt=Path(folder,'inputs');trt.mkdir()
    with labels.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['sentence','sentiment_label','control']);w.writeheader()
        for sid in range(1,166):w.writerow({'sentence':f'word {sid}','sentiment_label':[-1,0,1][(sid-1)%3],'control':0})
    for p,(name,_) in enumerate(audit.READERS):
        records={}
        for sid in range(1,166):
            vec=np.full(126,(p+1)*1000+sid,dtype=np.float32)
            words={0:{'TRT_total':vec}}
            if sid==1:words={i:{'TRT_total':vec+i/100} for i in range(49)}
            if sid==2:words={0:{'TRT_total':np.zeros(2)}}
            records[sid]={'sentenceId':sid,'persian_sentence':f'word {sid}','word':words}
        with (trt/(name+'_trt_total.pickle')).open('wb') as f:pickle.dump(records,f)
    return trt,labels


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.c=json.loads((ROOT/'configs/v4_independent_audit.json').read_text())
        self.codes=[c for _,c in audit.READERS]
        self.ids=np.arange(1,111);self.y=np.asarray([i%2 for i in range(110)])

    def test_independent_metrics_known_cases(self):
        truth=np.asarray([0,0,1,1])
        perfect=np.eye(2)[truth];wrong=perfect[:,::-1];constant=np.tile([1.,0.],(4,1))
        self.assertEqual(audit.score(truth,perfect)['macro_f1'],1.)
        self.assertEqual(audit.score(truth,wrong)['macro_f1'],0.)
        self.assertAlmostEqual(audit.score(truth,constant)['macro_f1'],1/3)
        self.assertEqual(audit.score(truth,constant)['accuracy'],.5)
        for bad in [np.full((4,2),np.nan),np.ones((4,2)),np.ones((3,2))]:
            with self.assertRaises(ValueError):audit.score(truth,bad)

    def test_new_seeds_targets_and_reader_splits(self):
        audit.validate_config(self.c)
        self.assertFalse(set(self.c['seeds'])&{42,45,52,62,72})
        targets=audit.targets_for(self.y,self.c)
        self.assertEqual(targets['arbitrary_sentence_labels'].sum(),55)
        self.assertFalse(np.array_equal(targets['sentiment'],targets['arbitrary_sentence_labels']))
        jobs=audit.jobs_for(self.codes,self.c);self.assertEqual(len(jobs),72)
        X=np.arange(1320).reshape(12,110,1,1)
        for job in jobs:
            xf,yf,xv,yv,xt,yt=audit.inputs_for(X,self.codes,self.y,job)
            self.assertEqual((len(xf),len(xv),len(xt)),(1100,110,110))
            self.assertEqual((sum(yf),sum(yv),sum(yt)),(550,55,55))
            self.assertFalse(set(xf.ravel())&set(xt.ravel()))
            self.assertFalse(set(xv.ravel())&set(xt.ravel()))
        bad={**jobs[0],'fit_readers':jobs[0]['fit_readers']+[jobs[0]['test_reader']]}
        with self.assertRaises(ValueError):audit.inputs_for(X,self.codes,self.y,bad)

    def test_independent_loader_matches_legacy_and_rejects_misalignment(self):
        from data import build_binary_dataset
        with tempfile.TemporaryDirectory() as temp:
            trt,labels=fixture(temp)
            X,y,codes,ids,checks=audit.load_independent(trt,labels)
            ox,oy,oc,oi=build_binary_dataset(str(trt),str(labels),return_sentence_ids=True)
            self.assertTrue(np.array_equal(X,ox));self.assertTrue(np.array_equal(y,oy))
            self.assertEqual(codes,oc);self.assertTrue(np.array_equal(ids,oi))
            self.assertEqual(X.shape,(12,110,49,126))
            path=trt/(audit.READERS[1][0]+'_trt_total.pickle')
            with path.open('rb') as f:r=pickle.load(f)
            r[1]['persian_sentence']='wrong sentence'
            with path.open('wb') as f:pickle.dump(r,f)
            with self.assertRaises(ValueError):audit.load_independent(trt,labels)

    def test_duplicate_and_mean_imputation_screens_and_zero_retention(self):
        X=np.random.default_rng(5).normal(size=(12,110,2,2)).astype(np.float32)
        jobs=audit.jobs_for(self.codes,self.c)
        X[0,0]=0
        result=audit.audit_inputs(X,self.codes,self.ids,jobs)
        self.assertEqual(result['all_zero_inputs_by_reader'][self.codes[0]],1)
        bad=X.copy();bad[0,1]=bad[1,1]
        with self.assertRaises(ValueError):audit.audit_inputs(bad,self.codes,self.ids,jobs)
        bad=X.copy();bad[0,2]=bad[1:,2].mean(0)
        with self.assertRaises(ValueError):audit.audit_inputs(bad,self.codes,self.ids,jobs)

    def test_own_checkpoint_ignores_warmup_and_restores_validation_best(self):
        tf=types.ModuleType('tensorflow');tf.keras=types.SimpleNamespace(callbacks=types.SimpleNamespace(Callback=object))
        truth=np.asarray([0,0,1,1]);perfect=np.eye(2)[truth]
        class Model:
            stop_training=False
            def predict(self,x,verbose=0):return np.tile([1.,0.],(4,1)) if self.epoch==11 else perfect
            def get_weights(self):return [self.epoch]
            def set_weights(self,w):self.epoch=w[0]
            def fit(self,xf,yf,**kwargs):
                cb=kwargs['callbacks'][0];cb.model=self
                for ep in range(kwargs['epochs']):
                    self.epoch=ep+1
                    cb.on_epoch_end(ep,{'loss':1.,'accuracy':.5,'val_loss':1.,'val_accuracy':.5})
                    if self.stop_training:break
                cb.on_train_end()
        m=Model()
        with patch.dict(sys.modules,{'tensorflow':tf}):
            history,selected=audit.fit_independent(m,np.zeros((4,1)),truth,np.zeros((4,1)),truth,self.c)
        self.assertEqual(selected['best_epoch'],12)
        self.assertEqual(len(history),52)
        self.assertEqual(m.epoch,12)

    def test_output_paths_protect_original_and_drive_primary(self):
        for local,mirror,ref in [('/tmp/local','/tmp/mirror','/tmp/ref'),
                                ('/content/drive/local','/content/drive/new','/content/drive/ref'),
                                ('/content/local','/content/drive/ref','/content/drive/ref')]:
            with self.assertRaises(ValueError):audit.validate_output_paths(local,mirror,ref)
        audit.validate_output_paths('/content/local','/content/drive/new','/content/drive/ref')

    def test_fake_full72_run_no_refit_resume_and_corruption_stop(self):
        import pandas as pd
        tf=types.ModuleType('tensorflow');tf.__version__='fake'
        tf.config=types.SimpleNamespace(list_physical_devices=lambda *a:['fake GPU'])
        tf.keras=types.SimpleNamespace(backend=types.SimpleNamespace(clear_session=lambda:None))
        sklearn=types.ModuleType('sklearn');sklearn.__version__='fake'
        sklearn_metrics=types.ModuleType('sklearn.metrics')
        sklearn_metrics.f1_score=lambda y,p,**k: audit.score(y,np.eye(2)[p])['macro_f1']
        calls=[]
        class Model:
            def predict(self,x,verbose=0):
                sid=np.rint(x[:,0,0,0]).astype(int)%1000
                pp=np.where(sid%3==0,.8,.2);return np.column_stack([1-pp,pp])
            def count_params(self):return 2
        def fit(m,xf,yf,xv,yv,c):
            self.assertEqual((len(xf),len(xv)),(1100,110));calls.append(1)
            sc=audit.score(yv,m.predict(xv))
            history=[{'epoch':e,'val_macro_f1':sc['macro_f1']} for e in range(1,12)]
            return history,{'best_epoch':11,'best_validation_macro_f1':sc['macro_f1'],'stopping_epoch':11}
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);trt,labels=fixture(temp)
            root=folder/'output';mirror=folder/'mirror';ref=folder/'reference';ref.mkdir()
            paths=[labels]+[trt/(n+'_trt_total.pickle') for n,_ in audit.READERS]
            config=folder/'config.json';config.write_text(json.dumps(self.c))
            old=json.loads((ROOT/'configs/v4_low_lr_confirmation.json').read_text())
            metadata={'definition':{'config':old,'data_files':[
                {'name':p.name,'size_bytes':p.stat().st_size,'sha256':audit.digest(p)} for p in paths]},
                'tensorflow':'fake','numpy':np.__version__,'pandas':pd.__version__,'scikit_learn':'fake'}
            audit.write_json(ref/'run_metadata.json',metadata)
            original=audit.digest(ref/'run_metadata.json')
            args=argparse.Namespace(config=str(config),local_dir=str(root),mirror_dir=str(mirror),reference_dir=str(ref),
                trt_dir=str(trt),labels_csv=str(labels),prepare_only=True,verify_only=False)
            with patch.dict(sys.modules,{'tensorflow':tf,'sklearn':sklearn,'sklearn.metrics':sklearn_metrics}),\
                 patch.object(audit,'validate_output_paths',lambda *a:None),\
                 patch.object(audit,'verify_architecture',lambda:[{'checked':'synthetic'}]),\
                 patch.object(audit,'set_seed',lambda *a:None),\
                 patch.object(audit,'build_independent',Model),patch.object(audit,'fit_independent',fit),\
                 contextlib.redirect_stdout(io.StringIO()):
                audit.run(args);self.assertEqual(calls,[])
                args.prepare_only=False;args.verify_only=True
                with self.assertRaises(ValueError):audit.run(args)
                args.verify_only=False;audit.run(args);self.assertEqual(len(calls),72)
                status=json.loads((root/'completion.json').read_text());self.assertTrue(status['complete'])
                summary=pd.read_csv(root/'summary_overall.csv');self.assertEqual(set(summary.predictions),{3960})
                self.assertEqual(len(list((mirror/'jobs').glob('*.zip'))),72)
                self.assertEqual(audit.digest(ref/'run_metadata.json'),original)
                (root/'summary_overall.csv').unlink();args.verify_only=True;audit.run(args)
                self.assertTrue((root/'summary_overall.csv').exists());self.assertEqual(len(calls),72)
                path=next((root/'jobs').glob('*.zip'))
                with zipfile.ZipFile(path) as z:self.assertNotIn('weights.npz',z.namelist())
                path.write_bytes(b'bad archive')
                with self.assertRaises(zipfile.BadZipFile):audit.run(args)
                self.assertEqual(len(calls),72)


if __name__=='__main__':unittest.main()
