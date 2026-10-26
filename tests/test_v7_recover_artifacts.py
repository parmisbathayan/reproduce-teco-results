"""Synthetic recovery tests: no downloads, installs, TensorFlow or ML fitting."""
import contextlib
import hashlib
import importlib.util
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import v7_recover_artifacts as recovery
from v7_unseen_sentences import make_sentence_plan, partition_indices, reader_plan, VIEWS


def scores(truth, probabilities):
    truth = np.asarray(truth)
    predicted = np.argmax(probabilities,axis=1)
    tn = int(sum((truth==0)&(predicted==0)))
    tp = int(sum((truth==1)&(predicted==1)))
    fp = int(sum((truth==0)&(predicted==1)))
    fn = int(sum((truth==1)&(predicted==0)))
    return dict(tn=tn,tp=tp,fp=fp,fn=fn,accuracy=(tn+tp)/len(truth),
        macro_f1=(2*tn/(2*tn+fp+fn)+2*tp/(2*tp+fp+fn))/2,
        collapsed=len(set(predicted))==1)


class RecoveryTests(unittest.TestCase):
    def test_storage_never_allows_original_or_drive_as_local_primary(self):
        for paths in [('/content/drive/a','/content/drive/b','/content/drive/c'),
                      ('/tmp/a','/content/drive/b','/content/drive/c'),
                      ('/content/a','/content/drive/c/sub','/content/drive/c')]:
            args=types.SimpleNamespace(**dict(zip(('local_dir','mirror_dir','original_dir'),paths)))
            with self.assertRaises(ValueError): recovery.validate_storage_paths(args)

    def test_import_10_regenerate_38_verify_resume_and_reject_tampering(self):
        config=json.loads((ROOT/'configs/v7_unseen_sentences.json').read_text())
        codes=[f'P{i:02d}' for i in range(12)]
        ids=np.arange(1,111)
        y=np.arange(110)%2
        texts=[f'stimulus_{i}' for i in ids]
        plan=make_sentence_plan(y,ids,texts,config)
        X=np.zeros((12,110,49,126),dtype=np.float32)
        X[:,:,0,0]=np.arange(1,1321).reshape(12,110)
        calls=[]
        data=types.ModuleType('data')
        data.PARTICIPANTS=[(p,p) for p in codes]
        data.build_binary_dataset=lambda *a,**k: None
        data._norm=lambda t:t
        data._read_subject=lambda *a:{i:dict(sentenceId=int(s),persian_sentence=t) for i,(s,t) in enumerate(zip(ids,texts))}
        tf=types.ModuleType('tensorflow')
        tf.__version__='synthetic'
        tf.keras=types.SimpleNamespace(backend=types.SimpleNamespace(clear_session=lambda:None))
        sklearn=types.ModuleType('sklearn')
        sklearn.__version__='synthetic'

        class Network:
            def __init__(self):self.weights=[np.asarray([.2,.8])]
            def predict(self,values,verbose=0):
                positive=np.where(((values[:,0,0,0].astype(int)-1)%110)%2,self.weights[0][1],self.weights[0][0])
                return np.column_stack([1-positive,positive])
            def get_weights(self):return [w.copy() for w in self.weights]
            def set_weights(self,values):self.weights=[w.copy() for w in values]
            def count_params(self):return 2

        model=types.ModuleType('model')
        model.build_cnn=lambda *a,**k:Network()
        modules={'data':data,'tensorflow':tf,'sklearn':sklearn,'model':model}
        with tempfile.TemporaryDirectory(prefix='v7-recovery-test-') as temp, patch.dict(sys.modules,modules):
            root=Path(temp)
            original,local,mirror,reference=[root/n for n in ('original','local','mirror','reference')]
            original.mkdir(); reference.mkdir()
            spec=importlib.util.spec_from_file_location('recovery_common_test', ROOT/'src/followup_common.py')
            common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
            metadata=dict(definition=dict(config=config,data_files=[dict(sha256='synthetic')],
                source_files=[dict(name=n,sha256=hashlib.sha256((ROOT/'src'/n).read_bytes()).hexdigest())
                              for n in ('v7_unseen_sentences.py','model.py','data.py','followup_common.py')]),
                tensorflow='synthetic',scikit_learn='synthetic',numpy=np.__version__,pandas=pd.__version__)
            for folder in (original,reference):common.atomic_json(metadata,str(folder/'run_metadata.json'))
            common.atomic_json(plan,str(original/'sentence_plan.json'))
            keys=[(c,s,p) for s in config['seeds'] for p in codes for c in config['conditions']]
            all_rows=[]
            for j,key in enumerate(keys):
                c,s,p=key
                fit_codes,val=reader_plan(codes,p)
                history=pd.DataFrame([dict(epoch=1)])
                for view,partition in [('seen_sentences','fit'),('unseen_sentences','unseen_test')]:
                    indices=partition_indices(plan,partition)
                    truth=np.asarray([plan[i][c] for i in indices])
                    prob=Network().predict(X[codes.index(p),indices,...,None])
                    identifiers=dict(zip(recovery.KEYS,[c,s,p,view]))
                    if j<10:
                        common.save_predictions(str(original/'predictions'/(recovery.stem(key)+'_'+view+'.csv')),
                                                truth,ids[indices],prob,identifiers)
                    metric=scores(truth,prob)
                    all_rows.append(dict(**identifiers,n_fit=660,n_validation=22,n_test=len(indices),
                        n_test_negative=len(indices)//2,n_test_positive=len(indices)//2,
                        fit_participants=','.join(fit_codes),validation_participant=val,preprocessing='raw',
                        learning_rate=.0001,stopping_epoch=1,best_epoch=1,correct=metric['tn']+metric['tp'],**metric))
                if j<10:common.atomic_csv(history,str(original/'histories'/(recovery.stem(key)+'.csv')))
            common.atomic_csv(pd.DataFrame(all_rows),str(original/'fold_metrics.csv'))
            original_hashes={str(p.relative_to(original)):common.sha256(str(p)) for p in original.rglob('*') if p.is_file()}

            def prepare(*args,**kwargs):
                if not (local/'run_metadata.json').exists():common.atomic_json(metadata,str(local/'run_metadata.json'))
                return X,y,codes,ids
            def fit(network,x_fit,y_fit,x_val,y_val,configuration):
                self.assertEqual((len(x_fit),len(x_val)),(660,22))
                self.assertEqual(int(sum(y_fit)),330)
                calls.append(1)
                network.set_weights([np.asarray([.3,.7])])
                return network,pd.DataFrame([dict(epoch=1)]),dict(best_epoch=1,stopping_epoch=1)
            common.prepare,common.fit_network,common.macro_metrics=prepare,fit,scores
            common.set_seed=lambda *a:None
            args=types.SimpleNamespace(local_dir=str(local),mirror_dir=str(mirror),original_dir=str(original),
                reference_dir=str(reference),config=str(ROOT/'configs/v7_unseen_sentences.json'),
                trt_dir='synthetic',labels_csv='synthetic',prepare_only=True,verify_only=False,storage_checked=False)
            with patch.dict(sys.modules,{'followup_common':common}), patch.object(recovery,'validate_storage_paths',return_value=(local,mirror,original)),contextlib.redirect_stdout(io.StringIO()):
                recovery.run(args)
                self.assertEqual(calls,[])
                self.assertEqual(len(list((local/'jobs').glob('*.zip'))),10)
                args.prepare_only=False
                with self.assertRaisesRegex(ValueError,'Storage gate'):recovery.run(args)
                self.assertEqual(calls,[])
                args.storage_checked=True
                recovery.run(args)
                self.assertEqual(len(calls),38)
                self.assertEqual(len(list((local/'jobs').glob('*.zip'))),48)
                args.verify_only=True
                recovery.run(args)
                self.assertEqual(len(calls),38)
                frame=pd.read_csv(local/'fold_metrics.csv')
                self.assertEqual(len(frame),96)
                self.assertEqual(frame.artifact_origin.value_counts().to_dict(),{'regenerated':76,'original_verified':20})
                # Local losses restore from a verified mirror, never refit or trust summary rows.
                path=local/'jobs'/(recovery.stem(keys[-1])+'.zip')
                path.unlink()
                recovery.run(args)
                self.assertTrue(path.exists());self.assertEqual(len(calls),38)
                self.assertEqual(original_hashes,{str(p.relative_to(original)):common.sha256(str(p)) for p in original.rglob('*') if p.is_file()})
                provenance=json.loads((local/'recovery_provenance.json').read_text())
                with zipfile.ZipFile(path) as z:payloads={n:z.read(n) for n in z.namelist()}
                payloads['history.csv']=b'epoch\n2\n'
                bad=root/'bad.zip'
                with zipfile.ZipFile(bad,'w') as z:
                    for n,payload in payloads.items():z.writestr(n,payload)
                with self.assertRaises(ValueError):recovery.verify_archive(bad,keys[-1],codes,plan,config,provenance)
                # Every code cell compiles, no training is executed from the notebook in tests.
                nb=json.loads((ROOT/'notebooks/07b_v7_artifact_recovery.ipynb').read_text())
                for cell in nb['cells']:
                    if cell['cell_type']=='code':compile(''.join(cell['source']),'recovery notebook','exec')


if __name__=='__main__':unittest.main()
