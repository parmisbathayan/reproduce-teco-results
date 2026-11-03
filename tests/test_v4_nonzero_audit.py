"""Variable-length filtering and saved-result integrity; no TensorFlow or training."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import v4_independent_audit as a


class NonzeroTests(unittest.TestCase):
    def setUp(self):
        self.c=json.loads((ROOT/'configs/v4_independent_nonzero.json').read_text())
        self.codes=[c for _,c in a.READERS]
        self.ids=np.arange(1,111)
        self.y=np.arange(110)%2
        self.X=np.arange(1,1321).reshape(12,110,1,1).astype(np.float32)
        # Different exclusions, including first/middle/last native IDs.
        for p in range(12):self.X[p,[0,p+1,109]]=0
        self.targets=a.targets_for(self.y,self.c)
        self.jobs=a.jobs_for(self.codes,self.c)
        self.counts,self.excluded=a.nonzero_jobs(self.X,self.y,self.codes,self.ids,self.targets,self.jobs)

    def test_masks_class_counts_and_no_cross_reader_deletion(self):
        a.validate_config(self.c)
        self.assertEqual(len(self.excluded),36)
        self.assertEqual(len(self.counts),48)
        for j in self.jobs:
            xf,yf,xv,yv,xt,yt=a.inputs_for(self.X,self.codes,self.targets[j['condition']],j)
            self.assertEqual((len(xf),len(xv),len(xt)),(1070,107,107))
            self.assertTrue(np.all(xf!=0) and np.all(xv!=0) and np.all(xt!=0))
            p=self.codes.index(j['test_reader'])
            keep=np.any(self.X[p]!=0,axis=(1,2))
            np.testing.assert_array_equal(yt,self.targets[j['condition']][keep])
            np.testing.assert_array_equal(np.asarray(j['test_indices']),np.flatnonzero(keep))
        bad=self.X.copy();bad[0,self.y==1]=0
        with self.assertRaises(ValueError):
            a.nonzero_jobs(bad,self.y,self.codes,self.ids,self.targets,a.jobs_for(self.codes,self.c))

    def test_all72_variable_length_archives_and_summary_denominators(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            for j in self.jobs:
                target=self.targets[j['condition']][j['test_indices']]
                prob=np.eye(2)[target]
                sc=a.score(target,prob)
                history=[{'epoch':e,'val_macro_f1':1.} for e in range(1,12)]
                metrics={'job':j,'n_fit':j['n_fit'],'n_validation':j['n_validation'],**sc,
                    'stopping_epoch':11,'best_epoch':11,'best_validation_macro_f1':1.}
                rows=[{'condition':j['condition'],'seed':j['seed'],'participant':j['test_reader'],
                    'sentence_id':int(self.ids[k]),'true_label':int(y),'predicted_label':int(y),
                    'probability_negative':float(p[0]),'probability_positive':float(p[1])}
                    for k,y,p in zip(j['test_indices'],target,prob)]
                a.save_job(root/'jobs',j,rows,history,metrics,self.ids,self.targets,'fixed')
            done=a.collect(root,self.jobs,self.ids,self.targets,'fixed',self.c,require_complete=True)
            self.assertEqual(len(done),72)
            summary=a.csv_rows((root/'summary_overall.csv').read_bytes())
            for row in summary:
                self.assertEqual(int(row['predictions']),3852)
                self.assertEqual(float(row['accuracy']),1.)
                self.assertEqual(float(row['mean_reader_seed_macro_f1']),1.)
            # Retained IDs must remain reader-specific; changing the expected mask must fail.
            job={**self.jobs[0],'test_indices':list(range(107))}
            with self.assertRaises(ValueError):
                a.verify_job(root/'jobs'/(a.stem(job)+'.zip'),job,self.ids,self.targets,'fixed')

    def test_notebook_config_and_syntax(self):
        n=json.loads((ROOT/'notebooks/04e_v4_nonzero_audit.ipynb').read_text())
        for cell in n['cells']:
            if cell['cell_type']=='code':compile(''.join(cell['source']),'<cell>','exec')
        setup=''.join(n['cells'][2]['source'])
        self.assertIn('v4-nonzero-audit-v1',setup)
        self.assertIn('configs/v4_independent_nonzero.json',setup)
        self.assertIn('/v4_independent_nonzero_11_23_37',setup)
        q=a.feature_quality(self.X,self.codes)
        self.assertEqual(q[0]['all_zero_trials'],3)


if __name__=='__main__':unittest.main()
