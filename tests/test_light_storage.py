import contextlib
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from light_storage import copy_file,restore_mirror,sync_mirror,validate_paths


class LightStorageTests(unittest.TestCase):
    def test_copy_restore_and_conflicts(self):
        common=types.ModuleType('followup_common')
        common.sha256=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as folder,patch.dict(sys.modules,{'followup_common':common}):
            root=Path(folder);local=root/'local';mirror=root/'mirror'
            (local/'jobs').mkdir(parents=True)
            (local/'jobs'/'one.zip').write_bytes(b'synthetic-result')
            (local/'summary.csv').write_text('first')
            sync_mirror(local,mirror)
            (local/'summary.csv').write_text('updated')
            sync_mirror(local,mirror)
            self.assertEqual((mirror/'summary.csv').read_text(),'updated')
            restored=root/'restored'
            restore_mirror(restored,mirror)
            self.assertEqual((restored/'jobs'/'one.zip').read_bytes(),b'synthetic-result')
            (local/'jobs'/'one.zip').write_bytes(b'different')
            with self.assertRaises(ValueError):sync_mirror(local,mirror)

    def test_protected_outputs_and_drive_primary_rejected(self):
        for local,mirror,old in [('/content/drive/x','/content/drive/y','/content/drive/z'),
                                  ('/content/x','/content/drive/old/sub','/content/drive/old')]:
            with self.assertRaises(ValueError):validate_paths(local,mirror,[old])
        validate_paths('/content/local','/content/drive/new',['/content/drive/old'])

    def test_both_notebooks_compile_and_have_no_automatic_training_or_download(self):
        for name in ('07c_v7_light_recovery.ipynb','08b_v8_light_results.ipynb'):
            notebook=json.loads((ROOT/'notebooks'/name).read_text())
            code='\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code')
            for cell in notebook['cells']:
                if cell['cell_type']=='code':compile(''.join(cell['source']),name,'exec')
            self.assertIn('DRIVE_PERSISTENCE_CHECKED = False',code)
            self.assertNotIn('files.download',code)
            self.assertNotIn('pip install',code)


if __name__=='__main__':unittest.main()
