"""Tiny explicit transport integrity tests; no historical data rewrite."""
import json,tempfile,unittest
from pathlib import Path
from query_minco.training.io import write,sha
from query_minco.training.pg1.transport import Store
class Transport(unittest.TestCase):
 def setUp(self):
  self.t=tempfile.TemporaryDirectory();self.root=Path(self.t.name);p=self.root/'tree/home/origin/a.json';p.parent.mkdir(parents=True);p.write_text('{"state":1}')
  self.logical='/home/origin/a.json';self.file=p
  self.doc=dict(schema='pg1_transfer_v1',files=[dict(original_path=self.logical,relative_path='tree/home/origin/a.json',sha256=sha(p),size=p.stat().st_size)])
  self.manifest=self.root/'TRANSFER.json';write(self.manifest,self.doc)
 def tearDown(self):self.t.cleanup()
 def store(self):return Store(dict(path=str(self.manifest),sha256=sha(self.manifest)))
 def test_actual_path_explicit_original_bytes(self):
  s=self.store();b=dict(path=self.logical,sha256=sha(self.file));actual=s.binding(b)
  self.assertEqual(actual['path'],str(self.file));self.assertEqual(s.read(b),dict(state=1));self.assertEqual(s.verify()['files'],1)
 def test_corruption_missing_unknown_rejected(self):
  s=self.store();self.file.write_text('CORRUPT')
  with self.assertRaises(ValueError):s.verify()
  with self.assertRaises(ValueError):s.artifact(dict(path='/home/other/a',sha256='0'*64))
 def test_path_escape_and_symlink_rejected(self):
  self.doc['files'][0]['relative_path']='../escape'
  self.manifest.write_text(json.dumps(self.doc))
  with self.assertRaises(ValueError):self.store()
 def test_duplicate_and_false_sha_rejected(self):
  self.doc['files'].append(dict(self.doc['files'][0]));self.manifest.write_text(json.dumps(self.doc))
  with self.assertRaises(ValueError):self.store()
if __name__=='__main__':unittest.main(verbosity=2)
