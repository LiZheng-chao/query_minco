"""Exact reference identity; no implicit latest and no historical imports."""
import hashlib,json
from pathlib import Path
METHOD_ID="query_minco_b2c1_c1_e1_v1"
ROOT=Path(__file__).resolve().parents[1]
MANIFEST_SHA='5a43db66fcc26d4dfc24861a1bb457d30228c73a30329a157049e740c282f7d7'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()
def describe(method_id=METHOD_ID):
 if method_id!=METHOD_ID:raise ValueError("UNKNOWN_METHOD")
 p=ROOT/"query_minco/methods/reference.json"
 if sha(p)!=MANIFEST_SHA:raise ValueError("MANIFEST_SHA")
 m=json.loads(p.read_text())
 for n,h in m["portable_source_sha256"].items():
  if sha(ROOT/n)!=h:raise ValueError("SOURCE_SHA "+n)
 return m
def registration(method_id,checkpoint,expected_checkpoint_sha256=None):
 m=describe(method_id)
 binding=dict(checkpoint=str(Path(checkpoint).resolve()),sha256="2352de6dd1e4a839b78ec12c115b5319b34da3c61cada105dbdeb24a4faec6d1",weights_sha256="68570b63e15f21ae9e9d512393391c9a4f3cf8569dfc730cfbeb9cb8f436d63b",seed=42,arm="C1_orthogonal")
 if expected_checkpoint_sha256 is not None and expected_checkpoint_sha256!=binding["sha256"]:raise ValueError("CHECKPOINT_SHA")
 if sha(checkpoint)!=binding["sha256"]:raise ValueError("CHECKPOINT_SHA")
 return binding,m
if __name__=="__main__":print(json.dumps(describe(),indent=2))
