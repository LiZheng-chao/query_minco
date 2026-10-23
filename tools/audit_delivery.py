"""Read-only verification of this delivery, never an experiment launcher."""
import json,hashlib,sys
from pathlib import Path
N=Path(__file__).resolve().parents[1];O=N/'research_outputs/migration_v1'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 seal=json.loads((O/'DELIVERY_SEAL.json').read_text());errors=[]
 for rel,z in seal['files'].items():
  p=O/rel
  if not p.is_file() or sha(p)!=z['sha256']:errors.append('OUTPUT_SHA '+rel)
 for rel,h in seal['source_sha256'].items():
  if not (N/rel).is_file() or sha(N/rel)!=h:errors.append('SOURCE_SHA '+rel)
 for name in ['comparisons/inference001/RESULT.json','comparisons/training001/RESULT.json',
 'comparisons/six_terms001/RESULT.json','package_fixture004/RESULT.json','audit/HISTORY_AFTER.json']:
  if not json.loads((O/name).read_text())['passed']:errors.append('CONTENT '+name)
 gates=json.loads((O/'GATE_STATUS.json').read_text())
 result=dict(delivery_integrity_passed=not errors,gates=gates,errors=errors,
 full_current_history_rescan_performed=False,formal_training_performed=False,remote_push_performed=False)
 print(json.dumps(result,indent=2))
 if errors:raise SystemExit(1)
if __name__=='__main__':main()
