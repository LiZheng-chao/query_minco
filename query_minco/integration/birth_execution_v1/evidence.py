"""New-chain predecessor includes native truth-failure-to-Plant-stop qualification."""
import json
from pathlib import Path
from query_minco.evaluation.mirror_holdout_v1 import sha
from query_minco.integration.yopo_execution_v4.evidence import CASES as OLD_CASES,expected_fault as old_expected,predecessor as old_predecessor,legacy_window
CASES=OLD_CASES+['truth_failure']
def expected_fault(case,reason):
 return str(reason) in ['TRUTH_COLLISION_UNKNOWN_OR_STALE',"REQUEST ValueError('BIRTH_TRUTH_INVALID')"] if case=='truth_failure' else old_expected(case,reason)
def predecessor(path,stage,binding):
 if stage!='C_CAMPAIGN':return old_predecessor(path,stage,binding)
 d=json.loads(Path(path).read_text())
 if d.get('stage')!=stage or d.get('status')!='PASS' or d.get('source_binding')!=binding or [r['case'] for r in d.get('episodes',[])]!=CASES:raise PermissionError('NEW_SEVEN_CASE_BINDING_REQUIRED')
 for row in d['episodes']:
  p=Path(row['result_path']);e=json.loads(p.read_text())
  if sha(p)!=row['result_sha256'] or row['status']!='PASS' or e.get('status')!='PASS' or not e.get('physical_feedback') or e.get('source_binding')!=binding:raise PermissionError('FAULT_EPISODE_IDENTITY')
 return d
