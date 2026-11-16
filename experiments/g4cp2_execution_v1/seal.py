from pathlib import Path
import sys,json,ast,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from query_minco.evaluation.mirror_holdout_v1 import sha,write
R=Path.cwd();O=R/'research_outputs/g4cp2_execution_20261111_v1'
entry=json.loads((O/'ENTRY.json').read_text());binding=json.loads((O/'SOURCE_BINDING.json').read_text())
assert all(sha(p)==h for p,h in binding.items())
assert all(sha(p)==h for p,h in entry['source_before'].items())
assert json.loads((O/'HISTORICAL_PROTECTION.json').read_text())['status']=='PASS'
assert json.loads((O/'FINAL_INTERFACE_CHECKS.json').read_text())['status']=='PASS'
for p in (O/'logs').glob('*/PROCESS_START.json'):assert (p.parent/'PROCESS_RESULT.json').exists()
new=[]
for base in ['query_minco','experiments','tests','docs','tools','examples','ros']:
 for p in (R/base).rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts and str(p) not in entry['source_before']:
   assert any(str(p.relative_to(R)).startswith(a) for a in entry['allowed_new']);new.append(p)
write(O/'FILES_AND_SHA.json',dict(new_sources={str(p):sha(p) for p in new},output_sha256={str(p.relative_to(O)):sha(p) for p in O.rglob('*') if p.is_file()},
 source_binding_sha256=sha(O/'SOURCE_BINDING.json'),protected_history_index_sha256=sha(O/'ENTRY.json'),new_network_updates=0,actual_physics_executions_this_turn=0,
 exclusions=['FILES_AND_SHA.json(self)','G4C_P2_AUDIT.zip(afterwards)','ARCHIVE_SHA256.txt(afterwards)']))
with zipfile.ZipFile(O/'G4C_P2_AUDIT.zip','x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in new:z.write(p,'new_source/'+str(p.relative_to(R)))
 for p in O.rglob('*'):
  if p.is_file() and p.name!='G4C_P2_AUDIT.zip':z.write(p,'evidence/'+str(p.relative_to(O)))
 for name in ['g4cp1_manual_held_20261111_164653_1005','g4cp1_manual_hover_retry_20261111_165004_1005']:
  root=R/'research_outputs'/name
  for pattern in ['ACCEPTANCE_*.json','preparation/EVENTS.jsonl','prepare/stderr.log','*/PROCESS_RESULT.json']:
   for p in root.glob(pattern):z.write(p,'historical_readonly_evidence/'+name+'/'+str(p.relative_to(root)))
with (O/'ARCHIVE_SHA256.txt').open('x') as f:f.write(sha(O/'G4C_P2_AUDIT.zip')+'  G4C_P2_AUDIT.zip\n')
with zipfile.ZipFile(O/'G4C_P2_AUDIT.zip') as z:assert z.testzip() is None
print('P2_SEALED',len(new),'SOURCE_FILES',(O/'G4C_P2_AUDIT.zip').stat().st_size)
