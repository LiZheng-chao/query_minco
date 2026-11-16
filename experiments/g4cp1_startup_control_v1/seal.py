from pathlib import Path
import sys,json,zipfile,ast,subprocess
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from query_minco.evaluation.mirror_holdout_v1 import sha,write
R=Path.cwd();O=R/'research_outputs/g4cp1_startup_control_20261111_v1'
e=json.loads((O/'ENTRY.json').read_text());binding=json.loads((O/'MANUAL_SOURCE_BINDING.json').read_text())
assert all(sha(p)==h for p,h in e['source_before'].items())
assert all(sha(p)==h for p,h in binding.items())
assert json.loads((O/'HISTORICAL_PROTECTION.json').read_text())['status']=='PASS'
for p in (O/'logs').glob('*/PROCESS_START.json'):assert (p.parent/'PROCESS_RESULT.json').exists()
new=[]
for base in ['query_minco','experiments','tests','docs','tools','examples','ros']:
 for p in (R/base).rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts and str(p) not in e['source_before']:
   name=str(p.relative_to(R));assert any(name.startswith(x) for x in e['allowed_new']);new.append(p)
   if p.suffix=='.py':ast.parse(p.read_text(),filename=str(p))
write(O/'FILES_AND_SHA.json',dict(head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),new_source_sha256={str(p):sha(p) for p in new},
 output_sha256={str(p.relative_to(O)):sha(p) for p in O.rglob('*') if p.is_file()},old_protection_index=str(O/'ENTRY.json'),old_protection_index_sha256=sha(O/'ENTRY.json'),
 exclusions=['FILES_AND_SHA.json(self)','G4C_P1_AUDIT.zip(afterwards)','ARCHIVE_SHA256.txt(afterwards)'],network_updates=0,physics_executions=0))
with zipfile.ZipFile(O/'G4C_P1_AUDIT.zip','x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in new:z.write(p,'new_source/'+str(p.relative_to(R)))
 for p in O.rglob('*'):
  if p.is_file() and p.name!='G4C_P1_AUDIT.zip':z.write(p,'evidence/'+str(p.relative_to(O)))
with (O/'ARCHIVE_SHA256.txt').open('x') as f:f.write(sha(O/'G4C_P1_AUDIT.zip')+'  G4C_P1_AUDIT.zip\n')
with zipfile.ZipFile(O/'G4C_P1_AUDIT.zip') as z:assert z.testzip() is None
print('P1_SEALED',len(new),'NEW_SOURCE_FILES',len(list(O.rglob('*'))),'ENTRIES')
