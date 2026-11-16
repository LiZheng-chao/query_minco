"""Last read-only identity check and exclusive audit archive."""
from pathlib import Path
import json,hashlib,subprocess,sys,zipfile,ast
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from query_minco.evaluation.mirror_holdout_v1 import sha,write
R=Path.cwd();O=R/'research_outputs/g4c_startup_and_dynamics_20261111_v1'
entry=json.loads((O/'ENTRY_PROTECTION.json').read_text());bound=json.loads((O/'MANUAL_STARTUP_SOURCE_BINDING.json').read_text())
assert not [p for p,h in entry['source_before'].items() if sha(p)!=h]
assert not [p for p,h in bound.items() if sha(p)!=h]
assert subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True).strip()=='4a2d962f599550d0e48f1b9dbfb127565d2c1594'
new=[]
for base in ['query_minco','experiments','tests','docs','tools','examples','ros']:
 for p in (R/base).rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts and str(p) not in entry['source_before']:new.append(str(p.relative_to(R)))
assert all(any(n.startswith(a) if a.endswith('/') else n==a for a in entry['allowed_new']) for n in new)
for n in new:
 p=R/n
 if p.suffix=='.py':ast.parse(p.read_text(),filename=n)
assert json.loads((O/'HISTORICAL_PROTECTION.json').read_text())['status']=='PASS'
for p in (O/'logs').glob('*/PROCESS_START.json'):
 assert (p.parent/'PROCESS_RESULT.json').exists(),'UNFINISHED_PROCESS '+str(p.parent)
ledger=json.loads((O/'engineering_peak_probe/ATOMIC_LEDGER.json').read_text())
print('ENGINEERING_LEDGER',ledger)
assert ledger['attempted']==2 and ledger['completed']==2 and not ledger['pending']
identity=json.loads((O/'ASSEMBLY_SOURCE_IDENTITY.json').read_text())
identity['output_sha256']={str(p.relative_to(O)):sha(p) for p in O.rglob('*') if p.is_file()}
identity['new_source_sha256']={str(R/n):sha(R/n) for n in new}
identity['manifest_exclusions']=['FILES_AND_SHA.json(self)','G4C_AUDIT.zip(created afterwards)','ARCHIVE_SHA256.txt(created afterwards)']
write(O/'FILES_AND_SHA.json',identity)
with zipfile.ZipFile(O/'G4C_AUDIT.zip','x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for n in new:z.write(R/n,'current_new_source/'+n)
 for p in O.rglob('*'):
  if p.is_file() and p.name not in ['G4C_AUDIT.zip','ARCHIVE_SHA256.txt']:z.write(p,'evidence/'+str(p.relative_to(O)))
with (O/'ARCHIVE_SHA256.txt').open('x') as f:f.write(sha(O/'G4C_AUDIT.zip')+'  G4C_AUDIT.zip\n')
print('DELIVERY_SEALED',len(new),'NEW_SOURCE_FILES',sum(1 for p in O.rglob('*') if p.is_file()),'FILES','ZIP_BYTES',(O/'G4C_AUDIT.zip').stat().st_size)
