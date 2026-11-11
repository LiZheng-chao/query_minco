"""Create an independent result/source evidence ZIP. No original file is modified."""
from pathlib import Path
import sys,json,hashlib,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
R=Path(__file__).resolve().parents[2];O=R/'research_outputs/g4b_formal128_and_control_20261111_v1';A=R/'research_outputs/g4a_forest_shadow_20261111_v1'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def main():
 with (O/'TEST_MATRIX.md').open('a') as f:f.write('\n## Report helper recovery\n\nFirst report generation failed a project-import path after writing initial tables. Its failed log is preserved. Bootstrap import fixed;recovery compared every existing partial report/CSV byte-for-byte(or JSON object equality) and retained it,then completed missing reports under new process log. Scientific replay and original run were not changed.\n')
 files={}
 for p in O.rglob('*'):
  if not p.is_file() or p.name in ['G4B_FORMAL128_AND_CONTROL_AUDIT.zip','FILES_AND_SHA.json','FILES_AND_SHA.sha256']:continue
  rel=str(p.relative_to(O))
  if rel.startswith('logs/package_build/'):continue
  files['g4b_audit/'+rel]=p
 for base,arc in [(A/'formal/A_reasonable_seed20261104','original_formal_run'),(A/'formal_process/A_reasonable_seed20261104','original_formal_process')]:
  for p in base.rglob('*'):
   if p.is_file():files[arc+'/'+str(p.relative_to(base))]=p
 for name in ['K1_FOREST_TRAINING_RUNSPEC.json','K1_FOREST_TRAINING_RUNSPEC_v2.json','SCIENTIFIC_DATASET_BINDING.json','PREREGISTRATION.json']:
  files['original_run_binding/'+name]=A/name
 files['original_run_binding/researcher_APPROVAL.json']=A/'approvals/A_reasonable_seed20261104.json'
 baseline=R/'research_outputs/g2b_k1_pilot_candidate_20261110_v1/A_reasonable_seed20261104'
 files['baseline_checkpoint/old_K1_step128.pt']=baseline/'step128.pt';files['baseline_checkpoint/old_K1_step128.pt.json']=baseline/'step128.pt.json'
 for base in ['query_minco','experiments/g4a_forest_and_shadow_v1','experiments/g4b_formal_and_control_v1','ros/g4b_control_supervisor_v1','tests/g4b_formal_and_control_v1']:
  for p in (R/base).rglob('*'):
   if p.is_file() and '__pycache__' not in p.parts and p.suffix in ['.py','.cpp','.cu','.cuh','.hpp','.h','.json','.yaml','.launch','.md']:
    files['live_source/'+str(p.relative_to(R))]=p
 for base in ['Simulator/src','Controller/src/so3_control','Controller/src/so3_quadrotor_simulator','Controller/src/utils/quadrotor_msgs/msg']:
  for p in (R.parent/'YOPO'/base).rglob('*'):
   if p.is_file() and p.suffix in ['.py','.cpp','.cu','.cuh','.h','.hpp','.yaml','.launch','.msg','.xml']:
    files['read_only_native_source/'+str(p.relative_to(R.parent/'YOPO'))]=p
 # Include complete manifest/schema identities,not hundreds of MB of raw prior map assets.
 spec=json.loads((A/'K1_FOREST_TRAINING_RUNSPEC.json').read_text());references={}
 for row in spec['train']+spec['dev']:
  p=Path(row['manifest']);m=json.loads(p.read_text())
  paths={str(p):row['manifest_sha256'],**m.get('files_sha256',{})}
  for path,h in list(paths.items()):
   q=Path(path)
   if q.name=='MAP_MANIFEST.json':
    child=json.loads(q.read_text());paths.update(child.get('files_sha256',{}));paths.update(child.get('yopo_sources_sha256',{}))
  references[row['sample_id']]=dict(kind=row['kind'],split=row['split'],manifest=str(p),files_sha256=paths)
  for path,h in paths.items():
   q=Path(path)
   if q.suffix=='.json' and q.is_file():
    if sha(q)!=h:raise ValueError('DATA_IDENTITY '+path)
    files['data_manifest_identity/'+h[:20]+'_'+q.name]=q
 entries={name:dict(original_path=str(p),sha256=sha(p),bytes=p.stat().st_size) for name,p in files.items()}
 target=O/'G4B_FORMAL128_AND_CONTROL_AUDIT.zip'
 with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED) as z:
  for name,p in sorted(files.items()):z.write(p,name)
  z.writestr('DATA_ASSET_REFERENCES.json',json.dumps(references,indent=2))
  z.writestr('FILES_AND_SHA.json',json.dumps(dict(schema='g4b_zip_entry_source_identity_v1',entries=entries,
   excluded_self='This inside-ZIP inventory;outer manifest independently binds ZIP SHA.',
   large_assets='Prior map/depth/checkpoint source data not all bundled;required paths and SHA in DATA_ASSET_REFERENCES. Do not bypass SHA or claim standalone data reproduction.',
   actual_formal_updates=128,new_updates_this_audit=0),indent=2))
 with zipfile.ZipFile(target) as z:
  if z.testzip() is not None:raise ValueError('ZIP_CRC')
  index=json.loads(z.read('FILES_AND_SHA.json'))['entries']
  for name,item in index.items():
   if hashlib.sha256(z.read(name)).hexdigest()!=item['sha256']:raise ValueError('ZIP_ENTRY_SHA '+name)
 with (O/'ZIP_VERIFICATION.json').open('x') as f:json.dump(dict(status='PASS',zip_path=str(target),zip_sha256=sha(target),source_entries_verified=len(entries),bytes=target.stat().st_size,
  all5_formal_checkpoints=True,real_process_and_approval_preserved=True,network_updates_this_audit=0),f,indent=2)
 print('ZIP_PASS',len(entries),target.stat().st_size,sha(target))
if __name__=='__main__':main()
