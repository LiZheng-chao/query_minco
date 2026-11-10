from pathlib import Path
import sys,json,hashlib,subprocess,zipfile,difflib,platform
import torch,numpy,scipy
R=Path(__file__).resolve().parents[2];O=R/'research_outputs/g2b_k1_visual_audit_20261110_v1';G2=R/'research_outputs/g2a_k1_trainer_qualification_20261110_v1';G1=R/'research_outputs/g1_k1_current_baseline_20261110_v2';RUN=R/'research_outputs/g2b_k1_pilot_candidate_20261110_v1/A_reasonable_seed20261104'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def write(n,d):
 with (O/n).open('x') as f:json.dump(d,f,indent=2)
assert read(O/'HISTORICAL_PROTECTION.json')['status']=='PASS'
# Correct display-only reconstructed argv typo; never touch old evidence.
q=O/'RUN_COMMAND_AND_PROCESS_GAPS.md';q.write_text(q.read_text().replace('--seed20261104','--seed 20261104'))
spec=read(G2/'K1_TRAINING_RUNSPEC_DRAFT.json');newg2=read(G2/'ENTRY_AND_WHITELIST.json')
changes=dict(g1_pre_g2_sources_unchanged=all(sha(k)==v for k,v in newg2['source_before'].items()),g2_new_files={k:sha(R/k) for k in newg2['allowed_new']},g1_manifest_path=str(G1/'FILES_AND_SHA.json'),g1_manifest_sha256=sha(G1/'FILES_AND_SHA.json'),limits='G1_TO_G2: G2 added these8 files; no G1 prior source content edited. SOURCE_DIFF.patch is current tracked diff against HEAD, not a fabricated G0/G1 commit diff.')
write('G1_TO_G2_SOURCE_CHANGES.json',changes)
env=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),torch=torch.__version__,numpy=numpy.__version__,scipy=scipy.__version__,device='cpu',audit_network_updates=0)
write('ENVIRONMENT.json',env)
git=[]
for args in [['rev-parse','HEAD'],['status','--short'],['diff','HEAD','--'],['ls-files','--others','--exclude-standard']]:
 s=subprocess.run(['git','--no-optional-locks','-C',str(R)]+args,capture_output=True,text=True);git.append(dict(argv=s.args,exit=s.returncode,stdout=s.stdout,stderr=s.stderr))
write('GIT_IDENTITY.json',git)
(O/'SOURCE_DIFF.patch').write_text(git[2]['stdout'])
allowed={'.py','.cpp','.cu','.cuh','.c','.hpp','.h','.json','.yaml','.yml','.md','.toml','.txt','.cfg','.ini','.cmake'}
sources={}
for base in ['query_minco','experiments','tests','docs','tools','examples']:
 for p in (R/base).rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts and '.git' not in p.parts and p.suffix.lower() in allowed:sources[str(p.relative_to(R))]=p
for name in ['README.md','pyproject.toml','requirements.txt','environment.yml','setup.py','setup.cfg','pytest.ini']:
 p=R/name
 if p.is_file():sources[name]=p
source_hash={k:sha(p) for k,p in sorted(sources.items())}
write('CURRENT_SOURCE_SHA.json',source_hash)
untracked={k:source_hash[k] for k in sources if k in git[3]['stdout'].splitlines()}
write('UNTRACKED_SOURCE_SHA.json',untracked)
with (O/'UNTRACKED_SOURCE.patch').open('x') as f:
 for name in sorted(untracked):
  try:text=sources[name].read_text()
  except UnicodeDecodeError:continue
  f.writelines(difflib.unified_diff([],text.splitlines(True),fromfile='/dev/null',tofile='b/'+name))
# References preserve full origin and bytes for every source-bound data file, with no data/checkpoint duplication.
refs={};metadata={}
def ref(path,expected=None,role='bound_dependency'):
 p=Path(path);k=str(p)
 if k not in refs:
  actual=sha(p);refs[k]=dict(path=k,sha256=actual,bytes=p.stat().st_size,role=role,included=False)
 if expected is not None and refs[k]['sha256']!=expected:raise ValueError('REFERENCE_SHA_DRIFT '+k)
 if p.suffix.lower() in allowed and p.stat().st_size<=1048576:
  metadata[k]=p
 return refs[k]
for row in spec['train']+spec['dev']:
 ref(row['manifest'],row['manifest_sha256'],'qualified_sample_manifest')
 m=read(row['manifest'])
 for path,h in m['files_sha256'].items():ref(path,h)
 for k in ['previous','original_p0c_manifest']:
  if k in row:ref(row[k],row[k+'_sha256'] if k+'_sha256' in row else row['previous_sha256'] if k=='previous' else row['original_p0c_sha256'],'historical_manifest_parent')
for mode,records in spec['initial_states'].items():
 for seed,v in records.items():ref(v['path'],v['sha256'],'frozen_initial_state_'+mode+'_'+seed)
for p in RUN.glob('*.pt'):ref(p,read(str(p)+'.json')['file_sha256'],'existing_formal_checkpoint')
write('EXTERNAL_ARTIFACT_REFERENCES.json',refs)
status=dict(FULL_SOURCE_AUDIT='PASS',FORMAL_128STEP_PROVENANCE='UNVERIFIED',VISUAL_CAUSALITY='INCONCLUSIVE',VISUAL_INPUT_DEPENDENCE='SUPPORTED',FIXED_REFERENCE_BASELINE='EVALUATED',MIRROR_HOLDOUT='NOT_QUALIFIED',K1_HARD_API_COMPATIBILITY='FAIL',NEW_NETWORK_UPDATES=0,READY_FOR_NEXT_TRAINING_ABLATION='NO',HISTORICAL_PROTECTION='PASS',LEGACY_C1_REGISTRY='BLOCKED')
write('FINAL_STATUS.json',status)
matrix=['# Actual test matrix','All actual argv, stdout/stderr, exits and RSS are in logs/*.json; do not confuse historical training with this zero-update audit.','',
'|execution|exit|seconds|peak RSS bytes|result|','|---|---:|---:|---:|---|']
for p in sorted((O/'logs').glob('*.json')):
 d=read(p);matrix.append('|%s|%s|%.4f|%s|%s|'%(p.name,d['exit'],d['elapsed_s'],d['sampled_process_tree_peak_rss_bytes'],'PASS' if d['exit']==0 else 'SETUP_FAILURE_RETAINED'))
matrix+=['','Main attempt02:105 exact task replays;360 same-state paired interventions;105 reference/model evaluations;78 OOD outputs; original K1 helper failure reproduced;6 K6 helper/direct matches;6 illegal duration rejections.',
'Restore:5 actual checkpoints, model/optimizer/cursor and next RNG draws exact, all128 logged dead hashes checked; no step.',
'Continuous extrema:360 polynomial-difference root computations; mirror inventory6 scenes. Mirror generation/feasibility/inference NOT_RUN.',
'First main setup failed ModuleNotFoundError; first continuous/inventory setup failed SyntaxError. Corrected only new audit scripts; both failed original logs retained.',
'Repair implementation/new API acceptance NOT_RUN. New training/optimizer updates/other seed/ROS/controller NOT_RUN. Prior G2-A181 suite is included as prior evidence, not re-executed.',
'Formal process argv/stdout/stderr/exit unavailable, not a newly executed test. Full provenance UNVERIFIED.']
(O/'TEST_MATRIX.md').write_text('\n'.join(matrix)+'\n')
(O/'README.md').write_text("""# G2-B audit handoff
Current source reviewed and frozen, no network updates. See FINAL_STATUS.json.
Read FULL_SOURCE_AUDIT.md -> FORMAL_128STEP_AUDIT.md -> VISUAL_CAUSALITY_REPORT.md -> FIXED_REFERENCE_BASELINES.md -> MIRROR_HOLDOUT_PLAN.md -> HARD_API_REPAIR_PROPOSAL.md.
Artifact consistency PASS and exact105-task replay PASS; formal runtime argv/process logs absent: provenance UNVERIFIED.
Input dependence supported; useful visual avoidance inconclusive. Original fixed reference20/21 vs learned21/21;90 final swaps alter paths but no Hard outcomes.
Auxiliary Hard K1 API FAIL, current production direct path PASS. Repair proposal only, original SHA preserved.
No independent mirror holdout qualified; no new training ablation readiness.
ZIP includes live source, tracked/untracked patches, all required G1/G2 manifests/reports/test logs, formal128 step/evaluation logs and sidecars, original approval, exact current audit raw arrays/logs and small data/schema metadata.
Large data/initial NPZ/formal PT files excluded; EXTERNAL_ARTIFACT_REFERENCES.json binds exact paths, bytes, source and SHA.
Archive is source/evidence handoff, not a self-contained large-data execution image. Absolute manifest paths are intentionally preserved; rehosting must explicitly remap and requalify, never silently skip SHA checks.
""")
entries={}
def add(path,name):
 check_name=Path(name)
 if check_name.is_absolute() or '..' in check_name.parts:raise ValueError('ZIP_PATH')
 if name in entries and entries[name]!=Path(path):raise ValueError('ZIP_DUPLICATE')
 entries[name]=Path(path)
for name,p in sources.items():add(p,'source/'+name)
for p in RUN.iterdir():
 if p.is_file() and p.suffix!='.pt':add(p,'evidence/formal_run/'+p.name)
for directory,label in [(G1,'G1'),(G2,'G2A')]:
 for p in directory.glob('*'):
  if p.is_file() and p.suffix in ['.md','.json'] and p.name not in ['ENTRY_AND_WHITELIST.json','HISTORY_BEFORE.json']:
   add(p,'evidence/'+label+'/'+p.name)
 for p in (directory/'logs').glob('*'):
  if p.is_file() and p.suffix in ['.json','.log']:add(p,'evidence/'+label+'/logs/'+p.name)
for p in O.rglob('*'):
 if p.is_file() and O/'audit_package' not in p.parents and 'tmp' not in p.parts:
  add(p,'audit/'+str(p.relative_to(O)))
for i,(name,p) in enumerate(sorted(metadata.items())):
 member='metadata/bound/'+str(i).zfill(4)+'_'+p.name;add(p,member);refs[name]['included']=True;refs[name]['zip_member']=member
# Update existing new reference record before archive hash enumeration.
(O/'EXTERNAL_ARTIFACT_REFERENCES.json').write_text(json.dumps(refs,indent=2))
attachment=list(Path('/mnt/c/Users/Lizheng/Downloads').glob('G2_K1_*20261110.md'))
if len(attachment)==1:add(attachment[0],'evidence/researcher_local_audit.md')
manifest={name:dict(sha256=sha(path),bytes=path.stat().st_size,origin=str(path)) for name,path in sorted(entries.items())}
write('FILES_AND_SHA.json',dict(schema='g2b_audit_archive_manifest_v1',source_sha256=source_hash,archive_members=manifest,formal_run=str(RUN),omissions='Large PT/NPZ/data referenced by exact SHA; unavailable formal process logs explicitly UNVERIFIED',new_network_updates=0))
add(O/'FILES_AND_SHA.json','FILES_AND_SHA.json')
target=O/'audit_package/CURRENT_K1_G1_G2_SOURCE_AND_EVIDENCE_20261110_v1.zip'
with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for name,path in sorted(entries.items()):z.write(path,name)
with zipfile.ZipFile(target) as z:
 assert z.testzip() is None
 assert set(z.namelist())==set(entries)
 for name,v in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==v['sha256'],name
 package_info=dict(status='PASS',zip_path=str(target),zip_sha256=sha(target),zip_bytes=target.stat().st_size,members=len(entries),source_files=len(sources),untracked_source_files=len(untracked),all_member_sha_verified=True,core_source_changed=False,new_network_updates=0)
write('PACKAGE_VERIFICATION.json',package_info)
print(json.dumps(package_info,indent=2))
