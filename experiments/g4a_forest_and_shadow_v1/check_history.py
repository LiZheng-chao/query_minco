from pathlib import Path
import sys,json,hashlib,subprocess,time
R=Path(__file__).resolve().parents[2];O=R/'research_outputs/g4a_forest_shadow_20261111_v1'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
e=json.loads((O/'ENTRY_PROTECTION.json').read_text())
def check(m):return [p for p,h in m.items() if not Path(p).is_file() or sha(p)!=h]
t=time.monotonic()
r=dict(source_changes=check(e['source_before']),historical_output_changes=check(e['history_before']),interrupted_migration_changes=check(e['interrupted_migration_sha']),git_identity_changes=[],repositories_after={})
for repo,commands in e['repositories_before'].items():
 r['repositories_after'][repo]={}
 for key,v in commands.items():
  p=subprocess.run(v['argv'],capture_output=True,text=True)
  r['repositories_after'][repo][key]=dict(exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr)
  if p.returncode!=v['exit_code'] or (p.stdout!=v['stdout'] and not(repo==str(R) and 'status' in key)):r['git_identity_changes'].append([repo,key])
prior=json.loads((R/'research_outputs/g2a_k1_trainer_qualification_20261110_v1/ENTRY_AND_WHITELIST.json').read_text())
r['outside_source_changes']=check(prior['outside_source_sha_before'])
r['outside_metadata_changes']=[p for p,v in prior['outside_metadata_before'].items() if not Path(p).is_file() or Path(p).stat().st_size!=v['size'] or Path(p).stat().st_mtime_ns!=v['mtime_ns']]
new=[]
for base in ['query_minco','experiments','tests','docs','tools','examples','ros']:
 for p in (R/base).rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts and str(p) not in e['source_before']:new.append(str(p.relative_to(R)))
def allowed(n):return any(n.startswith(a) if a.endswith('/') else n==a for a in e['allowed_new'])
r['unexpected_new_sources']=[p for p in new if not allowed(p)];r['new_sources']=new
r['counts']=dict(old_source=len(e['source_before']),old_outputs=len(e['history_before']),outside_source=len(prior['outside_source_sha_before']),outside_metadata=len(prior['outside_metadata_before']),interrupted_migration_files=len(e['interrupted_migration_sha']))
r['status']='PASS' if not any(r[k] for k in ['source_changes','historical_output_changes','interrupted_migration_changes','git_identity_changes','outside_source_changes','outside_metadata_changes','unexpected_new_sources']) else 'FAIL'
r['elapsed_s']=time.monotonic()-t;r['network_updates']=0
with (O/'HISTORICAL_PROTECTION.json').open('x') as f:json.dump(r,f,indent=2)
print(r['status'],r['counts'],r['elapsed_s'])
assert r['status']=='PASS'
