from pathlib import Path
import sys,json,hashlib,subprocess,time
R=Path.cwd();O=R/'research_outputs/g4cp3_observed_20261111_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
e=json.loads((O/'ENTRY.json').read_text());t=time.monotonic()
def changes(d):return [p for p,h in d.items() if not Path(p).is_file() or sha(p)!=h]
r=dict(source_changes=changes(e['source_before']),historical_changes=changes(e['history_before']),git_changes=[],repositories_after={})
for repo,records in e['repositories_before'].items():
 r['repositories_after'][repo]={}
 for key,before in records.items():
  p=subprocess.run(before['argv'],capture_output=True,text=True);r['repositories_after'][repo][key]=dict(exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr)
  if p.returncode!=before['exit_code'] or(p.stdout!=before['stdout'] and not(repo==str(R) and key=='status')):r['git_changes'].append([repo,key])
prior=json.loads((R/'research_outputs/g2a_k1_trainer_qualification_20261110_v1/ENTRY_AND_WHITELIST.json').read_text())
r['outside_source_changes']=changes(prior['outside_source_sha_before'])
r['outside_metadata_changes']=[p for p,v in prior['outside_metadata_before'].items() if not Path(p).is_file() or Path(p).stat().st_size!=v['size'] or Path(p).stat().st_mtime_ns!=v['mtime_ns']]
new=[]
for base in ['query_minco','experiments','tests','docs','tools','examples','ros']:
 for p in (R/base).rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts and str(p) not in e['source_before']:new.append(str(p.relative_to(R)))
r['new_sources']=new;r['unexpected_new_sources']=[n for n in new if not any(n.startswith(a) for a in e['allowed_new'])]
r['status']='PASS' if not any(r[k] for k in ['source_changes','historical_changes','git_changes','outside_source_changes','outside_metadata_changes','unexpected_new_sources']) else 'FAIL'
r['counts']=dict(protected_sources=len(e['source_before']),protected_output_files=len(e['history_before']),outside_sources=len(prior['outside_source_sha_before']),outside_metadata=len(prior['outside_metadata_before']))
r['elapsed_s']=time.monotonic()-t
with (O/'HISTORICAL_PROTECTION.json').open('x') as f:json.dump(r,f,indent=2)
print('HISTORICAL_PROTECTION',r['status'],r['counts']);assert r['status']=='PASS'
