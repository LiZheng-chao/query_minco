from pathlib import Path
import json,hashlib,subprocess,time
r=Path(__file__).resolve().parents[2];p=r/'research_outputs/g2b_k1_visual_audit_20261110_v1'
def sha(x):
 h=hashlib.sha256()
 with open(x,'rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
e=json.loads((p/'ENTRY_PROTECTION.json').read_text())
def changed(m):return [k for k,v in m.items() if not Path(k).is_file() or sha(k)!=v]
start=time.monotonic();source=changed(e['source_before']);outputs=changed(e['outputs_before'])
repos={};issues=[]
for repo,checks in e['repositories_before'].items():
 repos[repo]={}
 for key,v in checks.items():
  q=subprocess.run(v['argv'],capture_output=True,text=True);repos[repo][key]=dict(exit=q.returncode,stdout=q.stdout,stderr=q.stderr)
  if q.returncode!=v['exit'] or (q.stdout!=v['stdout'] and not (repo==str(r) and 'status' in key)):issues.append([repo,key])
g2=json.loads((r/'research_outputs/g2a_k1_trainer_qualification_20261110_v1/ENTRY_AND_WHITELIST.json').read_text())
outside_source=changed(g2['outside_source_sha_before'])
outside_meta=[k for k,v in g2['outside_metadata_before'].items() if not Path(k).is_file() or Path(k).stat().st_size!=v['size'] or Path(k).stat().st_mtime_ns!=v['mtime_ns']]
result=dict(status='PASS' if not(source or outputs or issues or outside_source or outside_meta) else 'FAIL',source_changes=source,historical_output_changes=outputs,git_identity_changes=issues,outside_source_changes=outside_source,outside_metadata_changes=outside_meta,counts=dict(old_source=len(e['source_before']),historical_outputs=len(e['outputs_before']),outside_sources=len(g2['outside_source_sha_before']),outside_metadata=len(g2['outside_metadata_before'])),repositories_after=repos,elapsed_s=time.monotonic()-start,new_network_updates=0)
with (p/'HISTORICAL_PROTECTION.json').open('x') as f:json.dump(result,f,indent=2)
print(result['status'],result['counts'],'elapsed',result['elapsed_s'])
assert result['status']=='PASS'
