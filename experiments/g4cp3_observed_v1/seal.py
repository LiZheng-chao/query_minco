"""Seal new source/reports and completed raw logs,without changing historical files."""
from pathlib import Path
import sys,json,subprocess,difflib,hashlib,zipfile,ast
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from query_minco.evaluation.mirror_holdout_v1 import sha,write
R=Path.cwd();O=R/'research_outputs/g4cp3_observed_20261111_v1'
entry=json.loads((O/'ENTRY.json').read_text());protection=json.loads((O/'HISTORICAL_PROTECTION.json').read_text())
assert protection['status']=='PASS'
for name in ['E0_SOURCE_BINDING.json','runtime_v5/SOURCE_BINDING.json']:
 binding=json.loads((O/name).read_text())
 assert all(sha(path)==h for path,h in binding.items())
overlays=[]
for row in json.loads((O/'RUNTIME_V5_DERIVATION.json').read_text())['rows']:
 old=R/row['source'];new=R/row['target'];dest=O/(new.stem+'_FINAL.patch')
 text=''.join(difflib.unified_diff(old.read_text().splitlines(True),new.read_text().splitlines(True),fromfile=str(old),tofile=str(new)))
 with dest.open('x') as f:f.write(text)
 overlays.append(dict(source=str(old),source_sha256=sha(old),target=str(new),target_sha256=sha(new),patch=str(dest)))
write(O/'FINAL_SOURCE_OVERLAYS.json',overlays)
q=json.loads((O/'logs/interface_qualification/PROCESS_RESULT.json').read_text())
assert q['exit_code']==1 and 'CARVE_DEADLINE' in (O/'logs/interface_qualification/stderr.log').read_text()
write(O/'RESOURCE_QUALIFICATION.json',dict(status='FAIL',reason='Real CARVE_DEADLINE and cold curve deadline recorded;not realtime qualified',
 original_sampler_129_single_measurement_s=.05650060199695872,query_budget_s=.05,update_budget_s=.15,
 actual_D_sensor_replay=json.loads((O/'replay_final2/ACTUAL_D_SENSOR_REPLAY.json').read_text())['update_s'],
 sampling_caveat='One CPU measurement,not a realtime guarantee. Full test process RSS is not map-only RSS.',live_5_to_10Hz='NOT_RUN'))
checks=[]
for argv in [['bash','-n','experiments/g4cp3_observed_v1/MANUAL_E0_SENSOR_ONLY.sh'],['git','diff','--check']]:
 p=subprocess.run(argv,capture_output=True,text=True);checks.append(dict(argv=argv,cwd=str(R),stdout=p.stdout,stderr=p.stderr,exit_code=p.returncode));assert p.returncode==0
# Every old primary source is rechecked after final new-only edits.
changes=[path for path,h in entry['source_before'].items() if sha(path)!=h]
assert not changes
checks.append(dict(action='final protected old-source SHA recheck',status='STATIC_PASS',files=len(entry['source_before'])))
for root in entry['repositories_before']:
 for key in ['head','diff','cached']:
  b=entry['repositories_before'][root][key];p=subprocess.run(b['argv'],capture_output=True,text=True)
  assert p.returncode==b['exit_code'] and p.stdout==b['stdout']
write(O/'FINAL_SEAL_CHECKS.json',checks)
rows=[]
for p in sorted((O/'logs').glob('*/PROCESS_RESULT.json')):
 d=json.loads(p.read_text());rows.append(dict(log_dir=str(p.parent),argv=d['argv'],cwd=d['cwd'],exit_code=d['exit_code'],timed_out=d['timed_out'],
 elapsed_s=d['elapsed_s'],sampled_process_tree_peak_rss_bytes=d['sampled_process_tree_peak_rss_bytes'],stdout_sha256=d['stdout_sha256'],stderr_sha256=d['stderr_sha256']))
write(O/'COMMANDS_AND_RESULTS.json',rows)
table='# Actual logged test commands\n\n| Log | Exit | Seconds | Peak sampled process-tree RSS MiB |\n|---|---:|---:|---:|\n'
for r in rows:table+='|'+Path(r['log_dir']).name+'|'+str(r['exit_code'])+'|'+str(round(r['elapsed_s'],3))+'|'+str(round(r['sampled_process_tree_peak_rss_bytes']/2**20,2))+'|\n'
table+='\nSee COMMANDS_AND_RESULTS.json for actual argv;raw stdout/stderr and environment remain alongside each PROCESS_RESULT. Exit1 failures are preserved. Static assertion PASS never overrides performance or physical NOT_RUN.\n'
with (O/'TEST_MATRIX.md').open('x') as f:f.write(table)
roots=[R/p for p in entry['allowed_new'] if not p.startswith('research_outputs')]
new=[]
for root in roots:
 new.extend(p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
for p in new:
 if p.suffix=='.py':compile(p.read_text(),str(p),'exec')
files={str(p.relative_to(R)):dict(sha256=sha(p),bytes=p.stat().st_size,role='new_source') for p in sorted(new)}
for p in sorted(O.rglob('*')):
 if not p.is_file() or (O/'logs/seal') in p.parents:continue
 files[str(p.relative_to(R))]=dict(sha256=sha(p),bytes=p.stat().st_size,role='report_or_evidence')
write(O/'FILES_AND_SHA.json',dict(method_id='g4cp3_sensor_ledger_minco_execution_v5',files=files,
 binding_files=['E0_SOURCE_BINDING.json','runtime_v5/SOURCE_BINDING.json'],
 exclusions=['this manifest self hash','archive and archive sidecar','logs/seal (live closing process;independently hashed below after completion)'],
 history_protection=protection['status'],new_network_updates=0,new_physics_runs=0))
archive=O/'G4C_P3_AUDIT.zip'
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
 for rel in files:z.write(R/rel,rel)
 z.write(O/'FILES_AND_SHA.json',str((O/'FILES_AND_SHA.json').relative_to(R)))
with (O/'ARCHIVE_SHA256.txt').open('x') as f:f.write(sha(archive)+'  '+archive.name+'\n')
print('SEALED',len(files),'FILES',archive,'SHA',sha(archive))
print('D_PHYSICAL_PASS;FOREST_MOTION_NO_GO;STATIC58;NETWORK_UPDATES0;HISTORICAL_PROTECTION_PASS')
