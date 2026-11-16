"""Source-bound static preflights only. No ROS process or native executable is started."""
from pathlib import Path
import sys,json,subprocess,difflib,time,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from query_minco.evaluation.mirror_holdout_v1 import sha,write
from query_minco.integration.observed_space_v1.map import ObservedMap,Config,BudgetExceeded
from query_minco.integration.yopo_execution_v4.trajectory import build
R=Path.cwd();O=R/'research_outputs/g4cp3_observed_20261111_v1';V=O/'runtime_v5'
base=json.loads((R/'research_outputs/g4cp2_execution_20261111_v1/SOURCE_BINDING.json').read_text())
critical=[p for folder in ['query_minco/integration/observed_space_v1','ros/g4cp3_observed_v1'] for p in (R/folder).glob('*.py')]
critical += [R/'experiments/g4cp3_observed_v1'/name for name in ['manual_e0.py','manual_v5.py']]
critical += [R/'research_outputs/g4cp2_execution_20261111_v1/SCENES.json',O/'D_PHYSICAL_REVIEW.json',V/'SCENES.json',V/'LEGACY_B_PHYSICAL_WINDOW.json']
binding=dict(base);binding.update({str(p):sha(p) for p in critical})
write(O/'E0_SOURCE_BINDING.json',binding);write(V/'SOURCE_BINDING.json',binding)
for path,h in binding.items():assert sha(path)==h
# Record final overlays,including subsequent explicit expiry fencing additions.
comparisons=[
 (R/'ros/g4cp2_execution_v4/supervisor_node.py',R/'ros/g4cp3_observed_v1/supervisor_v5.py'),
 (R/'ros/g4cp2_execution_v4/reference_node.py',R/'ros/g4cp3_observed_v1/reference_v5.py'),
 (R/'experiments/g4cp2_execution_v1/manual_physics.py',R/'experiments/g4cp3_observed_v1/manual_v5.py')]
# Use paths from derivation manifest if prior helper naming differs.
derivation=json.loads((O/'RUNTIME_V5_DERIVATION.json').read_text())
overlays=[]
for old,new in comparisons:
 if not old.is_file():continue
 diff=''.join(difflib.unified_diff(old.read_text().splitlines(True),new.read_text().splitlines(True),fromfile=str(old),tofile=str(new)))
 dest=O/(new.stem+'_FINAL_OVERLAY.patch');dest.write_text(diff)
 overlays.append(dict(source=str(old),source_sha256=sha(old),target=str(new),target_sha256=sha(new),patch=str(dest)))
write(O/'FINAL_RUNTIME_IDENTITIES.json',dict(status='STATIC_PASS',binding_entries=len(binding),overlays=overlays,legacy_C_D='unchanged original v4 identity;not promoted to v5',v5_C_D_physical='NOT_RUN'))
results=[]
commands=[
 [sys.executable,'-B','experiments/g4cp3_observed_v1/manual_e0.py','--allow-sensors','RESEARCHER_MANUAL_E0_SENSORS_NO_ARM','--output',str(O/'UNUSED_CHECK_ONLY'),'--check-only'],
 [sys.executable,'-B','experiments/g4cp3_observed_v1/manual_v5.py','--stage','C','--allow-physics','RESEARCHER_MANUAL_C_ONLY','--output',str(O/'UNUSED_CHECK_ONLY_C'),'--check-only'],
 [sys.executable,'-B','experiments/g4cp3_observed_v1/manual_v5.py','--stage','D','--allow-physics','RESEARCHER_MANUAL_D_ONLY','--prior',str(R/'research_outputs/g4cp2_manual_C_20261111_180435_1005/run/RESULT.json'),'--output',str(O/'UNUSED_CHECK_ONLY_D'),'--check-only']]
for i,argv in enumerate(commands):
 p=subprocess.run(argv,capture_output=True,text=True)
 results.append(dict(argv=argv,cwd=str(R),stdout=p.stdout,stderr=p.stderr,exit_code=p.returncode,
 expected='success without ROS' if i<2 else 'reject historical v4 C under new v5 source binding'))
 assert (p.returncode==0) if i<2 else (p.returncode!=0 and 'SOURCE_BINDING' in p.stderr.upper())
write(O/'CLI_PREFLIGHT_RESULTS.json',results)
print('STATIC_CLI',[(r['exit_code'],r['expected']) for r in results])
# Explicit resource diagnostic;failure is preserved,not converted to motion readiness.
m=ObservedMap(Config(shape=(32,32,24)))
for i in range(31):
 t=10+i*.05;x=-1.5+i*.05
 m.add(np.full((90,160),3.5-x,np.float32),dict(p=[x,0,0],q=[0,0,0,1]),t,t,t);s=m.snapshot(t)
plan=build(np.zeros(3),np.zeros(3),np.zeros(3),[0,0,0,1],np.array([.4,0,0]),t)
begin=time.perf_counter();_=[plan.sample(x) for x in np.linspace(0,2,129)];sampler=time.perf_counter()-begin
begin=time.perf_counter()
try:proof=s.certify(plan,t);cert=dict(status='STATIC_PASS',proof=proof)
except BudgetExceeded as e:cert=dict(status='FAIL',reason=str(e))
cert.update(elapsed_s=time.perf_counter()-begin,original_sampler_129_s=sampler,geometry_fixture_shape=[32,32,24],
 scope='single CPU measurement;not real-time or physical qualification',prior_default_grid_history_overrun='FAIL preserved in logs/final_static_tests')
write(O/'RESOURCE_QUALIFICATION.json',cert);print('RESOURCE_QUALIFICATION',cert)
