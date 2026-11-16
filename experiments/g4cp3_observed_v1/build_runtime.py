"""Versioned minimal ROS overlays;original v4 files never modified."""
from pathlib import Path
import sys,json,difflib,ast
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from query_minco.evaluation.mirror_holdout_v1 import sha,write
R=Path.cwd();O=R/'research_outputs/g4cp3_observed_20261111_v1'
def emit(source,dest,transform):
 old=(R/source).read_text();new=transform(old);ast.parse(new,filename=dest)
 p=R/dest;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:f.write(new)
 with (O/(p.stem+'_OVERLAY.patch')).open('x') as f:f.writelines(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile=source,tofile=dest))
 return dict(source=source,source_sha256=sha(R/source),target=dest,target_sha256=sha(p))
def supervisor(s):
 s=s.replace('from query_minco.integration.yopo_execution_v4.sensor_space import ObservedSpace','from query_minco.integration.observed_space_v1.map import ObservedMap\n from query_minco.integration.observed_space_v1.space import MapSpace,bootstrap')
 s=s.replace("stop_confirmed=None","stop_confirmed=None;ledger=ObservedMap(scene_id='frozen_forest3101_native_camera');space=MapSpace(ledger)")
 s=s.replace("space=ObservedSpace(d,poses[s],s)","ledger.add(d,poses[s],s,s,rospy.Time.now().to_sec())")
 s=s.replace("pva=gate.check(envelope,actual,rospy.Time.now().to_sec(),certify=cert)","pva=gate.check(envelope,actual,rospy.Time.now().to_sec(),certify=cert)\n    if a.forest and gate.plan is not None:space.certify(gate.plan)")
 s=s.replace("atomic_json(out/'HEARTBEAT.json',dict(", "boot=None\n    if a.forest and ledger.frames and 'odom' in last:\n     try:boot=bootstrap(space.snapshot(),last['odom']['p'],rospy.Time.now().to_sec())\n     except Exception as e:boot=dict(status='FAIL',reason=repr(e))\n    atomic_json(out/'HEARTBEAT.json',dict(bootstrap=boot,")
 return s
def reference(s):
 s=s.replace("choices=['C','D','E']","choices=['C','D','E1','E2']")
 s=s.replace('from query_minco.integration.yopo_execution_v4.sensor_space import ObservedSpace','from query_minco.integration.observed_space_v1.map import ObservedMap\n from query_minco.integration.observed_space_v1.space import MapSpace')
 s=s.replace("motion_start=None","motion_start=None;ledger=ObservedMap(scene_id='frozen_forest3101_native_camera');observed_space=MapSpace(ledger)")
 s=s.replace("if a.stage=='D' else 3.","if a.stage in ['D','E1'] else 3.")
 s=s.replace("a.stage in ['D','E']","a.stage in ['D','E1','E2']")
 s=s.replace("a.stage=='E' and now-last_plan>=.2","a.stage=='E2' and now-last_plan>=.2")
 s=s.replace("if a.stage=='E':","if a.stage in ['E1','E2']:")
 s=s.replace("if a.stage=='D' and now-plan.epoch>2.5","if a.stage in ['D','E1'] and now-plan.epoch>2.5")
 s=s.replace("if a.stage=='E' and replans>=20","if a.stage=='E2' and replans>=20")
 s=s.replace("space=ObservedSpace(d,pose_cache[image.header.stamp.to_sec()],image.header.stamp.to_sec())","space=observed_space")
 # Ingest actual matched source images before movement;callbacks only populate cache.
 marker="   if anchor is None:"
 add="""   if a.stage in ['E1','E2'] and 'depth' in c:
    image=c['depth'];ds=image.header.stamp.to_sec()
    if ds in pose_cache and ds>ledger.last_stamp:
     data=np.ndarray((image.height,image.width),dtype='<f4',buffer=image.data,strides=(image.step,4)).copy()
     ledger.add(data,pose_cache[ds],ds,ds,now)
"""
 s=s.replace(marker,add+marker)
 return s
def owner(s):
 s=s.replace("O=R/'research_outputs/g4cp2_execution_20261111_v1'","O=R/'research_outputs/g4cp3_observed_20261111_v1/runtime_v5'")
 s=s.replace("from query_minco.integration.yopo_execution_v4.evidence import CASES","from query_minco.integration.observed_space_v1.evidence import verify_e0\nfrom query_minco.integration.yopo_execution_v4.evidence import CASES")
 s=s.replace("root,stage,case,binding,env,scene","root,stage,case,binding,env,scene")
 s=s.replace("stage=='E'","stage in ['E1','E2']").replace("stage in ['D','E']","stage in ['D','E1','E2']")
 s=s.replace("choices=['C','D','E']","choices=['C','D','E1','E2']")
 s=s.replace("p.add_argument('--check-only',action='store_true');","p.add_argument('--check-only',action='store_true');p.add_argument('--e0-evidence');")
 old=" if a.stage in ['E1','E2'] and json.loads((O/'FOREST_SENSOR_ROUTE_QUALIFICATION.json').read_text())['status']!='PASS':raise PermissionError('E_INITIAL_OBSERVED_ROUTE_NOT_QUALIFIED')"
 assert old in s;s=s.replace(old," if a.stage in ['E1','E2']:\n  if not a.e0_evidence:raise PermissionError('E0_EVIDENCE_REQUIRED')\n  verify_e0(a.e0_evidence)")
 s=s.replace("ros/g4cp2_execution_v4/supervisor.py","ros/g4cp3_observed_v1/supervisor_v5.py").replace("ros/g4cp2_execution_v4/reference.py","ros/g4cp3_observed_v1/reference_v5.py")
 s=s.replace("if stage=='D' and not(","if stage in ['D','E1'] and not(").replace("if stage in ['E1','E2'] and not(progress>=2", "if stage=='E2' and not(progress>=2")
 # New live map must independently cover the startup body/tube immediately before ARM.
 marker="     launch('arm_permit',"
 s=s.replace(marker,"     if stage in ['E1','E2'] and (h.get('bootstrap') or {}).get('status')!='STATIC_PASS':raise PermissionError('LIVE_INITIAL_TUBE_UNKNOWN_NO_ARM')\n"+marker)
 s=s.replace('11349','11353')
 return s
def main():
 rows=[]
 rows.append(emit('ros/g4cp2_execution_v4/supervisor.py','ros/g4cp3_observed_v1/supervisor_v5.py',supervisor))
 rows.append(emit('ros/g4cp2_execution_v4/reference.py','ros/g4cp3_observed_v1/reference_v5.py',reference))
 rows.append(emit('experiments/g4cp2_execution_v1/manual.py','experiments/g4cp3_observed_v1/manual_v5.py',owner))
 output=O/'runtime_v5';output.mkdir()
 for name in ['SCENES.json','LEGACY_B_PHYSICAL_WINDOW.json']:
  with (output/name).open('xb') as f:f.write((R/'research_outputs/g4cp2_execution_20261111_v1'/name).read_bytes())
 write(O/'RUNTIME_V5_DERIVATION.json',dict(method_id='g4cp3_sensor_ledger_minco_execution_v5',rows=rows,old_C_D_source_preserved=True,new_execution_chain_requalification='NOT_RUN;old exact-binding PASS is never re-labelled',actual_physics_executions=0))
 print('V5_E1_E2_SOURCE_GENERATED_NO_EXECUTION')
if __name__=='__main__':main()
