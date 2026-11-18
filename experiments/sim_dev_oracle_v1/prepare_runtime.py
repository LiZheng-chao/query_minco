"""Generate only isolated SIM integration adapters/config; no ROS launch."""
from pathlib import Path
import sys,json,hashlib,subprocess
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R));O=R/'research_outputs/sim_dev_oracle_20261118_v1';D=Path(__file__).parent
from query_minco.integration.native_grid_truth_v1.grid import Grid

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def replace_once(s,a,b):
 if s.count(a)!=1:raise ValueError('UPSTREAM_PATCH_CONTEXT '+a)
 return s.replace(a,b)
def main():
 g=O/'generated_v2';g.mkdir();runtime=O/'runtime_v2';runtime.mkdir();records=[]
 source=R/'ros/g4cp7b_fix2_v1/supervisor.py';text=source.read_text()
 text=replace_once(text,'from query_minco.integration.depth_minco_t2_v1.ros_bridge import SupervisorBridge','from experiments.sim_dev_oracle_v1.bridge import SupervisorBridge')
 text=replace_once(text,'sys.path.insert(0,str(Path(__file__).resolve().parents[2]))','sys.path.insert(0,'+repr(str(R))+')')
 text=replace_once(text,"run_nonce=runtime.get('run_nonce');command_counter=0;","run_nonce=runtime.get('run_nonce');\n if runtime.get('experiment_mode')!='SIM_DEV_ORACLE':raise ValueError('SIM_MODE_IDENTITY')\n command_counter=0;")
 (g/'supervisor.py').write_text(text);compile(text,str(g/'supervisor.py'),'exec');records.append(dict(source=str(source),source_sha=sha(source),generated=str(g/'supervisor.py'),generated_sha=sha(g/'supervisor.py')))
 source=R/'experiments/stop_latency_fix_v1/manual.py';text=source.read_text().replace("'F1'","'SIM_DEV_ORACLE'")
 text=replace_once(text,"'ros/g4cp7b_fix2_v1/supervisor.py'",repr(str(g/'supervisor.py')))
 text=replace_once(text,"'ros/g4cp7b_fix2_v1/reference.py'","("+repr(str(D/'reference.py'))+" if stage=='SIM_DEV_ORACLE' else 'ros/g4cp7b_fix2_v1/reference.py')")
 text=replace_once(text,'from experiments.forest_f1_v1.manual import qualification','from experiments.sim_dev_oracle_v1.manual import qualification')
 text=replace_once(text,'from query_minco.integration.f0_continuity_v1.execution_evidence import assess','from experiments.sim_dev_oracle_v1.evidence import assess')
 text=replace_once(text,"e.get('payload',{}).get('event')=='CERTIFIED_COMMITTED'","e.get('payload',{}).get('event')=='CANDIDATE_GENERATED'")
 text=replace_once(text,"watchdog_s=.2,run_nonce=nonce)","watchdog_s=.2,run_nonce=nonce,experiment_mode='SIM_DEV_ORACLE')")
 old="if stage in ['E','SIM_DEV_ORACLE'] and not(progress>=2 and ref['replans']>=20 and metrics['maxima']['tracking_error']<=.15):qualification=False"
 text=replace_once(text,old,"if stage=='E' and not(progress>=2 and ref['replans']>=20 and metrics['maxima']['tracking_error']<=.15):qualification=False\n   if stage=='SIM_DEV_ORACLE' and not(progress>=.75 and ref['replans']>=2 and metrics['maxima']['tracking_error']<=.15):qualification=False")
 text=replace_once(text,"write(out/'OWNER_STOP_TIMELINE.json',owner_events);write(out/'RESULT.json',outcome);return outcome","outcome.update(experiment_mode='SIM_DEV_ORACLE',strict_depth_qualified=False,real_uav_deployment_qualified=False)\n write(out/'OWNER_STOP_TIMELINE.json',owner_events);write(out/'RESULT.json',outcome);return outcome")
 (g/'manual.py').write_text(text);compile(text,str(g/'manual.py'),'exec');records.append(dict(source=str(source),source_sha=sha(source),generated=str(g/'manual.py'),generated_sha=sha(g/'manual.py')))
 old=R/'research_outputs/f0_controller_consumption_20261118_v1/runtime';scenes=json.loads((old/'SCENES.json').read_text());scene=scenes['forest'];scene.update(sensor_binary=str(R/'research_outputs/forest_dev_sparse_seed3_dist6_fixedbirth_v1/build/native_renderer'),spawn=[.25,-.25,2.],experiment_mode='SIM_DEV_ORACLE',seed=3,maze_type=5,tree_dist=6,scene_kind='native_forest_seed3_maze5_tree_dist6_sim_dev_oracle')
 (runtime/'SCENES.json').write_text(json.dumps(scenes,indent=2));grid=Grid(R/'research_outputs/forest_dev_sparse_seed3_dist6_fixedbirth_v1/capture_ordered/native_grid_GT_NOT_USED');clear=grid.clearance([.25,-.25,2.]);assert clear>=.6,'FIXED_BIRTH_GT_NOT_CLEAR'
 birth=json.loads((old/'BIRTH_SELECTION.json').read_text());birth['map_id']=grid.map_id;birth['selected'].update(position=[.25,-.25,2.],clearance_lower_bound_m=clear);birth['selection_scope']='FIXED_BY_RESEARCHER_NO_SPAWN_SEARCH_SIM_ONLY';(runtime/'BIRTH_SELECTION.json').write_text(json.dumps(birth,indent=2))
 (runtime/'LEGACY_B_PHYSICAL_WINDOW.json').write_bytes((old/'LEGACY_B_PHYSICAL_WINDOW.json').read_bytes())
 (O/'GENERATED_ADAPTER_IDENTITY.json').write_text(json.dumps(dict(records=records,scene=scene,fixed_birth_clearance=clear,no_original_source_changed=True,physical_qualification_inherited=False),indent=2));print('SIM_RUNTIME_GENERATED_NO_ROS_NO_ARM',clear)
if __name__=='__main__':main()
