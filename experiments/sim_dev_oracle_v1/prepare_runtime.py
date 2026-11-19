"""Generate only isolated SIM integration adapters/config; no ROS launch."""
from pathlib import Path
import sys,json,hashlib,subprocess
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R));O=R/'research_outputs/sim_dev_oracle_20261118_v1';D=Path(__file__).parent
from query_minco.integration.native_grid_truth_v1.grid import Grid

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def replace_once(s,a,b):
 if s.count(a)!=1:raise ValueError('UPSTREAM_PATCH_CONTEXT '+a)
 return s.replace(a,b)
def state_pair_adapter(text):
 text=replace_once(text," def accept(k,stamp,d):\n  ok=r.observe", " def accept(k,stamp,d):\n  if forest_bridge and k in ('odom','imu'):\n   try:forest_bridge.sensor(k,stamp,d,rospy.Time.now().to_sec())\n   except ValueError as ex:\n    abort('SIM_STATE '+str(ex));log('state_input_rejected',stamp,dict(reason=str(ex),source=k),False);return False\n  ok=r.observe")
 text=replace_once(text,"log('controller_command',s,dict(pva=pva.tolist(),plan_id=None,authorization=None,", "log('controller_command',s,dict(pva=pva.tolist(),plan_id=None,authorization=None,state_pair=forest_bridge.engine.actual_pair,")
 return text

def arm_handshake_adapter(text):
 text=replace_once(text," command=rospy.Publisher('/p2/position_cmd',PositionCommand,queue_size=1);final=rospy.Publisher('/p2/so3_cmd',SO3Command,queue_size=1)", " command=rospy.Publisher('/p2/position_cmd',PositionCommand,queue_size=1);final=rospy.Publisher('/p2/so3_cmd',SO3Command,queue_size=1)\n sim_start=rospy.Publisher('/p2/start_permit',String,queue_size=1) if forest_bridge else None\n def sim_arm_request(message):\n  with journal.callback() as active:\n   if active and mode!='TERMINATING' and forest_bridge:\n    try:forest_bridge.engine.request_arm(message.data)\n    except Exception as ex:abort('SIM_ARM '+repr(ex))")
 text=replace_once(text," rate=rospy.Rate(100);errors=[]", " if forest_bridge:subs.append(rospy.Subscriber('/p2/sim_arm_request',String,sim_arm_request,queue_size=1))\n rate=rospy.Rate(100);errors=[]")
 text=replace_once(text,"    if forest_bridge and r.mode=='ARMED':\n     try:\n      forest_bridge.state(", "    if forest_bridge and r.mode=='ACK':\n     try:forest_bridge.engine.with_arm_ready(rospy.Time.now().to_sec(),lambda proof:sim_start.publish(String(data=run_nonce)))\n     except Exception as ex:abort('SIM_ARM '+repr(ex))\n    if forest_bridge and r.mode=='ARMED':\n     try:\n      forest_bridge.state(")
 return text

def goal_queue_adapter(text):
 return replace_once(text,"rospy.Subscriber('/p2/request',String,request,queue_size=10,tcp_nodelay=True)","rospy.Subscriber('/p2/request',String,request,queue_size=(1 if forest_bridge else 10),tcp_nodelay=True)")

def arm_cleanup_adapter(text):
 text=replace_once(text,'from query_minco.integration.owned_stop_v2.stop import OwnedProcessStop','from query_minco.integration.owned_stop_v2.stop import OwnedProcessStop\nfrom experiments.sim_dev_oracle_v1.arm_cleanup import capture as capture_arm,cleanup as cleanup_arm')
 text=replace_once(text,'owned_identities={};owner_events=[];','owned_identities={};arm_owners={};expected_arm_exits=[];owner_events=[];')
 text=replace_once(text,"  except BaseException as error:\n   so.close();se.close();write(folder/'LAUNCH_ERROR.json',dict(error=repr(error)));raise", "  except BaseException as error:\n   so.close();se.close();write(folder/'LAUNCH_ERROR.json',dict(error=repr(error)));raise\n  if stage=='SIM_DEV_ORACLE' and name in ['arm_permit','prepare_permit']:\n   arm_owners[name]=capture_arm(child,argv);write(folder/'ARM_PROCESS_OWNER.json',arm_owners[name])")
 start=text.index('   if p is not plant and p.poll() is None:')
 end=text.index('   so.close();se.close();row=',start)
 original=text[start:end]
 replacement="   arm_evidence=None\n   if stage=='SIM_DEV_ORACLE' and name in ['arm_permit','prepare_permit']:\n    try:code,arm_evidence=cleanup_arm(p,arm_owners.get(name),terminal)\n    except Exception as ex:code=p.poll();arm_evidence=dict(error=repr(ex),parent_initiated_sigint_expected=False)\n    if arm_evidence.get('error') or arm_evidence.get('residual_owned') or arm_evidence.get('forced'):stop_errors.append('ARM_HELPER_CLEANUP_'+name)\n    if arm_evidence.get('parent_initiated_sigint_expected'):expected_arm_exits.append(name)\n   else:\n"+'\n'.join(' '+line for line in original.rstrip().splitlines())+'\n'
 text=replace_once(text,original,replacement.replace('\\n','\n'))
 text=replace_once(text,"   write(folder/'PROCESS_RESULT.json',row);children[name]=row;", "   if arm_evidence is not None:row['arm_cleanup_evidence']=arm_evidence\n   write(folder/'PROCESS_RESULT.json',row);children[name]=row;")
 text=replace_once(text," expected_children=['plant']+", " expected_children=expected_arm_exits+['plant']+")
 return text

def raw_latest_adapter(text):
 text=replace_once(text,'from query_minco.integration.yopo_execution_v4.lifecycle import Journal,atomic_json','from query_minco.integration.yopo_execution_v4.lifecycle import Journal,atomic_json\n from experiments.sim_dev_oracle_v1.raw_delivery import LatestRaw')
 text=replace_once(text,' def raw(m):'," raw_slot=LatestRaw(run_nonce) if forest_bridge else None\n def receive_raw(m):\n  raw_slot.offer(m,rospy.Time.now().to_sec(),time.monotonic())\n def raw(m):")
 text=replace_once(text,"rospy.Subscriber('/p2/so3_raw',SO3Command,raw,queue_size=10,tcp_nodelay=True)","rospy.Subscriber('/p2/so3_raw',SO3Command,receive_raw if forest_bridge else raw,queue_size=(1 if forest_bridge else 10),tcp_nodelay=True)")
 text=replace_once(text,"    if forest_bridge and r.mode=='ACK':", "    if forest_bridge:\n     try:raw_slot.service(rospy.Time.now().to_sec(),raw,journal.append)\n     except ValueError as ex:abort(str(ex))\n    if forest_bridge and r.mode=='ACK':")
 text=replace_once(text," finally:\n  if forest_bridge:"," finally:\n  if raw_slot is not None:raw_slot.close(rospy.Time.now().to_sec(),journal.append)\n  if forest_bridge:")
 return text

def main():
 g=O/'generated_v2';g.mkdir();runtime=O/'runtime_v2';runtime.mkdir();records=[]
 source=R/'ros/g4cp7b_fix2_v1/supervisor.py';text=source.read_text()
 text=replace_once(text,'from query_minco.integration.depth_minco_t2_v1.ros_bridge import SupervisorBridge','from experiments.sim_dev_oracle_v1.bridge import SupervisorBridge')
 text=replace_once(text,'sys.path.insert(0,str(Path(__file__).resolve().parents[2]))','sys.path.insert(0,'+repr(str(R))+')')
 text=replace_once(text,"run_nonce=runtime.get('run_nonce');command_counter=0;","run_nonce=runtime.get('run_nonce');\n if runtime.get('experiment_mode')!='SIM_DEV_ORACLE':raise ValueError('SIM_MODE_IDENTITY')\n command_counter=0;")
 text=raw_latest_adapter(goal_queue_adapter(arm_handshake_adapter(state_pair_adapter(text))))
 (g/'supervisor.py').write_text(text);compile(text,str(g/'supervisor.py'),'exec');records.append(dict(source=str(source),source_sha=sha(source),generated=str(g/'supervisor.py'),generated_sha=sha(g/'supervisor.py')))
 source=R/'experiments/stop_latency_fix_v1/manual.py';text=source.read_text().replace("'F1'","'SIM_DEV_ORACLE'")
 text=arm_cleanup_adapter(text)
 text=replace_once(text,"'ros/g4cp7b_fix2_v1/supervisor.py'",repr(str(g/'supervisor.py')))
 text=replace_once(text,"'/p2/start_permit'","('/p2/sim_arm_request' if stage=='SIM_DEV_ORACLE' else '/p2/start_permit')")
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
 scene['goal']=[3.25,-.25,2.];(runtime/'SCENES.json').write_text(json.dumps(scenes,indent=2));grid=Grid(R/'research_outputs/forest_dev_sparse_seed3_dist6_fixedbirth_v1/capture_ordered/native_grid_GT_NOT_USED');clear=grid.clearance([.25,-.25,2.]);assert clear>=.6,'FIXED_BIRTH_GT_NOT_CLEAR'
 birth=json.loads((old/'BIRTH_SELECTION.json').read_text());birth['map_id']=grid.map_id;birth['selected'].update(position=[.25,-.25,2.],clearance_lower_bound_m=clear);birth['selection_scope']='FIXED_BY_RESEARCHER_NO_SPAWN_SEARCH_SIM_ONLY';(runtime/'BIRTH_SELECTION.json').write_text(json.dumps(birth,indent=2))
 (runtime/'LEGACY_B_PHYSICAL_WINDOW.json').write_bytes((old/'LEGACY_B_PHYSICAL_WINDOW.json').read_bytes())
 (O/'GENERATED_ADAPTER_IDENTITY.json').write_text(json.dumps(dict(records=records,scene=scene,fixed_birth_clearance=clear,no_original_source_changed=True,physical_qualification_inherited=False),indent=2));print('SIM_RUNTIME_GENERATED_NO_ROS_NO_ARM',clear)
if __name__=='__main__':main()
