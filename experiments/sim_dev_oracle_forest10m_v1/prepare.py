"""Generate a separate native SIM owner from the previously exercised debug owner."""
import json
from pathlib import Path
from experiments.sim_dev_oracle_v1.prepare_runtime import replace_once as patch

R = Path(__file__).resolve().parents[2]
D = Path(__file__).resolve().parent
O = R/'research_outputs/sim_dev_oracle_forest10m_v1'
OLD = R/'research_outputs/sim_dev_raw_latest_fix_20261118_v1/runtime_bundle'

def generate(b, *, proof_gap=False, refresh=False, p0=False, p1=False):
    if (p0 or p1) and not proof_gap:
        raise ValueError("FOREST10M_PATCHES_REQUIRE_PROOF_GAP_BASE")
    g = b/'generated_v2'
    rt = b/'runtime_v2'
    g.mkdir(parents=True,exist_ok=refresh)
    rt.mkdir(exist_ok=refresh)
    cfg = json.loads((D/'config.json').read_text())
    scenes = json.loads((OLD/'runtime_v2/SCENES.json').read_text())
    scenes['forest'].update(goal=cfg['goal'], forest10m=cfg)
    (rt/'SCENES.json').write_text(json.dumps(scenes, indent=2))
    for name in ('BIRTH_SELECTION.json','LEGACY_B_PHYSICAL_WINDOW.json'):
        (rt/name).write_bytes((OLD/'runtime_v2'/name).read_bytes())
    bridge = (R/'experiments/sim_dev_oracle_v1/bridge.py').read_text()
    bridge = patch(bridge, 'expected=[3.25,-.25,2.]', "expected=self.goal_config")
    bridge = patch(bridge, "  self.sensor_dir=self.run/", "  self.goal_config=json.loads((self.run.parent/'SCENE_IDENTITY.json').read_text())['scene']['forest10m']['goal']\n  self.sensor_dir=self.run/")
    if proof_gap:
        import inspect
        from query_minco.integration.yopo_execution_v4.sensor_space import ObservedSpace
        search = inspect.getsource(ObservedSpace.search_goal)
        search = patch(search, "    new=cost[u]+h*np.hypot(dx,dy)\n", "")
        search = patch(search, "    n=max(2,int(np.linalg.norm(q-p)/.05)+1)",
                       "    new=cost[u]+h*np.hypot(dx,dy)\n    if new>=cost.get(w,float('inf')):continue\n    n=max(2,int(np.linalg.norm(q-p)/.05)+1)")
        # Outdated heap entries cannot improve any edge after a better entry
        # for the same node has already been expanded.
        search = patch(search, "parent={};end=None", "parent={};end=None;expanded={}")
        search = patch(search, "   _,u=heapq.heappop(frontier);p=", "   _,u=heapq.heappop(frontier)\n   if cost[u]>=expanded.get(u,float('inf')):continue\n   expanded[u]=cost[u]\n   p=")
        planner = (R/'experiments/sim_dev_oracle_v1/planner.py').read_text()
        from textwrap import dedent
        planner = patch(planner, "class CandidatePlanner:", "import heapq\n"+dedent(search)+"\nclass CandidatePlanner:")
        planner = patch(planner, "local=ObservedSpace.search_goal(space,np.array(state['p']),horizon);astar_s=time.perf_counter()-tick\n   delta=", "local=search_goal(space,np.array(state['p']),horizon);astar_s=time.perf_counter()-tick\n   delta=")
        planner = patch(planner, "local_goal=local.tolist(),fixed_lookahead=.4", "local_goal=local.tolist(),search_origin=list(state['p']),clipped_goal=horizon.tolist(),fixed_lookahead=.4")
        if p1:
            planner=patch(planner,"def search_goal(self,start,goal):","def search_goal(self,start,goal,*,first_edge=False):")
            planner=patch(planner," return next((p for p in points[1:]", " if first_edge:return points[1] if len(points)>1 else points[0]\n return next((p for p in points[1:]")
            planner=patch(planner,"import heapq\n","import heapq\nfrom experiments.sim_dev_oracle_forest10m_v1.local_goal import choose_local_goal\n")
            planner=patch(planner,"horizon=np.clip(goal,space.origin+.33,space.origin+np.array(space.cfg.shape)*.1-.33);local=search_goal(space,np.array(state['p']),horizon)",
                          "local,horizon,target_mode=choose_local_goal(space,np.array(state['p']),goal,lambda s,p,g:search_goal(s,p,g,first_edge=True))")
            planner=patch(planner,"clipped_goal=horizon.tolist()", "ray_target=horizon.tolist(),target_mode=target_mode")
        (g/'candidate_fast.py').write_text(planner)
        bridge = patch(bridge, "from experiments.sim_dev_oracle_v1.planner import CandidatePlanner,validate_input", "from candidate_fast import CandidatePlanner,validate_input")
        bridge = patch(bridge, "   try:\n    if not self.guardian_ready:", "   proposal=None\n   try:\n    if not self.guardian_ready:")
        bridge = patch(bridge, "self.proposed is not None and self.proposed['plan'].id", "self.proposed is not None and self.proposed is proposal and self.proposed['plan'].id")
    if p0:
        from experiments.sim_dev_oracle_forest10m_v1.proof_continuity import adapt_bridge
        bridge=adapt_bridge(bridge)
    (g/'bridge.py').write_text(bridge)
    sup = (OLD/'generated_v2/supervisor.py').read_text()
    sup = patch(sup, 'journal=Journal(', 'from experiments.sim_dev_oracle_forest10m_v1.audit import AsyncJournal\n journal=AsyncJournal(')
    sup = patch(sup, 'from experiments.sim_dev_oracle_v1.bridge import SupervisorBridge', 'from bridge import SupervisorBridge')
    sup = patch(sup, ' def abort(reason):', """ from experiments.sim_dev_oracle_forest10m_v1.startup import Startup
 if not forest_bridge:raise ValueError('FOREST10M_ONLY')
 cfg=json.loads((run.parent/'SCENE_IDENTITY.json').read_text())['scene']['forest10m']
 r=Startup(cfg['startup_wait_limit_s'],forest_bridge.engine,time.monotonic,log)
 arm_intent=False
 def abort(reason):""")
    start=sup.index("  if forest_bridge and k in ('odom','imu'):")
    end=sup.index('  if ok:last[k]=d',start)
    sup=sup[:start]+"""  now=rospy.Time.now().to_sec()
  ok=r.observe(k,stamp,now,d);log(k,stamp,d,ok)
  if ok and k in ('odom','imu'):
   try:forest_bridge.sensor(k,stamp,d,now)
   except ValueError as ex:
    abort('SIM_STATE '+str(ex));return False
"""+sup[end:]
    sup=patch(sup,"    if 'odom' not in last or 'imu' not in last:return","""    now=rospy.Time.now().to_sec()
    if r.mode!='ARMED' and (r.wait is not None or not r.sensors_ready(now)):
     return
    if 'odom' not in r.samples or 'imu' not in r.samples:return""")
    sup=patch(sup," def sim_arm_request(message):"," def sim_arm_request(message):\n  nonlocal arm_intent")
    sup=patch(sup,'try:forest_bridge.engine.request_arm(message.data)',"""try:
     if message.data!=run_nonce:raise ValueError('SIM_ARM_NONCE')
     arm_intent=True""")
    sup=patch(sup,"    if forest_bridge and r.mode=='ACK':","""    r.tick(rospy.Time.now().to_sec())
    if journal.failure:abort(journal.failure)
    if arm_intent and r.mode=='ACK' and r.wait is None and not journal.failure:
     forest_bridge.engine.request_arm(run_nonce)""")
    # Keep receipt timestamps at callback entry, independent of journal contention.
    for kind in ('odom','imu'):
        sup=patch(sup, f' def {kind}(m):', f' def {kind}(m):\n  receipt_ros=rospy.Time.now().to_sec()\n  receipt_mono=time.monotonic()')
        next_fn=sup.index(' def ',sup.index(f' def {kind}(m):')+6)
        first=sup.index('  with journal.callback()',sup.index(f' def {kind}(m):'))
        insert=sup.index(' as active:',first)+len(' as active:')
        sup=sup[:insert]+f"\n   if active:journal.append(dict(kind='{kind}_receipt',source_stamp=m.header.stamp.to_sec(),receipt_ros=receipt_ros,receipt_monotonic=receipt_mono,processing_monotonic=time.monotonic()))"+sup[insert:]
    # Record the wait flag without inventing a Plant phase.
    sup=patch(sup,'reason=r.reason,counts=r.counts','reason=r.reason,waiting_for_sensors=r.wait is not None,counts=r.counts')
    sup=patch(sup,"    atomic_json(out/'HEARTBEAT.json',dict(", "    heartbeat=dict(")
    sup=patch(sup,"planner=forest_bridge.engine.summary() if forest_bridge else None))\n   rate.sleep()", "planner=forest_bridge.engine.summary() if forest_bridge else None)\n   atomic_json(out/'HEARTBEAT.json',heartbeat)\n   rate.sleep()")
    (g/'supervisor.py').write_text(sup)
    owner=(OLD/'generated_v2/debug_owner.py').read_text()
    owner=patch(owner,str(OLD/'generated_v2/supervisor.py'),str(g/'supervisor.py'))
    owner=patch(owner,str(R/'experiments/sim_dev_oracle_v1/reference.py'),str(D/'reference.py'))
    owner=patch(owner,"(70 if stage in ['E','SIM_DEV_ORACLE'] else 35)", "scene['forest10m']['startup_wait_limit_s']+scene['forest10m']['mission_s']")
    owner=patch(owner,"    if stage!='S' and h['mode']=='ACK' and not arm_sent:", "    if not h.get('waiting_for_sensors') and stage!='S' and h['mode']=='ACK' and not arm_sent:")
    owner=patch(owner,"     if armed_at is None:armed_at=now;anchor=h['state']['p']","""     if armed_at is None:armed_at=now;anchor=h['state']['p']
     if now-armed_at>=scene['forest10m']['mission_s']:reason='MISSION_TIME_LIMIT';break""")
    owner=patch(owner,"   if stage=='SIM_DEV_ORACLE' and not(progress>=.75 and ref['replans']>=2 and metrics['maxima']['tracking_error']<=.15):qualification=False","   if stage=='SIM_DEV_ORACLE' and not(error<=.15 and speed<=.10 and ref['replans']>=2 and metrics['maxima']['tracking_error']<=.15):qualification=False")
    owner=patch(owner,'  from experiments.sim_dev_oracle_v1.manual import qualification\n  qualification(O,binding)\n','')
    owner=patch(owner,"'TARGETED_TEST_RESULTS.json'","'TARGETED_NON_HOLD_TESTS.json'")
    owner=patch(owner," write(out/'RESULT.json',final);print", """ status_path=out/'none/reference_data/STATUS.json'
 if status_path.exists():final['mission']=json.loads(status_path.read_text())
 final['mission_outcome']='GOAL_REACHED' if final.get('mission',{}).get('done') else ('TIME_LIMIT' if any(f.get('reason') in ('MISSION_TIME_LIMIT','EPISODE_DEADLINE') for f in final.get('faults',[])) else 'STOPPED')
 write(out/'RESULT.json',final);print""")
    if proof_gap:
        owner=patch(owner,"'TARGETED_NON_HOLD_TESTS.json'","'TARGETED_PROOF_GAP_TESTS.json'")
        owner=patch(owner," write(out/'RESULT.json',final);print", """ from experiments.sim_dev_oracle_forest10m_v1.proof_evidence import observed
 final['observed_execution']=observed(out,final)
 final.setdefault('mission',{})['ever_armed']=final['observed_execution']['ever_armed']
 write(out/'RESULT.json',final);print""")
    if p0 or p1:
        owner=patch(owner, "'TARGETED_PROOF_GAP_TESTS.json'", "'TARGETED_P0_P1_TESTS.json'")
    (g/'debug_owner.py').write_text(owner)
    for path in g.glob('*.py'):
        compile(path.read_text(), str(path), 'exec')
