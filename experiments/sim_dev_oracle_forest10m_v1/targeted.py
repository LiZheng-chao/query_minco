"""Targeted nonphysical checks of the generated 10m software."""
import ast
import inspect
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from threading import RLock
from tempfile import TemporaryDirectory
import unittest
from .startup import Startup
from .audit import AsyncJournal
from .prepare import O
from .reference import source, config

class Tests(unittest.TestCase):
    def fixture(self):
        mono=[0.]
        engine=SimpleNamespace(lock=RLock(),arm_sent=False,committed='old',arm_request_stamp=99.)
        r=Startup(20.,engine,lambda:mono[0],lambda *a:None)
        r.mode='ACK'
        r.last_sent=99.9
        r.sent={99.9}
        r.acks={99.9}
        p=dict(phase='ACK',integration_steps=0,final_command_stamp=99.9,final_count=8,command_valid=True)
        r.samples['plant']=(100.,100.,p)
        r.samples['imu']=(99.7,99.7,{})
        return r,engine,mono,p

    def test_short_gap_needs_new_plant_ack(self):
        r,e,mono,p=self.fixture()
        self.assertEqual(r.tick(100.),'PREPARE')
        self.assertIsNotNone(r.wait)
        self.assertIsNone(e.committed)
        self.assertIsNone(e.arm_request_stamp)
        self.assertFalse(r.forward_allowed(100.))
        for k in ['odom','imu','depth','request','raw','plant']:
            r.counts[k]=5
            r.samples[k]=(100.01,100.01,dict(pva=[[.25,-.25,2.],[0,0,0],[0,0,0]]) if k=='request' else p)
        self.assertEqual(r.tick(100.02),'PREPARE')
        self.assertFalse(r.acks)
        for i in range(5):
            stamp=100.03+i*.01
            r.forwarded(stamp)
            p['final_command_stamp']=stamp
            r.samples['plant']=(stamp,stamp,p)
            r.tick(stamp)
        self.assertEqual(r.mode,'ACK')
        self.assertEqual(len(r.acks),5)

    def test_timeout_and_armed_staleness(self):
        r,e,mono,p=self.fixture()
        r.tick(100.)
        mono[0]=20.
        self.assertEqual(r.tick(100.),'ABORT')
        self.assertEqual(r.reason,'STARTUP_SENSOR_WAIT_TIMEOUT')
        r,e,mono,p=self.fixture()
        r.mode='ARMED'
        self.assertEqual(r.tick(100.),'ABORT')
        self.assertEqual(r.reason,'imu_STALE')
        r,e,mono,p=self.fixture()
        e.arm_sent=True
        self.assertEqual(r.tick(100.),'ABORT')

    def test_bound_goal_in_reference_and_engine(self):
        from query_minco.integration.depth_minco_t2_v1.ros_bridge import reference_main
        cfg=json.loads((O/'runtime_bundle_v5/runtime_v2/SCENES.json').read_text())['forest']['forest10m']
        text=source(inspect.getsource(reference_main))
        compile(text,'<reference>','exec')
        self.assertIn("target=config(run)['goal']",text)
        with TemporaryDirectory() as tmp:
            run=Path(tmp)/'none';run.mkdir()
            (run.parent/'SCENE_IDENTITY.json').write_text(json.dumps(dict(scene=dict(forest10m=cfg))))
            goal=config(run)['goal']
            self.assertEqual(goal,[10.5,.75,2.])
        file=O/'runtime_bundle_v5/generated_v2/bridge.py'
        spec=importlib.util.spec_from_file_location('forest10m_generated_bridge',file)
        m=importlib.util.module_from_spec(spec)
        import sys
        sys.path.insert(0,str(file.parent))
        try:spec.loader.exec_module(m)
        finally:sys.path.remove(str(file.parent))
        e=m.Engine.__new__(m.Engine)
        e.clock=lambda:100.;e.goal_config=goal;e.lock=RLock();e.record=lambda *a,**kw:None
        e.set_goal(goal,100.)
        self.assertEqual(e.goal,(goal,100.))
        with self.assertRaisesRegex(ValueError,'SIM_GOAL_COORDINATES'):
            e.set_goal([3.25,-.25,2.],100.)

    def test_async_write_and_display_namespace(self):
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'audit.jsonl'
            j=AsyncJournal(path)
            with j.lock:
                self.assertTrue(j.append(dict(kind='imu',stamp=100.)))
                self.assertTrue(j.append(dict(kind='ARM',stamp=101.)))
            self.assertEqual(j.close(),[])
            self.assertEqual([json.loads(x)['kind'] for x in path.read_text().splitlines()],['imu','ARM','JOURNAL_CLOSED'])
        display=Path(__file__).with_name('display.py').read_text()
        self.assertIn("prefix='/forest10m/display/'",display)
        self.assertNotIn('/p2/',display)

    def test_startup_request_contract(self):
        pva=[[.2507617663553498,-.2492382336446502,2.],
             [.016499331407276663,.016499331407276663,0.],
             [.22921462763421396,.22921462763421396,0.]]
        for mode in ('HELD','PREPARE','ACK','ARMED'):
            with self.subTest(mode=mode):
                readiness,*_=self.fixture()
                readiness.mode=mode
                accepted=readiness.observe('request',101.,101.,dict(frame='world',pva=pva))
                self.assertEqual(accepted,mode=='ARMED')
                if mode!='ARMED':
                    self.assertEqual(readiness.reason,'request_NON_HOLD_REQUEST')

        for v,a in ((1.01,0.),(0.,8.01),(float('nan'),0.)):
            with self.subTest(v=v,a=a):
                readiness,*_=self.fixture()
                readiness.mode='ARMED'
                self.assertFalse(readiness.observe('request',101.,101.,dict(
                    frame='world',pva=[[0.,0.,0.],[v,0.,0.],[a,0.,0.]])))
                self.assertEqual(readiness.reason,'request_REQUEST_LIMIT')

    def test_generated_supervisor_armed_request(self):
        from types import SimpleNamespace as NS
        import numpy as np
        file=O/'runtime_bundle_v5/generated_v2/supervisor.py'
        module=ast.parse(file.read_text())
        main=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        setup=[n for n in main.body if
               (isinstance(n,ast.ImportFrom) and n.module==Startup.__module__)
               or (isinstance(n,ast.Assign) and isinstance(n.value,ast.Call)
                   and isinstance(n.value.func,ast.Name) and n.value.func.id=='Startup')]
        accept=next(n for n in main.body if isinstance(n,ast.FunctionDef) and n.name=='accept')
        forest_bridge=NS(engine=NS())
        ns=dict(cfg=dict(startup_wait_limit_s=20.),forest_bridge=forest_bridge,
                time=NS(monotonic=lambda:100.),log=lambda *a:None,last={},
                rospy=NS(Time=NS(now=lambda:NS(to_sec=lambda:100.))))
        exec(compile(ast.Module(body=[*setup,accept],type_ignores=[]),str(file),'exec'),ns)
        self.assertIs(type(ns['r']),Startup)
        ns['r'].mode='ARMED'
        # Execute the actual generated ARMED block as a closure, with the
        # Guardian gate either offering a valid envelope or refusing it.
        armed=next(n for n in ast.walk(main) if isinstance(n,ast.If)
                   and any(isinstance(x,ast.FunctionDef) and x.name=="send_position" for x in ast.walk(n))
                   and ast.dump(n.test)==ast.dump(ast.parse("forest_bridge and r.mode=='ARMED'",mode="eval").body))
        wrapper=ast.parse('def exercise():\n command_counter=0\n return None')
        wrapper.body[0].body.insert(1,armed)
        ns['r'].samples.update(odom=(100.,),imu=(100.,))
        published=[]; faults=[]
        forest_bridge.state=lambda *a:None
        forest_bridge.engine.published=lambda *a:None
        ns.update(np=np,abort=faults.append,run_nonce='test',
                  PositionCommand=object,command=NS(publish=published.append),
                  position_command=lambda *a:NS(header=NS(stamp=NS(to_nsec=lambda:100000000000)),trajectory_id=1,yaw=0.,yaw_dot=0.))
        ns['rospy'].Time.from_sec=lambda value:value
        exec(compile(ast.fix_missing_locations(wrapper),str(file),'exec'),ns)
        forest_bridge.engine.with_current=lambda now,action:None
        ns['exercise']()
        self.assertEqual(faults,['T2_NO_CURRENT_PROOF'])
        self.assertEqual(published,[])
        faults.clear()
        envelope=dict(stamp=100.,pva=[[.25,-.25,2.],[.01,0.,0.],[.2,0.,0.]])
        forest_bridge.engine.with_current=lambda now,action:(action(envelope),envelope)[1]
        ns['exercise']()
        self.assertEqual(faults,[])
        self.assertEqual(len(published),1)

    def test_real_guardian_renews_and_expired_proof_stops_publication(self):
        import sys
        import numpy as np
        from collections import deque
        from experiments.sim_dev_oracle_v1.guardian import Guardian
        from query_minco.integration.yopo_execution_v4.trajectory import build
        g=O/'runtime_bundle_v5/generated_v2'
        sys.path.insert(0,str(g))
        try:
            spec=importlib.util.spec_from_file_location('forest10m_renewal_bridge',g/'bridge.py')
            bridge=importlib.util.module_from_spec(spec)
            spec.loader.exec_module(bridge)
        finally:
            sys.path.remove(str(g))
        run=O/'runtime_bundle_v3/runs/FOREST10M_20261119_123103_7375/none'
        e=bridge.Engine.__new__(bridge.Engine)
        t=[100.]
        e.clock=lambda:t[0]
        e.lock=RLock()
        e.events=deque()
        e.audit_overflow=e.evidence_overflow=e.input_fault=e.state_pending=False
        e.nonce='nonphysical-renewal'
        e.guardian_ready=True
        e.committed=e.last_executed=None
        guardian=Guardian(str(run/'native_grid'),e.nonce)
        class Step:
            used=False
            def wait(self,delay):
                if self.used:return True
                self.used=True
                return False
        class IPC:
            online=True
            def send(self,request):
                if self.online:
                    try:self.reply=dict(ok=True,proof=guardian.check(**request))
                    except ValueError as exc:self.reply=dict(ok=False,reason=str(exc))
            def poll(self,timeout):return self.online
            def recv(self):return self.reply
        e.conn=IPC()
        p=np.array([.25,-.25,2.])
        z=np.zeros(3)
        state=dict(p=p.tolist(),v=z.tolist(),a=z.tolist())
        ids=[]
        for i in range(31):
            now=100.+i*.04
            t[0]=now
            if i%5==0:
                plan=build(p,z,z,[0,0,0,1],p,now)
                e.proposed=dict(plan=plan,planner={})
            e.last_stamp=now-.04
            e.revision=i
            e.actual=(state,now,now)
            e.actual_pair=dict(state_pair_id=str(i),odom_stamp=now,imu_stamp=now)
            e.goal=([10.5,.75,2.],now)
            if e.committed:self.assertIsNotNone(e.command(now))
            e.stop=Step()
            e.guard_loop()
            self.assertEqual(e.committed['proof']['issued'],now)
            self.assertLessEqual(e.committed['proof']['expires'],now+.10)
            self.assertIsNotNone(e.command(now+.03))
            ids.append(e.committed['proof']['identity'])
            e.last_executed=plan
        self.assertEqual(len(set(ids)),31)
        last=e.committed['proof']
        t[0]+=0.02
        bad=dict(state,v=[.12,0.,0.])
        e.actual=(bad,t[0],t[0]);e.last_stamp=t[0];e.goal=(e.goal[0],t[0])
        e.actual_pair=dict(state_pair_id='tracking',odom_stamp=t[0],imu_stamp=t[0])
        e.stop=Step();e.guard_loop()
        self.assertTrue(e.urgent_replan)
        self.assertIs(e.committed['proof'],last)
        self.assertEqual(sum(x['event']=='EARLY_REPLAN_REQUESTED' for x in e.events),1)
        e.stop=Step();e.guard_loop()
        self.assertEqual(sum(x['event']=='EARLY_REPLAN_REQUESTED' for x in e.events),1)
        # A new plan can be considered even with the last committed pair.
        # All original tracking and switch tests still run.
        replacement=build(p,z,z,[0,0,0,1],p,t[0])
        e.proposed=dict(plan=replacement,planner={})
        e.actual=(state,*e.actual[1:])
        e.actual_pair=e.committed['state_pair']
        e.stop=Step();e.guard_loop()
        self.assertEqual(e.committed['plan'].id,replacement.id)
        self.assertIsNotNone(e.command(t[0]))
        last=e.committed['proof']
        t[0]=last['expires']+.001
        e.last_stamp=t[0];e.goal=(e.goal[0],t[0]);e.actual=(state,t[0],t[0])
        e.actual_pair=dict(state_pair_id='lost',odom_stamp=t[0],imu_stamp=t[0])
        e.conn.online=False;e.stop=Step();e.guard_loop()
        sent=[]
        self.assertIsNone(e.with_current(t[0],sent.append))
        self.assertEqual(sent,[])
        self.assertIs(e.committed['proof'],last)
        self.assertIsNone(e.with_current(t[0],sent.append))
        rejected=[x for x in e.events if x['event']=='COMMAND_REJECTED']
        self.assertEqual(len(rejected),1)
        self.assertEqual(rejected[0]['reason'],'PROOF_EXPIRED')
        self.assertEqual(rejected[0]['now'],t[0])

    def test_search_preserves_recorded_routes(self):
        import sys
        import numpy as np
        g=O/'runtime_bundle_v5/generated_v2'
        sys.path.insert(0,str(g))
        try:from candidate_fast import search_goal,LatestHitSpace,ObservedSpace
        finally:sys.path.remove(str(g))
        index=O/'runtime_bundle_v3/runs/FOREST10M_20261119_123103_7375/none/sensors/INDEX.jsonl'
        frames=[json.loads(s) for s in index.read_text().splitlines()][-45:]
        for row in frames[::4]:
            depth=np.load(row['path'])
            def search(fn):
                space=LatestHitSpace(depth,row['pose'],row['stamp'])
                goal=np.clip([10.5,.75,2.],space.origin+.33,space.origin+np.array(space.cfg.shape)*.1-.33)
                # Equivalence is independent of host timing. Runtime retains 50ms.
                space.deadline=float('inf')
                return fn(space,np.asarray(row['pose']['p']),goal)
            np.testing.assert_array_equal(search(ObservedSpace.search_goal),search(search_goal))

    def test_ray_target_and_blocked_segment(self):
        import sys
        import numpy as np
        from .local_goal import choose_local_goal,ray_box_exit_distance
        g=O/'runtime_bundle_v5/generated_v2'
        sys.path.insert(0,str(g))
        try:from candidate_fast import search_goal,CandidatePlanner
        finally:sys.path.remove(str(g))
        p=np.array([.25,-.25,2.]);goal=np.array([10.5,.75,2.])
        pose=dict(frame='world',p=p.tolist(),v=[0.,0.,0.],a=[0.,0.,0.],q=[0.,0.,0.,1.])
        depth=np.full((90,160),20.,dtype=np.float32)
        plan,info=CandidatePlanner().propose(depth,pose,pose,goal,100.,allow_diagnostic=True,prefer_strict=False)
        self.assertEqual(info['target_mode'],'ray')
        np.testing.assert_allclose(plan.record['goal'],p+.4*(goal-p)/np.linalg.norm(goal-p))
        self.assertAlmostEqual(ray_box_exit_distance([0,0,0],[0,-1,0],[-1,-2,-1],[1,1,1]),2.)
        self.assertAlmostEqual(ray_box_exit_distance([0,0,0],[0,0,1],[-1,-2,-1],[1,1,.3]),.3)
        for point,lo,hi in [([2,0,0],[-1]*3,[1]*3),([0,0,0],[0]*3,[0]*3),([float('nan'),0,0],[-1]*3,[1]*3)]:
            with self.assertRaisesRegex(ValueError,'LOCAL_RAY_OUT_OF_BOUNDS'):
                ray_box_exit_distance(point,[1,0,0],lo,hi)
        # Veto only the middle of the straight segment. Both endpoints are free.
        def ball_free(q,r):
            self.assertAlmostEqual(r,.33)
            return not (.37<q[0]<.50 and abs(q[1]-p[1])<.05)
        space=SimpleNamespace(origin=np.array([-1.,-1.,1.]),cfg=SimpleNamespace(shape=(32,24,20)),ball_free=ball_free)
        local,target,mode=choose_local_goal(space,p,goal,lambda s,p,g:search_goal(s,p,g,first_edge=True))
        self.assertEqual(mode,'astar')
        self.assertTrue(ball_free(target,.33))
        self.assertGreater(abs(local[1]-p[1]),.05)
        from experiments.sim_dev_oracle_v1.guardian import Guardian
        from query_minco.integration.yopo_execution_v4.trajectory import build
        detour=build(p,np.zeros(3),np.zeros(3),pose['q'],local,100.)
        run=O/'runtime_bundle_v4/runs/FOREST10M_20261119_125240_7375/none'
        proof=Guardian(str(run/'native_grid'),'nonphysical-ray-case').check(detour.record,pose,100.,100.,1)
        self.assertEqual(proof['plan_id'],detour.id)
        with self.assertRaisesRegex(ValueError,'NO_SENSOR_CERTIFIED_LOCAL_ROUTE'):
            choose_local_goal(space,p,goal,lambda *_:p)
