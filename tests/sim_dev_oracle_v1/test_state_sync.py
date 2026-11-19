"""Real SIM bridge/pairing/command methods; no ROS, Plant or control publisher."""
import ast
import copy
import json
import threading
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from experiments.sim_dev_oracle_v1.bridge import Engine, SupervisorBridge
from experiments.sim_dev_oracle_v1.state_sync import StatePairs, StateInvalid
from experiments.sim_dev_oracle_v1.prepare_runtime import state_pair_adapter
from experiments.sim_dev_oracle_v1.guardian import Guardian
from query_minco.integration.yopo_execution_v4.trajectory import build

ROOT = Path(__file__).resolve().parents[2]
ODOM = dict(frame='world', p=[.25,-.25,2.], v=[0.,0.,0.], q=[0.,0.,0.,1.])
IMU = dict(frame='world', a_world=[0.,0.,0.])
class Empty:
    map_id = 'SYNTHETIC_TEST_NOT_FLIGHT_QUALIFICATION'
    def clearance(self,p): return 10.

def bridge(now=100.):
    clock=[now]
    e=Engine.__new__(Engine)
    e.clock=lambda:clock[0];e.lock=threading.RLock();e.events=deque();e.audit_overflow=False;e.evidence_overflow=False;e.input_fault=False
    e.actual=None;e.actual_pair=None;e.state_pending=False;e.goal=None;e.committed=None;e.proposed=None;e.last_stamp=now;e.nonce='abc';e.last_executed=None;e.published_ids=set();e.published_plan_count=0
    b=SupervisorBridge.__new__(SupervisorBridge);b.engine=e;b.clock=e.clock;b.pairs=StatePairs()
    return b,clock

def feed(b,kind,stamp,data=None):
    b.sensor(kind,stamp,copy.deepcopy(data if data is not None else (ODOM if kind=='odom' else IMU)),b.clock())

def request(b,armed=False):
    return b.request(dict(kind='forest_goal',frame='world',goal=[3.25,-.25,2.],stamp=b.clock()),{},b.clock(),armed,0.,0.)

def commit(b):
    e=b.engine;now=b.clock();e.set_goal([3.25,-.25,2.],now);e.last_stamp=now
    plan=build(np.array(ODOM['p']),np.zeros(3),np.zeros(3),ODOM['q'],np.array([.85,-.25,2.]),now)
    proof=Guardian(Empty(),e.nonce).check(plan.record,e.actual[0],now,now,1)
    e.committed=dict(plan=plan,proof=proof,planner={},state_pair=e.actual_pair)
    return proof

class Sync(unittest.TestCase):
    def test_synchronous_real_bridge(self):
        b,c=bridge();feed(b,'odom',100.);feed(b,'imu',100.)
        self.assertTrue(np.array_equal(request(b)[0],ODOM['p']));self.assertEqual(b.engine.actual[1:],(100.,100.))
    def test_latest_30ms_uses_received_history(self):
        b,c=bridge(100.1);feed(b,'imu',100.07);feed(b,'odom',100.07);feed(b,'imu',100.1)
        p=b.state({},0,0);self.assertEqual(p['status'],'STATE_PAIR_READY');self.assertAlmostEqual(p['raw_latest_skew'],.03);self.assertEqual((p['odom_stamp'],p['imu_stamp']),(100.07,100.07))
    def test_nearest_imu_wins(self):
        b,c=bridge(100.);feed(b,'imu',99.98);feed(b,'imu',99.99);feed(b,'odom',99.989)
        self.assertEqual(b.state({},0,0)['imu_stamp'],99.99)
    def test_19ms_accepted(self):
        b,c=bridge();feed(b,'imu',99.981);feed(b,'odom',100.);self.assertIsNotNone(request(b))
    def test_21ms_pending_no_actual_no_command(self):
        b,c=bridge();feed(b,'imu',99.979);feed(b,'odom',100.)
        self.assertIsNone(request(b));self.assertIsNone(b.engine.actual);self.assertIsNone(b.engine.command(100.))
    def test_running_missing_stream_pending(self):
        b,c=bridge();feed(b,'odom',100.);self.assertIsNone(request(b));self.assertEqual(b.engine.take_events()[-1]['event'],'STATE_PAIR_PENDING')
    def test_unpaired_streams_cannot_start_and_existing_deadline_expires(self):
        b,c=bridge()
        for i in range(703):
            c[0]=100.+i*.1;feed(b,'odom',c[0]-.03);feed(b,'imu',c[0]);self.assertIsNone(request(b));b.engine.take_events()
        self.assertIsNone(b.engine.actual);self.assertFalse(b.engine.summary()['ready'])
        src=(ROOT/'experiments/stop_latency_fix_v1/manual.py').read_text();ep=next(n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name=='episode')
        condition=next(n.test for n in ast.walk(ep) if isinstance(n,ast.While) and any(isinstance(x,ast.Name) and x.id=='stage' for x in ast.walk(n.test)))
        expr=compile(ast.Expression(body=condition),'<actual owner deadline>','eval')
        self.assertFalse(eval(expr,dict(time=SimpleNamespace(monotonic=lambda:170.001),begin=100.,stage='SIM_DEV_ORACLE')))
    def test_duplicate_does_not_refresh_actual_or_state_stamp(self):
        b,c=bridge();feed(b,'odom',100.);feed(b,'imu',100.);request(b);n=sum(e['event']=='STATE_UPDATED' for e in b.engine.events)
        c[0]=100.01;request(b);self.assertEqual(sum(e['event']=='STATE_UPDATED' for e in b.engine.events),n);self.assertEqual(b.engine.actual[1:],(100.,100.))
    def test_armed_pending_does_not_extend_proof(self):
        b,c=bridge();feed(b,'odom',100.);feed(b,'imu',100.);request(b);proof=commit(b);old=copy.deepcopy(proof)
        c[0]=100.05;feed(b,'odom',100.05);feed(b,'imu',100.021);request(b,True)
        self.assertEqual(b.engine.committed['proof'],old)
        called=[];self.assertIsNone(b.engine.with_current(100.201,called.append));self.assertEqual(called,[])
    def test_expired_proof_no_publish_even_fresh_state(self):
        b,c=bridge();feed(b,'odom',100.);feed(b,'imu',100.);request(b);proof=commit(b)
        called=[];self.assertIsNone(b.engine.with_current(proof['expires']+.001,called.append));self.assertEqual(called,[])
    def test_imu_freshness_checked_at_command(self):
        b,c=bridge();feed(b,'odom',100.);feed(b,'imu',99.981);request(b);proof=commit(b)
        c[0]=100.19;proof.update(issued=100.19,expires=100.29);b.engine.last_stamp=c[0];b.engine.set_goal([3.25,-.25,2.],c[0])
        self.assertIsNone(b.engine.command(c[0]))
    def test_invalid_latest_cannot_fall_back_to_history(self):
        for kind,field,value,reason in [('odom','p',[float('nan'),0,0],'NONFINITE'),('imu','a_world',[0,float('inf'),0],'NONFINITE'),('odom','frame','camera_link','FRAME'),('imu','frame','body','FRAME'),('odom','v',[0,0],'SHAPE'),('odom','q',[0,0,0,0],'QUATERNION')]:
            with self.subTest(kind=kind,field=field):
                b,c=bridge();feed(b,'odom',99.99);feed(b,'imu',99.99);request(b);commit(b);bad=copy.deepcopy(ODOM if kind=='odom' else IMU);bad[field]=value
                with self.assertRaisesRegex(StateInvalid,reason):feed(b,kind,100.,bad)
                self.assertTrue(b.engine.input_fault);self.assertIsNone(b.engine.actual);self.assertIsNone(b.engine.committed)
                with self.assertRaises(StateInvalid):request(b)
    def test_stale_future_rollback_each_stream(self):
        for kind in ['odom','imu']:
            for stamp,reason in [(99.79,'STALE'),(100.001,'FUTURE'),(float('nan'),'TIMESTAMP_NONFINITE'),(99.98,'ROLLBACK')]:
                with self.subTest(kind=kind,reason=reason):
                    b,c=bridge();feed(b,kind,99.99)
                    with self.assertRaisesRegex(StateInvalid,reason):feed(b,kind,stamp)
    def test_expired_received_sample_rejected_at_selection(self):
        b,c=bridge();feed(b,'odom',100.);feed(b,'imu',100.);c[0]=100.201
        with self.assertRaisesRegex(StateInvalid,'STALE'):request(b)
    def test_timestamp_conflict_is_invalid(self):
        b,c=bridge();feed(b,'odom',100.);bad=copy.deepcopy(ODOM);bad['p'][0]+=.1
        with self.assertRaisesRegex(StateInvalid,'DUPLICATE_CONFLICT'):feed(b,'odom',100.,bad)
    def test_bounded_history_and_original_stamps(self):
        b,c=bridge()
        for i in range(100):
            c[0]=100+i*.001;feed(b,'odom',c[0]);feed(b,'imu',c[0])
        self.assertEqual(len(b.pairs.buffers['odom']),32);p=b.state({},0,0);self.assertEqual(p['odom_stamp'],c[0]);self.assertEqual(p['imu_stamp'],c[0])
    def test_pair_command_plan_log_association(self):
        b,c=bridge();feed(b,'odom',100.);feed(b,'imu',100.);request(b);commit(b);cmd=b.engine.command(100.)
        self.assertEqual(cmd['state_pair']['state_pair_id'],b.engine.actual_pair['state_pair_id']);self.assertEqual(cmd['guardian_state_pair'],cmd['state_pair'])
        self.assertEqual(cmd['plan']['epoch'],100.);self.assertTrue(np.allclose(cmd['pva'],b.engine.committed['plan'].at(100.)))
        b.engine.published(cmd);self.assertEqual(b.engine.last_executed.id,cmd['plan_id'])
    def test_receipt_clock_cannot_use_not_yet_received_sample(self):
        b,c=bridge(100.1);feed(b,'odom',100.);feed(b,'imu',100.);c[0]=100.05
        with self.assertRaisesRegex(StateInvalid,'RECEIPT_FUTURE'):request(b)
    def test_guard_loop_does_not_renew_without_new_pair(self):
        b,c=bridge();feed(b,'odom',100.);feed(b,'imu',100.);request(b);commit(b)
        class Pulse:
            def __init__(self):self.calls=0
            def wait(self,delay):self.calls+=1;return self.calls>1
        class NoRequest:
            def send(self,data):raise AssertionError('must not renew an old state pair')
        e=b.engine;e.stop=Pulse();e.guardian_ready=True;e.conn=NoRequest();e.revision=1
        e.guard_loop()
        self.assertFalse(any(x['event']=='SIM_REJECTED' for x in e.events))
    def test_guard_loop_pending_does_not_renew_old_lease(self):
        b,c=bridge();feed(b,'odom',100.);feed(b,'imu',100.);request(b);commit(b);old=copy.deepcopy(b.engine.committed['proof'])
        # Explicit pending blocks new guard checks, but existing command remains limited by the original proof.
        b.engine.state_pending=True
        class Pulse:
            def __init__(self):self.calls=0
            def wait(self,delay):self.calls+=1;return self.calls>1
        e=b.engine;e.stop=Pulse();e.guardian_ready=True;e.conn=None;e.revision=1;e.guard_loop()
        self.assertEqual(e.committed['proof'],old)
        self.assertIsNone(e.command(old['expires']+.001))
    def test_pending_during_guardian_reply_cannot_commit(self):
        b,c=bridge();feed(b,'odom',100.);feed(b,'imu',100.);request(b);proof=commit(b);e=b.engine;plan=e.committed['plan'];e.committed=None;e.proposed=dict(plan=plan,planner={})
        class Pulse:
            def __init__(self):self.calls=0
            def wait(self,delay):self.calls+=1;return self.calls>1
        class Reply:
            def send(self,data):pass
            def poll(self,timeout):return True
            def recv(self):e.state_pending=True;return dict(ok=True,proof=proof)
        e.stop=Pulse();e.guardian_ready=True;e.conn=Reply();e.revision=1;e.guard_loop()
        self.assertIsNone(e.committed)
    def test_guardian_pair_expiry_checked_even_if_actual_pair_is_new(self):
        b,c=bridge();feed(b,'odom',100.);feed(b,'imu',99.981);request(b);proof=commit(b)
        c[0]=100.19;feed(b,'odom',100.19);feed(b,'imu',100.19);request(b)
        proof.update(issued=100.19,expires=100.29);b.engine.last_stamp=100.19
        self.assertEqual(b.engine.actual[1:],(100.19,100.19));self.assertIsNone(b.engine.command(100.19))
    def test_real_generator_accept_hook_and_s_c7_isolation(self):
        source=(ROOT/'ros/g4cp7b_fix2_v1/supervisor.py').read_text();adapted=state_pair_adapter(source);compile(adapted,'<generated supervisor>','exec')
        main=next(n for n in ast.parse(adapted).body if isinstance(n,ast.FunctionDef) and n.name=='main');fn=next(n for n in ast.walk(main) if isinstance(n,ast.FunctionDef) and n.name=='accept')
        b,c=bridge();logs=[];aborts=[];last={};r=SimpleNamespace(observe=lambda *a:True)
        ns=dict(forest_bridge=b,rospy=SimpleNamespace(Time=SimpleNamespace(now=lambda:SimpleNamespace(to_sec=lambda:c[0]))),r=r,log=lambda *a:logs.append(a),abort=aborts.append,last=last)
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'<actual adapter accept>','exec'),ns)
        self.assertTrue(ns['accept']('odom',100.,ODOM));self.assertEqual(len(b.pairs.buffers['odom']),1)
        self.assertFalse(ns['accept']('imu',100.001,IMU));self.assertTrue(aborts)
        ns['forest_bridge']=None;self.assertTrue(ns['accept']('odom',100.,ODOM))
        with self.assertRaisesRegex(ValueError,'UPSTREAM_PATCH_CONTEXT'):state_pair_adapter(adapted)

if __name__ == '__main__':unittest.main()
