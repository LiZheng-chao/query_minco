"""Synthetic counterexamples / mocked certified snapshots, NO physics."""
import time,threading,unittest
from unittest.mock import patch
import numpy as np
import torch
torch.set_num_threads(1)
from query_minco.integration.observed_space_v1.map import Snapshot,Config,UNKNOWN,KNOWN_FREE,OCCUPIED
from query_minco.integration.depth_minco_t2_v1.space import BirthGrant,authorized_snapshot,Space
from query_minco.integration.depth_minco_t2_v1.engine import Engine
from query_minco.integration.yopo_execution_v4.trajectory import build
from query_minco.integration.observed_space_v2.core import Prepared
class Tests(unittest.TestCase):
 def setUp(self):
  self.now=100.;self.rev=1;self.cfg=Config()
 def snapshot(self,states=None):
  return Snapshot(np.zeros(self.cfg.shape,np.uint8) if states is None else states,
    np.array([-2.,-2.,-1.5]),self.cfg,[dict(id='SYNTHETIC_ONLY',stamp=self.now)],self.now,self.rev,lambda:self.rev)
 def grant(self):return BirthGrant((0.,0.,0.),.60,100.2,'TEST_SYNTHETIC_NOT_RUNTIME_AUTHORITY')
 def test_unknown_and_birth_finite(self):
  snap=self.snapshot();self.assertFalse(snap.ball([0,0,0],.3,self.now)['passed'])
  union=authorized_snapshot(snap,self.grant(),self.now,1,lambda:self.rev)
  self.assertTrue(union.ball([0,0,0],.3,self.now)['passed'])
  self.assertTrue(union.ball([.28,0,0],.3,self.now)['passed'])
  self.assertFalse(union.ball([.5,0,0],.3,self.now)['passed'])
  self.assertEqual(snap.states.sum(),0)
 def test_occupied_wins_over_birth(self):
  s=np.zeros(self.cfg.shape,np.uint8);s[20,20,15]=OCCUPIED
  union=authorized_snapshot(self.snapshot(s),self.grant(),self.now,1,lambda:self.rev)
  self.assertFalse(union.ball([0,0,0],.3,self.now)['passed'])
 def test_expiry_and_epoch(self):
  snap=authorized_snapshot(self.snapshot(),self.grant(),self.now,1,lambda:self.rev)
  with self.assertRaises(ValueError):snap.current(100.21)
  self.rev+=1
  with self.assertRaises(ValueError):snap.current(100.)
 def test_mirror_routes_or_safe_reject(self):
  # Synthetic certified-cell fixture only; it is not actual sensor evidence.
  outcomes=[]
  for mirror in [1,-1]:
   states=np.full(self.cfg.shape,KNOWN_FREE,np.uint8)
   ij=np.indices(states.shape).transpose(1,2,3,0)
   xyz=np.array([-2.,-2.,-1.5])+(ij+.5)*.1
   wall=(xyz[...,0]>.5)&(xyz[...,0]<.9)&(xyz[...,1]*mirror<.4)
   states[wall]=OCCUPIED
   sp=Space(self.snapshot(states),lambda:self.now)
   try:
    self.assertTrue(sp.ball_free(np.zeros(3),.33))
    route=sp.search_goal(np.zeros(3),np.array([1.4,0,0]))
    self.assertGreater(route[1]*mirror,0)
    outcomes.append('DIRECTION_CORRECT')
   except ValueError:outcomes.append('SAFE_REJECT')
  self.assertEqual(len(outcomes),2)
 def test_minco_and_curve_unknown(self):
  plan=build(np.zeros(3),np.zeros(3),np.zeros(3),[0,0,0,1],[.28,0,0],self.now)
  self.assertTrue(all(a<=b for a,b in zip(plan.maxima,[1.,8.,100.])))
  with self.assertRaises(ValueError):self.snapshot().certify(Prepared(plan),self.now)
 def engine(self):
  e=Engine(lambda now:None,clock=lambda:self.now)
  state=dict(p=[0.,0.,0.],v=[0.,0.,0.],a=[0.,0.,0.],q=[0.,0.,0.,1.])
  e.observe(np.full((90,160),5.,np.float32),state,self.now,self.now)
  e.set_state(state,self.now);e.set_goal([.6,0,0],self.now)
  return e,state
 def test_admission_publish_revoke(self):
  e,state=self.engine()
  def full_snapshot(now):return Snapshot(np.full(self.cfg.shape,KNOWN_FREE,np.uint8),[-2.,-2.,-1.5],self.cfg,[dict(id='SYNTHETIC_SENSOR_FIXTURE',stamp=self.now)],self.now,e.ledger.epoch,lambda:e.ledger.epoch)
  with patch.object(e.ledger,'snapshot',side_effect=full_snapshot):
   # Warm only MINCO libraries, never change any criterion after outcome.
   build(np.zeros(3),np.zeros(3),np.zeros(3),[0,0,0,1],[.1,0,0],self.now)
   self.assertTrue(e.run_once(),list(e.events))
   cmd=e.command(self.now);self.assertIsNotNone(cmd)
   for k in ['plan_id','source_stamp','map_epoch','acceptance_id']:self.assertIn(k,cmd)
   sent=[];e.with_current(self.now,lambda x:sent.append(x))
   self.assertEqual(len(sent),1)
   # New content revokes synchronously even before geometry reconstruction.
   self.now+=.03;e.observe(np.full((90,160),.5,np.float32),state,self.now,self.now)
   self.assertIsNone(e.command(self.now))
 def test_identical_depth_no_epoch_churn(self):
  e,state=self.engine();rev=e.revision
  self.now+=.03;e.observe(np.full((90,160),5.,np.float32),state,self.now,self.now)
  self.assertEqual(e.revision,rev)
 def test_worker_failure_no_publish(self):
  e,state=self.engine()
  with patch.object(e.ledger,'snapshot',side_effect=ValueError('MAP_DEADLINE')):
   self.assertFalse(e.run_once())
  called=[];self.assertIsNone(e.with_current(self.now,lambda x:called.append(x)))
  self.assertFalse(called)
 def test_control_does_not_wait_for_worker(self):
  e,state=self.engine();started=threading.Event();release=threading.Event()
  def blocked(now):started.set();release.wait(1);raise ValueError('DEADLINE')
  with patch.object(e.ledger,'snapshot',side_effect=blocked):
   worker=threading.Thread(target=e.run_once);worker.start();self.assertTrue(started.wait(1))
   t=time.perf_counter()
   for i in range(100):self.assertIsNone(e.command(self.now))
   elapsed=time.perf_counter()-t;release.set();worker.join(1)
   self.assertLess(elapsed,.1)

 def test_chunked_geometry_matches_original(self):
  import ast,subprocess
  import query_minco.integration.observed_space_v1.map as module
  source=subprocess.check_output(['git','show','HEAD:query_minco/integration/observed_space_v1/map.py'],text=True)
  cls=next(x for x in ast.parse(source).body if isinstance(x,ast.ClassDef) and x.name=='Snapshot')
  fn=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='certify')
  ns=dict(module.__dict__);exec(compile(ast.Module(body=[fn],type_ignores=[]),'old_reference','exec'),ns)
  old=ns['certify'];rng=np.random.RandomState(44)
  plan=Prepared(build(np.zeros(3),np.zeros(3),np.zeros(3),[0,0,0,1],[.28,0,0],self.now))
  def outcome(fn,snap):
   try:fn(snap,plan,self.now);return ('PASS',)
   except module.CertificationError as e:
    return tuple((k,str(e.record.get(k))) for k in ['reason','trajectory_time','world','radius','unknown','occupied'])
  # Math-only clock freeze avoids mixing geometry equivalence with performance.
  # Separate production-clock timing remains enabled in replay.
  with patch.object(module.time,'perf_counter',return_value=1.):
   for i in range(12):
    states=np.full(self.cfg.shape,KNOWN_FREE,np.uint8)
    for j in range(i):
     pos=tuple(rng.randint(0,n) for n in self.cfg.shape);states[pos]=UNKNOWN if j%2 else OCCUPIED
    self.assertEqual(outcome(old,self.snapshot(states)),outcome(type(self.snapshot()).certify,self.snapshot(states)))
 def test_birth_loader_rejects_boolean_bypass(self):
  import json
  d=dict(scope='ENGINEERING_BIRTH_BALL_ONLY',center_world=[0.,0.,0.],radius_m=.6,map_id='map',run_nonce='nonce',issued=99.,expires=129.,owner_verified=True)
  selected=dict(selected=dict(position=[0.,0.,0.],clearance_lower_bound_m=.7),map_id='map')
  truth=dict(status='FAIL',map_id='map',monotonic=10.)
  with self.assertRaises(ValueError):
   BirthGrant.from_records(json.dumps(d).encode(),truth,['_run_nonce:=nonce'],selected,100.,10.)

 def test_expired_execution_proof_revokes_commit(self):
  e,state=self.engine()
  plan=build(np.zeros(3),np.zeros(3),np.zeros(3),[0,0,0,1],[.1,0,0],100.)
  e.committed=dict(plan=plan,map_epoch=e.revision,source_stamp=100.,
     proof=dict(valid_until=100.1),acceptance_id='UNIT_PROOF',snapshot='UNIT')
  self.now=100.11
  self.assertIsNone(e.command(self.now));self.assertIsNone(e.committed)
 def test_bad_dynamics_never_publishes(self):
  e,state=self.engine();state['v']=[1.2,0,0];e.set_state(state,self.now)
  def full(now):return Snapshot(np.full(self.cfg.shape,KNOWN_FREE,np.uint8),[-2.,-2.,-1.5],self.cfg,[dict(id='UNIT',stamp=100.)],100.,e.ledger.epoch,lambda:e.ledger.epoch)
  with patch.object(e.ledger,'snapshot',side_effect=full):self.assertFalse(e.run_once())
  self.assertIsNone(e.command(self.now))

 def test_rejected_proposals_are_rate_limited(self):
  e,state=self.engine()
  def full(now):return Snapshot(np.full(self.cfg.shape,KNOWN_FREE,np.uint8),[-2.,-2.,-1.5],self.cfg,[dict(id='UNIT',stamp=self.now)],self.now,e.ledger.epoch,lambda:e.ledger.epoch)
  with patch.object(e.ledger,'snapshot',side_effect=full),patch('query_minco.integration.depth_minco_t2_v1.engine.DepthPlanner.propose',side_effect=ValueError('NO_ROUTE')) as propose:
   self.assertFalse(e.run_once())
   self.now+=.1;e.set_state(state,self.now);e.set_goal([.6,0,0],self.now)
   self.assertFalse(e.run_once());self.assertEqual(propose.call_count,1)
   self.now+=.11;e.set_state(state,self.now);e.set_goal([.6,0,0],self.now)
   e.observe(np.full((90,160),5.,np.float32),state,self.now,self.now)
   self.assertFalse(e.run_once());self.assertEqual(propose.call_count,2)

 def test_accelerated_ball_exact_equivalence(self):
  rng=np.random.RandomState(17)
  states=np.full(self.cfg.shape,KNOWN_FREE,np.uint8)
  states[20:25,15:25,12:18]=OCCUPIED;states[26:28,:,:]=UNKNOWN
  snap=self.snapshot(states);space=Space(snap,lambda:self.now)
  for i in range(100):
   point=rng.uniform(-1.8,1.8,3);point[2]*=.5
   self.assertEqual(space.ball_free(point,.33),snap.ball(point,.33,self.now)['passed'])
  # Cached results cannot outlive the original snapshot validity.
  self.now=100.21
  with self.assertRaises(ValueError):space.ball_free(point,.33)

 def test_actual_tracking_stays_in_certified_envelope(self):
  e,state=self.engine()
  plan=build(np.zeros(3),np.zeros(3),np.zeros(3),[0,0,0,1],[.1,0,0],100.)
  e.committed=dict(plan=plan,map_epoch=e.revision,source_stamp=100.,
     proof=dict(valid_until=100.2),acceptance_id='UNIT_PROOF',snapshot='UNIT')
  state['p']=[.2,0,0];e.set_state(state,100.)
  self.assertIsNone(e.command(100.));self.assertIsNone(e.committed)

 def test_bounded_audit_overflow_is_fail_closed(self):
  e,state=self.engine()
  for i in range(2049):e.record('UNIT_EVENT',index=i)
  self.assertTrue(e.audit_overflow);self.assertIsNone(e.command(100.));self.assertFalse(e.summary()['ready'])
  rows=e.take_events();self.assertEqual(len(rows),2048);self.assertEqual(len(e.events),0)

 def test_epoch_revocation_cannot_bypass_switch_check(self):
  e,state=self.engine()
  e.last_executed_plan=build(np.zeros(3),np.zeros(3),np.zeros(3),[0,0,0,1],[.6,0,0],99.)
  e.revoke('NEW_DEPTH')
  e.observe(np.full((90,160),5.,np.float32),state,self.now+.001,self.now+.001) if False else None
  # New source frame carries the new epoch; old execution is reference only.
  self.now+=.01;e.set_state(state,self.now);e.set_goal([.6,0,0],self.now)
  e.observe(np.full((90,160),5.,np.float32),state,self.now,self.now)
  def full(now):return Snapshot(np.full(self.cfg.shape,KNOWN_FREE,np.uint8),[-2.,-2.,-1.5],self.cfg,[dict(id='UNIT',stamp=self.now)],self.now,e.ledger.epoch,lambda:e.ledger.epoch)
  with patch.object(e.ledger,'snapshot',side_effect=full):
   self.assertFalse(e.run_once())
  self.assertIn('SWITCH_DISCONTINUITY',str(list(e.events)))
  self.assertIsNone(e.command(self.now))
