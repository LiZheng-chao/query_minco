import unittest,copy,json
from pathlib import Path
from query_minco.integration.yopo_execution_v4.evidence import expected_fault,depth_loss_causality
class Contract(unittest.TestCase):
 def fixture(self,reason="REQUEST ValueError('F0_DEPTH_STALE')"):
  f=dict(case='depth_loss',armed=True,steps=1000,monotonic=10.,ros_time=100.)
  term=dict(reason=reason,terminal_state='OWNED_PROCESS_TERMINATED',actual_exit_code=-9,pid=123,start_ticks='TEST')
  ev=[dict(kind='plant',accepted=True,mode='ARMED',monotonic=9.99,payload=dict(integration_steps=1000,state_source='DYNAMICS_INTEGRATED')),dict(kind='depth',accepted=True,stamp=99.98,monotonic=9.98,payload=dict(source_odom_stamp=99.98)),dict(kind='fault',monotonic=10.2,ros_now=100.2,payload=dict(reason=reason))]
  return ev,f,term,10.21
 def test_old(self):self.assertTrue(depth_loss_causality(*self.fixture('depth_STALE')))
 def test_new(self):self.assertTrue(depth_loss_causality(*self.fixture()))
 def test_wrong_causes(self):
  for reason in ['request_STALE',"REQUEST ValueError('BIRTH_EXPIRED')","REQUEST ValueError('F0_NEW_OCCUPIED_EVIDENCE')","REQUEST ValueError('F0_DEPTH_STAMP')","REQUEST ValueError('other')",'depth_UNRELATED']:
   with self.subTest(reason=reason):
    self.assertFalse(expected_fault('depth_loss',reason));self.assertFalse(depth_loss_causality(*self.fixture(reason)))
 def test_no_injection(self):ev,f,t,s=self.fixture();self.assertFalse(depth_loss_causality(ev,{},t,s))
 def test_unarmed(self):ev,f,t,s=self.fixture();f['armed']=False;self.assertFalse(depth_loss_causality(ev,f,t,s))
 def test_no_physical_feedback(self):ev,f,t,s=self.fixture();ev[0]['payload']['integration_steps']=0;self.assertFalse(depth_loss_causality(ev,f,t,s))
 def test_late(self):ev,f,t,s=self.fixture();self.assertFalse(depth_loss_causality(ev,f,t,10.351))
 def test_depth_not_stale(self):ev,f,t,s=self.fixture();ev[2]['ros_now']=100.1;self.assertFalse(depth_loss_causality(ev,f,t,s))
 def test_depth_continues(self):ev,f,t,s=self.fixture();ev.insert(2,dict(kind='depth',accepted=True,stamp=100.1,monotonic=10.1,payload=dict(source_odom_stamp=100.1)));self.assertFalse(depth_loss_causality(ev,f,t,s))
 def test_mismatched_pose(self):ev,f,t,s=self.fixture();ev[1]['payload']['source_odom_stamp']=99.;self.assertFalse(depth_loss_causality(ev,f,t,s))
 def test_not_exited(self):ev,f,t,s=self.fixture();t['terminal_state']='STOP_NOT_CONFIRMED';self.assertFalse(depth_loss_causality(ev,f,t,s))
 def test_nonfinite(self):ev,f,t,s=self.fixture();ev[2]['ros_now']=float('inf');self.assertFalse(depth_loss_causality(ev,f,t,s))
