import unittest,copy,math
from query_minco.integration.yopo_startup_v3.readiness import Readiness,FRAMES
class Handshake(unittest.TestCase):
 def fixture(self):
  return dict(odom=dict(frame='world',p=[0,0,2],v=[0,0,0],q=[0,0,0,1]),imu=dict(frame='world',a_world=[0,0,0]),
   depth=dict(frame='camera_link',stamp=10.,source_odom_stamp=10.,valid_count=100),
   request=dict(frame='world',pva=[[0,0,2],[0,0,0],[0,0,0]]),
   raw=dict(frame='/quadrotor',force_world=[0,0,9.6138],q=[0,0,0,1],kR=[1.5,1.5,1],kOm=[.13,.13,.1],motors_enabled=True),
   plant=dict(frame='world',phase='HELD',integration_steps=0,mass=.98,gravity=9.81,final_command_stamp=0.,final_count=0,command_valid=False,motor_initial_state='zero',kf=8.98132e-9,motor_tau_s=1/30,rpm_actual=[0.,0.,0.,0.]))
 def feed(self,r,stamp,changes=None,omit=None):
  data=self.fixture()
  if changes:
   for k,v in changes.items():data[k].update(v)
  for k,d in data.items():
   if k==omit:continue
   if k=='depth':d['stamp']=stamp;d['source_odom_stamp']=stamp
   r.observe(k,stamp,stamp,d)
 def prepare(self,arm=True):
  r=Readiness(True,arm)
  for i in range(5):self.feed(r,10+i*.01)
  self.assertEqual(r.tick(10.04),'PREPARE')
  self.feed(r,10.05,dict(plant=dict(phase='PREPARE')))
  return r
 def acknowledge(self,r):
  for i in range(5):
   t=10.06+i*.01;r.forwarded(t)
   self.feed(r,t,dict(plant=dict(phase='PREPARE' if i<4 else 'ACK',command_valid=True,final_command_stamp=t,final_count=i+1)))
   r.tick(t)
  self.assertEqual(r.mode,'ACK')
 def test_explicit_equilibrium_initial_condition(self):
  rpm=math.sqrt(.98*9.81/(4*8.98132e-9));r=Readiness(True,True,'equilibrium_hover')
  for i in range(5):self.feed(r,10+i*.01,dict(plant=dict(motor_initial_state='equilibrium_hover',rpm_actual=[rpm]*4)))
  self.assertEqual(r.tick(10.04),'PREPARE')
  r=Readiness(True,True,'equilibrium_hover');self.feed(r,10.);self.assertEqual(r.mode,'ABORT')
 def test_duplicate_observations_cannot_count(self):
  r=Readiness(True)
  for _ in range(5):self.feed(r,10.)
  self.assertEqual(r.tick(10.),'HELD');self.assertTrue(all(v==1 for v in r.counts.values()));self.assertFalse(r.forward_allowed(10.))
 def test_duplicate_cannot_refresh_liveness(self):
  r=Readiness(True);self.feed(r,10.);self.feed(r,10.);self.assertEqual(r.tick(10.201),'ABORT')
 def test_backward_any_stream(self):
  for k in FRAMES:
   r=Readiness(True);self.feed(r,10.02);d=self.fixture()[k]
   r.observe(k,10.01,10.02,d);self.assertEqual(r.mode,'ABORT',k)
 def test_zero_imu_missing_depth_and_request(self):
  for omit in ['depth','request']:
   r=Readiness(True)
   for i in range(5):self.feed(r,10+i*.01,omit=omit)
   self.assertEqual(r.tick(10.04),'HELD');self.assertFalse(r.forward_allowed(10.04))
  r=Readiness(True);r.observe('imu',0.,10.,self.fixture()['imu']);self.assertEqual(r.mode,'ABORT')
 def test_explicit_authority(self):
  r=Readiness()
  for i in range(5):self.feed(r,10+i*.01)
  self.assertEqual(r.tick(10.04),'HELD')
 def test_ordered_full_protocol(self):
  r=self.prepare();self.assertTrue(r.forward_allowed(10.05));self.acknowledge(r)
  self.feed(r,10.11,dict(plant=dict(phase='ARMED',integration_steps=10,command_valid=True,final_count=6,final_command_stamp=10.10)))
  self.assertEqual(r.tick(10.11),'ARMED')
  self.assertEqual([x['after'] for x in r.transitions],['PREPARE','ACK','ARMED'])
 def test_no_final_ack_or_replayed_ack(self):
  r=self.prepare()
  for i in range(5):
   t=10.06+i*.01;self.feed(r,t,dict(plant=dict(phase='ACK' if i==4 else 'PREPARE',final_count=5,command_valid=False,motor_initial_state='zero',kf=8.98132e-9,motor_tau_s=1/30,rpm_actual=[0.,0.,0.,0.])))
   r.tick(t)
  self.assertEqual(r.mode,'PREPARE')
  r=self.prepare();r.forwarded(10.06)
  for i in range(5):
   t=10.06+i*.01;self.feed(r,t,dict(plant=dict(phase='PREPARE',final_count=5,command_valid=True,final_command_stamp=10.06)));r.tick(t)
  self.assertEqual(r.mode,'PREPARE');self.assertEqual(len(r.acks),1)
 def test_forged_ack(self):
  r=self.prepare();self.feed(r,10.06,dict(plant=dict(phase='PREPARE',command_valid=True,final_command_stamp=10.06,final_count=5)));self.assertEqual(r.tick(10.06),'ABORT')
 def test_unauthorized_arm_and_step_before_ready(self):
  r=self.prepare(False);self.acknowledge(r);self.feed(r,10.11,dict(plant=dict(phase='ARMED',integration_steps=1)))
  self.assertEqual(r.mode,'ABORT')
  r=Readiness(True);self.feed(r,10.,dict(plant=dict(phase='ARMED',integration_steps=1)));self.assertEqual(r.mode,'ABORT')
 def test_phase_skip(self):
  r=Readiness(True);self.feed(r,10.);self.feed(r,10.01,dict(plant=dict(phase='ACK')));self.assertEqual(r.mode,'ABORT')
 def test_frames_all_streams(self):
  for k in FRAMES:
   r=Readiness(True);d=self.fixture()[k];d['frame']='wrong';r.observe(k,10,10,d);self.assertEqual(r.mode,'ABORT',k)
 def test_gap_nan_force_wrong_mass(self):
  r=self.prepare();self.feed(r,10.30);self.assertEqual(r.mode,'ABORT')
  for key,value in [('raw',dict(force_world=[float('nan'),0,9.])),('plant',dict(mass=1.))]:
   r=Readiness(True);self.feed(r,10,dict([(key,value)]));self.assertEqual(r.mode,'ABORT')
 def test_fault_latch(self):
  r=self.prepare();r.abort('FAILURE');self.feed(r,10.06);self.assertEqual(r.tick(10.06),'ABORT');self.assertFalse(r.forward_allowed(10.06))
 def test_invalid_command_shape(self):
  r=Readiness(True);self.feed(r,10,dict(request=dict(pva=[[0,0,2],[.1,0,0],[0,0,0]])));self.assertEqual(r.mode,'ABORT')
 def test_force_direction_attitude(self):
  for changes in [dict(raw=dict(force_world=[0,0,-9.6138])),dict(raw=dict(q=[1,0,0,0]))]:
   r=Readiness(True)
   for i in range(5):self.feed(r,10+i*.01,changes)
   self.assertEqual(r.tick(10.04),'ABORT')
if __name__=='__main__':unittest.main(verbosity=2)
