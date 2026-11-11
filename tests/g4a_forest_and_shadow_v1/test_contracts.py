"""No-update safety,split,authorization and live shadow contract regressions."""
from pathlib import Path
import sys,json,tempfile,unittest,io
from unittest.mock import patch
sys.path.extend(['/opt/ros/noetic/lib/python3/dist-packages','/home/lizheng/YOPO/Controller/devel/lib/python3/dist-packages'])
import numpy as np,torch
from scipy.spatial.transform import Rotation
from quadrotor_msgs.msg import PositionCommand
from query_minco.integration.yopo_shadow_v1.sensor_contract import observation,rotation,sample_world,OPTICAL_TO_BODY
from query_minco.integration.yopo_shadow_v1.command_contract import CommandGate,position_command
from query_minco.training.k1_forest_v1 import trainer
from query_minco.training.k1_forest_v1.data import load
from query_minco.evaluation.validation import numerical_reference,evaluate
from query_minco.audit.run_provenance_v1 import AtomicAttemptLedger
R=Path(__file__).resolve().parents[2];O=R/'research_outputs/g4a_forest_shadow_20261111_v1'
class Contracts(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  torch.set_num_threads(1);cls.spec_path=O/'K1_FOREST_TRAINING_RUNSPEC_v2.json';cls.spec=trainer.verify(cls.spec_path)
 def setUp(self):
  self.guard=patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('NO_OPTIMIZER_TEST_UPDATE'));self.guard.start()
 def tearDown(self):self.guard.stop()
 def test_dynamic_state_and_optical_world_frame(self):
  q=Rotation.from_euler('xyz',[.2,-.3,.4]).as_quat();rot=rotation(q);p=np.array([1.,2,3]);v=np.array([.3,-.2,.1]);a=np.array([.2,.1,-.3]);g=p+np.array([2,1,-.2])
  x,s,_=observation(np.full((90,160),2,np.float32),q,v,a,p,g,[80.,0,80.,0,80.,45.,0,0,1.])
  np.testing.assert_allclose(s[0,:3]@rot.T,v,atol=1e-7);np.testing.assert_allclose(s[0,3:6]@rot.T,a,atol=1e-7)
  np.testing.assert_allclose(s[0,6:9]@rot.T,g-p,atol=1e-6);np.testing.assert_allclose(s[0,9:12]@rot.T,[0,0,-1],atol=1e-7)
  ray=np.array([.2,-.1,2]);world=p+rot@OPTICAL_TO_BODY@ray
  np.testing.assert_allclose(OPTICAL_TO_BODY.T@rot.T@(world-p),ray,atol=1e-12)
  self.assertFalse(np.allclose(rot,np.eye(3)));self.assertEqual(x.shape,(1,2,128,160))
 def test_invalid_observation_and_no_return(self):
  K=[80.,0,80.,0,80.,45.,0,0,1.];d=np.full((90,160),2,np.float32);d[0,0]=20;d[0,1]=np.nan
  x,_,_=observation(d,[0,0,0,1],[0,0,0],[0,0,0],[0,0,0],[2,0,0],K)
  self.assertEqual(float(x[0,1,19,0]),0);self.assertEqual(float(x[0,0,19,1]),0);self.assertTrue((x[:,:,:19]==0).all())
  for q in [[0,0,0,0],[np.nan,0,0,1],[0,0,0,2]]:
   with self.assertRaises(ValueError):rotation(q)
  with self.assertRaises(ValueError):observation(np.full((90,160),20,np.float32),[0,0,0,1],[0]*3,[0]*3,[0]*3,[2,0,0],K)
  with self.assertRaises(ValueError):observation(d,[0,0,0,1],[0]*3,[np.nan,0,0],[0]*3,[2,0,0],K)
 def fixture(self):
  dt=np.full(3,2/3);head=np.array([[0,0,0],[.2,0,0],[0,0,0]],float);tail=np.array([[.4,0,0],[0,0,0],[0,0,0]],float)
  # Actual-PVA unique quintic samples,then original numerical MINCO.
  from query_minco.trajectory.state_goal_reference_v1 import reference_raw
  from query_minco.training.physics_only_v1.decoder import FixedTimeDecoder
  from query_minco.training.physics_only_v1_2.spec import PhysicsV12Spec
  s=torch.tensor([[.2,0,0,0,0,0,.4,0,0,0,0,-1]],dtype=torch.float32)
  p=FixedTimeDecoder(PhysicsV12Spec(phase='A',aggregation='foundation'),num_candidates=1)(reference_raw(s,'state_goal_quintic'),s)
  c=numerical_reference(p.head_pva[0,0].numpy(),p.tail_pva[0,0].numpy(),p.inner_pts[0,0].numpy(),dt)
  return c,dt,head
 def test_command_world_pva_and_actual_message_serialization(self):
  c,d,head=self.fixture();rot=Rotation.from_euler('xyz',[.1,.2,.3]).as_matrix();origin=np.array([1.,2.,3.]);actual=sample_world(c,d,origin,rot,0)
  np.testing.assert_allclose(actual[0],origin,atol=1e-12);np.testing.assert_allclose(actual[1],rot@head[1],atol=1e-8)
  target=CommandGate().sample(c,d,origin,rot,10,10,10,10,actual)
  import rospy
  msg=position_command(PositionCommand,target,rospy.Time.from_sec(10),.2,1);stream=io.BytesIO();msg.serialize(stream);copy=PositionCommand().deserialize(stream.getvalue())
  self.assertEqual(copy.header.frame_id,'world');self.assertEqual(copy.trajectory_flag,PositionCommand.TRAJECTORY_STATUS_READY)
  np.testing.assert_allclose([copy.velocity.x,copy.velocity.y,copy.velocity.z],actual[1],atol=1e-12)
  self.assertEqual(copy._md5sum,'4712f0609ca29a79af79a35ca3e3967a')
 def test_stale_nan_switch_and_latched_abort(self):
  c,d,head=self.fixture();origin=np.zeros(3);rot=np.eye(3)
  gate=CommandGate()
  with self.assertRaisesRegex(ValueError,'ABORT'):gate.sample(c,d,origin,rot,10,11,10,10,head)
  with self.assertRaisesRegex(ValueError,'ABORT_LATCHED'):gate.sample(c,d,origin,rot,10,10,10,10,head)
  for bad in [np.full_like(c,np.nan),np.zeros((2,3,6))]:
   with self.assertRaises(ValueError):CommandGate().sample(bad,d,origin,rot,10,10,10,10,head)
  with self.assertRaises(ValueError):CommandGate().sample(c,d,origin,np.full((3,3),np.nan),10,10,10,10,head)
  with self.assertRaisesRegex(ValueError,'SWITCH_DISCONTINUITY'):CommandGate().sample(c,d,origin,rot,10,10,10,10,head,head+1)
  with self.assertRaisesRegex(ValueError,'STATE_INVALID'):CommandGate().sample(c,d,origin,rot,10,10,10,10,np.full_like(head,np.nan))
 def test_exact_approval_default_forbidden_and_output_claim(self):
  s=self.spec;output=s['formal_run_directories']['A_reasonable_seed20261104']
  with self.assertRaises(PermissionError):trainer.authorize(self.spec_path,None,output,20261104)
  with tempfile.TemporaryDirectory(dir=O) as tmp:
   p=Path(tmp)/'approval.json';p.write_text(json.dumps(dict(decision='APPROVE_EXACT_RUNSPEC',runspec_sha256='0'*64)))
   with self.assertRaises(PermissionError):trainer.authorize(self.spec_path,p,output,20261104)
   p.write_text(json.dumps(dict(decision='APPROVE_EXACT_RUNSPEC',runspec_sha256=trainer.sha(self.spec_path),run_id='A_reasonable_seed20261104',researcher_statement='SYNTHETIC_AUTHORIZATION_TEST_ONLY',resume_sha256=None)))
   trainer.authorize(self.spec_path,p,output,20261104)
   with self.assertRaises(PermissionError):trainer.authorize(self.spec_path,p,output+'_wrong',20261104)
   with self.assertRaises(FileExistsError):trainer.claim(tmp)
 def test_dev_and_seed_family_leakage_boundaries(self):
  for row in self.spec['dev']:
   with self.assertRaises(PermissionError):load(row,'train')
  self.assertEqual(len(self.spec['train']),16);self.assertEqual(len(self.spec['dev']),12)
  old={3101,3102,3201,3202,3301,3302};tr={r['map_seed'] for r in self.spec['train'] if r['kind']=='forest'};dv={r['map_seed'] for r in self.spec['dev'] if r['kind']=='forest'}
  self.assertFalse(tr&dv);self.assertFalse((tr|dv)&old)
 def test_real_preflight_and_two_process_resume_evidence(self):
  pre=json.loads((O/'preflight001/RESULT.json').read_text());self.assertEqual(pre['status'],'PASS');self.assertEqual(len(pre['train_backward']),16)
  self.assertEqual(pre['model_sha_before'],pre['model_sha_after']);self.assertEqual(pre['evaluation']['dev_backward_calls'],0)
  a=json.loads((O/'engineering_step001/RESULT.json').read_text());b=json.loads((O/'engineering_step002/RESULT.json').read_text());ledger=json.loads((O/'ENGINEERING_ATOMIC_LEDGER.json').read_text())
  self.assertEqual(ledger['attempted'],2);self.assertEqual(ledger['completed'],2);self.assertEqual(ledger['pending'],[])
  self.assertEqual(a['report']['model_sha_after'],b['report']['model_sha_before']);self.assertEqual(b['resume']['status'],'PASS')
  for e in [a,b]:
   self.assertEqual(e['report']['dead_sha_before'],e['report']['dead_sha_after']);self.assertGreater(e['report']['live_head_update'],0)
   self.assertTrue(all(v['nonzero']==0 for v in e['report']['isolation']['state_after'].values()))
  with self.assertRaises(PermissionError):AtomicAttemptLedger(O/'ENGINEERING_ATOMIC_LEDGER.json',2).reserve()
 def test_observed_ros_delivery_and_dynamic_geometry(self):
  attempt=O/'shadow_attempt003';r=json.loads((attempt/'planner/RESULT.json').read_text());receiver=json.loads((attempt/'receiver_data/RESULT.json').read_text())
  self.assertEqual(r['frames'],30);self.assertFalse(r['control_published']);self.assertEqual(receiver['status'],'PASS')
  for m in ['fixed_forward','state_goal_quintic','K1_step128']:self.assertGreaterEqual(len(receiver['records']['/g4a/shadow/path/'+m]),10)
  for row in r['records']:
   self.assertEqual(row['depth_odom_delta_s'],0);self.assertEqual(row['camera_delta_s'],0)
  self.assertEqual(json.loads((O/'DYNAMIC_FRAME_AUDIT.json').read_text())['status'],'PASS')
if __name__=='__main__':unittest.main(verbosity=2)
