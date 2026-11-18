"""Run actual production callback AST with non-ROS message/journal stubs."""
import ast,json,unittest,types
from contextlib import nullcontext
from pathlib import Path
from query_minco.integration.depth_minco_t2_v1.engine import Engine
from query_minco.integration.depth_minco_t2_v1.ros_bridge import SupervisorBridge
class Tests(unittest.TestCase):
 def callback(self,name,ns):
  main=next(x for x in ast.parse(Path('ros/g4cp7b_fix2_v1/supervisor.py').read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='main')
  fn=next(x for x in main.body if isinstance(x,ast.FunctionDef) and x.name==name)
  wrapper=ast.parse('def _factory():\n command_counter=0\n').body[0]
  wrapper.body.extend([fn,ast.Return(value=ast.Name(id=name,ctx=ast.Load()))])
  tree=ast.fix_missing_locations(ast.Module(body=[wrapper],type_ignores=[]))
  exec(compile(tree,'production_callback','exec'),ns)
  return ns['_factory']()
 def test_goal_request_without_pva_is_not_direct_control(self):
  b=SupervisorBridge.__new__(SupervisorBridge);b.engine=Engine(lambda now:None,clock=lambda:100.)
  r=types.SimpleNamespace(mode='ARMED',samples={'odom':(100.,),'imu':(100.,)})
  last=dict(odom=dict(p=[0,0,0],v=[0,0,0],q=[0,0,0,1]),imu=dict(a_world=[0,0,0]))
  errors=[];logs=[]
  ns=dict(journal=types.SimpleNamespace(callback=lambda:nullcontext(True)),mode='RUNNING',
          json=json,rospy=types.SimpleNamespace(Time=types.SimpleNamespace(now=lambda:types.SimpleNamespace(to_sec=lambda:100.))),
          last=last,forest_bridge=b,r=r,abort=errors.append,log=lambda *a:logs.append(a))
  cb=self.callback('request',ns)
  cb(types.SimpleNamespace(data=json.dumps(dict(kind='forest_goal',frame='world',stamp=100.,goal=[1,0,0]))))
  self.assertFalse(errors);self.assertEqual(len(logs),1)
  self.assertEqual(b.engine.goal.tolist(),[1,0,0])
  self.assertIsNone(b.engine.command(100.))
 def test_raw_so3_does_not_bypass_empty_proof(self):
  e=Engine(lambda now:None,clock=lambda:100.)
  r=types.SimpleNamespace(mode='ARMED',forward_allowed=lambda x:True,forwarded=lambda x:None)
  sent=[];errors=[]
  vector=types.SimpleNamespace(x=0.,y=0.,z=9.6)
  m=types.SimpleNamespace(header=types.SimpleNamespace(stamp=types.SimpleNamespace(to_sec=lambda:100.,to_nsec=lambda:100000000000),frame_id='world'),
      force=vector,orientation=types.SimpleNamespace(x=0,y=0,z=0,w=1),kR=[],kOm=[],aux=types.SimpleNamespace(enable_motors=True,current_yaw=0.,kf_correction=0.,angle_corrections=[0.,0.],use_external_yaw=False))
  ns=dict(journal=types.SimpleNamespace(callback=lambda:nullcontext(True)),mode='RUNNING',
       r=r,accept=lambda *x:True,forest_bridge=types.SimpleNamespace(engine=e),
       rospy=types.SimpleNamespace(Time=types.SimpleNamespace(now=lambda:types.SimpleNamespace(to_sec=lambda:100.))),
       final=types.SimpleNamespace(publish=sent.append),abort=errors.append,log=lambda *a:None)
  self.callback('raw',ns)(m)
  self.assertFalse(sent);self.assertEqual(errors,['T2_NO_CURRENT_PROOF'])
 def test_imu_alignment_rejected(self):
  b=SupervisorBridge.__new__(SupervisorBridge);b.engine=Engine(lambda now:None,clock=lambda:100.)
  last=dict(odom=dict(p=[0,0,0],v=[0,0,0],q=[0,0,0,1]),imu=dict(a_world=[0,0,0]))
  with self.assertRaises(ValueError):b.state(last,100.,99.9)

 def test_f0_cannot_leave_certified_tracking_tube(self):
  import numpy as np
  errors=[]
  ns=dict(journal=types.SimpleNamespace(callback=lambda:nullcontext(True)),mode='RUNNING',json=json,np=np,
      rospy=types.SimpleNamespace(Time=types.SimpleNamespace(now=lambda:types.SimpleNamespace(to_sec=lambda:100.))),
      last=dict(odom=dict(p=[.16,0,0],v=[0,0,0]),imu=dict(a_world=[0,0,0])),
      forest_bridge=None,r=types.SimpleNamespace(mode='ARMED'),a=types.SimpleNamespace(forest=False),
      gate=types.SimpleNamespace(check=lambda *a,**k:np.zeros((3,3))),abort=errors.append)
  self.callback('request',ns)(types.SimpleNamespace(data=json.dumps(dict(kind='hold',frame='world',stamp=100.,pva=[[0,0,0]]*3))))
  self.assertTrue(any('CERTIFIED_TRACKING_ENVELOPE' in x for x in errors))
