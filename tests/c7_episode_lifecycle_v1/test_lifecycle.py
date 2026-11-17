"""NONPHYSICAL: actual rosmaster registry and actual launcher AST with owned fakes."""
import ast,types,tempfile,json,hashlib,signal,subprocess,sys,unittest
from pathlib import Path
sys.path.insert(0,'/opt/ros/noetic/lib/python3/dist-packages')
from rosmaster.registrations import RegistrationManager
class Registry(unittest.TestCase):
 def manager(self):
  tasks=[];pool=types.SimpleNamespace(queue_task=lambda *args:tasks.append(args))
  return RegistrationManager(pool),tasks
 def test_old_shared_master_reproduces_wrong_target(self):
  m,t=self.manager();m._register_node_api('/p2_deterministic_reference','http://127.0.0.1:36369/')
  # Reference is SIGKILLed without unregistering; its old URI is reused by Controller.
  m._register_node_api('/p6_native_position_controller','http://127.0.0.1:36369/')
  m._register_node_api('/p2_deterministic_reference','http://127.0.0.1:40173/')
  self.assertEqual(t[0][0],'http://127.0.0.1:36369/')
  self.assertEqual(t[0][2][1],'/p2_deterministic_reference')
 def test_fresh_master_has_no_cross_episode_shutdown(self):
  old,_=self.manager();old._register_node_api('/p2_deterministic_reference','http://127.0.0.1:36369/')
  fresh,t=self.manager();fresh._register_node_api('/p6_native_position_controller','http://127.0.0.1:36369/')
  fresh._register_node_api('/p2_deterministic_reference','http://127.0.0.1:40173/')
  self.assertEqual(t,[])
class Launcher(unittest.TestCase):
 def exercise(self,residual=False,episode_fail=False,identity_fail=False,stage='C'):
  tree=ast.parse(Path('experiments/stop_latency_fix_v1/manual.py').read_text())
  main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
  i=next(i for i,n in enumerate(main.body) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='case')
  nodes=main.body[i-2:i+2]
  active=set();procs=[];timeline=[];clock=[0.]
  class Proc:
   def __init__(self,pid):self.pid=pid;self.done=False
   def poll(self):return 0 if self.done else None
   def wait(self,timeout):
    timeline.append(('master_exit',self.pid));self.done=True;active.discard(self.pid);return 0
  def launch(argv,**kwargs):
   self.assertFalse(active,'next master started while owned process remained')
   p=Proc(4000+len(procs)*10);p.logs=(kwargs['stdout'],kwargs['stderr']);procs.append(p);active.add(p.pid)
   if residual:active.add(p.pid+1)
   timeline.append(('master_start',p.pid));return p
  def family(pid):
   return [dict(pid=x,start_ticks=str(x)) for x in sorted(active) if x in [pid,pid+1]]
  def alive(row):return row['pid'] in active and not(identity_fail and row['pid']%10==0)
  def killpg(pid,sig):self.assertIn(pid,active);timeline.append(('signal_group',pid));return None
  def kill(pid,sig):self.assertIn(pid,active);active.discard(pid);timeline.append(('owned_child_killed',pid))
  class Sock:
   def __enter__(self):return self
   def __exit__(self,*a):pass
   def setsockopt(self,*a):pass
   def bind(self,*a):pass
  def mono():clock[0]+=.02;return clock[0]
  def episode(out,*args):
   out.mkdir();d=dict(status='FAIL' if episode_fail else 'PASS');(out/'RESULT.json').write_text(json.dumps(d));timeline.append(('episode',out.name));return d
  rpc=types.SimpleNamespace(getPid=lambda *a:[1,'',procs[-1].pid],setParam=lambda *a:None)
  with tempfile.TemporaryDirectory() as td:
   ns=dict(out=Path(td),env=dict(ROS_MASTER_URI='mock-no-network'),a=types.SimpleNamespace(stage=stage),CASES=['request_timeout','depth_loss','odom_rollback','nan_control','reference_exit','controller_exit','truth_failure'],socket=types.SimpleNamespace(socket=Sock,SOL_SOCKET=1,SO_REUSEADDR=2),subprocess=types.SimpleNamespace(Popen=launch,TimeoutExpired=subprocess.TimeoutExpired),parent_death_preexec=lambda:None,family=family,alive=alive,os=types.SimpleNamespace(getpgid=lambda p:p,killpg=killpg,kill=kill),signal=signal,time=types.SimpleNamespace(time=lambda:100.,monotonic=mono,sleep=lambda t:None),R=Path.cwd(),xmlrpc=types.SimpleNamespace(client=types.SimpleNamespace(ServerProxy=lambda uri:rpc)),write=lambda p,d:p.write_text(json.dumps(d)),sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest(),episode=episode,binding={},scene={})
   if identity_fail:
    with self.assertRaisesRegex(RuntimeError,'OWNERSHIP_CHANGED'):
     exec(compile(ast.Module(body=nodes,type_ignores=[]),'actual_launcher','exec'),ns)
    self.assertFalse(any(x[0]=='signal_group' for x in timeline))
   else:exec(compile(ast.Module(body=nodes,type_ignores=[]),'actual_launcher','exec'),ns)
   self.assertTrue(all(all(f.closed for f in p.logs) for p in procs))
   ns['master_records']=[json.loads(p.read_text()) for p in Path(td).glob('master_*/PROCESS_RESULT.json')]
   return ns,procs,timeline
 def test_seven_sequential_masters(self):
  n,p,t=self.exercise();self.assertEqual(len(p),7);self.assertEqual(n['master_codes'],[0]*7)
  self.assertEqual([x[0] for x in t].count('master_exit'),7)
 def test_s_one_master(self):
  n,p,t=self.exercise(stage='S');self.assertEqual(len(p),1)
 def test_failed_episode_no_next_start(self):
  n,p,t=self.exercise(episode_fail=True);self.assertEqual(len(p),1)
 def test_owned_residual_blocks_next_episode(self):
  n,p,t=self.exercise(residual=True);self.assertEqual(len(p),1);self.assertEqual(n['code'],-1);self.assertEqual(n['master_records'][0]['actual_exit_code'],0);self.assertFalse(n['master_records'][0]['master_cleanup_pass'])
 def test_identity_change_never_signals(self):self.exercise(identity_fail=True)
