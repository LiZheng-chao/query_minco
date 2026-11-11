"""No network/physics updates;real message evidence plus owned-OS-child safety tests."""
from pathlib import Path
import unittest,json,tempfile,subprocess,time,os,signal,sys,importlib.util
from unittest.mock import patch
import numpy as np,torch
from query_minco.integration.yopo_control_supervisor_v1.lease import Lease,OwnedProcessStop,parent_death_preexec
from query_minco.training.k1_forest_v1.trainer import verify
R=Path(__file__).resolve().parents[2];O=R/'research_outputs/g4b_formal128_and_control_20261111_v1'
class Contracts(unittest.TestCase):
 def setUp(self):
  self.p=patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('NEW_NETWORK_UPDATE_FORBIDDEN'));self.p.start()
 def tearDown(self):self.p.stop()
 def active(self):
  lease=Lease();lease.odom(1,10,10,[0,0,2],[0,0,0]);lease.depth(1,10,10,True);lease.raw(1,10,10,[0,0,9.6,0,0,0,1]);lease.command(1,10,10,[[.02,0,2],[0,0,0],[0,0,0]])
  self.assertEqual(lease.mode,'ACTIVE');return lease
 def test_formal128_identity_and_replayed_points(self):
  d=json.loads((O/'FORMAL128_IDENTITY_ACCEPTANCE.json').read_text());self.assertEqual(d['status'],'PASS');self.assertEqual(d['verified_updates'],128)
  self.assertEqual([r['step'] for r in d['points']],[0,8,32,64,128]);self.assertTrue(all(r['evaluation_exact'] for r in d['points']))
  self.assertEqual(d['new_network_updates'],0)
 def test_actual_native_consumer_all_six_faults(self):
  d=json.loads((O/'controller_consumption004/RESULT.json').read_text());self.assertEqual(d['status'],'PASS');self.assertTrue(d['no_quadrotor_physics_started'])
  self.assertEqual(len(d['cases']),6)
  for row in d['cases']:
   r=row['result'];self.assertTrue(r['actual_native_controller_consumed']);self.assertTrue(r['hold_message_verified']);self.assertTrue(r['actual_so3_old_reference_replaced'])
   self.assertTrue(r['gain_overrides_zero']);self.assertEqual(r['final_so3_after_abort'],0);self.assertLess(r['detection_delay_s'],.4)
 def test_timeout_latches_and_late_packets_do_not_rearm(self):
  x=self.active();x.tick(1.21);self.assertEqual(x.mode,'ABORT')
  x.command(1.22,10.22,10.22,[[0,0,2],[0,0,0],[0,0,0]]);self.assertEqual(x.mode,'ABORT');self.assertFalse(x.forward_allowed(1.22))
  y=self.active();y.command(1.21,10.21,10.21,[[0,0,2],[0,0,0],[0,0,0]])
  self.assertEqual(y.reason,'request_TIMEOUT_LATE')
 def test_nonfinite_clock_depth_and_switch_rejection(self):
  for kind in ['request','odom','controller']:
   x=self.active();x.touch(kind,1.01,10.01,10.01,[np.nan]);self.assertEqual(x.mode,'ABORT')
  x=self.active();x.depth(1.01,10.01,10.01,False);self.assertEqual(x.reason,'depth_INVALID')
  x=self.active();x.command(1.01,10.01,10.01,[[.2,0,2],[0,0,0],[0,0,0]]);self.assertEqual(x.reason,'SWITCH_DISCONTINUITY')
  x=self.active();x.odom(1.01,9.9,10.3,[0,0,2],[0,0,0]);self.assertEqual(x.mode,'ABORT')
 def test_one_meter_per_second_gate(self):
  x=self.active();x.command(1.01,10.01,10.01,[[0,0,2],[1.01,0,0],[0,0,0]]);self.assertEqual(x.reason,'request_LIMIT')
  x=self.active();x.odom(1.01,10.01,10.01,[0,0,2],[1.01,0,0]);self.assertEqual(x.reason,'MEASURED_SPEED')
 def test_owned_child_pause_and_terminate_real_exit(self):
  child=subprocess.Popen([sys.executable,'-B','-c','import time;time.sleep(60)'],start_new_session=True)
  result=OwnedProcessStop(child).stop('SYNTHETIC_OS_CHILD_TEST_NOT_PHYSICS')
  print('OWNED_OS_CHILD_STOP',json.dumps(result));self.assertTrue(result['stop_state_observed']);self.assertEqual(result['actual_exit_code'],-9)
 def test_guardian_death_kills_owned_child(self):
  with tempfile.TemporaryDirectory(dir=O) as d:
   record=Path(d)/'child.pid'
   code="from query_minco.integration.yopo_control_supervisor_v1.lease import parent_death_preexec;import subprocess,sys,time;from pathlib import Path;p=subprocess.Popen([sys.executable,'-B','-c','import time;time.sleep(60)'],start_new_session=True,preexec_fn=parent_death_preexec());Path("+repr(str(record))+").write_text(str(p.pid));time.sleep(60)"
   parent=subprocess.Popen([sys.executable,'-B','-c',code],cwd=R,start_new_session=True)
   pid=None
   try:
    for _ in range(300):
     if record.exists():pid=int(record.read_text());break
     time.sleep(.01)
    self.assertIsNotNone(pid);os.killpg(parent.pid,signal.SIGKILL);parent.wait()
    state=None
    for _ in range(300):
     p=Path('/proc/%d/stat'%pid)
     state=p.read_text().rsplit(')',1)[1].split()[0] if p.exists() else 'ABSENT'
     if state in ['Z','X','ABSENT']:break
     time.sleep(.01)
    print('GUARDIAN_DEATH_OS_CHILD_STATE',state);self.assertIn(state,['Z','X','ABSENT'])
   finally:
    if parent.poll() is None:os.killpg(parent.pid,signal.SIGKILL);parent.wait()
    if pid and Path('/proc/%d'%pid).exists():
     try:os.killpg(pid,signal.SIGKILL)
     except ProcessLookupError:pass
 def test_manual_physics_requires_actual_safety_evidence_and_explicit_flag(self):
  path=R/'experiments/g4b_formal_and_control_v1/run_manual_physics.py';spec=importlib.util.spec_from_file_location('g4b_manual_test',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  with tempfile.TemporaryDirectory(dir=O) as d:
   evidence=Path(d)/'fake.json';evidence.write_text(json.dumps(dict(status='PASS',actual_physics=False)))
   with self.assertRaises(PermissionError):module.verify_evidence(evidence,{})
   argv=['manual','--stage','track','--output',str(Path(d)/'DO_NOT_CREATE')]
   with patch.object(sys,'argv',argv),patch.object(subprocess,'Popen',side_effect=AssertionError('PHYSICS_MUST_NOT_LAUNCH')):
    with self.assertRaisesRegex(PermissionError,'MANUAL_PHYSICS'):module.main()
   self.assertFalse((Path(d)/'DO_NOT_CREATE').exists())
 def test_old_math_and_registered_training_sources_unchanged(self):
  verify(R/'research_outputs/g4a_forest_shadow_20261111_v1/K1_FOREST_TRAINING_RUNSPEC.json')
  entry=json.loads((O/'ENTRY_PROTECTION.json').read_text())
  import hashlib
  for file,h in entry['source_before'].items():self.assertEqual(hashlib.sha256(Path(file).read_bytes()).hexdigest(),h)
if __name__=='__main__':unittest.main(verbosity=2)
