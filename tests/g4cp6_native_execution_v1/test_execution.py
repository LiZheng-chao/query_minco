import unittest,subprocess,sys
from pathlib import Path
import xml.etree.ElementTree as ET
from query_minco.integration.yopo_control_supervisor_v1.lease import OwnedProcessStop
class Tests(unittest.TestCase):
 def test_native_controller_gains_preserved(self):
  old=ET.parse('/home/lizheng/YOPO/Controller/src/so3_quadrotor_simulator/launch/simulator_position_control.launch').getroot().find("node[@pkg='nodelet']")
  new=ET.parse('ros/g4cp6_native_execution_v1/native_position_controller.launch').getroot().find('node')
  attrs=lambda n:{e.attrib['name']:e.attrib.get('value') for e in n.findall('param') if 'init_state_' not in e.attrib['name']}
  self.assertEqual(attrs(old),attrs(new))
  self.assertEqual(old.attrib['args'],new.attrib['args'])
 def test_owned_real_process_stop(self):
  p=subprocess.Popen([sys.executable,'-B','-c','import time;time.sleep(60)'],start_new_session=True)
  try:
   r=OwnedProcessStop(p).stop('NONPHYSICAL_UNIT_TEST')
   self.assertEqual(r['terminal_state'],'OWNED_PROCESS_TERMINATED');self.assertIsNotNone(p.poll())
  finally:
   if p.poll() is None:p.kill();p.wait()
 def test_fixed_target_not_rolling(self):
  s=Path('ros/g4cp2_execution_v4/reference.py').read_text()
  self.assertIn('if anchor is None:anchor=pos.copy();target=',s)
  self.assertIn("target=anchor+np.array([.4",s)
 def test_new_owner_no_forest_bypass(self):
  s=Path('experiments/g4cp6_native_execution_v1/manual.py').read_text()
  compile(s,'owner','exec');self.assertIn("choices=['C','D']",s)
  self.assertIn("predecessor(a.prior,'C_CAMPAIGN'",s);self.assertIn("len(set(h['ack_stamps']))<5",s)
 def test_native_imu_producer_fixed_in_overlay(self):
  s=Path('research_outputs/g4cp1_startup_control_20261111_v1/build_equilibrium_v3/gated_host.cpp').read_text()
  self.assertIn('imu.header.stamp=tnow;',s.replace(' ',''))
