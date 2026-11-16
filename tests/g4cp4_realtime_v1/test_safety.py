import unittest,tempfile,json,hashlib
from pathlib import Path
import numpy as np
from query_minco.integration.observed_space_v2.core import ObservedMap,AsyncCertificates,Prepared
from query_minco.integration.observed_space_v2.bootstrap import validate
from query_minco.integration.yopo_execution_v4.trajectory import build
class Safety(unittest.TestCase):
 def test_measured_volume_is_bounded_expiring_and_never_arm(self):
  with tempfile.TemporaryDirectory() as folder:
   raw=Path(folder)/'measurement';raw.write_bytes(b'synthetic test only')
   d=dict(device_id='SYNTHETIC_ONLY',calibration_sha256='0'*64,measurement_files=[dict(path=str(raw),sha256=hashlib.sha256(raw.read_bytes()).hexdigest())],
    center_world=[0,0,0],measured_radius_m=.4,error_bound_m=.02,valid_from=10,valid_until=20,scene_id='s',reviewer='synthetic',scope='INITIAL_BALL_ONLY')
   p=Path(folder)/'evidence.json'
   def check(now=11,point=[0,0,0]):
    p.write_text(json.dumps(d));return validate(p,now,'s',point)
   self.assertFalse(check()['motion_authority'])
   for now,point in [(21,[0,0,0]),(11,[.09,0,0])]:
    with self.assertRaises(ValueError):check(now,point)
   d['scope']='CORRIDOR'
   with self.assertRaises(ValueError):check()
   d['scope']='INITIAL_BALL_ONLY';raw.write_bytes(b'changed')
   with self.assertRaises(ValueError):check()
 def test_async_valid_then_epoch_and_expiry_revoke(self):
  now=[10.]
  m=ObservedMap();pose=dict(p=[-1.2,0,0],q=[0,0,0,1])
  m.add(np.full((90,160),5,np.float32),pose,10,10,10)
  p=build(np.zeros(3),np.zeros(3),np.zeros(3),[0,0,0,1],np.array([.4,0,0]),10)
  w=AsyncCertificates(m,clock=lambda:now[0])
  try:
   w.submit(p);proof=w.future.result(timeout=2)
   self.assertIsNotNone(w.consume(p.id));self.assertIsNone(w.consume('wrong-plan'))
   now[0]=10.21;self.assertIsNone(w.consume(p.id))
   now[0]=10.1;m.add(np.full((90,160),1,np.float32),pose,10.1,10.1,10.1)
   self.assertIsNone(w.consume(p.id))
  finally:w.close()
