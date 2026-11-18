"""Synthetic chain format only; real Controller probe is a separate artifact."""
import unittest,copy,importlib.util
from pathlib import Path
import numpy as np
from experiments.sim_dev_oracle_v1.evidence import assess
from experiments.sim_dev_oracle_v1.guardian import Guardian
from query_minco.integration.yopo_execution_v4.trajectory import build
class Empty:
 map_id='SYNTHETIC_GT_NOT_FLIGHT'
 def clearance(self,p):return 10.
def fixture():
 spec=importlib.util.spec_from_file_location('old_fixture','tests/f0_controller_consumption_v1/test_evidence.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);events,refs,logs,identity=m.fixture();guardian=Guardian(Empty(),m.NONCE)
 for i in range(2):
  now=101.+i;cmd=events[3*i]['payload'];pva=np.array(cmd['pva']);plan=build(pva[0],pva[1],pva[2],[0,0,0,1],pva[0]+[.1,0,0],now);proof=guardian.check(plan.record,dict(p=pva[0],v=pva[1],a=pva[2]),now,now,i)
  cmd.update(mode='SIM_DEV_ORACLE',stamp=now,source_stamp=now,plan_id=plan.id,map_epoch=i,snapshot_id=proof['map_id'],certificate_id=proof['identity'],acceptance_id=proof['identity'],authorization=dict(type='SIM_DEV_ORACLE_PREFIX_BRAKE',geometry='SIM_DEV_ORACLE_PREFIX_BRAKE',oracle=proof,plan_id=plan.id,certificate_expiry=proof['expires']))
  refs[i]=dict(kind='plan',plan=plan.record,plan_id=plan.id);events[3*i+1]['payload']['observed_authority']=copy.deepcopy(cmd)
 return events,refs,logs,identity,m.NONCE
def check(f):
 e,r,l,i,n=f;return assess(e,0,4,r,l,n,i)
class OracleEvidence(unittest.TestCase):
 def test_synthetic_two_causal_chains(self):
  d=check(fixture());self.assertEqual(d['status'],'PASS',d);self.assertFalse(d['strict_depth_qualified'])
 def test_missing_plant_does_not_pass(self):
  f=fixture();f[0].pop();self.assertNotEqual(check(f)['status'],'PASS')
 def test_wrong_plan_does_not_pass(self):
  f=fixture();f[0][3]['payload']['plan_id']='BAD';self.assertNotEqual(check(f)['status'],'PASS')
 def test_renewed_forward_lease_still_requires_scene(self):
  f=fixture();f[0][1]['payload']['observed_authority']['snapshot_id']='BAD_SCENE';self.assertNotEqual(check(f)['status'],'PASS')
 def test_bad_nonce_does_not_pass(self):
  f=fixture();f[0][0]['payload']['authorization']['oracle']['nonce']='bad';self.assertNotEqual(check(f)['status'],'PASS')
 def test_expired_consumption_does_not_pass(self):
  f=fixture();f[2][1]['received_ros']+=1;self.assertNotEqual(check(f)['status'],'PASS')
 def test_missing_controller_consumption_does_not_pass(self):
  f=fixture();f[2][1]['kind']='MISSING';self.assertNotEqual(check(f)['status'],'PASS')
if __name__=='__main__':unittest.main(verbosity=2)
