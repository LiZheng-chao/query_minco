import unittest,copy,ast,json
from pathlib import Path
import numpy as np
from query_minco.integration.yopo_execution_v4.trajectory import build,digest
from query_minco.integration.f0_continuity_v1.execution_evidence import validate_observed_authorization
from experiments.forest_f1_v1.manual import historical_f0,qualification
class Contract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.plan=build(np.zeros(3),np.zeros(3),np.zeros(3),[0,0,0,1],[.28,0,0],100.)
    def envelope(self):
        p=self.plan;cert=digest(dict(plan=p.id,snapshot='snapshot',expiry=100.2))
        proof=dict(type='OBSERVED_SPACE_FULL_TUBE_V1',plan_id=p.id,map_epoch=2,snapshot_id='snapshot',certificate_expiry=100.2,certificate_id=cert,acceptance_id='acceptance',source_stamp=100.,evidence=dict(passed=True,epoch=2,snapshot='snapshot',valid_until=100.2))
        return dict(plan_id=p.id,map_epoch=2,snapshot_id='snapshot',source_stamp=100.,stamp=100.01,certificate_id=cert,acceptance_id='acceptance',authorization=proof,pva=p.at(100.01).tolist())
    def test_valid_observed_proof(self):self.assertIsNone(validate_observed_authorization(self.envelope(),self.plan.record))
    def test_wrong_epoch(self):
        e=self.envelope();e['map_epoch']=3;self.assertEqual(validate_observed_authorization(e,self.plan.record),'OBSERVED_MAP_EPOCH')
    def test_wrong_content(self):
        e=self.envelope();e['pva'][0][0]+=1;self.assertEqual(validate_observed_authorization(e,self.plan.record),'OBSERVED_COMMAND_CONTENT')
    def test_expired(self):
        e=self.envelope();e['stamp']=100.21;self.assertEqual(validate_observed_authorization(e,self.plan.record),'OBSERVED_EXPIRY')
    def test_no_geometry_permission(self):
        e=self.envelope();e['authorization']['evidence']['passed']=False;self.assertEqual(validate_observed_authorization(e,self.plan.record),'OBSERVED_GEOMETRY_NOT_PASSED')
    def test_wrong_snapshot(self):
        e=self.envelope();e['snapshot_id']='other';self.assertEqual(validate_observed_authorization(e,self.plan.record),'OBSERVED_SNAPSHOT')
    def test_no_forged_f1_qualification(self):
        with self.assertRaises(PermissionError):qualification(Path('research_outputs/forest_f1_20261118_v1'),{})
    def test_f0_historical_scope_only(self):self.assertIn('NOT_NEW_BINDING',historical_f0()['scope'])
    def test_shadow_no_publisher(self):
        src=Path('experiments/forest_f1_v1/native_shadow.py').read_text();self.assertNotIn('rospy.Publisher(',src)
    def test_revision_invalidation_preserved(self):
        from query_minco.integration.depth_minco_t2_v1.engine import Engine
        e=Engine(lambda now:None,lambda:100.)
        d=np.full((90,160),5.,np.float32);pose=dict(p=[0,0,0],q=[0,0,0,1])
        e.observe(d,pose,100.,100.);e.committed={'dummy':True};pose['p']=[.001,0,0]
        e.clock=lambda:100.01;e.observe(d,pose,100.01,100.01);self.assertIsNone(e.committed)
