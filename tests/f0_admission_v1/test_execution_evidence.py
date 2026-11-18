import hashlib
import json
import unittest
from query_minco.integration.f0_continuity_v1.execution_evidence import assess as reduce_evidence

RECORDS = [dict(coefficients=[value],durations=[2/3]*3) for value in [1,2]]
A, B = [hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest() for r in RECORDS]
PLANS = [dict(kind='plan',plan_id=i,plan=r) for i,r in zip([A,B],RECORDS)]
def assess(events,begin,end):
    return reduce_evidence(events,begin,end,PLANS)

def command(identity, mono):
    return dict(kind='controller_command', monotonic=mono, mode='ARMED', accepted=True,
                stamp=100+mono, ros_now=100+mono,
                payload=dict(plan_id=identity, authorization=dict(plan_id=identity,
                    geometry='FINITE_BIRTH_FULL_TUBE_129_WITH_OCCUPIED_VETO',
                    certificate_expiry=101+mono)))

def feedback():
    return [dict(kind='final_so3', monotonic=3, mode='ARMED', accepted=True,
                 stamp=103., payload={}),
            dict(kind='plant', monotonic=3.1, mode='ARMED', accepted=True, stamp=103.1,
                 payload=dict(phase='ARMED', state_source='DYNAMICS_INTEGRATED',
                              command_valid=True, integration_steps=10, final_command_stamp=103.))]

class Evidence(unittest.TestCase):
    def test_replans_only_fails(self):
        self.assertEqual(assess([dict(kind='plan', replans=2, monotonic=1)],0,4)['status'],'FAIL')
    def test_repeated_plan_fails(self):
        self.assertEqual(assess([command(A,1),command(A,2)]+feedback(),0,4)['status'],'FAIL')
    def test_admitted_switch_and_consumed_so3_still_unknown_lineage(self):
        result=assess([command(A,1),command(B,2)]+feedback(),0,4)
        self.assertEqual(result['status'],'UNKNOWN')
        self.assertEqual(result['accepted_plan_ids'],[A,B])
        self.assertEqual(len(result['switches']),1)
        self.assertEqual(result['consumed_so3_stamps'],[103.])
    def test_missing_feedback_fails(self):
        self.assertEqual(assess([command(A,1),command(B,2)],0,4)['status'],'FAIL')
    def test_wrong_plant_stamp_fails(self):
        rows=feedback();rows[-1]['payload']['final_command_stamp']=102.
        self.assertEqual(assess([command(A,1),command(B,2)]+rows,0,4)['status'],'FAIL')
    def test_outside_window_switch_fails(self):
        self.assertEqual(assess([command(A,1),command(B,2)]+feedback(),1.5,4)['status'],'FAIL')
    def test_hold_proof_is_not_trajectory(self):
        row=command(B,2);row['payload']['authorization']['geometry']='HOLD'
        self.assertEqual(assess([command(A,1),row]+feedback(),0,4)['status'],'FAIL')
    def test_expired_or_rejected_command_fails(self):
        for invalid in ['expired','rejected','not_armed']:
            row=command(B,2)
            if invalid=='expired':row['payload']['authorization']['certificate_expiry']=0
            if invalid=='rejected':row['accepted']=False
            if invalid=='not_armed':row['mode']='ACK'
            self.assertEqual(assess([command(A,1),row]+feedback(),0,4)['status'],'FAIL')
    def test_hold_digest_without_reference_plan_fails(self):
        row=command('c'*64,2)
        self.assertEqual(assess([command(A,1),row]+feedback(),0,4)['status'],'FAIL')
    def test_rollback_fails(self):
        self.assertEqual(assess([command(A,2),command(B,1)]+feedback(),0,4)['status'],'FAIL')
    def test_unknown_blocks_runtime_pass(self):
        from pathlib import Path
        source=Path('experiments/stop_latency_fix_v1/manual.py').read_text()
        self.assertIn("execution_evidence['status']!='PASS':qualification=False",source)
