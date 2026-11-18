import unittest
import numpy as np
from query_minco.integration.depth_minco_t2_v1.engine import Engine
from query_minco.integration.depth_minco_t2_v1.space import BirthGrant
class RootCause(unittest.TestCase):
    def test_static_feasible_map_commits(self):
        e=Engine(lambda n:BirthGrant((0,0,0),.6,100.2,'SYNTHETIC_ONLY'),lambda:100.)
        try:
            p=dict(p=[0,0,0],v=[0,0,0],q=[0,0,0,1]);e.observe(np.full((90,160),5.,np.float32),p,100.,100.);e.set_state(dict(p,a=[0,0,0]),100.);e.set_goal([.28,0,0],100.)
            self.assertTrue(e.run_once());self.assertIsNotNone(e.command(100.))
        finally:e.close()
    def test_snapshot_failure_cannot_ack_unprocessed_geometry(self):
        now=[100.];e=Engine(lambda n:None,lambda:now[0]);p=dict(p=[0,0,0],v=[0,0,0],q=[0,0,0,1])
        def fail(n):raise ValueError('SYNTHETIC_SNAPSHOT_FAILURE')
        e.ledger.snapshot=fail
        try:
            for i in range(8):
                now[0]=100.+i*.03;e.observe(np.full((90,160),5.,np.float32),p,now[0],now[0]);e.set_state(dict(p,a=[0,0,0]),now[0]);e.set_goal([.28,0,0],now[0]);self.assertFalse(e.run_once())
            self.assertEqual(len(e.pending),8);now[0]+=.03
            with self.assertRaisesRegex(ValueError,'EVIDENCE_QUEUE_OVERFLOW'):e.observe(np.full((90,160),5.,np.float32),p,now[0],now[0])
            self.assertTrue(e.evidence_overflow);self.assertIsNone(e.command(now[0]))
        finally:e.close()
