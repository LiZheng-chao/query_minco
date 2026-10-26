"""CPU logic fixtures. They are NOT native ROS or tracking qualification."""
import unittest,math,numpy as np
from query_minco.inference.committed_reference import *
from query_minco.inference.observed_space import FreeVolume,box_cell,Scope
from query_minco.inference.trajectory_execution import Program
class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        c=np.zeros((1,3,6));c[0,:,0]=[0,0,2];c[0,0,1]=.5;c[0,0,2]=.025
        cls.prefix=Program(c,[.4])
        cls.volume=FreeVolume((box_cell([-2.8,-2.8,.2],[6.8,2.8,3.8],200,'fixture'),),Scope.EXTERNAL_SUPERVISOR)
        cls.bundle=prepare_reference(cls.prefix,cls.volume,10.,10.)
    def state(self,pva,t):return StateSnapshot(t,t,tuple(map(tuple,pva)),topology_received_mono=t)
    def arm(self):
        k=ReferenceKernel();k.submit(self.bundle,10.1,10.1)
        k.tick(10.1,10.1,self.state(self.prefix.pva(0),10.1));return k
    def test_immutable_reference_bytes(self):
        a=np.frombuffer(self.bundle.prefix.coefficients,dtype='<f8')
        with self.assertRaises(ValueError):a[0]=10
        with self.assertRaises(ValueError):a.setflags(write=True)
        with self.assertRaises(Exception):self.bundle.stops=()
    def test_full_backups_are_prepared(self):
        self.assertEqual(len(self.bundle.stops),21)
        for e in self.bundle.stops:
            np.testing.assert_allclose(e.polynomial.pva(0),self.prefix.pva(e.phase),atol=1e-8)
            np.testing.assert_allclose(e.polynomial.pva(e.polynomial.duration)[1:],0,atol=1e-8)
    def test_watchdog_uses_existing_stop_no_search(self):
        k=self.arm()
        import unittest.mock as mock
        with mock.patch('query_minco.inference.committed_reference.soft_brake',side_effect=AssertionError('RT_SEARCH')),mock.patch.object(FreeVolume,'certify',side_effect=AssertionError('RT_CERT')):
            for t in np.arange(10.12,14.8,.02):
                expected=k._reference(float(t));k.tick(float(t),float(t),self.state(expected,float(t)),watchdog=abs(t-10.2)<.001)
        self.assertIn(k.mode,['BRAKING','STOP_REFERENCE_COMPLETE'])
        self.assertTrue(any(x['reason']=='BACKUP_ACTIVATED' for x in k.events))
    def test_early_does_not_activate(self):
        k=ReferenceKernel();k.submit(self.bundle,10.1,10.1)
        self.assertIsNone(k.tick(10.09,10.09,self.state(self.prefix.pva(0),10.09)))
    def test_late_new_plan_keeps_old_commitment(self):
        k=self.arm();old=k.active;k.submit(self.bundle,10.11,10.11)
        self.assertIsNotNone(k.tick(10.135,10.135,self.state(k._reference(10.135),10.135)))
        self.assertIs(k.active,old)
        self.assertTrue(any(e['reason']=='PROPOSAL_REJECT' for e in k.events))
    def test_dropped_state_never_claims_tracking(self):
        k=self.arm();k.tick(10.12,10.12,None)
        self.assertTrue(k.qualification_failed)
        self.assertEqual(k.mode,'BACKUP_TRACKING_UNPROVEN')
    def test_conflicting_authority_never_claims_pass(self):
        k=self.arm();s=StateSnapshot(10.12,10.12,tuple(map(tuple,k._reference(10.12))),('/intruder',),10.12)
        k.tick(10.12,10.12,s);self.assertTrue(k.qualification_failed)
    def test_clock_jump_and_delayed_scheduler_latch_failure(self):
        for delta in [.05,-.05]:
            k=self.arm();k.tick(10.12+delta,10.12,self.state(k._reference(10.12),10.12))
            self.assertTrue(k.qualification_failed)
        k=self.arm();k.tick(10.16,10.16,self.state(k._reference(10.16),10.16))
        self.assertTrue(k.qualification_failed)
    def test_unknown_space_cannot_enter(self):
        with self.assertRaises(ValueError):prepare_reference(self.prefix,FreeVolume((),Scope.NATIVE_RAYS),10,10)
    def test_no_backup_no_execution(self):
        k=ReferenceKernel();self.assertIsNone(k.tick(10.,10.,None,watchdog=True))
    def test_positive_acceleration_stop_limits(self):
        h=np.array([[0,0,2],[.4,0,0],[.5,0,0]])
        p=reserved_stop(h);p.validate(MotionLimits())
        np.testing.assert_allclose(p.pva(0),h,atol=1e-10)
        np.testing.assert_allclose(p.pva(p.duration)[1:],0,atol=1e-9)
if __name__=='__main__':unittest.main()
