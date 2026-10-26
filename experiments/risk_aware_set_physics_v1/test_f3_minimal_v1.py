"""Deterministic CPU/ideal-reference tests. No ROS, network training or old Plant edits."""
import unittest
import numpy as np
from query_minco.inference.observed_space import *
from query_minco.inference.trajectory_execution import *
from query_minco.inference.selector_safety import select_with_safety
from pathlib import Path

def head(x=0,v=.8,a=0,z=2):return np.array([[x,0,z],[v,0,0],[a,0,0]],float)
def space(expiry=100,scope=Scope.EXTERNAL_SUPERVISOR):
    return FreeVolume((box_cell([-2,-2,0],[8,2,4],expiry,'analytic-empty-cell'),),scope)
def line(v=.8,T=2.,origin=(0,0,0)):
    c=np.zeros((1,3,6));c[0,:,0]=origin;c[0,0,1]=v;return Program(c,np.array([T]))
def fixture_depth():
    d=np.zeros((2,128,160),np.float32);d[0]=.5;d[1]=1.;return d
class GeometryTests(unittest.TestCase):
    def test_native_rays_never_volume(self):
        history=DepthHistory();history.add(DepthFrame(1.,fixture_depth(),[0,0,2],np.eye(3),'one'),K)
        r=history.volume(1.).certify(line(origin=(1,0,2)).coefficients,[3.],1.)
        self.assertEqual(r['status'],'UNKNOWN')
        with self.assertRaises(ValueError):FreeVolume(space().cells,Scope.NATIVE_RAYS)
    def test_calibration_pose_and_clock(self):
        hist=DepthHistory();f=DepthFrame(1.,fixture_depth(),[0,0,2],np.eye(3),'one')
        bad=K.copy();bad[0,0]+=1
        with self.assertRaises(ValueError):hist.add(f,bad)
        with self.assertRaises(ValueError):DepthFrame(1.,fixture_depth(),[0,0,2],np.eye(3)*2,'bad')
        hist.add(f,K)
        with self.assertRaises(ValueError):hist.add(f,K)
        self.assertEqual(hist.frames,[])
    def test_past_pose_and_expiry(self):
        h=DepthHistory();d=fixture_depth();d[1,:,70:90]=0;d[0,:,70:90]=0
        h.add(DepthFrame(1.,d,[0,0,2],np.eye(3),'past'),K)
        h.add(DepthFrame(1.2,fixture_depth(),[.16,0,2],np.eye(3),'now'),K)
        a=h.diagnostics([[2,0,2],[-1,0,2],[12,0,2]],1.2)
        self.assertEqual(a['fresh_frames'],2);self.assertGreater(a['point_occluded_fraction'],0)
        self.assertFalse(a['continuous_volume_available'])
        self.assertEqual(h.diagnostics([[1,0,2]],2.)['fresh_frames'],0)
    def test_positive_continuous_fixture(self):
        p=line(origin=(1,0,2));r=space(scope=Scope.ANALYTIC_FIXTURE).certify(p.coefficients,p.durations,0,required_until=4)
        self.assertEqual(r['status'],'CERTIFIED_FREE');self.assertEqual(r['scope'],'ANALYTIC_FIXTURE')
        self.assertGreater(r['minimum_halfspace_slack_m'],0)
    def test_whole_body_outside_point_free_cell(self):
        c=box_cell([0,-.05,1.95],[5,.05,2.05],100,'thin-free-ray-tube')
        p=line(origin=(1,0,2))
        r=FreeVolume((c,),Scope.ANALYTIC_FIXTURE).certify(p.coefficients,p.durations,0,max_depth=5)
        self.assertEqual(r['status'],'UNKNOWN')
    def test_thin_gap_between_sample_points(self):
        # Endpoints are in separate known-free cells; a 1mm unknown gap is crossed.
        cells=(box_cell([-.5,-2,0],[1.4995,2,4],100,'left'),
               box_cell([1.5005,-2,0],[4,2,4],100,'right'))
        p=line(1.,3.,(0,0,2));r=FreeVolume(cells,Scope.ANALYTIC_FIXTURE).certify(p.coefficients,p.durations,0,max_depth=10)
        self.assertEqual(r['status'],'UNKNOWN')
    def test_future_expiry(self):
        p=line(origin=(1,0,2));r=space(.1).certify(p.coefficients,p.durations,0,required_until=1)
        self.assertEqual(r['status'],'UNKNOWN')
    def test_camera_rear_body_unknown(self):
        c=box_cell([.04,-2,0],[5,2,4],100,'forward-only')
        p=line(origin=(0,0,2));r=FreeVolume((c,),Scope.ANALYTIC_FIXTURE).certify(p.coefficients,p.durations,0,max_depth=3)
        self.assertEqual(r['status'],'UNKNOWN')
    def test_errors_shrink_space(self):
        p=line(origin=(1,0,2));cell=box_cell([0,-.3,1.7],[5,.3,2.3],100,'tight')
        small=FreeVolume((cell,),Scope.ANALYTIC_FIXTURE,ErrorBudget(rotation_rad=0))
        large=FreeVolume((cell,),Scope.ANALYTIC_FIXTURE,ErrorBudget(tracking=.5))
        self.assertEqual(small.certify(p.coefficients,p.durations,0)['status'],'CERTIFIED_FREE')
        self.assertEqual(large.certify(p.coefficients,p.durations,0,max_depth=2)['status'],'UNKNOWN')
    def test_budget_fails_closed(self):
        p=line(origin=(1,0,2));tiny=FreeVolume((box_cell([0,0,0],[.1,.1,.1],100,'tiny'),),Scope.ANALYTIC_FIXTURE)
        self.assertEqual(tiny.certify(p.coefficients,p.durations,0,max_nodes=1)['status'],'UNKNOWN')

class ExecutionTests(unittest.TestCase):
    def test_quintic_nonzero_pva(self):
        h=head(v=.8,a=.2);tail=head(x=1,v=0)
        p=quintic(h,tail,2.)
        np.testing.assert_allclose(p.pva(0),h,atol=1e-12);np.testing.assert_allclose(p.pva(2),tail,atol=1e-12)
    def test_brake_nonzero_acceleration(self):
        h=head(v=.8,a=.2);p,r=brake(h)
        np.testing.assert_allclose(p.pva(0),h,atol=1e-12)
        self.assertLess(np.linalg.norm(p.pva(p.duration)[1:]),1e-10)
        self.assertGreater(r['endpoint_displacement_m'],0)
        self.assertGreaterEqual(r['travel_distance_upper_bound_m'],r['endpoint_displacement_m'])
        p.validate(MotionLimits())
    def test_impossible_brake(self):
        with self.assertRaises(ValueError):brake(head(v=10),max_time=.4)
    def test_trim_no_replay_of_elapsed_prefix(self):
        p=line(origin=(0,0,2));t=trim(p,.5,1.)
        np.testing.assert_allclose(t.pva(0),p.pva(.5));self.assertAlmostEqual(t.duration,.5)
        with self.assertRaises(ValueError):p.pva(4.)
    def test_activation_splice_against_active_reference(self):
        p=line();active=Scheduled(Program(world_coefficients(p.coefficients,[0,0,2],np.eye(3)),p.durations),0,0,2,{},p)
        m=Measurement(.10,head(x=.08))
        s=prepare(p,0,[0,0,2],np.eye(3),m,.10,.16,space(),active=active)
        np.testing.assert_allclose(s.program.pva(0),active.pva(.16),atol=1e-12)
        self.assertGreater(s.program.pva(0)[0,0],0)
        self.assertAlmostEqual(s.activation,.16)
        self.assertAlmostEqual(s.program.duration,.35)
        s.program.validate(MotionLimits())
    def test_activation_pva_mismatch_rejected(self):
        p=line();a=Scheduled(line(origin=(0,0,2)),0,0,2,{},p)
        with self.assertRaisesRegex(ValueError,'TRACKING'):prepare(p,0,[0,0,2],np.eye(3),Measurement(.1,head(x=1.)),.1,.16,space(),active=a)
    def test_source_age_late_and_brake_space_reject(self):
        p=line();m=Measurement(.3,head(x=.24))
        with self.assertRaisesRegex(ValueError,'STALE'):prepare(p,0,[0,0,2],np.eye(3),m,.3,.36,space())
        with self.assertRaisesRegex(ValueError,'LATE'):prepare(p,.3,[0,0,2],np.eye(3),m,.3,.29,space())
        with self.assertRaisesRegex(ValueError,'SPACE'):prepare(p,.3,[.24,0,2],np.eye(3),m,.3,.36,FreeVolume((),Scope.NATIVE_RAYS))
    def test_50hz_exact_reference_and_reserved_brake(self):
        # Explicit exact-reference integration is a CPU fixture, NOT a ROS Plant.
        p=line();m=Measurement(0,head())
        s=prepare(p,0,[0,0,2],np.eye(3),m,0,.06,space())
        e=DryExecutor();e.arm_dryrun([],0,1);e.submit(s,0)
        modes=[]
        for t in np.arange(.06,1.001,.02):
            t=round(float(t),8)
            ref=s.program.pva(t-s.activation) if t<s.expiry else s.brake_reserve.pva(min(t-s.expiry,s.brake_reserve.duration))
            out=e.tick(float(t),Measurement(float(t),ref),space())
            self.assertIsNotNone(out['pva']);self.assertFalse(out['ros_published']);modes.append(out['mode'])
            np.testing.assert_allclose(out['pva'],ref,atol=1e-10)
        self.assertIn('BRAKING',modes)
        self.assertTrue(any(x['type']=='RESERVED_BRAKE' for x in e.events))
    def test_pending_activation_does_not_fake_hold(self):
        p=line();s=prepare(p,0,[0,0,2],np.eye(3),Measurement(0,head()),0,.06,space())
        e=DryExecutor();e.arm_dryrun([],0);e.submit(s,0)
        r=e.tick(.02,Measurement(.02,head(x=.016)),space())
        self.assertEqual(r['mode'],'WAITING_EXTERNAL_REFERENCE')
        self.assertIsNone(r['pva']);self.assertIsNotNone(e.pending)
    def test_bad_source_horizon_and_frame_rejected(self):
        with self.assertRaisesRegex(ValueError,'TIME_CONTRACT'):
            prepare(line(T=3),0,[0,0,2],np.eye(3),Measurement(0,head()),0,.06,space())
        with self.assertRaisesRegex(ValueError,'POSE'):
            prepare(line(),0,[0,0,2],np.eye(3)*2,Measurement(0,head()),0,.06,space())
    def test_watchdog_empty_set_brakes_not_hover(self):
        e=DryExecutor();e.arm_dryrun([],0)
        out=e.tick(.02,Measurement(.02,head(v=.8,a=.2)),space())
        self.assertEqual(out['mode'],'BRAKING');self.assertGreater(np.linalg.norm(out['pva'][1]),.7)
    def test_watchdog_unknown_brake_unproven(self):
        e=DryExecutor();e.arm_dryrun([],0)
        out=e.tick(.02,Measurement(.02,head()),FreeVolume((),Scope.NATIVE_RAYS))
        self.assertEqual(out['mode'],'EMERGENCY_STOP_UNPROVEN');self.assertIsNone(out['pva'])
        self.assertFalse(out['physical_stop_guaranteed'])
    def test_missing_odom_clockjump_and_conflict(self):
        e=DryExecutor();e.arm_dryrun([],0);o=e.tick(.1,Measurement(0,head()),space())
        self.assertEqual(o['mode'],'EMERGENCY_STOP_UNPROVEN')
        e=DryExecutor();e.arm_dryrun([],0);e.last_now=.2
        self.assertEqual(e.tick(.1,Measurement(.1,head()),space())['mode'],'CLOCK_FAULT')
        e=DryExecutor();e.arm_dryrun([],0)
        self.assertEqual(e.tick(.1,Measurement(.1,head()),space(),['/yopo_net'])['mode'],'AUTHORITY_FAULT')
    def test_missed_ticks_and_expired_lease(self):
        e=DryExecutor();e.arm_dryrun([],0);e.last_now=0.
        o=e.tick(.1,Measurement(.1,head()),space());self.assertEqual(o['mode'],'BRAKING')
        self.assertEqual(e.events[-1]['reason'],'MISSED_50HZ_TICKS')
        e=DryExecutor();e.arm_dryrun([],0,.1)
        self.assertEqual(e.tick(.2,Measurement(.2,head()),space())['mode'],'AUTHORITY_FAULT')
    def test_handover_and_no_ros_path(self):
        a=Authority()
        with self.assertRaises(ValueError):a.acquire_dryrun(['/yopo_net'],0)
        token=a.acquire_dryrun([],0);a.release('query_executor_DRYRUN')
        with self.assertRaises(ValueError):a.verify([],0,token)
        with self.assertRaises(PermissionError):a.publish_ros('anything')
    def test_terminal_goal_and_stop(self):
        g=local_goal([.7,0,2],np.eye(3),[3,0,2]);self.assertEqual(g['mode'],'TERMINAL_ANALYTIC_ONLY');self.assertIsNone(g['network_goal'])
        p=terminal(head(x=.7,v=.8,a=.1),[3,0,2]);p.validate(MotionLimits())
        np.testing.assert_allclose(p.pva(p.duration),head(x=3,v=0),atol=1e-10)
        self.assertEqual(local_goal([0,0,2],np.eye(3),[3,0,2])['mode'],'NETWORK_ROLLING')
    def test_terminal_overshoot_counterexample(self):
        # Too little room at nonzero speed must reject, not overshoot then return.
        with self.assertRaisesRegex(ValueError,'MONOTONE_TERMINAL'):
            terminal(head(x=2.99,v=.8,a=.1),[3,0,2])
    def test_terminal_measured_dwell(self):
        m=TerminalMonitor([3,0,2])
        self.assertFalse(m.update(Measurement(0,head(x=3,v=.2))))
        flags=[m.update(Measurement(round(float(t),8),head(x=3,v=0))) for t in np.arange(.02,.561,.02)]
        self.assertTrue(flags[-1])
        self.assertFalse(m.update(Measurement(.8,head(x=3,v=0)))) # missing samples reset dwell
        self.assertFalse(m.update(Measurement(.82,head(x=3,v=.1))))
        with self.assertRaises(ValueError):m.update(Measurement(.2,head(x=3,v=0)))
    def test_authority_nonfinite(self):
        a=Authority();epoch=a.acquire_dryrun([],0)
        with self.assertRaises(ValueError):a.renew_dryrun([],float('nan'),epoch)

class IntegrationTests(unittest.TestCase):
    def test_continuous_footprint_fixture_and_native_reject(self):
        f=DepthFrame(0,fixture_depth(),[0,0,2],np.eye(3),'analytic-patch')
        with self.assertRaises(ValueError):footprint_cell(f,[0,159],[0,127],10,10,Scope.NATIVE_RAYS,'fake')
        cell=footprint_cell(f,[0,159],[0,127],10,10,Scope.ANALYTIC_FIXTURE,'analytic-continuous-patch')
        p=line(origin=(1,0,2))
        r=FreeVolume((cell,),Scope.ANALYTIC_FIXTURE).certify(p.coefficients,p.durations,0)
        self.assertEqual(r['status'],'CERTIFIED_FREE')
    def test_external_evidence_cannot_enter_online_ranking(self):
        path=Path(__file__).resolve().parents[2]/'research_outputs/rsps_forest_shadow_v1/low_speed_shadow001/frame_000080.npz'
        with np.load(path) as z:
            d=z['depth_input'];s=z['state'];o={k[3:]:z[k] for k in z.files if k.startswith('C1_')}
            class ExternalHistory:
                def volume(self,now):
                    return FreeVolume((box_cell([-50,-10,-10],[50,10,10],100,'external'),),Scope.EXTERNAL_SUPERVISOR)
            r=select_with_safety(d,s,o,.02,ExternalHistory(),0,z['position_W'],z['R_WB0'])
            self.assertEqual(r['eligible_ranking'],[]);self.assertIsNone(r['selected_query'])
            self.assertTrue(all(x['status']=='UNKNOWN' for x in r['candidates']))
    def test_raw19_20_do_not_change_gates(self):
        path=Path(__file__).resolve().parents[2]/'research_outputs/rsps_forest_shadow_v1/low_speed_shadow001/frame_000080.npz'
        with np.load(path) as z:
            d=z['depth_input'];s=z['state'];o={k[3:]:z[k] for k in z.files if k.startswith('C1_')}
            h=DepthHistory();h.add(DepthFrame(0,d,z['position_W'],z['R_WB0'],'raw'),K)
            a=select_with_safety(d,s,o,.02,h,.02,z['position_W'],z['R_WB0'])
            o['raw']=o['raw'].copy();o['raw'][:,19:]=100000
            b=select_with_safety(d,s,o,.02,h,.02,z['position_W'],z['R_WB0'])
            self.assertEqual(a,b)

if __name__=='__main__':unittest.main(verbosity=2)
