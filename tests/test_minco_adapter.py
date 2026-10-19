import unittest
from unittest.mock import patch
import torch
import numpy as np
from query_minco.trajectory import MincoAdapter, NumericalSolveError, TrajectorySampler, evaluate_piece
from query_minco.tests.trajectory_common import fixture
from query_minco.tests.trajectory_reference import reference_solve, reference_evaluate, reference_energy

# Fixed before measurement. Numerical FP32 checks; not flight safety tolerances.
REFERENCE_ATOL = {"coeffs": .003, "p": .0003, "v": .001, "a": .003, "jerk": .03}
JOINT_ATOL = (.0003, .001, .003, .03, .3)


class MincoAdapterTests(unittest.TestCase):
    def test_boundary_waypoints_and_c4_cpu(self):
        for batch in (1,2):
            h,t,q,d=fixture(batch)
            out=MincoAdapter()(h,t,q,d)
            self.assertEqual(tuple(out.coeffs.shape),(batch,6,3,3,6))
            self.assertTrue(bool(out.solve_valid.all()))
            for order,atol in enumerate(JOINT_ATOL):
                zero=torch.zeros(batch,6,1)
                if order<3:
                    torch.testing.assert_close(evaluate_piece(out.coeffs[...,0,:,:],zero,order).squeeze(-2),
                                               h[...,order,:],atol=atol,rtol=0)
                    torch.testing.assert_close(evaluate_piece(out.coeffs[...,2,:,:],d[...,2,None],order).squeeze(-2),
                                               t[...,order,:],atol=atol,rtol=0)
                for i in range(2):
                    left=evaluate_piece(out.coeffs[...,i,:,:],d[...,i,None],order).squeeze(-2)
                    right=evaluate_piece(out.coeffs[...,i+1,:,:],zero,order).squeeze(-2)
                    torch.testing.assert_close(left,right,atol=atol,rtol=0)
                    if order==0: torch.testing.assert_close(left,q[...,i,:],atol=atol,rtol=0)

    def test_independent_fp64_kkt_coeffs_and_physical_samples(self):
        h,t,q,d=fixture()
        out=MincoAdapter()(h,t,q,d)
        samples=TrajectorySampler()(out,num_samples=65)
        for candidate in range(6):
            ref=reference_solve(h[0,candidate],t[0,candidate],q[0,candidate],d[0,candidate])
            np.testing.assert_allclose(out.coeffs[0,candidate].numpy(),ref,atol=REFERENCE_ATOL["coeffs"],rtol=0)
            for order,name in enumerate(("p","v","a","jerk")):
                expected=reference_evaluate(ref,d[0,candidate].numpy(),samples.times[0,candidate].numpy(),order)
                np.testing.assert_allclose(getattr(samples,name)[0,candidate].numpy(),expected,
                                           atol=REFERENCE_ATOL[name],rtol=0)

    def test_kkt_reference_minimum_energy_against_feasible_perturbation(self):
        # Independent feasible variation vanishes in P/V/A at endpoints and P at joints.
        # Nullspace of the C2/waypoint constraints provides an alternative curve.
        h,t,q,d=fixture()
        ref=reference_solve(h[0,0],t[0,0],q[0,0],d[0,0])
        # Test energy via a known C2 knot-bump: normalized B-spline-like variation
        # is built from the constraint nullspace below, independent of MINCO.
        from query_minco.tests.trajectory_reference import basis
        ds=d[0,0].numpy().astype(np.float64)
        rows=[]
        def r(seg,u,k):
            row=np.zeros(18); row[6*seg:6*seg+6]=basis(u,k)/ds[seg]**k; return row
        for k in range(3): rows.extend((r(0,0,k),r(2,1,k)))
        for i in range(2):
            rows.extend((r(i,1,0),r(i+1,0,0),r(i,1,1)-r(i+1,0,1),r(i,1,2)-r(i+1,0,2)))
        _,_,vh=np.linalg.svd(np.stack(rows),full_matrices=True)
        perturb=vh[-1].reshape(3,6)/ds[:,None]**np.arange(6)
        for sign in (-1,1):
            alt=ref.copy(); alt[:,0,:]+=sign*.1*perturb
            self.assertGreater(reference_energy(alt,ds),reference_energy(ref,ds))

    def test_constant_and_uniform_velocity_analytic(self):
        h,t,q,d=fixture()
        for moving in (False,True):
            h.zero_(); t.zero_(); q.zero_()
            velocity=torch.tensor([.8,-.2,.1]) if moving else torch.zeros(3)
            h[...,1,:]=velocity; t[...,1,:]=velocity
            t[...,0,:]=d.sum(-1)[...,None]*velocity
            q[...,0,:]=d[...,0,None]*velocity
            q[...,1,:]=d[...,:2].sum(-1)[...,None]*velocity
            out=MincoAdapter()(h,t,q,d)
            s=TrajectorySampler()(out)
            torch.testing.assert_close(s.p,s.times[...,None]*velocity,atol=3e-5,rtol=0)
            torch.testing.assert_close(s.v,velocity.expand_as(s.v),atol=5e-5,rtol=0)
            torch.testing.assert_close(s.a,torch.zeros_like(s.a),atol=3e-4,rtol=0)

    def test_per_call_solver_isolation(self):
        h,t,q,d=fixture()
        adapter=MincoAdapter(); first=adapter(h,t,q,d)
        frozen=first.coeffs.clone()
        second=adapter(h,t,q+.1,d)
        torch.testing.assert_close(first.coeffs,frozen,rtol=0,atol=0)
        self.assertFalse(torch.equal(first.coeffs,second.coeffs))

    def test_invalid_inputs_and_numerical_domain(self):
        h,t,q,d=fixture()
        for bad in (0.,-.1,1e-8,11.):
            durations=d.clone(); durations[0,0,0]=bad
            with self.assertRaises(ValueError): MincoAdapter()(h,t,q,durations)
        with self.assertRaises(TypeError): MincoAdapter()(h.double(),t,q,d)
        with self.assertRaises(ValueError): MincoAdapter()(h,t,q[...,:1,:],d)
        with self.assertRaises(ValueError): MincoAdapter()(h,t,q,d[:,:,:2])
        for v in (float("nan"),float("inf")):
            durations=d.clone(); durations[0,0,0]=v
            with self.assertRaises(ValueError): MincoAdapter()(h,t,q,durations)

    def test_solver_exception_nonfinite_and_boundary_failure_are_explicit(self):
        h,t,q,d=fixture()
        with patch("query_minco.trajectory.minco_adapter.MincoS3NU.set_parameters",
                   side_effect=RuntimeError("singular system fixture")):
            with self.assertRaisesRegex(NumericalSolveError,"linear solve failed"): MincoAdapter()(h,t,q,d)
        def bad_coeffs(solver,*args):
            solver.coeffs=torch.full((6,18,3),float("nan"))
        with patch("query_minco.trajectory.minco_adapter.MincoS3NU.set_parameters",bad_coeffs):
            with self.assertRaisesRegex(NumericalSolveError,"nonfinite"): MincoAdapter()(h,t,q,d)
        def wrong_coeffs(solver,*args): solver.coeffs=torch.zeros(6,18,3)
        with patch("query_minco.trajectory.minco_adapter.MincoS3NU.set_parameters",wrong_coeffs):
            with self.assertRaisesRegex(NumericalSolveError,"residual failed"): MincoAdapter()(h,t,q,d)

    @unittest.skipUnless(torch.cuda.is_available(),"CUDA unavailable")
    def test_cuda_amp_solver_is_fp32_and_reference(self):
        h,t,q,d=fixture(device="cuda")
        with torch.autocast("cuda",dtype=torch.float16):
            out=MincoAdapter()(h,t,q,d)
            samples=TrajectorySampler()(out)
        self.assertEqual(out.coeffs.dtype,torch.float32)
        self.assertEqual(samples.p.dtype,torch.float32)
        ref=reference_solve(h[0,0].cpu(),t[0,0].cpu(),q[0,0].cpu(),d[0,0].cpu())
        np.testing.assert_allclose(out.coeffs[0,0].cpu().numpy(),ref,atol=REFERENCE_ATOL["coeffs"],rtol=0)
        with self.assertRaises(ValueError): MincoAdapter()(h.cpu(),t,q,d)

    def test_actual_ill_conditioned_duration_rejects_without_fallback(self):
        h,t,q,d=fixture()
        d[:]=torch.tensor([.001,10.,.001])
        with self.assertRaises(NumericalSolveError) as raised:
            MincoAdapter()(h,t,q,d)
        self.assertIn("solve_valid",raised.exception.diagnostics)
        self.assertFalse(bool(raised.exception.diagnostics["solve_valid"].any()))

    def test_short_default_domain_segments_remain_numerically_checked(self):
        h,t,q,d=fixture()
        d[:]=torch.tensor([.12,.12,.56])
        out=MincoAdapter()(h,t,q,d)
        self.assertTrue(bool(out.solve_valid.all()))

    def test_same_adapter_concurrent_calls_do_not_share_solver_state(self):
        from concurrent.futures import ThreadPoolExecutor
        h,t,q,d=fixture()
        adapter=MincoAdapter()
        with ThreadPoolExecutor(max_workers=2) as pool:
            first=pool.submit(adapter,h,t,q,d)
            second=pool.submit(adapter,h,t,q+.1,d)
            a,b=first.result(),second.result()
        torch.testing.assert_close(a.coeffs,adapter(h,t,q,d).coeffs,rtol=0,atol=0)
        torch.testing.assert_close(b.coeffs,adapter(h,t,q+.1,d).coeffs,rtol=0,atol=0)

    @unittest.skipUnless(torch.cuda.is_available(),"CUDA unavailable")
    def test_cuda_internal_matrix_and_constants_remain_fp32(self):
        from YOPO.minco.minco import MincoS3NU as OriginalSolver
        created=[]
        def factory(piece_num,device):
            solver=OriginalSolver(piece_num,device=device)
            created.append(solver)
            return solver
        h,t,q,d=fixture(device="cuda")
        with patch("query_minco.trajectory.minco_adapter.MincoS3NU",side_effect=factory):
            with torch.autocast("cuda",dtype=torch.float16):
                MincoAdapter()(h,t,q,d)
        solver=created[0]
        for value in (solver.A_inv,solver.coeffs,solver._A_const,solver._exps,solver.durations):
            self.assertEqual(value.dtype,torch.float32)
            self.assertEqual(value.device.type,"cuda")
