import unittest
import math
import torch
from query_minco.trajectory import TrajectorySampler, MincoAdapter, MincoTrajectory, evaluate_piece
from query_minco.tests.trajectory_common import fixture


class TrajectorySamplerTests(unittest.TestCase):
    def test_global_uniform_endpoints_and_unequal_segments(self):
        h,t,q,d=fixture(2)
        trajectory=MincoAdapter()(h,t,q,d)
        samples=TrajectorySampler()(trajectory,num_samples=64)
        self.assertEqual(tuple(samples.times.shape),(2,6,64))
        for x in (samples.p,samples.v,samples.a,samples.jerk):
            self.assertEqual(tuple(x.shape),(2,6,64,3))
            self.assertTrue(bool(torch.isfinite(x).all()))
        torch.testing.assert_close(samples.times[...,0],torch.zeros(2,6),rtol=0,atol=0)
        torch.testing.assert_close(samples.times[...,-1],d.sum(-1),rtol=0,atol=0)
        torch.testing.assert_close(samples.times.diff(dim=-1),
                                   (d.sum(-1)/63)[...,None].expand(2,6,63),atol=3e-7,rtol=0)
        torch.testing.assert_close(samples.p[...,0,:],h[...,0,:],atol=3e-4,rtol=0)
        torch.testing.assert_close(samples.p[...,-1,:],t[...,0,:],atol=3e-4,rtol=0)

    def test_physical_derivatives_and_exact_junction_piece_selection(self):
        _,_,_,d=fixture()
        starts=torch.cat((torch.zeros(1,6,1),d.cumsum(-1)[...,:2]),dim=-1)
        coeffs=torch.zeros(1,6,3,3,6)
        # Global p=(t**5, 2*t**5, -t**5), expressed in each local segment basis.
        axes=torch.tensor([1.,2.,-1.])
        for k in range(6): coeffs[...,k]=math.comb(5,k)*starts[...,None]**(5-k)*axes
        tr=MincoTrajectory(coeffs,d,torch.ones(1,6,dtype=torch.bool),{})
        times=torch.stack((torch.zeros(1,6),d[...,0],d[...,:2].sum(-1),d.sum(-1)),dim=-1)
        out=TrajectorySampler()(tr,times=times)
        self.assertTrue(torch.equal(out.segment_indices,torch.tensor([0,1,2,2]).expand(1,6,4)))
        for derivative,x in enumerate((out.p,out.v,out.a,out.jerk)):
            factor=math.factorial(5)/math.factorial(5-derivative)
            expected=factor*times[...,None]**(5-derivative)*axes
            torch.testing.assert_close(x,expected,atol=5e-4,rtol=2e-6)

    def test_piece_derivatives_zero_through_five(self):
        coeffs=torch.randn(2,3,6)
        times=torch.tensor([[0.,.2,.7],[.1,.3,.8]])
        for order in range(6):
            expected=sum(coeffs[...,k][:,None,:]*math.factorial(k)/math.factorial(k-order)*
                         times[...,None]**(k-order) for k in range(order,6))
            torch.testing.assert_close(evaluate_piece(coeffs,times,order),expected,atol=2e-4,rtol=2e-6)

    def test_invalid_sampling_queries_and_solution(self):
        h,t,q,d=fixture(); tr=MincoAdapter()(h,t,q,d)
        for count in (0,1,1.5,True):
            with self.assertRaises(ValueError): TrajectorySampler()(tr,num_samples=count)
        for times in (torch.zeros(1,5,10),torch.full((1,6,2),-1.),
                      torch.full((1,6,2),10.),torch.full((1,6,2),float("nan")),
                      torch.zeros(1,6,2,dtype=torch.float64)):
            with self.assertRaises(ValueError): TrajectorySampler()(tr,times=times)
        invalid=MincoTrajectory(tr.coeffs,d,torch.zeros_like(tr.solve_valid),{})
        with self.assertRaises(ValueError): TrajectorySampler()(invalid)
        bad=MincoTrajectory(tr.coeffs,torch.zeros_like(d),tr.solve_valid,{})
        with self.assertRaises(ValueError): TrajectorySampler()(bad)
        with self.assertRaises(ValueError): evaluate_piece(tr.coeffs[...,0,:,:],torch.zeros(1,6,1),6)
