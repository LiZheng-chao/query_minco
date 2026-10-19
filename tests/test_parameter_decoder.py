import unittest
import torch
from query_minco.trajectory import PhysicalParameterConfig, PhysicalParameterDecoder
from query_minco.tests.trajectory_common import raw_fixture


class ParameterDecoderTests(unittest.TestCase):
    def test_batch_one_two_shapes_and_zero_logits(self):
        for batch in (1, 2):
            raw = torch.zeros(batch, 6, 21)
            state = torch.randn(batch, 12)
            result = PhysicalParameterDecoder()(raw, state)
            for value, shape in [(result.head_pva,(batch,6,3,3)),(result.tail_pva,(batch,6,3,3)),
                                 (result.inner_pts,(batch,6,2,3)),(result.durations,(batch,6,3)),
                                 (result.score,(batch,6)),(result.feas_logit,(batch,6))]:
                self.assertEqual(tuple(value.shape),shape)
            torch.testing.assert_close(result.inner_pts, torch.tensor([5.,0.,0.]).expand(batch,6,2,3))
            torch.testing.assert_close(result.head_pva[...,0,:],torch.zeros(batch,6,3))
            torch.testing.assert_close(result.head_pva[...,1,:],state[:,None,:3].expand(-1,6,-1))
            torch.testing.assert_close(result.head_pva[...,2,:],state[:,None,3:6].expand(-1,6,-1))
            torch.testing.assert_close(result.durations,torch.full((batch,6,3),.5))
            torch.testing.assert_close(result.total_time,torch.full((batch,6),1.5))

    def test_absolute_points_tail_pva_and_passthrough(self):
        raw,state=raw_fixture()
        out=PhysicalParameterDecoder()(raw,state)
        lo=torch.tensor([0.,-4.,-2.]); hi=torch.tensor([10.,4.,2.])
        points=lo+(raw[...,:9].reshape(1,6,3,3).tanh()+1)*.5*(hi-lo)
        torch.testing.assert_close(out.inner_pts,points[...,:2,:])
        torch.testing.assert_close(out.tail_pva[...,0,:],points[...,2,:])
        torch.testing.assert_close(out.tail_pva[...,1,:],raw[...,9:12].tanh()*torch.tensor([6.,4.,3.]))
        torch.testing.assert_close(out.tail_pva[...,2,:],raw[...,12:15].tanh()*torch.tensor([6.,6.,4.]))
        self.assertTrue(torch.equal(out.score,raw[...,19]))
        self.assertTrue(torch.equal(out.feas_logit,raw[...,20]))

    def test_extreme_logits_and_time_gradient(self):
        for extreme in (-1e4,1e4):
            raw=torch.full((2,6,21),extreme)
            raw[...,16:19]=torch.tensor([extreme,0.,-extreme])
            out=PhysicalParameterDecoder()(raw,torch.zeros(2,12))
            self.assertTrue(bool(torch.isfinite(out.durations).all()))
            self.assertTrue(bool((out.durations>=.12).all()))
            torch.testing.assert_close(out.durations.sum(-1),out.total_time,atol=2e-6,rtol=0)
        raw,state=raw_fixture(); raw.requires_grad_()
        out=PhysicalParameterDecoder()(raw,state)
        (out.durations*torch.tensor([1.,2.,4.])).sum().backward()
        self.assertGreater(float(raw.grad[...,15].abs().sum()),0)
        self.assertGreater(float(raw.grad[...,16:19].abs().sum()),0)

    def test_anisotropic_custom_bounds_and_axis_not_norm(self):
        cfg=PhysicalParameterConfig(position_min_m=(-1.,-2.,-.5),position_max_m=(7.,2.,1.5))
        raw=torch.zeros(1,6,21); raw[...,9:15]=1000
        out=PhysicalParameterDecoder(cfg)(raw,torch.zeros(1,12))
        torch.testing.assert_close(out.inner_pts,torch.tensor([3.,0.,.5]).expand(1,6,2,3))
        self.assertGreater(float(out.diagnostics["terminal_speed_norm_mps"].max()),6)
        self.assertGreater(float(out.diagnostics["terminal_acceleration_norm_mps2"].max()),6)

    def test_head_state_is_not_clipped(self):
        raw=torch.zeros(1,6,21); state=torch.zeros(1,12); state[:,:6]=100.
        out=PhysicalParameterDecoder()(raw,state)
        self.assertTrue(bool((out.head_pva[...,1:,:]==100).all()))

    def test_invalid_shapes_dtypes_values_and_config(self):
        raw,state=raw_fixture()
        cases=[(raw[:,:,:20],state,ValueError),(raw,state[:,:9],ValueError),
               (raw,torch.zeros(2,12),ValueError),(raw.long(),state,TypeError),
               (raw.double(),state,TypeError),(None,state,TypeError)]
        for a,b,error in cases:
            with self.assertRaises(error): PhysicalParameterDecoder()(a,b)
        for value in (float("nan"),float("inf")):
            a=raw.clone(); a[0,0,0]=value
            with self.assertRaises(ValueError): PhysicalParameterDecoder()(a,state)
            b=state.clone(); b[0,0]=value
            with self.assertRaises(ValueError): PhysicalParameterDecoder()(raw,b)
        for kwargs in ({"min_total_time_s":.3},{"min_segment_time_s":0},
                       {"position_min_m":(0.,0.,0.),"position_max_m":(0.,1.,1.)},
                       {"terminal_velocity_axis_max_mps":(0.,1.,1.)}):
            with self.assertRaises(ValueError): PhysicalParameterConfig(**kwargs)

    @unittest.skipUnless(torch.cuda.is_available(),"CUDA unavailable")
    def test_cuda_amp_decode_and_device_mismatch(self):
        raw,state=raw_fixture("cuda")
        with torch.autocast("cuda",dtype=torch.float16):
            out=PhysicalParameterDecoder()(raw.half(),state)
        self.assertEqual(out.durations.dtype,torch.float32)
        self.assertEqual(out.head_pva.device.type,"cuda")
        self.assertEqual(out.score.dtype,torch.float16)
        with self.assertRaises(ValueError): PhysicalParameterDecoder()(raw,state.cpu())
