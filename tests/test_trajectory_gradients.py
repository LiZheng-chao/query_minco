import unittest
import torch
from query_minco.trajectory import TrajectoryPostprocessor
from query_minco.tests.common import model_and_inputs, gradient_summary
from query_minco.tests.trajectory_common import raw_fixture, trajectory_loss
from query_minco.tests.trajectory_gradient_checks import GROUPS,directional_checks


class TrajectoryGradientTests(unittest.TestCase):
    def test_raw_all_physical_groups_and_unused_scores(self):
        raw,state=raw_fixture(); raw.requires_grad_()
        loss=trajectory_loss(raw,state); loss.backward()
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertTrue(bool(torch.isfinite(raw.grad).all()))
        for name,(lo,hi) in GROUPS.items():
            with self.subTest(group=name): self.assertGreater(float(raw.grad[...,lo:hi].norm()),1e-7)
        self.assertTrue(bool((raw.grad[...,19:21]==0).all()))
        torch.testing.assert_close(raw.grad[...,16:19].sum(-1),torch.zeros(1,6),atol=1e-5,rtol=0)

    def test_directional_differences_multiple_steps_cpu(self):
        raw,state=raw_fixture()
        _,records=directional_checks(raw,state)
        for row in records:
            with self.subTest(group=row["group"],step=row["step"]):
                self.assertLessEqual(row["absolute_error"],.003+.03*abs(row["autograd"]),row)

    def network_backward(self,device,amp=False):
        model,depth,state=model_and_inputs(device=device)
        with torch.autocast(device_type=device,dtype=torch.float16,enabled=amp):
            raw=model(depth,state); raw.retain_grad()
            result=TrajectoryPostprocessor()(raw,state,num_samples=32)
            self.assertEqual(result.trajectory.coeffs.dtype,torch.float32)
            self.assertEqual(result.samples.p.dtype,torch.float32)
            loss=result.samples.p.square().mean()+.03*(result.samples.v-.2).square().mean()
        loss.backward()
        self.assertTrue(bool(torch.isfinite(raw.grad).all()))
        self.assertTrue(bool((raw.grad[...,19:21]==0).all()))
        for group,summary in gradient_summary(model).items():
            with self.subTest(group=group):
                self.assertTrue(summary["all_finite"],summary)
                self.assertEqual(summary["with_gradient"],summary["parameter_tensors"])
                self.assertGreater(summary["l2_norm"],0)

    def test_network_through_sampler_backward_cpu(self):
        self.network_backward("cpu")

    @unittest.skipUnless(torch.cuda.is_available(),"CUDA unavailable")
    def test_network_through_sampler_backward_cuda(self):
        self.network_backward("cuda")

    @unittest.skipUnless(torch.cuda.is_available(),"CUDA unavailable")
    def test_network_amp_solver_fp32_backward_cuda(self):
        self.network_backward("cuda",amp=True)

    @unittest.skipUnless(torch.cuda.is_available(),"CUDA unavailable")
    def test_directional_differences_cuda(self):
        raw,state=raw_fixture("cuda")
        _,records=directional_checks(raw,state)
        for row in records:
            with self.subTest(group=row["group"],step=row["step"]):
                self.assertLessEqual(row["absolute_error"],.003+.03*abs(row["autograd"]),row)

    def test_each_physical_group_has_resolved_directional_difference(self):
        for device in (["cpu","cuda"] if torch.cuda.is_available() else ["cpu"]):
            raw,state=raw_fixture(device)
            _,records=directional_checks(raw,state)
            for name in GROUPS:
                rows=[x for x in records if x["group"]==name]
                with self.subTest(device=device,group=name):
                    self.assertLess(min(x["scaled_relative_error"] for x in rows),.01)
                    for row in rows:
                        if row["step"]>=.01:
                            self.assertLess(row["scaled_relative_error"],.02,row)
