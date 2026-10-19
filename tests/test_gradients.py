import unittest
import torch
from query_minco.tests.common import model_and_inputs, gradient_summary


class GradientTests(unittest.TestCase):
    def assert_gradients(self, model):
        for name, summary in gradient_summary(model).items():
            with self.subTest(module=name):
                self.assertTrue(summary["all_finite"], summary)
                self.assertEqual(summary["with_gradient"], summary["parameter_tensors"])
                self.assertGreater(summary["nonzero_tensors"], 0)
                self.assertGreater(summary["l2_norm"], 0)

    def backward_once(self, device="cpu", amp=False, diagnostics=False):
        model, depth, state = model_and_inputs(device=device)
        depth.requires_grad_()
        state.requires_grad_()
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=amp):
            result = model(depth, state, return_diagnostics=diagnostics)
            raw = result[0] if diagnostics else result
            target = torch.randn_like(raw)
            loss = (raw.float() - target.float()).square().mean()
        self.assertTrue(bool(torch.isfinite(raw).all()))
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        self.assert_gradients(model)
        for tensor in (depth, state):
            self.assertTrue(bool(torch.isfinite(tensor.grad).all()))
            self.assertGreater(float(tensor.grad.abs().sum()), 0)

    def test_cpu_all_major_module_and_input_gradients(self):
        self.backward_once()

    def test_cpu_diagnostic_path_backward(self):
        self.backward_once(diagnostics=True)

    def test_state_and_depth_sensitivity(self):
        model, depth, state = model_and_inputs()
        model.eval()
        with torch.no_grad():
            reference = model(depth, state)
            changed_state = model(depth, state + 0.25)
            changed_depth = depth.clone()
            changed_depth[:, 0] = 1 - changed_depth[:, 0]
            changed_image = model(changed_depth, state)
        self.assertGreater(float((reference - changed_state).abs().max()), 1e-7)
        self.assertGreater(float((reference - changed_image).abs().max()), 1e-7)

    def test_explicit_query_conditioning(self):
        from unittest.mock import patch
        model, _, _ = model_and_inputs()
        decoder = model.query_decoder
        context = torch.randn(2, 81, 128)
        state_embed = torch.randn(2, 128)
        seen = []
        handle = decoder.layers[0].register_forward_pre_hook(
            lambda module, args: seen.append(args[0].detach().clone()))
        try:
            decoder(context, state_embed)
        finally:
            handle.remove()
        torch.testing.assert_close(seen[0], decoder.learned_queries[None] + state_embed[:, None])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_forward_backward(self):
        self.backward_once(device="cuda")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_amp_forward_backward(self):
        self.backward_once(device="cuda", amp=True)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_migration_and_mixed_device_errors(self):
        model, depth, state = model_and_inputs()
        model = model.cuda()
        with self.assertRaisesRegex(ValueError, "model must be on"):
            model(depth, state)
        with self.assertRaisesRegex(ValueError, "same device"):
            model(depth.cuda(), state)
        with torch.no_grad():
            raw = model(depth.cuda(), state.cuda())
        self.assertTrue(bool(torch.isfinite(raw).all()))
        model = model.cpu()
        with torch.no_grad():
            self.assertEqual(tuple(model(depth, state).shape), (2, 6, 21))


if __name__ == "__main__":
    unittest.main()
