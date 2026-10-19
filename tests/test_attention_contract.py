import unittest
from unittest.mock import patch
import torch
from query_minco.tests.common import model_and_inputs


class AttentionTests(unittest.TestCase):
    def test_diagnostic_shape_and_normalization(self):
        for batch in (1, 2):
            with self.subTest(batch=batch):
                model, depth, state = model_and_inputs(batch)
                model.eval()
                with torch.no_grad():
                    raw, diag = model(depth, state, return_diagnostics=True)
                weights = diag["cross_attention"]
                self.assertEqual(tuple(weights.shape), (batch, 2, 4, 6, 81))
                self.assertTrue(bool(torch.isfinite(weights).all()))
                self.assertTrue(bool((weights >= 0).all()))
                torch.testing.assert_close(weights.sum(-1), torch.ones_like(weights.sum(-1)))
                self.assertEqual(tuple(weights[..., :80].reshape(batch, 2, 4, 6, 8, 10).shape),
                                 (batch, 2, 4, 6, 8, 10))
                self.assertEqual(diag["state_token_index"], 80)
                torch.testing.assert_close(diag["context"][:, :80], diag["image_tokens"])
                torch.testing.assert_close(diag["context"][:, 80:], diag["state_token"])

    def test_default_and_diagnostics_eval_equivalence(self):
        # Nonzero configured dropout must still be disabled by eval().
        model, depth, state = model_and_inputs(dropout=0.1)
        model.eval()
        with torch.no_grad():
            plain = model(depth, state)
            debug, _ = model(depth, state, return_diagnostics=True)
        self.assertIsInstance(plain, torch.Tensor)
        torch.testing.assert_close(plain, debug, atol=1e-5, rtol=1e-5)

    def test_default_does_not_request_or_return_attention(self):
        model, depth, state = model_and_inputs()
        model.eval()
        for layer in model.query_decoder.layers:
            original = layer.cross_attention.forward
            with patch.object(layer.cross_attention, "forward", wraps=original) as spy:
                with torch.no_grad():
                    result = model(depth, state)
                self.assertIsInstance(result, torch.Tensor)
                self.assertFalse(spy.call_args.kwargs["need_weights"])
                self.assertFalse(spy.call_args.kwargs["average_attn_weights"])

    def test_diagnostics_requests_per_head_weights(self):
        model, depth, state = model_and_inputs()
        model.eval()
        for layer in model.query_decoder.layers:
            original = layer.cross_attention.forward
            with patch.object(layer.cross_attention, "forward", wraps=original) as spy:
                with torch.no_grad():
                    model(depth, state, return_diagnostics=True)
                self.assertTrue(spy.call_args.kwargs["need_weights"])
                self.assertFalse(spy.call_args.kwargs["average_attn_weights"])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_eval_attention_equivalence(self):
        model, depth, state = model_and_inputs(device="cuda")
        model.eval()
        with torch.no_grad():
            plain = model(depth, state)
            debug, diag = model(depth, state, return_diagnostics=True)
        self.assertEqual(tuple(diag["cross_attention"].shape), (2, 2, 4, 6, 81))
        torch.testing.assert_close(plain, debug, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
