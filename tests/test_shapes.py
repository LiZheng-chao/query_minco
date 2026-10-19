import unittest
import torch
from dataclasses import asdict
from unittest.mock import patch
from query_minco import QueryMincoConfig, QueryMincoPlanner
from query_minco.depth_encoder import image_position_encoding
from query_minco.tests.common import model_and_inputs


class ShapeTests(unittest.TestCase):
    def test_cpu_batch_one_and_two_all_stages(self):
        expected = {"d1": (24, 64, 80), "d2": (32, 32, 40), "d3": (64, 16, 20),
                    "d4": (96, 8, 10), "d5": (96, 8, 10), "d6": (128, 8, 10)}
        for batch in (1, 2):
            with self.subTest(batch=batch):
                model, depth, state = model_and_inputs(batch)
                observed = {}
                handles = [getattr(model.depth_encoder, name).register_forward_hook(
                    lambda module, args, output, key=name: observed.update({key: tuple(output.shape)}))
                    for name in expected]
                try:
                    raw, diag = model(depth, state, return_diagnostics=True)
                finally:
                    for handle in handles:
                        handle.remove()
                self.assertEqual(tuple(raw.shape), (batch, 6, 21))
                self.assertTrue(bool(torch.isfinite(raw).all()))
                for name, shape in expected.items():
                    self.assertEqual(observed[name], (batch,) + shape)
                for key, shape in {"image_tokens": (80, 128), "state_token": (1, 128),
                                   "context": (81, 128), "decoder_tokens": (6, 128)}.items():
                    self.assertEqual(tuple(diag[key].shape), (batch,) + shape)

    def test_wrong_resolution_channels_state_and_rank(self):
        model, depth, state = model_and_inputs()
        cases = [
            (torch.zeros(2, 2, 96, 160), state, "depth_input"),
            (torch.zeros(2, 1, 128, 160), state, "depth_input"),
            (torch.zeros(2, 2, 128, 159), state, "depth_input"),
            (torch.zeros(2, 128, 160), state, "depth_input"),
            (depth, torch.zeros(2, 16), "state"),
            (depth, torch.zeros(2, 12, 1), "state"),
        ]
        for d, s, message in cases:
            with self.subTest(depth=tuple(d.shape), state=tuple(s.shape)):
                with self.assertRaisesRegex(ValueError, message):
                    model(d, s)

    def test_batch_mismatch_and_empty(self):
        model, depth, state = model_and_inputs()
        with self.assertRaisesRegex(ValueError, "batch sizes"):
            model(depth, state[:1])
        with self.assertRaisesRegex(ValueError, "nonempty"):
            model(depth[:0], state[:0])

    def test_non_tensor_integer_and_mismatched_dtype(self):
        model, depth, state = model_and_inputs()
        for d, s, message in [(None, state, "torch.Tensor"),
                              (depth.long(), state, "floating"),
                              (depth, state.double(), "same dtype"),
                              (depth.double(), state.double(), "model dtype")]:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, message):
                    model(d, s)
        with self.assertRaisesRegex(TypeError, "return_diagnostics"):
            model(depth, state, return_diagnostics=1)

    def test_nonfinite_inputs(self):
        model, depth, state = model_and_inputs()
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                d = depth.clone()
                d[0, 0, 0, 0] = value
                with self.assertRaisesRegex(ValueError, "depth_input contains"):
                    model(d, state)
                s = state.clone()
                s[0, 0] = value
                with self.assertRaisesRegex(ValueError, "state contains"):
                    model(depth, s)

    def test_fixed_config_rejects_old_architecture(self):
        for key, value in [("num_queries", 8), ("state_dim", 16), ("depth_height", 96),
                           ("decoder_layers", 3), ("output_dim", 14), ("attention_heads", True),
                           ("dropout", float("nan")), ("dropout", 1.0)]:
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    QueryMincoConfig(**{key: value})
        with self.assertRaises(TypeError):
            QueryMincoPlanner(config={})

    def test_configuration_snapshot(self):
        # YAML is a flat snapshot; no YAML library is needed by the network or tests.
        from pathlib import Path
        entries = {}
        path = Path(__file__).resolve().parents[1] / "configs/v1_network.yaml"
        for line in path.read_text().splitlines():
            if line.strip() and not line.lstrip().startswith("#"):
                key, value = line.split(":")
                entries[key.strip()] = float(value) if "." in value else int(value)
        self.assertEqual(entries, asdict(QueryMincoConfig()))

    def test_position_encoding_layout_and_buffer(self):
        model, _, _ = model_and_inputs()
        pe = image_position_encoding()[0]
        self.assertEqual(tuple(pe.shape), (80, 128))
        torch.testing.assert_close(pe[0, :64], pe[1, :64])
        torch.testing.assert_close(pe[0, 64:], pe[10, 64:])
        self.assertFalse(torch.equal(pe[0, :64], pe[10, :64]))
        self.assertFalse(torch.equal(pe[0, 64:], pe[1, 64:]))
        self.assertIn("depth_encoder.position_encoding", dict(model.named_buffers()))
        self.assertNotIn("depth_encoder.position_encoding", dict(model.named_parameters()))

    def test_module_dtype_migration(self):
        model, depth, state = model_and_inputs(1)
        model = model.double()
        with torch.no_grad():
            raw = model(depth.double(), state.double())
        self.assertEqual(raw.dtype, torch.float64)
        self.assertTrue(bool(torch.isfinite(raw).all()))

    def test_no_baseline_imports(self):
        import subprocess, sys
        program = ("import sys; from query_minco import QueryMincoPlanner; "
                   "assert not any(n == 'rospy' or n == 'minco' or n == 'policy' "
                   "or n.startswith(('YOPO.', 'policy.', 'loss.', 'neural_planner_v1.')) "
                   "for n in sys.modules)")
        subprocess.run([sys.executable, "-B", "-c", program], check=True)


if __name__ == "__main__":
    unittest.main()
