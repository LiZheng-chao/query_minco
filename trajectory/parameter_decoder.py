"""Raw logits to B0 absolute-local waypoints and physical seconds, in FP32."""
from dataclasses import dataclass
from math import isfinite
from typing import Tuple
import torch
from ._validation import tensor, matching
from .types import PhysicalParameters


@dataclass(frozen=True)
class PhysicalParameterConfig:
    # Illustrative forward workspace in metres, not a calibrated flight envelope.
    position_min_m: Tuple[float, float, float] = (0.0, -4.0, -2.0)
    position_max_m: Tuple[float, float, float] = (10.0, 4.0, 2.0)
    # Per-axis limits: vector norms can reach sqrt(sum(axis_limit**2)).
    terminal_velocity_axis_max_mps: Tuple[float, float, float] = (6.0, 4.0, 3.0)
    terminal_acceleration_axis_max_mps2: Tuple[float, float, float] = (6.0, 6.0, 4.0)
    min_total_time_s: float = 0.8
    max_total_time_s: float = 2.2
    min_segment_time_s: float = 0.12

    def __post_init__(self):
        vectors = (self.position_min_m, self.position_max_m,
                   self.terminal_velocity_axis_max_mps, self.terminal_acceleration_axis_max_mps2)
        for vector in vectors:
            if len(vector) != 3 or not all(isfinite(x) for x in vector):
                raise ValueError("all per-axis bounds require three finite values")
        if not all(a < b for a, b in zip(self.position_min_m, self.position_max_m)):
            raise ValueError("position lower bounds must be below upper bounds")
        if not all(x > 0 for vector in vectors[2:] for x in vector):
            raise ValueError("terminal axis limits must be positive")
        times = (self.min_total_time_s, self.max_total_time_s, self.min_segment_time_s)
        if not all(isfinite(t) for t in times):
            raise ValueError("time bounds must be finite")
        if not 0 < self.min_segment_time_s * 3 < self.min_total_time_s < self.max_total_time_s:
            raise ValueError("require 0 < 3*Tsegmin < Tmin < Tmax")


class PhysicalParameterDecoder:
    def __init__(self, config=None):
        self.config = config if config is not None else PhysicalParameterConfig()
        if not isinstance(self.config, PhysicalParameterConfig):
            raise TypeError("config must be PhysicalParameterConfig")

    def __call__(self, raw, state):
        tensor(raw, "raw", (6, 21))
        tensor(state, "state", (12,))
        matching((raw, state))
        cfg = self.config
        with torch.autocast(device_type=raw.device.type, enabled=False):
            r, s = raw.float(), state.float()
            lower = r.new_tensor(cfg.position_min_m)
            upper = r.new_tensor(cfg.position_max_m)
            points = lower + (r[..., :9].reshape(-1, 6, 3, 3).tanh() + 1) * 0.5 * (upper - lower)
            velocity = r[..., 9:12].tanh() * r.new_tensor(cfg.terminal_velocity_axis_max_mps)
            acceleration = r[..., 12:15].tanh() * r.new_tensor(cfg.terminal_acceleration_axis_max_mps2)
            total = cfg.min_total_time_s + (cfg.max_total_time_s - cfg.min_total_time_s) * r[..., 15].sigmoid()
            weights = r[..., 16:19].softmax(dim=-1)
            durations = cfg.min_segment_time_s + (total - 3 * cfg.min_segment_time_s)[..., None] * weights
            head = torch.stack((torch.zeros_like(s[:, :3]), s[:, :3], s[:, 3:6]), dim=1)
            head = head[:, None].expand(-1, 6, -1, -1)
            tail = torch.stack((points[..., 2, :], velocity, acceleration), dim=-2)
            finite = torch.isfinite(head).all(dim=(-2, -1)) & torch.isfinite(tail).all(dim=(-2, -1))
            positive = (durations > 0).all(-1)
            sum_error = (durations.sum(-1) - total).abs()
            valid = finite & positive & (sum_error <= 2e-6)
            if not bool(valid.all()):
                raise ValueError("decoded parameters are not finite or durations are inconsistent")
        return PhysicalParameters(
            head, tail, points[..., :2, :], durations, total,
            raw[..., 19], raw[..., 20],
            {"parameter_valid": valid, "time_sum_error_s": sum_error,
             "terminal_speed_norm_mps": velocity.norm(dim=-1),
             "terminal_acceleration_norm_mps2": acceleration.norm(dim=-1)})
