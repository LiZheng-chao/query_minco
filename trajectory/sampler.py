"""Analytic sampling over global physical seconds; endpoints are included."""
import math
import torch
from ._validation import tensor, matching
from .types import MincoTrajectory, TrajectorySamples


def evaluate_piece(coeffs, local_times, derivative=0):
    """coeffs [...,xyz,6], times [...,S] -> [...,S,xyz], using Horner."""
    if type(derivative) is not int or not 0 <= derivative <= 5:
        raise ValueError("derivative must be an integer in [0,5]")
    if coeffs.shape[-2:] != (3, 6) or local_times.shape[:-1] != coeffs.shape[:-2]:
        raise ValueError("piece coefficients/times leading shapes must match")
    if coeffs.device != local_times.device or coeffs.dtype != local_times.dtype:
        raise ValueError("piece coefficients/times device and dtype must match")
    t = local_times.unsqueeze(-1)
    out = torch.zeros_like(t * coeffs[..., 0, 0].unsqueeze(-1).unsqueeze(-1))
    for k in range(5, derivative - 1, -1):
        scale = math.factorial(k) / math.factorial(k - derivative)
        out = out * t + coeffs[..., :, k].unsqueeze(-2) * scale
    return out


class TrajectorySampler:
    def __call__(self, trajectory, num_samples=64, times=None):
        if not isinstance(trajectory, MincoTrajectory):
            raise TypeError("trajectory must be MincoTrajectory")
        coeffs, durations = trajectory.coeffs, trajectory.durations
        tensor(coeffs, "coeffs", (6, 3, 3, 6), fp32=True)
        tensor(durations, "durations", (6, 3), fp32=True)
        matching((coeffs, durations))
        if not bool((durations > 0).all()):
            raise ValueError("durations must be positive")
        if trajectory.solve_valid.shape != durations.shape[:2] or not bool(trajectory.solve_valid.all()):
            raise ValueError("cannot sample an invalid numerical solution")
        total = durations.sum(-1)
        if times is None:
            if type(num_samples) is not int or num_samples < 2:
                raise ValueError("num_samples must be an integer >=2")
            times = total[..., None] * torch.linspace(0, 1, num_samples, device=coeffs.device)
        else:
            if not isinstance(times, torch.Tensor) or times.ndim != 3 or times.shape[:2] != total.shape or times.shape[-1] < 1:
                raise ValueError("times must have shape [B,6,S], S>=1")
            if times.device != coeffs.device or times.dtype != torch.float32:
                raise ValueError("times must be float32 on the coefficient device")
            if not bool(torch.isfinite(times).all()) or not bool(((times >= 0) & (times <= total[..., None])).all()):
                raise ValueError("times must be finite and within [0,T]; no silent clamping")
        with torch.autocast(device_type=coeffs.device.type, enabled=False):
            boundaries = durations.cumsum(-1)
            indices = (times[..., None] >= boundaries[..., None, :2]).sum(-1)
            starts = torch.cat((torch.zeros_like(boundaries[..., :1]), boundaries[..., :2]), dim=-1)
            local = times - starts.gather(-1, indices)
            selected = coeffs.gather(2, indices[..., None, None].expand(-1, -1, -1, 3, 6))
            values = [evaluate_piece(selected, local[..., None], d).squeeze(-2) for d in range(4)]
            if not all(bool(torch.isfinite(x).all()) for x in values):
                raise ValueError("sampling overflow: nonfinite polynomial output")
        return TrajectorySamples(times, *values, indices, local)
