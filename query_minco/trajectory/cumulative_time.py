"""Execution revision E1: differentiable explicit prefix sums for exactly 3 pieces."""
import torch
def cumulative_three(durations):
    if durations.ndim < 1 or durations.shape[-1] != 3:
        raise ValueError("exactly three durations required")
    if durations.dtype not in (torch.float32, torch.float64):
        raise TypeError("FP32 or FP64 required")
    a, b, c = durations.unbind(-1)
    ab = a + b
    return torch.stack((a, ab, ab + c), dim=-1)
