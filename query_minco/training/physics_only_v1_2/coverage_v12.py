"""Pure offline valid-diversity interface, disconnected from generation loss.

Caller supplies qualification from independent Goal/collision/dynamics/domain checks.
This function cannot certify those labels. No selector is implemented or enabled.
"""
import math
import torch
from .goal_v12 import finite_tensor


def valid_diversity(paths, qualified, relative_times, *, minimum_distance=.5,
                    epsilon_distance=1e-6, epsilon_mass=1e-8):
    finite_tensor(paths, "coverage paths")
    finite_tensor(relative_times, "relative times")
    if paths.ndim != 4 or paths.shape[1] != 6 or paths.shape[-1] != 3 or not len(paths):
        raise ValueError("paths [B,6,M,3] required")
    n = paths.shape[2]
    if n < 2 or relative_times.shape != (n,) or relative_times.dtype != paths.dtype or relative_times.device != paths.device:
        raise ValueError("common relative time grid required")
    expected = torch.linspace(0, 1, n, dtype=paths.dtype, device=paths.device)
    if relative_times.requires_grad or not torch.allclose(relative_times, expected, atol=1e-7, rtol=0):
        raise ValueError("common uniform relative grid [0,1] required")
    if not isinstance(qualified, torch.Tensor) or qualified.shape != paths.shape[:2] or qualified.device != paths.device:
        raise ValueError("qualified [B,6] required")
    if qualified.dtype != torch.bool:
        finite_tensor(qualified, "qualification")
        if not bool(((qualified == 0) | (qualified == 1)).all()):
            raise ValueError("unknown/nonbinary qualification must not become valid")
    for x in (minimum_distance, epsilon_distance, epsilon_mass):
        if type(x) not in (int, float) or not math.isfinite(x) or x <= 0:
            raise ValueError("positive finite diversity scales required")
    a = qualified.detach().to(paths.dtype)
    ii, jj = torch.triu_indices(6, 6, 1, device=paths.device)
    difference = paths[:, ii] - paths[:, jj]
    distance = (difference.square().sum(-1).mean(-1) + epsilon_distance**2).sqrt()
    mass = a[:, ii] * a[:, jj]
    count = mass.sum(-1)
    per_scene = (mass * (minimum_distance - distance).clamp_min(0).square()).sum(-1) / (count + epsilon_mass)
    # Even zero mass keeps a differentiable exact zero in paths, never labels.
    per_scene = torch.where(count > 0, per_scene, paths.sum((1, 2, 3)) * 0)
    return per_scene.mean(), dict(per_scene=per_scene, qualified=a, pair_mass=count,
                                 pair_distances=distance, topology_claim=False)
