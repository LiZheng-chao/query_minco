from dataclasses import replace
import pytest
import torch
from query_minco.training.physics_only_v1.spec import PhysicsSpec
from query_minco.training.physics_only_v1.objective import PhysicsOnlyObjective, aggregate_costs
from query_minco.training.physics_only_v1.costs import SyntheticGeometry
from query_minco.training.losses.set_physical_risk import box_sdf
from query_minco.trajectory.minco_adapter import constraint_residuals


def test_fixed_horizon_rejects_changes():
    with pytest.raises(ValueError, match="fixes"):
        PhysicsSpec(total_time_s=2.1)


def test_aabb_sdf_sign_distance_and_gradient():
    p = torch.tensor([[.5, .5, .5], [1., .5, .5], [2., .5, .5], [2., 2., .5]], requires_grad=True)
    boxes = torch.tensor([[[0., 0., 0.], [1., 1., 1.]]])
    d = box_sdf(p, boxes)
    torch.testing.assert_close(d, torch.tensor([-.5, 0., 1., 2.**.5]))
    assert torch.isfinite(torch.autograd.grad(d.sum(), p)[0]).all()


@pytest.mark.parametrize("mode", ["mean", "softmin"])
def test_query_permutation_values_gradients_and_minco_continuity(mode):
    torch.manual_seed(11)
    raw = (torch.randn(1, 6, 21) * .03).requires_grad_()
    state = torch.zeros(1, 12)
    state[0, 6] = 4.; state[0, 11] = -1.
    geom = [SyntheticGeometry(torch.tensor([[[.9, -.3, -.3], [1.2, .3, .3]]]),
                              torch.tensor([[0., -4., -2.], [10., 4., 2.]]), False)]
    obj = PhysicsOnlyObjective(PhysicsSpec(aggregation=mode))
    first = obj(raw, state, geom)
    perm = torch.tensor([3, 0, 5, 2, 1, 4])
    permuted_raw = raw.detach()[:, perm].requires_grad_()
    second = obj(permuted_raw, state, geom)
    torch.testing.assert_close(first.loss, second.loss, rtol=2e-5, atol=1e-5)
    for key in ("J", "C", "F", "S", "G_task", "pi", "violation", "quality"):
        torch.testing.assert_close(first.metrics[key][:, perm], second.metrics[key], rtol=2e-5, atol=1e-5)
    g1 = torch.autograd.grad(first.loss, raw)[0]
    g2 = torch.autograd.grad(second.loss, permuted_raw)[0]
    torch.testing.assert_close(g1[:, perm], g2, rtol=2e-4, atol=1e-4)
    p, tr = first.parameters, first.trajectory
    residuals, valid = constraint_residuals(tr.coeffs, tr.durations, p.head_pva, p.tail_pva, p.inner_pts)
    assert valid.all()
    assert set(residuals) == {"p", "v", "a", "jerk", "snap"}


def test_teacher_independence_in_fresh_process():
    import subprocess
    import sys
    code = """
import builtins
original = builtins.__import__
def guard(name, *args, **kwargs):
    if any(x in name for x in ('teacher', 'datasets.manifest', 'geometry_sidecar', 'pg2_guided')):
        raise RuntimeError('forbidden dependency: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guard
from query_minco.training.physics_only_v1.dataset import PhysicsOnlyDataset
from query_minco.training.physics_only_v1.objective import PhysicsOnlyObjective
"""
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("distance", [0., 3.2, 3.4, 3.6, 6.])
def test_goal_analytic_gradient_matches_finite_difference(distance):
    from query_minco.training.physics_only_v1.costs import goal_cost
    endpoint = torch.tensor([[[.43, .11, -.07]]]*6, dtype=torch.float64).reshape(1, 6, 3).requires_grad_()
    velocity = torch.full_like(endpoint, .3, requires_grad=True)
    goal = torch.tensor([[distance, 0., 0.]], dtype=torch.float64)
    def function(p, v):
        task, progress, _ = goal_cost(p, v, goal, torch.ones(1, dtype=torch.bool), PhysicsSpec())
        return task + progress
    assert torch.autograd.gradcheck(function, (endpoint, velocity), eps=1e-6, atol=1e-5, rtol=1e-4)


def test_exact_jerk_gauss_integral_constant_polynomial():
    from types import SimpleNamespace
    from query_minco.training.pg1_v2_r1.losses import dynamic_quality_loss
    coeffs = torch.zeros(1, 6, 3, 3, 6)
    coeffs[..., 0, 3] = 1.  # x=t^3 -> jerk=6, integral over T=2 is 72.
    trajectory = SimpleNamespace(coeffs=coeffs, durations=torch.full((1, 6, 3), 2/3),
                                 solve_valid=torch.ones(1, 6, dtype=torch.bool))
    result = dynamic_quality_loss(trajectory, enabled=("jerk",))
    torch.testing.assert_close(result["integrals_si"]["jerk"], torch.full((1, 6), 72.))
