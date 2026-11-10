import dataclasses
import math
import pytest
import torch
from query_minco.training.physics_only_v1.spec import PhysicsSpec
from query_minco.training.physics_only_v1.decoder import FixedTimeDecoder
from query_minco.training.physics_only_v1.costs import SyntheticGeometry, goal_cost, collision_cost
from query_minco.training.physics_only_v1.objective import PhysicsOnlyObjective, aggregate_costs


def scene(*, blocked=False, certified=True):
    boxes = torch.tensor([[[.9, -.3, -.3], [1.2, .3, .3]]], dtype=torch.float32) if blocked else torch.empty(0, 2, 3)
    domain = torch.tensor([[0., -4., -2.], [10., 4., 2.]], dtype=torch.float32)
    return SyntheticGeometry(boxes, domain, certified)


def state(goal=(2.4, 0., 0.)):
    s = torch.zeros(1, 12)
    s[0, :3] = torch.tensor([.8, 0., 0.])
    s[0, 6:9] = torch.tensor(goal)
    s[0, 11] = -1.
    return s


def test_fixed_time_no_gradient_to_time_or_scoring():
    raw = torch.zeros(1, 6, 21, requires_grad=True)
    p = FixedTimeDecoder(PhysicsSpec())(raw, state())
    assert p.durations.shape == (1, 6, 3)
    torch.testing.assert_close(p.durations, torch.full_like(p.durations, 2/3), atol=1e-7, rtol=0)
    result = PhysicsOnlyObjective()(raw, state(), [scene()])
    assert result.trajectory.coeffs.shape == (1, 6, 3, 3, 6)
    assert torch.isfinite(result.loss)
    result.loss.backward()
    assert raw.grad is not None and torch.isfinite(raw.grad).all()
    assert raw.grad[..., :15].abs().sum() > 0
    assert raw.grad[..., 15:21].abs().max() == 0


def test_blocked_loss_changes_with_geometry_and_backprop():
    raw = torch.zeros(1, 6, 21, requires_grad=True)
    obj = PhysicsOnlyObjective()
    free = obj(raw, state(), [scene()])
    occupied = obj(raw, state(), [scene(blocked=True, certified=False)])
    assert (free.metrics['C'] == 0).all()
    assert (occupied.metrics['C'] > free.metrics['C']).all()
    g = torch.autograd.grad(occupied.metrics['C'].sum(), raw)[0]
    assert torch.isfinite(g).all() and g[..., :9].abs().sum() > 0


def test_missing_or_bad_geometry_fails_closed():
    raw = torch.zeros(1, 6, 21)
    obj = PhysicsOnlyObjective()
    for x in ([], [scene(certified=False)], [None], [scene(blocked=True, certified=True)]):
        with pytest.raises(ValueError):
            obj(raw, state(), x)


def test_goal_zero_overshoot_and_disabled_mask():
    spec = PhysicsSpec()
    goal = torch.zeros(1, 3)
    terminal = torch.zeros(1, 6, 3)
    p = torch.zeros(1, 6, 3, requires_grad=True)
    task, progress, _ = goal_cost(p, terminal, goal, torch.ones(1, dtype=torch.bool), spec)
    assert torch.equal(progress, torch.zeros_like(progress))
    assert torch.isfinite(torch.autograd.grad((task+progress).sum(),p)[0]).all()
    p2 = torch.ones(1, 6, 3)
    _, p2cost, _ = goal_cost(p2,terminal,goal,torch.ones(1,dtype=torch.bool),spec)
    assert (p2cost > progress).all()
    disabled = torch.zeros(1, dtype=torch.bool)
    t, prog, _ = goal_cost(p2,terminal,goal,disabled,spec)
    assert (t == 0).all() and (prog == 0).all()


def test_continuous_task_at_34_boundary():
    spec = PhysicsSpec()
    p = torch.zeros(1,6,3, requires_grad=True)
    p.data[...,0] = 1.
    v = torch.zeros_like(p)
    out = []
    for d in [3.4-1e-4, 3.4+1e-4]:
        val, progress, _ = goal_cost(p,v,torch.tensor([[d,0.,0.]]),torch.ones(1,dtype=torch.bool),spec)
        out.append(val.detach())
        assert torch.isfinite(torch.autograd.grad((val+progress).sum(),p)[0]).all()
    assert (out[0]-out[1]).abs().max() < .005


def test_set_bounds_and_softmax_grad():
    spec = PhysicsSpec(guard_weight=0.)
    base = torch.tensor([[0.2,1.,2.,3.,4.,5.]], dtype=torch.float64, requires_grad=True)
    zero = torch.zeros_like(base)
    costs = dict(C=base, F=zero, G_task=zero, Progress=zero, S=zero)
    L, diag = aggregate_costs(costs,spec)
    assert base.min() <= L <= base.mean()
    (grad,) = torch.autograd.grad(L,base)
    torch.testing.assert_close(grad, torch.softmax(-base/spec.set_tau,dim=1), rtol=1e-12, atol=1e-12)
    assert 1 <= diag['K_eff'].item() <= 6
    L2, _ = aggregate_costs({**costs, 'C': base.flip(1)}, spec)
    torch.testing.assert_close(L, L2)


def test_guard_all_queries_have_derivative():
    spec = PhysicsSpec(guard_weight=.1, wc=0., wg=0., wf=0., wj=0.)
    x = torch.arange(1,7,dtype=torch.float64).view(1,6).requires_grad_()
    zero = torch.zeros_like(x)
    L, _ = aggregate_costs(dict(C=x, F=zero, G_task=zero, Progress=zero, S=zero),spec)
    (grad,) = torch.autograd.grad(L,x)
    torch.testing.assert_close(grad,torch.full_like(grad,.1/6),atol=1e-12,rtol=1e-12)


def test_stable_softmin_extreme_values():
    spec = PhysicsSpec()
    x = torch.tensor([[0,1e5,1e7,1e10,1e20,1e25]],dtype=torch.float32)
    z = torch.zeros_like(x)
    L, d = aggregate_costs(dict(C=x,F=z,G_task=z,Progress=z,S=z),spec)
    assert torch.isfinite(L) and torch.isfinite(d['pi']).all()


def test_strict_no_teacher_module_imports():
    import ast
    from pathlib import Path
    path = Path(__file__).resolve().parents[1]
    for file in path.glob('*.py'):
        if file.name == '__init__.py': continue
        code = file.read_text()
        tree = ast.parse(code)
        names = [node.module or '' for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)]
        assert not any('teacher' in name.lower() or 'matching' in name.lower() or 'pg2_guided' in name for name in names)
        # teacher in explanatory docstrings/README is okay; not loaded by runtime.


def test_goal_ray_diagnostic_never_changes_duration():
    raw = torch.zeros(1,6,21,requires_grad=True)
    p = FixedTimeDecoder(PhysicsSpec(position_anchor='goal_ray'))(raw, state())
    assert p.inner_pts.shape == (1,6,2,3)
    assert torch.isfinite(p.inner_pts).all()
    assert (p.inner_pts[...,0] > 0).all()
    assert (p.durations == p.durations[..., :1]).all()
    p.inner_pts.sum().backward()
    assert raw.grad[..., :9].abs().sum() > 0
    assert raw.grad[..., 15:].abs().sum() == 0


def test_goal_progress_monotonic_in_far_task_pass_region():
    spec = PhysicsSpec()
    g = torch.tensor([[6.,0.,0.]])
    p = torch.zeros(1,6,3)
    v = torch.zeros_like(p)
    p[...,0] = 1.
    t1, r1, _ = goal_cost(p,v,g,torch.ones(1,dtype=torch.bool),spec)
    p[...,0] = 5.
    t2, r2, _ = goal_cost(p,v,g,torch.ones(1,dtype=torch.bool),spec)
    assert (r2 < r1).all()
    assert (t1 == 0).all() and (t2 == 0).all()


def test_sampling_physical_weights_sum_one_and_jerk_nonnegative():
    raw = torch.zeros(1,6,21)
    obj = PhysicsOnlyObjective()
    out = obj(raw, state(), [scene()])
    from query_minco.training.pg1.losses import analytic_samples
    ana = analytic_samples(out.trajectory, 64)
    torch.testing.assert_close(ana['weights'].sum(-1), torch.ones(1,6), atol=1e-6,rtol=0)
    assert out.metrics['S'].shape == (1,6)
    assert (out.metrics['S'] >= 0).all()
    assert (out.metrics['F'] >= 0).all()
