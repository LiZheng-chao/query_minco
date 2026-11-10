import math
import pytest
import torch

from query_minco.training.physics_only_v1_2.goal_v12 import phi2, goal_terms
from query_minco.training.physics_only_v1.costs import goal_cost as s0_goal
from query_minco.training.physics_only_v1.spec import PhysicsSpec as S0Spec
from query_minco.training.pg1.losses import goal_task_loss as strict_old_task


def values(distance=2.4, endpoint=.0, speed=.0, dtype=torch.float64):
    p = torch.zeros(1, 6, 3, dtype=dtype); p[..., 0] = endpoint
    v = torch.zeros_like(p); v[..., 0] = speed
    return p, v, torch.tensor([[distance, 0., 0.]], dtype=dtype), torch.ones(1, dtype=torch.bool)


def test_phi_value_small_limit_derivative_bound_and_gradcheck():
    z = torch.tensor([0., 1e-8, 1., 15., 1e6], dtype=torch.float64, requires_grad=True)
    actual = phi2(z)
    assert actual[0] == 0
    torch.testing.assert_close(actual[1], z[1].square(), rtol=1e-12, atol=1e-30)
    grad = torch.autograd.grad(actual.sum(), z)[0]
    torch.testing.assert_close(grad, 2*z/(1+(z/2).square()).sqrt())
    assert grad[0] == 0 and (grad >= 0).all() and (grad < 4).all()
    assert torch.autograd.gradcheck(phi2, (torch.tensor([.3, 2., 15.], dtype=torch.float64, requires_grad=True),))
    with pytest.raises(ValueError): phi2(torch.tensor([-1.]))
    with pytest.raises(ValueError): phi2(torch.tensor([float("nan")]))


def test_fixed_three_distinct_contracts_and_gradient():
    p, v, g, enabled = values()
    p.requires_grad_()
    new = goal_terms(p, v, g, enabled, ("normal",))
    strict, _ = strict_old_task(p, v, g, enabled)
    old, _, _ = s0_goal(p, v, g, enabled, S0Spec())
    assert strict[0, 0].item() == pytest.approx(225.)
    assert old[0, 0].item() == pytest.approx(.87890625)
    assert new["G_N"][0, 0].item() == pytest.approx(52.530983801686226, abs=1e-12)
    dx = torch.autograd.grad(new["G_N"][0, 0], p)[0][0, 0, 0]
    assert dx.item() == pytest.approx(-26.4327440182, abs=1e-9)


@pytest.mark.parametrize("distance", [1e-8, 2.4, 3.2-1e-5, 3.2, 3.2+1e-5,
                                     3.4-1e-5, 3.4, 3.4+1e-5,
                                     3.6-1e-5, 3.6, 3.6+1e-5, 6.])
def test_goal_double_gradcheck_and_shapes(distance):
    p, v, g, enabled = values(distance, .43, .31)
    p[..., 1] = .11; v[..., 2] = .09
    p.requires_grad_(); v.requires_grad_()
    def fn(pp, vv):
        d = goal_terms(pp, vv, g, enabled, ("normal",))
        return d["G_task_train"] + .25*d["P"]
    assert torch.autograd.gradcheck(fn, (p, v), eps=1e-6, atol=1e-5, rtol=1e-4)
    assert all(x.shape == (1, 6) and torch.isfinite(x).all() for x in goal_terms(p, v, g, enabled, ("normal",)).values())


@pytest.mark.parametrize("distance", [0., 1e-8, 2.4, 3.4, 6.])
@pytest.mark.parametrize("enabled_value", [False, True])
def test_zero_exact_arrival_overshoot_mask_and_progress_gradient(distance, enabled_value):
    p, v, g, mask = values(distance, distance, .0)
    mask[:] = enabled_value
    p.requires_grad_(); v.requires_grad_()
    tasks = ("hover",) if distance == 0 else ("normal",)
    d = goal_terms(p, v, g, mask, tasks)
    assert (d["P"] == 0).all()
    for key in ("P", "G_task_train"):
        gp, gv = torch.autograd.grad(d[key].sum(), (p, v), retain_graph=True, allow_unused=True)
        assert gp is not None and torch.isfinite(gp).all()
        assert not torch.count_nonzero(gp)  # at exact goal
        if gv is not None: assert not torch.count_nonzero(gv)
    offset = p.detach() + .2
    offset.requires_grad_()
    dd = goal_terms(offset, v, g, mask, tasks)
    gp = torch.autograd.grad(dd["P"].sum(), offset)[0]
    expected = (offset-g[:, None]) / (g.norm(dim=-1).clamp_min(1.)[:, None, None] *
               ((offset-g[:, None]).square().sum(-1, keepdim=True)+.05**2).sqrt())
    torch.testing.assert_close(gp, expected * enabled_value)
    if enabled_value: assert (dd["P"] > 0).all()
    else: assert (dd["P"] == 0).all() and (dd["G_task_train"] == 0).all()
    assert torch.autograd.gradcheck(
        lambda pp, vv: goal_terms(pp, vv, g, mask, tasks)["G_task_train"] +
                       goal_terms(pp, vv, g, mask, tasks)["P"], (offset, v))


def test_disabled_goal_direct_velocity_and_position_gradients_zero():
    p, v, g, enabled = values(2.4, .4, .8)
    p.requires_grad_(); v.requires_grad_(); enabled[:] = False
    d = goal_terms(p, v, g, enabled, ("normal",))
    grads = torch.autograd.grad((d["G_task_train"] + d["P"]).sum(), (p, v))
    assert all(torch.count_nonzero(x) == 0 for x in grads)


def test_progress_monotonic_inside_tolerance_and_after_far_pass():
    for distance, a, b in [(2.4, 2.3, 2.35), (6., 1., 5.)]:
        p, v, g, enabled = values(distance, a)
        first = goal_terms(p, v, g, enabled, ("normal",))
        p[..., 0] = b
        second = goal_terms(p, v, g, enabled, ("normal",))
        assert (second["P"] < first["P"]).all()
        assert (first["G_task_train"] == 0).all() and (second["G_task_train"] == 0).all()


def test_terminal_speed_transition_and_goal_derivative_continuity():
    for distance, expected in [(2.4, 1.), (3.4, .5), (6., 0.)]:
        p, v, g, enabled = values(distance, distance, .5)
        d = goal_terms(p, v, g, enabled, ("normal",))
        assert d["G_task_train"][0, 0].item() == pytest.approx(expected * 8*(math.sqrt(1.25)-1))
    for boundary in (3.2, 3.4, 3.6):
        outputs, gradients = [], []
        for side in (-1e-7, 1e-7):
            p, v, g, enabled = values(boundary+side, .5, .3)
            g.requires_grad_()
            d = goal_terms(p, v, g, enabled, ("normal",))
            outputs.append(d["G_task_train"][0, 0].item())
            gradients.append(torch.autograd.grad(d["G_task_train"][0, 0], g)[0])
        assert abs(outputs[0]-outputs[1]) < 1e-4
        torch.testing.assert_close(gradients[0], gradients[1], atol=1e-3, rtol=1e-4)


@pytest.mark.parametrize("fault", ["zero", "nan", "frame", "units", "task", "mask", "shape", "hover_nonzero"])
def test_illegal_goal_fails_closed(fault):
    p, v, g, enabled = values()
    tasks, kwargs = ("normal",), {}
    if fault == "zero": g.zero_()
    if fault == "nan": g[0, 0] = float("nan")
    if fault == "frame": kwargs["frame"] = "world"
    if fault == "units": kwargs["units"] = "cm"
    if fault == "task": tasks = ("unknown",)
    if fault == "mask": enabled = enabled.float()
    if fault == "shape": p = p[:, :5]
    if fault == "hover_nonzero": tasks = ("hover",)
    with pytest.raises(ValueError): goal_terms(p, v, g, enabled, tasks, **kwargs)
