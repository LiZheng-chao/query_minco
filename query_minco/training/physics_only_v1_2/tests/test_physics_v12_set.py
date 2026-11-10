from dataclasses import replace
import math
import pytest
import torch
from query_minco.training.physics_only_v1_2.set_objective_v12 import aggregate_terms
from query_minco.training.physics_only_v1_2.spec import PhysicsV12Spec


def terms():
    torch.manual_seed(4)
    return {k: torch.rand(1, 6, dtype=torch.float64, requires_grad=True)
            for k in ("C_viol", "F", "G_task_train", "C_shape", "P", "S")}


@pytest.mark.parametrize("mode", ["foundation", "mean", "mean_guard", "softmin_guard"])
def test_vq_set_formula_gradients_and_permutation(mode):
    cfg = PhysicsV12Spec(phase="A" if mode == "foundation" else "B", aggregation=mode)
    t = terms()
    loss, d = aggregate_terms(t, cfg)
    V = t["C_viol"] + t["F"] + t["G_task_train"]
    Q = t["C_shape"] + .25*t["P"] + t["S"]
    J = V + Q
    manual_soft = -.3*torch.log(torch.exp(-J/.3).mean(-1))
    if mode == "foundation": expected = V.mean() + .1*Q.mean()
    elif mode == "mean": expected = J.mean()
    elif mode == "mean_guard": expected = J.mean() + .1*V.mean()
    else: expected = manual_soft.mean() + .1*V.mean()
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(d["V"], V); torch.testing.assert_close(d["Q"], Q)
    assert J.min() <= d["softmin_per_scene"].item() <= J.mean()
    assert 1 <= d["K_eff"].item() <= 6
    gt_grad, p_grad = torch.autograd.grad(loss, (t["G_task_train"], t["P"]), retain_graph=True)
    if mode == "softmin_guard":
        torch.testing.assert_close(gt_grad, cfg.w_g*(d["pi"]+.1/6))
        torch.testing.assert_close(p_grad, cfg.w_p*d["pi"])
    if mode == "foundation":
        torch.testing.assert_close(gt_grad, torch.full_like(gt_grad, 1/6))
        torch.testing.assert_close(p_grad, torch.full_like(p_grad, .1*.25/6))
    perm = torch.tensor([4, 2, 0, 5, 1, 3])
    perm_terms = {k: v.detach()[:, perm].requires_grad_() for k, v in t.items()}
    perm_loss, pd = aggregate_terms(perm_terms, cfg)
    torch.testing.assert_close(loss, perm_loss)
    for key in ("V", "Q", "J", "pi"):
        torch.testing.assert_close(d[key][:, perm], pd[key])
    gradients = torch.autograd.grad(loss, tuple(t.values()))
    perm_gradients = torch.autograd.grad(perm_loss, tuple(perm_terms.values()))
    for a, b in zip(gradients, perm_gradients): torch.testing.assert_close(a[:, perm], b)


def test_all_query_goal_guard_and_winner_progress_gradients():
    t = {k: torch.zeros(1, 6, dtype=torch.float64, requires_grad=True) for k in terms()}
    t["G_task_train"] = torch.tensor([[1., 5., 10., 20., 100., 1e3]], dtype=torch.float64, requires_grad=True)
    loss, d = aggregate_terms(t, PhysicsV12Spec())
    gt, p = torch.autograd.grad(loss, (t["G_task_train"], t["P"]))
    assert (gt >= .1/6).all()
    assert p[0, 0] > .24 and p[0, -1] < 1e-15
    torch.testing.assert_close(gt, d["pi"] + .1/6)


def test_set_double_gradcheck_temperature_limits_and_large_J():
    x = torch.tensor([[.2, 1., 2., 3., 4., 5.]], dtype=torch.float64, requires_grad=True)
    zero = torch.zeros_like(x)
    def objective(xx):
        return aggregate_terms(dict(C_viol=xx, F=zero, G_task_train=zero, C_shape=zero, P=zero, S=zero),
                               PhysicsV12Spec(lambda_all=0.))[0]
    assert torch.autograd.gradcheck(objective, (x,))
    t = dict(C_viol=x, F=zero, G_task_train=zero, C_shape=zero, P=zero, S=zero)
    small, sd = aggregate_terms(t, PhysicsV12Spec(set_tau=1e-5, lambda_all=0.))
    large, ld = aggregate_terms(t, PhysicsV12Spec(set_tau=1e6, lambda_all=0.))
    assert abs(small.item()-x.min().item()) < 2e-5
    assert abs(large.item()-x.mean().item()) < 1e-4
    assert sd["K_eff"] < ld["K_eff"]
    extreme = torch.tensor([[0., 1e5, 1e10, 1e15, 1e20, 1e25]])
    loss, d = aggregate_terms({k: extreme for k in terms()}, PhysicsV12Spec())
    assert torch.isfinite(loss) and torch.isfinite(d["pi"]).all()


@pytest.mark.parametrize("kwargs", [
    dict(points_per_segment=63), dict(total_time_s=2.2), dict(position_anchor="goal_ray"),
    dict(w_g=float("nan")), dict(w_p=0.), dict(set_tau=0.), dict(phase="A"),
    dict(lambda_div=.1), dict(selector_enabled=True), dict(radius_m=.2), dict(w_f=-1.)])
def test_invalid_spec_rejected(kwargs):
    with pytest.raises(ValueError): PhysicsV12Spec(**kwargs)


def test_mean_means_no_guard_and_shape_nonfinite_rejected():
    t = terms()
    plain, _ = aggregate_terms(t, PhysicsV12Spec(aggregation="mean"))
    guarded, d = aggregate_terms(t, PhysicsV12Spec(aggregation="mean_guard"))
    torch.testing.assert_close(guarded-plain, .1*d["V"].mean())
    with pytest.raises(ValueError): aggregate_terms({**t, "F": torch.zeros(1, 5)}, PhysicsV12Spec())
    with pytest.raises(ValueError): aggregate_terms({**t, "F": torch.full((1, 6), float("nan"))}, PhysicsV12Spec())
