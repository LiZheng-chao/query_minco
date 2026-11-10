from types import SimpleNamespace
import pytest
import torch

from query_minco.training.physics_only_v1.costs import SyntheticGeometry
from query_minco.training.physics_only_v1_2.costs_v12 import (
    collision_terms, feasibility_terms, smoothness_terms, sample_weights)
from query_minco.training.physics_only_v1_2.spec import PhysicsV12Spec, DOMAIN_ABLATION_ID
from query_minco.training.pg1.losses import analytic_samples
from query_minco.evaluation.validation import certify_piece


def scene(dtype=torch.float64, *, empty=False, certified=None, width=10.):
    boxes = torch.empty(0, 2, 3, dtype=dtype) if empty else torch.tensor(
        [[[0., 0., 0.], [1., 1., 1.]]], dtype=dtype)
    return SyntheticGeometry(boxes, torch.tensor([[0., -4., -2.], [width, 4., 2.]], dtype=dtype),
                             empty if certified is None else certified)


def samples(x=2., n=2):
    p = torch.zeros(1, 6, n, 3, dtype=torch.float64)
    p[..., 0] = x
    return p, torch.full((1, 6, n), 1/n, dtype=p.dtype)


def test_collision_sdf_manual_formula_and_gradcheck():
    p, w = samples()
    p[..., 1:] = .5; p.requires_grad_()
    spec = PhysicsV12Spec()
    d = collision_terms(p, w, [scene()], spec)
    assert (d["C_viol"] == 0).all()
    expected = (.05/.15) * torch.nn.functional.softplus(torch.tensor((.15-1.)/.05, dtype=p.dtype))
    torch.testing.assert_close(d["C_shape"], torch.full((1, 6), expected.item(), dtype=p.dtype))
    assert torch.autograd.gradcheck(lambda pp: collision_terms(pp, w, [scene()], spec)["C_shape"], (p,))
    near = p.detach().clone(); near[..., 0] = 1.05; near.requires_grad_()
    violation = collision_terms(near, w, [scene()], spec)
    assert (violation["C_viol"] > 0).all()
    grad = torch.autograd.grad(violation["C_viol"].sum(), near)[0]
    assert (grad[..., 0] < 0).all()  # gradient descent moves outward toward +x
    assert torch.autograd.gradcheck(lambda pp: collision_terms(pp, w, [scene()], spec)["C_viol"], (near,))
    inside = p.detach().clone(); inside[..., 0] = .9
    assert (collision_terms(inside, w, [scene()], spec)["min_sdf_m"] < 0).all()


@pytest.mark.parametrize("fault", ["unknown", "uncertified", "contradiction", "frame", "units", "mass", "nan", "negative_weight"])
def test_collision_fail_closed(fault):
    from dataclasses import replace
    p, w = samples()
    geom = [scene()]
    if fault == "unknown": geom = []
    if fault == "uncertified": geom = [scene(empty=True, certified=False)]
    if fault == "contradiction": geom = [scene(empty=False, certified=True)]
    if fault == "frame": geom = [replace(scene(), frame="world")]
    if fault == "units": geom = [replace(scene(), units="cm")]
    if fault == "mass": w *= 2
    if fault == "nan": p[..., 0] = float("nan")
    if fault == "negative_weight": w[..., 0] = -.1; w[..., 1] = 1.1
    with pytest.raises(ValueError): collision_terms(p, w, geom, PhysicsV12Spec())


def test_certified_empty_exact_zero_and_quadrature_weights():
    p, w = samples()
    p.requires_grad_()
    d = collision_terms(p, w, [scene(empty=True)], PhysicsV12Spec())
    assert (d["C_viol"] == 0).all() and (d["C_shape"] == 0).all()
    assert torch.count_nonzero(torch.autograd.grad((d["C_viol"]+d["C_shape"]).sum(), p)[0]) == 0
    tr = SimpleNamespace(coeffs=torch.zeros(1, 6, 3, 3, 6), durations=torch.full((1, 6, 3), 2/3),
                         solve_valid=torch.ones(1, 6, dtype=torch.bool))
    ana = analytic_samples(tr, 64)
    alpha = sample_weights(ana["p"], ana["weights"])
    torch.testing.assert_close(alpha.sum(-1), torch.ones(1, 6))
    assert alpha[0, 0, 0] == pytest.approx(1/(6*63))
    assert alpha[0, 0, 63] == alpha[0, 0, 64]  # two segment half-weights, no extra time


def test_dynamic_limits_zero_positive_and_direction_gradients():
    p, w = samples()
    ana = dict(p=p, weights=w)
    for k in ("v", "a", "jerk"): ana[k] = torch.zeros_like(p, requires_grad=True)
    zero = feasibility_terms(ana, [scene(empty=True)], PhysicsV12Spec())
    assert (zero["F"] == 0).all()
    for k, limit in (("v", 3.), ("a", 8.), ("jerk", 100.)):
        x = torch.zeros_like(p); x[..., 0] = limit*1.1; x.requires_grad_()
        d = feasibility_terms({**ana, k: x}, [scene(empty=True)], PhysicsV12Spec())
        torch.testing.assert_close(d["F_dyn"], torch.full((1, 6), .01, dtype=p.dtype))
        assert (torch.autograd.grad(d["F"].sum(), x)[0][..., 0] > 0).all()


def test_domain_five_cm_gradient_and_named_ablation():
    p, w = samples(-.05); p.requires_grad_()
    ana = dict(p=p, weights=w, v=p*0, a=p*0, jerk=p*0)
    spec = PhysicsV12Spec()
    d = feasibility_terms(ana, [scene(empty=True)], spec)
    assert (d["F_domain"] > .1).all()
    grad = torch.autograd.grad(d["F_domain"][0, 0], p, retain_graph=True)[0]
    assert grad[0, 0, 0, 0] < -1.
    assert torch.autograd.gradcheck(lambda pp: feasibility_terms(
        {**ana, "p": pp}, [scene(empty=True)], spec)["F_domain"], (p,))
    wide = feasibility_terms(ana, [scene(empty=True, width=100.)], spec)
    torch.testing.assert_close(d["F_domain"], wide["F_domain"])
    ablation = PhysicsV12Spec(method_id=DOMAIN_ABLATION_ID, domain_mode="s0_domain_width_ablation")
    old = feasibility_terms(ana, [scene(empty=True)], ablation)
    torch.testing.assert_close(old["F_domain"], torch.full((1, 6), .000025, dtype=p.dtype))
    with pytest.raises(ValueError): PhysicsV12Spec(domain_mode="s0_domain_width_ablation")


def test_exact_jerk_integral_zero_and_cubic_gradient():
    c = torch.zeros(1, 6, 3, 3, 6, dtype=torch.float64)
    tr = SimpleNamespace(coeffs=c, durations=torch.full((1, 6, 3), 2/3, dtype=c.dtype),
                         solve_valid=torch.ones(1, 6, dtype=torch.bool))
    assert torch.count_nonzero(smoothness_terms(tr, PhysicsV12Spec())["S"]) == 0
    c[..., 0, 3] = 1.; c.requires_grad_()
    d = smoothness_terms(tr, PhysicsV12Spec())
    torch.testing.assert_close(d["jerk_energy_si"], torch.full((1, 6), 72., dtype=c.dtype))
    torch.testing.assert_close(d["S"], torch.full((1, 6), 72/22000, dtype=c.dtype))
    assert torch.autograd.grad(d["S"].sum(), c)[0][..., 0, 3].abs().sum() > 0


def test_sampled_collision_zero_can_disagree_with_independent_hard_check():
    # Dynamics-valid straight line; a grazing 0.2mm box lies between 64-point samples.
    c = torch.zeros(1, 6, 3, 3, 6)
    c[..., 0, 1] = 3.
    for s in range(3): c[:, :, s, 0, 0] = 2*s
    tr = SimpleNamespace(coeffs=c, durations=torch.full((1, 6, 3), 2/3),
                         solve_valid=torch.ones(1, 6, dtype=torch.bool))
    ana = analytic_samples(tr, 64)
    x = 1/63
    box = dict(min=[x-.0001, .14999, -.2], max=[x+.0001, .3, .2])
    geom = SyntheticGeometry(torch.tensor([[box["min"], box["max"]]]),
                             torch.tensor([[0., -4., -2.], [10., 4., 2.]]), False)
    d = collision_terms(ana["p"], ana["weights"], [geom], PhysicsV12Spec())
    assert torch.count_nonzero(d["C_viol"]) == 0
    certificate = certify_piece(c[0, 0, 0].double().numpy(), 2/3, [box])
    assert not certificate["passed"]  # discrete zero is explicitly not a certificate
