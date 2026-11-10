import pytest
import torch
from query_minco.training.physics_only_v1_2.coverage_v12 import valid_diversity


def fixture():
    paths = torch.zeros(1, 6, 5, 3, dtype=torch.float64)
    paths[:, 1, :, 0] = .1
    return paths.requires_grad_(), torch.linspace(0, 1, 5, dtype=torch.float64)


@pytest.mark.parametrize("count", [0, 1])
def test_no_pair_exact_zero_and_gradients(count):
    p, t = fixture()
    qualified = torch.zeros(1, 6, dtype=torch.bool); qualified[:, :count] = True
    loss, d = valid_diversity(p, qualified, t)
    assert loss.item() == 0 and d["pair_mass"].item() == 0
    assert torch.count_nonzero(torch.autograd.grad(loss, p)[0]) == 0


def test_two_feasible_rms_formula_gradcheck_detach_and_permutation():
    p, t = fixture()
    a = torch.tensor([[1., 1., 0., 0., 0., 0.]], dtype=p.dtype, requires_grad=True)
    loss, d = valid_diversity(p, a, t)
    expected = (.5-(.1**2+1e-12)**.5)**2/(1+1e-8)
    assert loss.item() == pytest.approx(expected)
    gp, ga = torch.autograd.grad(loss, (p, a), allow_unused=True, retain_graph=True)
    assert ga is None and gp.abs().sum() > 0
    assert torch.count_nonzero(gp[:, 2:]) == 0
    assert torch.autograd.gradcheck(lambda pp: valid_diversity(pp, a, t)[0], (p,))
    perm = torch.tensor([2, 5, 1, 4, 0, 3])
    other, _ = valid_diversity(p[:, perm], a[:, perm], t)
    torch.testing.assert_close(loss, other)
    assert d["topology_claim"] is False


@pytest.mark.parametrize("fault", ["unknown", "nonbinary", "shape", "grid"])
def test_coverage_invalid_qualifications_and_time_grid(fault):
    p, t = fixture()
    a = torch.ones(1, 6, dtype=p.dtype)
    if fault == "unknown": a[:, 0] = float("nan")
    if fault == "nonbinary": a[:, 0] = .5
    if fault == "shape": a = a[:, :5]
    if fault == "grid": t[2] = .7
    with pytest.raises(ValueError): valid_diversity(p, a, t)


def test_identical_feasible_paths_have_positive_penalty_not_a_second_route():
    p, t = fixture()
    p = p.detach()*0
    loss, _ = valid_diversity(p, torch.ones(1, 6, dtype=torch.bool), t)
    assert loss > 0
