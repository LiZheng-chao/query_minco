from dataclasses import replace
from types import SimpleNamespace
import subprocess
import sys
from unittest.mock import patch
import pytest
import torch
from query_minco.training.physics_only_v1.costs import SyntheticGeometry
from query_minco.training.physics_only_v1_2.spec import PhysicsV12Spec
from query_minco.training.physics_only_v1_2.set_objective_v12 import PhysicsV12Objective
from query_minco.training.physics_only_v1_2.diagnostics import hard_diagnostics
from query_minco.models import QueryMincoPlanner
from query_minco.training.physics_only_v1.smoke import parameter_hash
from query_minco.evaluation.goal import task_metrics


def fixture(goal=2.4):
    state = torch.zeros(1, 12); state[0, 0] = .8; state[0, 6] = goal; state[0, 11] = -1.
    geom = [SyntheticGeometry(torch.tensor([[[.9, -.3, -.3], [1.2, .3, .3]]]),
                              torch.tensor([[0., -4., -2.], [10., 4., 2.]]), False)]
    return state, geom, torch.ones(1, dtype=torch.bool)


@pytest.mark.parametrize("mode", ["foundation", "mean", "mean_guard", "softmin_guard"])
def test_actual_network_minco_zero_update_and_per_query_shapes(mode):
    torch.manual_seed(20261104)
    torch.set_num_threads(1)
    model = QueryMincoPlanner().cpu().eval()
    before = parameter_hash(model)
    state, geom, enabled = fixture()
    depth = torch.zeros(1, 2, 128, 160); depth[:, 1] = 1.
    cfg = PhysicsV12Spec(phase="A" if mode == "foundation" else "B", aggregation=mode)
    with patch("torch.optim.AdamW", side_effect=AssertionError("optimizer forbidden")), patch(
            "torch.optim.Adam", side_effect=AssertionError("optimizer forbidden")):
        raw = model(depth, state)
        out = PhysicsV12Objective(cfg)(raw, state, geom, goal_enabled=enabled, task_types=("normal",))
        grad, shared = torch.autograd.grad(out.loss, (raw, model.depth_encoder.d1[0].weight))
    assert torch.isfinite(grad).all() and torch.isfinite(shared).all()
    assert grad[..., :15].abs().sum() > 0
    assert torch.count_nonzero(grad[..., 15:21]) == 0
    assert out.trajectory.coeffs.dtype == torch.float32 and out.trajectory.solve_valid.all()
    torch.testing.assert_close(out.parameters.durations, torch.full((1, 6, 3), 2/3))
    for k in ("G_N", "G_F", "blend", "G_task_train", "P", "eta_P_times_P", "G_total",
              "C_viol", "C_shape", "F", "S", "V", "Q", "J", "pi"):
        assert out.metrics[k].shape == (1, 6)
    assert before == parameter_hash(model)
    hd = hard_diagnostics(out, state, geom, enabled, ("normal",))
    assert hd["HardGoalPass"].shape == (1, 6)
    assert (hd["hard_maxima_v_a_j"][..., 0] >= out.metrics["speed_sample_peak"].double()-1e-5).all()


def test_full_minco_query_permutation_and_unsafe_progress_not_detached():
    torch.manual_seed(9)
    raw = (torch.randn(1, 6, 21)*.02).requires_grad_()
    state, geom, enabled = fixture()
    obj = PhysicsV12Objective()
    out = obj(raw, state, geom, goal_enabled=enabled, task_types=("normal",))
    grad = torch.autograd.grad(out.loss, raw, retain_graph=True)[0]
    perm = torch.tensor([3, 1, 5, 0, 4, 2])
    rp = raw.detach()[:, perm].requires_grad_()
    other = obj(rp, state, geom, goal_enabled=enabled, task_types=("normal",))
    torch.testing.assert_close(out.loss, other.loss, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(grad[:, perm], torch.autograd.grad(other.loss, rp)[0], rtol=2e-4, atol=1e-4)
    assert (out.metrics["C_viol"] > 0).any()
    gp = torch.autograd.grad(out.metrics["P"].sum(), raw)[0]
    assert gp[..., :9].abs().sum() > 0 and torch.isfinite(gp).all()


def test_explicit_hover_has_new_outer_contract_legacy_stays_rejecting():
    state, geom, enabled = fixture(0.)
    zero = torch.zeros(1, 6, 3, 3)
    fake = SimpleNamespace(trajectory=SimpleNamespace(coeffs=torch.zeros(1, 6, 3, 3, 6),
                                                      durations=torch.full((1, 6, 3), 2/3)),
                           parameters=SimpleNamespace(tail_pva=zero))
    hd = hard_diagnostics(fake, state, geom, enabled, ("hover",))
    assert hd["HardGoalPass"].all() and hd["goal_contract_ids"] == ["hover_zero_goal_extension_v12"]
    with pytest.raises(ValueError): task_metrics([0, 0, 0], [0, 0, 0], [0, 0, 0])
    with pytest.raises(ValueError):
        PhysicsV12Objective()(torch.zeros(1, 6, 21), state, geom, goal_enabled=enabled, task_types=("normal",))


def test_loader_output_cannot_be_relabelled_as_bound_training_geometry():
    state, geom, enabled = fixture()
    obj = PhysicsV12Objective()
    with pytest.raises(ValueError):
        obj(torch.zeros(1, 6, 21), state, [SimpleNamespace(boxes=geom[0].boxes, domain=geom[0].domain)],
            goal_enabled=enabled, task_types=("normal",))


def test_fresh_process_teacher_import_guard_and_forward_backward():
    code = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if any(s in name.lower() for s in ('teacher', 'matching', 'imitation', 'pg2_guided', 'geometry_sidecar', 'datasets.manifest')):
        raise RuntimeError('forbidden import '+name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import torch
from query_minco.training.physics_only_v1.costs import SyntheticGeometry
from query_minco.training.physics_only_v1_2 import PhysicsV12Objective
raw=torch.zeros(1,6,21,requires_grad=True)
state=torch.zeros(1,12); state[0,6]=2.4;state[0,11]=-1.
g=SyntheticGeometry(torch.empty(0,2,3),torch.tensor([[0.,-4.,-2.],[10.,4.,2.]]),True)
out=PhysicsV12Objective()(raw,state,[g],goal_enabled=torch.ones(1,dtype=torch.bool),task_types=('normal',))
out.loss.backward()
assert torch.isfinite(raw.grad).all()
"""
    subprocess.run([sys.executable, "-B", "-c", code], check=True)
