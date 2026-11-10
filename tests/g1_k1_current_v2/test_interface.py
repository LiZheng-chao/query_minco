"""K1 interface contracts; no optimizer updates or legacy C1 registry bypass."""
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
import torch
from query_minco.models import QueryMincoPlanner
from query_minco.models.config import QueryMincoConfig
from query_minco.trajectory.parameter_decoder import PhysicalParameterDecoder
from query_minco.trajectory.minco_adapter import MincoAdapter
from query_minco.training.physics_only_v1.decoder import FixedTimeDecoder
from query_minco.training.physics_only_v1_2.spec import PhysicsV12Spec
from query_minco.training.physics_only_v1_2.set_objective_v12 import aggregate_terms
from query_minco.training.physics_only_v1_2_gate1.k1_interface import make_planner, make_objective, identity
from query_minco.training.physics_only_v1_2_gate1.e1_dataset_binding import load_e1_bound
from query_minco.training.physics_only_v1_2_gate1.roi_safe_training import RoiSafeTrainingGeometry
from query_minco.training.physics_only_v1_2_gate1.hard_validator import validate_polynomial, COMPONENT_ID
from query_minco.training.physics_only_v1_2_gate1.geometry import TypedBoundOccupancy, _SEAL
from experiments.g1_k1_current_v2.qualify import OUT, state_hash, numeric, load_rows

torch.set_num_threads(1)


@pytest.fixture(scope="module")
def bound():
    row = load_rows()[0]
    return load_e1_bound(row["manifest"], row["manifest_sha256"])


@pytest.fixture(scope="module")
def model_result(bound):
    torch.manual_seed(20261104)
    model = make_planner().eval()
    before = state_hash(model)
    with patch.object(torch.optim.Optimizer, "__init__", side_effect=AssertionError("NO_OPTIMIZER")):
        raw, diag = model(bound.depth, bound.state, True)
        raw.retain_grad()
        out = make_objective()(raw, bound)
        out.loss.backward()
    assert before == state_hash(model)
    return model, raw, diag, out


def test_true_single_query_raw21_and_identity(model_result):
    model, raw, diag, out = model_result
    assert raw.shape == (1, 1, 21)
    assert diag["decoder_tokens"].shape == (1, 1, 128)
    assert model.query_decoder.learned_queries.shape == (1, 128)
    assert sum(p.numel() for p in model.parameters()) == 469821
    assert len(model.query_decoder.layers) == 2
    assert identity()["architecture_id"] == model.config.architecture_id
    assert identity()["validator_id"] == COMPONENT_ID
    assert identity()["method_id"] != "query_minco_b2c1_c1_e1_v1"


def test_minco_pva_continuity_and_independent_numeric(model_result, bound):
    model, raw, diag, out = model_result
    assert out.trajectory.coeffs.shape == (1, 1, 3, 3, 6)
    numeric(out)
    expected = torch.stack([torch.zeros(3), bound.state[0, :3], bound.state[0, 3:6]])
    assert torch.equal(out.parameters.head_pva[0, 0], expected)
    assert out.parameters.total_time.item() == 2.
    assert out.trajectory.solve_valid.all()


def test_gradient_modules_and_dead_output_rows(model_result):
    model, raw, diag, out = model_result
    for module in [model.depth_encoder, model.state_encoder, model.query_decoder, model.trajectory_head]:
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
        assert torch.cat([p.grad.flatten() for p in module.parameters()]).norm() > 0
    assert raw.grad[..., :15].abs().sum() > 0
    assert raw.grad[..., 15:].count_nonzero() == 0
    assert model.trajectory_head.mlp[-1].weight.grad[15:].count_nonzero() == 0
    assert model.trajectory_head.mlp[-1].bias.grad[15:].count_nonzero() == 0


@pytest.mark.parametrize("k", [1, 6])
def test_phase_a_exact_component_gradient_and_set_normalization(k):
    spec = PhysicsV12Spec(phase="A", aggregation="foundation")
    terms = {n:torch.full((1, k), float(i+1), dtype=torch.float64, requires_grad=True)
             for i, n in enumerate(["C_viol", "F", "G_task_train", "C_shape", "P", "S"])}
    loss, metrics = aggregate_terms(terms, spec)
    weights = [1., 1., 1., .1, .025, .1]
    expected = sum(w*t.mean() for w,t in zip(weights, terms.values()))
    torch.testing.assert_close(loss, expected, atol=1e-14, rtol=1e-14)
    grads = torch.autograd.grad(loss, tuple(terms.values()))
    for g, w in zip(grads, weights):
        torch.testing.assert_close(g, torch.full_like(g, w/k), atol=1e-15, rtol=1e-15)
    torch.testing.assert_close(metrics["pi"], torch.full_like(metrics["pi"], 1/k))
    torch.testing.assert_close(metrics["K_eff"], torch.full_like(metrics["K_eff"], float(k)))
    if k == 1:
        assert metrics["entropy"].item() == 0.
        torch.testing.assert_close(metrics["softmin_per_scene"], metrics["J"][:, 0])


@pytest.mark.parametrize("aggregation", ["mean", "mean_guard", "softmin_guard"])
def test_k1_auxiliary_set_math_uses_log1(aggregation):
    spec = PhysicsV12Spec(phase="B", aggregation=aggregation)
    terms = {n:torch.ones(1, 1, dtype=torch.float64, requires_grad=True)
             for n in ["C_viol", "F", "G_task_train", "C_shape", "P", "S"]}
    loss, m = aggregate_terms(terms, spec)
    torch.testing.assert_close(m["softmin_per_scene"], m["J"][:, 0])
    expected = m["J"].mean() + (spec.lambda_all*m["V"].mean() if aggregation!="mean" else 0.)
    torch.testing.assert_close(loss, expected)
    assert m["K_eff"].item() == 1. and m["pi"].item() == 1.


@pytest.mark.parametrize("bad", [0, 2, 3, 5, -1, True, 1.0])
def test_invalid_k_rejected_at_explicit_configuration(bad):
    for call in [lambda:QueryMincoConfig(num_queries=bad),
                 lambda:PhysicalParameterDecoder(num_candidates=bad),
                 lambda:MincoAdapter(num_candidates=bad)]:
        with pytest.raises((ValueError, TypeError)):
            call()


@pytest.mark.parametrize("case", ["depth_shape", "state_shape", "depth_nan", "state_inf", "dtype"])
def test_network_bad_input_rejected(bound, case):
    model = make_planner()
    depth, state = bound.depth.clone(), bound.state.clone()
    if case == "depth_shape":depth=depth[:, :, :-1]
    elif case == "state_shape":state=state[:, :-1]
    elif case == "depth_nan":depth[0,0,0,0]=float("nan")
    elif case == "state_inf":state[0,0]=float("inf")
    else:state=state.double()
    with pytest.raises((ValueError, TypeError)):
        model(depth, state)


@pytest.mark.parametrize("shape", [(1, 6, 21), (1, 1, 20), (2, 1, 21)])
def test_k1_objective_raw_candidate_and_batch_contract(bound, shape):
    with pytest.raises(ValueError):
        make_objective()(torch.zeros(shape), bound)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_objective_nonfinite_rejection(bound, bad):
    with pytest.raises(ValueError):
        make_objective()(torch.full((1,1,21), bad), bound)


def test_k6_state_dict_strict_default_and_k1_mismatch():
    before = np.load(OUT / "K6_BEFORE.npz")
    state = {n[len("state::"):]:torch.from_numpy(before[n].copy()) for n in before.files if n.startswith("state::")}
    model = QueryMincoPlanner()
    result = model.load_state_dict(state, strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    with pytest.raises(RuntimeError, match="size mismatch"):
        make_planner().load_state_dict(state, strict=True)


@pytest.fixture(scope="module")
def geometry():
    counts = np.zeros((8,8,8), np.int32);counts[6,6,5]=1;counts[1,2,3]=1
    return TypedBoundOccupancy(seal=_SEAL,counts=counts,
        meta=dict(grid_origin=[0,0,0],resolution_m=.1,occupy_threshold=0),
        pose=dict(position_world=[0,0,0],R_WB0=np.eye(3).tolist()),
        roi=[[.15,0,.11],[.79,.79,.79]],domain=[[.15,0,.11],[.79,.79,.79]],
        identity={"fixture":"SYNTHETIC_TEST_ONLY G1 boundary"},revalidate=lambda:None)


@pytest.mark.parametrize("value", [-1., 0., float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("segment", [0,1,2])
def test_hard_v2_all_illegal_time_segments(geometry, value, segment):
    c=np.zeros((3,3,6));c[:,:,0]=[.4,.4,.4]
    dt=np.full(3,2/3);dt[segment]=value
    result=validate_polynomial(c,dt,geometry,[.4,.4,.4])
    assert not result["passed"] and result["reasons"]
    assert result["reasons"][0] in ["NONPOSITIVE_DURATION","INVALID_DURATION_NONFINITE"]


@pytest.mark.parametrize("dt", [1., [], [1.], [1.,1.], [1.]*4, [[1.,1.,1.]]])
def test_hard_v2_bad_shape(geometry, dt):
    assert not validate_polynomial(np.zeros((3,3,6)),dt,geometry,[.4,.4,.4])["passed"]


def test_p0c_outside_proxy_and_hard_unknown(geometry):
    point=torch.tensor([.149999,.25,.35],dtype=torch.float64,requires_grad=True)
    record=RoiSafeTrainingGeometry().query(point,geometry)
    assert record["outside_mask"] and record["training_distance"]<=0
    grad=torch.autograd.grad(record["training_distance"],point)[0]
    assert torch.isfinite(grad).all() and grad.norm()>0
    with pytest.raises(ValueError,match="UNKNOWN_OUTSIDE_CERTIFIED_ROI"):
        geometry.sdf(point)
    c=np.zeros((3,3,6));c[:,:,0]=point.detach().numpy()
    result=validate_polynomial(c,np.full(3,2/3),geometry,[.4,.4,.4])
    assert not result["passed"]


def test_no_privileged_input_or_optimizer_in_k1_factory():
    import inspect
    from query_minco.training.physics_only_v1_2_gate1 import k1_interface
    args=list(inspect.signature(QueryMincoPlanner.forward).parameters)
    assert args==["self","depth_input","state","return_diagnostics"]
    source=inspect.getsource(k1_interface)
    assert "optimizer.step(" not in source
    assert "imitation" not in source and "matching" not in source


@pytest.mark.parametrize("sample_id", ["train_free_roi_1101_p0_near", "train_free_roi_1101_p0_far",
                                     "dev_single_slit_2104_p0_near", "dev_two_portals_2103_p0_far"])
def test_known_real_legal_hard_results_unchanged(sample_id):
    import importlib.util
    root=Path(__file__).resolve().parents[2]
    previous=root/"research_outputs/bug01_hard_duration_contract_20261110_01/snapshots/hard_validator_before.py"
    assert hashlib.sha256(previous.read_bytes()).hexdigest()=="f55159ffc879b3b1d0062638bcde42690a40719cfcd3c23b34a9f8ec11cbeacf"
    spec=importlib.util.spec_from_file_location("_g1_legacy_hard_positive_control",previous)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    row=next(r for r in load_rows() if r["sample_id"]==sample_id)
    batch=load_e1_bound(row["manifest"],row["manifest_sha256"])
    idx=json.loads((root/"research_outputs/physics_only_v1_2_p0c_world_boundary_v1/_preparation/bindings_p0c_001/BINDING_INDEX.json").read_text())
    old_row=next(r for r in idx if r["sample_id"]==sample_id)
    ev=json.loads(Path(old_row["hard_evidence_path"]).read_text())
    src=Path(ev["source_evidence_path"])
    assert hashlib.sha256(src.read_bytes()).hexdigest()==ev["source_evidence_sha256"]
    witnesses=json.loads(src.read_text())["witnesses"]
    assert witnesses
    for w in witnesses.values():
        args=(np.asarray(w["coeffs_B0"]),np.asarray(w["durations_s"]),batch.geometry[0],batch.state[0,6:9].numpy())
        before=old.validate_polynomial(*args)
        assert before["passed"]
        assert validate_polynomial(*args)==before


def test_completed_zero_update_output_cannot_be_overwritten():
    from experiments.g1_k1_current_v2.qualify import zero
    path = OUT / "K1_INITIAL_STATE.npz"
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError, match="ZERO_UPDATE_OUTPUT_OCCUPIED"):
        zero()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
