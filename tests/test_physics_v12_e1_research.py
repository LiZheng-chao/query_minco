"""Descriptive geometry and gradient evidence contracts, no training."""
import copy
import numpy as np
import pytest
from query_minco.training.physics_only_v1_2_gate1.e1_research import (
    cosine,similarities,crossing_signature,scene_split_guard,choose_progress_weight)


def fixture():
    c=np.zeros((6,3,3,6));dt=np.full((6,3),2/3)
    for i in range(3):
        c[:,i,0,0]=i*2/3;c[:,i,0,1]=1
    return c,dt


def test_endpoint_equality_does_not_imply_full_trajectory_equality():
    c,dt=fixture()
    T=dt[0,0];c[1,0,1,1]=T;c[1,0,1,2]=-1
    r=similarities(c,dt)
    assert r["endpoint_distance_m"][0][1]==0
    assert r["physical_time_exact_rms_m"][0][1]>.01
    assert r["physical_time_polynomial_max_offset_m"][0][1]==pytest.approx(T*T/4,abs=1e-10)
    assert r["homotopy_class"]=="NOT_IDENTIFIED"


def test_exact_rms_constant_offset_and_symmetry():
    c,dt=fixture();c[1,:,1,0]=.02
    r=similarities(c,dt)
    assert r["physical_time_exact_rms_m"][0][1]==pytest.approx(.02,abs=1e-12)
    assert r["physical_time_polynomial_max_offset_m"][0][1]==pytest.approx(.02,abs=1e-12)
    np.testing.assert_array_equal(np.diag(r["physical_time_exact_rms_m"]),0)
    np.testing.assert_allclose(r["physical_time_exact_rms_m"],np.array(r["physical_time_exact_rms_m"]).T)
    assert r["full_trajectory_within_5cm"]


@pytest.mark.parametrize("fault",["shape","nan","time"])
def test_invalid_trajectory_evidence_rejected(fault):
    c,dt=fixture()
    if fault=="shape":c=c[:5]
    if fault=="nan":c[0,0,0,0]=np.nan
    if fault=="time":dt[1,0]=.7
    with pytest.raises(ValueError):similarities(c,dt)


def test_gate_fingerprint_not_certified_channel():
    c,dt=fixture()
    result=crossing_signature(c[0],dt[0],planes=(.5,1.5))
    assert all(len(g["crossings"])==1 for g in result["gates"])
    assert not result["topology_certified"] and result["feasible_channel_count"] is None
    assert result["channel_label"].startswith("UNKNOWN")


def test_cosine_requires_vector_and_undefined_zero():
    assert cosine([1,0],[-1,0])==-1
    assert cosine([1,0],[0,1])==0
    assert cosine([0,0],[1,0]) is None
    with pytest.raises(ValueError):cosine([1],[1,2])


@pytest.mark.parametrize("same_key",["scene_id","occupancy_semantic_sha256","canonical_geometry_sha256"])
def test_scene_geometry_cannot_cross_split(same_key):
    a=dict(split="train",scene_id="a",occupancy_semantic_sha256="b",canonical_geometry_sha256="c")
    b=dict(split="dev",scene_id="d",occupancy_semantic_sha256="e",canonical_geometry_sha256="f")
    b[same_key]=a[same_key]
    with pytest.raises(ValueError,match="CROSS_SPLIT"):scene_split_guard([a,b])


def test_all_poses_same_scene_can_remain_train():
    r=dict(split="train",scene_id="same",occupancy_semantic_sha256="grid",canonical_geometry_sha256="geometry")
    assert scene_split_guard([r,copy.deepcopy(r)])


def test_weight_selection_is_prescribed_not_optimal():
    x={k:dict(P=.1,S=.4,C_shape=.2) for k in ("raw","CNN","Transformer")}
    result=choose_progress_weight(x)
    assert result["candidate_w_p"]==1. and not result["tuned_or_optimal"]
    x["raw"]["P"]=0
    with pytest.raises(ValueError,match="CALIBRATION_UNDEFINED"):choose_progress_weight(x)


@pytest.mark.parametrize("dtype_name",["float32","float64"])
def test_gradient_statistics_match_frozen_fp64_accumulation(dtype_name):
    import torch
    from query_minco.training.physics_only_v1_2_gate1.e1_research import tensor_stats
    from query_minco.training.physics_only_v1_2_gate1.preflight import stats as frozen_stats
    dtype=getattr(torch,dtype_name)
    # Mixed scales/long reduction, with a non-contiguous view and autograd attached.
    x=torch.linspace(-3.,5.,450450,dtype=dtype).reshape(150150,3).T.requires_grad_()
    assert not x.is_contiguous()
    before=x.detach().clone()
    actual=tensor_stats(x);reference=frozen_stats(x)
    assert actual["norm"]==reference["norm"]
    assert actual["rms"]==reference["rms"]
    assert actual["zero_ratio"]==reference["zero_ratio"]
    assert x.grad is None
    torch.testing.assert_close(x.detach(),before,atol=0,rtol=0)


def test_float32_norm_overflow_cannot_replace_fp64_summary():
    import torch
    from query_minco.training.physics_only_v1_2_gate1.e1_research import tensor_stats
    from query_minco.training.physics_only_v1_2_gate1.preflight import stats as frozen_stats
    # Norm is representable, squared sum overflows FP32; double accumulation is required.
    x=torch.tensor([1e20,-1e20,1.,-2.],dtype=torch.float32)
    assert not torch.isfinite(x.norm())
    actual=tensor_stats(x)
    assert np.isfinite(actual["norm"]) and actual["norm"]==frozen_stats(x)["norm"]
    assert actual["rms"]==frozen_stats(x)["rms"]


def test_nonfinite_gradient_summary_is_rejected():
    import torch
    from query_minco.training.physics_only_v1_2_gate1.e1_research import tensor_stats
    for x in (torch.tensor([float("nan")]),torch.tensor([float("inf")]),torch.empty(0)):
        with pytest.raises(ValueError,match="NONFINITE_GRADIENT"):tensor_stats(x)


def test_checkpoint_comparison_uses_fp64_summary_without_relaxing_tolerance():
    import ast,inspect
    from query_minco.training.physics_only_v1_2_gate1.e1_research import checkpoint_gradients
    # Bind the numerical regression above to the actual checkpoint comparison call site.
    tree=ast.parse(inspect.getsource(checkpoint_gradients))
    matches=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and
       isinstance(n.func,ast.Attribute) and n.func.attr=="assert_allclose" and
       any(isinstance(k,ast.Constant) and k.value=="shared" for k in ast.walk(n))]
    assert len(matches)==1
    call=matches[0];norm=call.args[0]
    assert isinstance(norm,ast.Subscript) and isinstance(norm.value,ast.Call)
    assert norm.value.func.id=="tensor_stats"
    values={k.arg:ast.literal_eval(k.value) for k in call.keywords}
    assert values==dict(atol=1e-5,rtol=2e-6)
