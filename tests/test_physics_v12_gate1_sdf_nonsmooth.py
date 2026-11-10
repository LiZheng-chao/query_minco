"""Explicit smooth / near-switch contracts; original failed test remains unchanged."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch
from torch.autograd.gradcheck import GradcheckError
from query_minco.training.physics_only_v1_2_gate1.binding import load_bound,sha
from query_minco.training.physics_only_v1_2_gate1.geometry import TypedBoundOccupancy,_SEAL,voxel_boxes
from query_minco.training.physics_only_v1_2_gate1.preflight import (
    complete_branches,compare_full,old_counterexample,smooth_certificate)
from query_minco.training.physics_only_v1_2_gate1.objective import BoundPhysicsV12Objective

MANIFEST=Path("research_outputs/physics_only_v1_2_gate1_forest_v1/runs/bind_posecert_001/BOUND_MANIFEST.json")

@pytest.fixture(scope="module")
def bound():
    return load_bound(MANIFEST,"fd8237790ec5bb2d7b33fce6f8b66a4398c321b0d4ee6a01a2284f449b68445f")

def test_original_failure_unchanged_and_reproduced(bound):
    p=Path("tests/test_physics_v12_gate1_geometry.py")
    spec=importlib.util.spec_from_file_location("old_failed_test",p)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    with pytest.raises(GradcheckError):
        old.test_exact_distance_matches_independent_all_boxes(bound)
    r=old_counterexample(bound.geometry[0])
    assert r["original_gradcheck_reproduced"]
    assert r["nearest_gap_m"]<3e-9
    assert r["stable_branch_analytic_difference"]<1e-10
    assert r["repeatable"]

def test_proven_smooth_points_and_fp32_batch(bound):
    r=smooth_certificate(bound.geometry[0])
    assert len(r["points"])>=4
    assert all(p["nearest_gap_m"]>2*p["difference_radius_m"] for p in r["points"])
    assert r["fp64_gradcheck"] and r["fp32_finite"]

def test_real_complete_enumeration_and_kd_candidates(bound):
    g=bound.geometry[0]
    rng=np.random.RandomState(20261104)
    points=rng.uniform(g.roi[0]+.01,g.roi[1]-.01,size=(48,3))
    # Interior / edge / face, ground, mirror/ROI edge; all are new diagnostic points.
    b=g.boxes[np.argmin(np.linalg.norm(g.centers-g.origin,axis=1))]
    points=np.vstack([g.origin,points,b.mean(0),b[0],b[1],
                      [g.origin[0],g.origin[1],g.floor+.02],
                      g.roi[0]+.0001,g.roi[1]-.0001])
    r=compare_full(g,points)
    assert r["max_abs_error_m"]<1e-10 and r["kd_nearest_inclusion"]
    assert r["negative_count"]>0 and r["positive_count"]>0

def test_small_voxels_mirror_sign_reserve_and_manual_enumeration():
    counts=np.zeros((4,4,8),np.int32);counts[1,2,3]=1;counts[2,1,5]=2
    meta=dict(grid_origin=[0,0,0],resolution_m=.1,occupy_threshold=0)
    roi=np.array([[-.4,-.4,-.2],[.8,.8,.7]])
    g=TypedBoundOccupancy(seal=_SEAL,counts=counts,meta=meta,
       pose=dict(position_world=[0,0,0],R_WB0=np.eye(3).tolist()),roi=roi,
       domain=[[-.4,-.4,-.2],[.8,.8,.7]],identity={"fixture":"manual_not_production"},revalidate=lambda:None)
    h=float(np.float32(.1));manual=[]
    for i in range(-4,8):
      for j in range(-4,8):
       for k in range(1,7):
        def mirror(x):
            x=x%6
            return x if x<4 else 6-x
        if counts[mirror(i),mirror(j),k]>0:
            manual.append([[i*h-1e-5,j*h-1e-5,k*h-1e-5],
                           [(i+1)*h+1e-5,(j+1)*h+1e-5,(k+1)*h+1e-5]])
    # float32 h makes floor/ceil index bounds include one additional edge cell;
    # compare all generated boxes whose centers lie in this manual cell range.
    actual=g.boxes
    keep=(actual.mean(1)[:,0]>=-4*h)&(actual.mean(1)[:,0]<8*h)&(actual.mean(1)[:,1]>=-4*h)&(actual.mean(1)[:,1]<8*h)
    np.testing.assert_allclose(actual[keep],np.asarray(manual),atol=1e-12,rtol=0)
    pts=np.array([[.15,.25,.35],[.1-1e-5,.25,.35],[.08,.25,.35],
                  [-.15,.25,.35],[.4,.4,.11],[.7999,.2,.4]])
    result=compare_full(g,pts)
    assert result["max_abs_error_m"]<1e-10 and result["kd_nearest_inclusion"]
    assert float(g.sdf(torch.tensor(pts[0])))<0
    with pytest.raises(ValueError,match="UNKNOWN"):g.sdf(torch.tensor([.81,.2,.4]))
    with pytest.raises(ValueError,match="UNKNOWN_TOP"):
        TypedBoundOccupancy(seal=_SEAL,counts=counts,meta=meta,
          pose=dict(position_world=[0,0,0],R_WB0=np.eye(3).tolist()),roi=[[-.4,-.4,0],[.8,.8,1]],
          domain=roi,identity={},revalidate=lambda:None)

def test_full_bound_objective_raw_math_integration_only(bound):
    # No network before B pass; seeded ordinary raw parameters, no optimized candidate.
    gen=torch.Generator().manual_seed(20261104)
    raw=(torch.randn(1,6,21,generator=gen)*.03).requires_grad_()
    out=BoundPhysicsV12Objective()(raw,bound)
    grad=torch.autograd.grad(out.loss,raw)[0]
    assert torch.isfinite(out.loss) and torch.isfinite(grad).all()
    assert not grad[...,15:21].any() and grad[...,:15].abs().sum()>0


def test_exact_switch_directional_derivatives():
    from query_minco.training.physics_only_v1_2_gate1.preflight import exact_tie_fixture
    result=exact_tie_fixture()
    assert result["selected_gradient_is_active_branch"] and not result["classical_gradient_exists"]
