"""P0 design counterexamples only; none of these tests qualifies a training proxy."""
import numpy as np
import pytest
import torch
from query_minco.training.physics_only_v1_2_gate1.geometry import TypedBoundOccupancy, _SEAL, world_points
from query_minco.training.physics_only_v1_2_gate1.objective import collision_math
from query_minco.training.physics_only_v1_2.spec import PhysicsV12Spec

def roi_only_candidate(points, geometry):
    """REJECTED candidate, exclusively for auditing; NOT a production SDF."""
    world = world_points(points, torch.as_tensor(geometry.origin, dtype=points.dtype),
                         torch.as_tensor(geometry.rotation, dtype=points.dtype))
    lo = torch.as_tensor(geometry.roi[0], dtype=points.dtype)
    hi = torch.as_tensor(geometry.roi[1], dtype=points.dtype)
    return torch.minimum(world-lo, hi-world).amin(-1)

def fixture(occupied=True):
    counts = np.zeros((8,8,8), np.int32)
    counts[6,6,5] = 1
    if occupied:
        counts[1,2,3] = 1
    return TypedBoundOccupancy(seal=_SEAL, counts=counts,
        meta=dict(grid_origin=[0,0,0],resolution_m=.1,occupy_threshold=0),
        pose=dict(position_world=[0,0,0], R_WB0=np.eye(3).tolist()),
        roi=[[.15,0,.11],[.79,.79,.79]], domain=[[.15,0,.11],[.79,.79,.79]],
        identity={"fixture":"synthetic boundary counterexample, no production authority"},
        revalidate=lambda:None)

def costs(distance):
    return collision_math(distance.reshape(1,1),torch.ones((1,1),dtype=distance.dtype),
        PhysicsV12Spec(phase="A",aggregation="foundation"))

def synthetic_counterexample():
    g=fixture();empty=fixture(False)
    boundary=torch.tensor([.15,.25,.35],dtype=torch.float64)
    inside=boundary+torch.tensor([1e-6,0,0],dtype=torch.float64)
    outside=boundary-torch.tensor([1e-6,0,0],dtype=torch.float64)
    outside.requires_grad_()
    true=g.sdf(inside);proxy=roi_only_candidate(outside,g)
    gradient=torch.autograd.grad(costs(proxy)["C_viol"].sum(),outside)[0]
    moved=outside.detach()-1e-6*gradient
    error=None
    try:g.sdf(outside)
    except ValueError as ex:error=str(ex)
    return dict(roi=g.roi.tolist(),occupied_boundary_trace=float(g.sdf(boundary)),
        other_world_same_roi_boundary_trace=float(empty.sdf(boundary)),
        inside_actual_distance_m=float(true),outside_training_proxy_m=float(proxy),
        inside_C_viol=float(costs(true)["C_viol"]),outside_C_viol=float(costs(proxy)["C_viol"]),
        inside_C_shape=float(costs(true)["C_shape"]),outside_C_shape=float(costs(proxy)["C_shape"]),
        strict_outside_error=error,local_outside_gradient=gradient.tolist(),
        negative_gradient_moves_inside=bool(moved[0]>outside[0]),
        finite_local_gradient=bool(torch.isfinite(gradient).all()),
        boundary_discontinuous=True,proxy_qualified=False)

def real_floor_counterexample(batch):
    g=batch.geometry[0]
    q=np.array([g.origin[0],g.origin[1],g.roi[0,2]])
    eps=1e-6
    inside=q+np.array([0,0,eps]);outside=q-np.array([0,0,eps])
    a=torch.tensor((inside-g.origin)@g.rotation,dtype=torch.float64)
    b=torch.tensor((outside-g.origin)@g.rotation,dtype=torch.float64,requires_grad=True)
    actual=g.sdf(a);proxy=roi_only_candidate(b,g)
    error=None
    try:g.sdf(b)
    except ValueError as ex:error=str(ex)
    grad=torch.autograd.grad(costs(proxy)["C_viol"].sum(),b)[0]
    return dict(boundary_world_m=q.tolist(),inside_world_m=inside.tolist(),outside_world_m=outside.tolist(),
        roi_world_m=g.roi.tolist(),floor_world_z_m=g.floor,inside_actual_distance_m=float(actual),
        outside_training_proxy_m=float(proxy),inside_C_viol=float(costs(actual)["C_viol"]),
        outside_C_viol=float(costs(proxy)["C_viol"]),inside_C_shape=float(costs(actual)["C_shape"]),
        outside_C_shape=float(costs(proxy)["C_shape"]),strict_outside_error=error,
        outside_proxy_gradient_B0=grad.tolist(),finite_gradient=bool(torch.isfinite(grad).all()),
        proxy_qualified=False,model_forward_calls=0,Hard_revalidated=False)

def test_same_roi_different_strict_boundary_trace_blocks_roi_only_continuation():
    r=synthetic_counterexample()
    assert r["occupied_boundary_trace"]<-.049
    assert abs(r["other_world_same_roi_boundary_trace"])<1e-12

def test_candidate_has_inward_gradient_but_leaving_known_collision_lowers_cost():
    r=synthetic_counterexample()
    assert r["finite_local_gradient"] and r["negative_gradient_moves_inside"]
    assert r["inside_C_viol"]>r["outside_C_viol"]+.5
    assert r["inside_C_shape"]>r["outside_C_shape"]
    assert not r["proxy_qualified"]

def test_original_strict_outside_error_unchanged():
    assert synthetic_counterexample()["strict_outside_error"]=="UNKNOWN_OUTSIDE_CERTIFIED_ROI"

def test_different_boundary_trace_persists_as_epsilon_tends_zero():
    g=fixture()
    boundary=torch.tensor([.15,.25,.35],dtype=torch.float64)
    for eps in [1e-4,1e-6,1e-8]:
        actual=g.sdf(boundary+torch.tensor([eps,0,0],dtype=torch.float64))
        proxy=roi_only_candidate(boundary-torch.tensor([eps,0,0],dtype=torch.float64),g)
        assert float(proxy-actual)>.049

def test_world_transform_used_for_rejected_candidate_direction():
    g=fixture()
    g.origin=np.array([1.,-2.,.1])
    g.rotation=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    # No strict call on mutated synthetic fixture: only coordinate formula is under test.
    w=np.array([.14,.25,.35])
    p=torch.tensor((w-g.origin)@g.rotation,dtype=torch.float64,requires_grad=True)
    value=roi_only_candidate(p,g)
    grad=torch.autograd.grad(costs(value)["C_viol"].sum(),p)[0]
    moved=(p.detach()-1e-6*grad).numpy()@g.rotation.T+g.origin
    assert abs(float(value)+.01)<1e-12 and moved[0]>w[0]
