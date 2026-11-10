"""P0b diagnostic tests. Projected candidate is not a production adapter."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch
from query_minco.training.physics_only_v1_2_gate1.geometry import TypedBoundOccupancy,_SEAL,world_points
from query_minco.training.physics_only_v1_2_gate1.objective import collision_math
from query_minco.training.physics_only_v1_2.spec import PhysicsV12Spec
BASE=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("_frozen_p0_tests",BASE/"tests/test_physics_v12_p0_roi_proxy.py")
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
SPEC=PhysicsV12Spec(phase="A",aggregation="foundation")

def geometry(rotation=None,origin=None,floor_intersection=False):
    g=old.fixture()
    if rotation is None and origin is None and not floor_intersection:return g
    roi=g.roi.copy()
    if floor_intersection:roi[0,2]=.08
    return TypedBoundOccupancy(seal=_SEAL,counts=g.counts,meta=g.meta,
        pose=dict(position_world=[0.,0.,0.] if origin is None else origin,
                  R_WB0=np.eye(3).tolist() if rotation is None else rotation),
        roi=roi,domain=g.domain.tolist(),identity={"fixture":"P0b synthetic math only"},revalidate=lambda:None)

def packet(distance,N=192):
    if distance.ndim==0:
        distance=distance.expand(1,6,N)
    if distance.ndim!=3 or distance.shape[1]!=6 or distance.shape[-1]<2:raise ValueError("TEST_PACKET_SHAPE")
    N=distance.shape[-1]
    alpha=torch.arange(1,N+1,dtype=distance.dtype,device=distance.device).expand_as(distance)
    alpha=alpha/alpha.sum(-1,keepdim=True)
    return collision_math(distance,alpha,SPEC),alpha

def scalar_cost(distance,N=192,key="C_viol"):
    return packet(distance,N)[0][key][0,0]

def query_world(g,points):
    return world_points(points,torch.as_tensor(g.origin,dtype=points.dtype),
                        torch.as_tensor(g.rotation,dtype=points.dtype))

def projected_candidate(points,g,queries=None):
    """Exact candidate under review. No nudge/fallback/tolerance for strict queries."""
    g.validate()
    if points.device.type!="cpu" or points.dtype not in (torch.float32,torch.float64) or points.shape[-1]!=3 or not torch.isfinite(points).all():
        raise ValueError("P0B_POINT_TYPE_OR_FINITE")
    if not np.isfinite(g.roi).all() or (g.roi[1]<=g.roi[0]).any():
        raise ValueError("P0B_INVALID_ROI")
    shape=points.shape[:-1];flat=points.reshape(-1,3)
    w=query_world(g,flat)
    # Same exact ROI comparison as the original strict SDF, not rounded FP32 bounds.
    numeric=w.detach().double().numpy()
    outside=torch.as_tensor(((numeric<g.roi[0])|(numeric>g.roi[1])).any(-1),dtype=torch.bool)
    values=torch.zeros(len(flat),dtype=points.dtype)
    def strict(p,label):
        actual=query_world(g,p).detach().double().numpy()
        legal=not bool(((actual<g.roi[0])|(actual>g.roi[1])).any())
        if queries is not None:queries.append(dict(kind=label,points_B0=p.detach().tolist(),actual_world=actual.tolist(),legal=legal))
        if not legal:raise ValueError("P0B_PROJECTED_QUERY_ROUNDTRIP_OUTSIDE_ROI")
        return g.sdf(p)
    if (~outside).any():values[~outside]=strict(flat[~outside],"original_known")
    deltas=torch.zeros(len(flat),dtype=points.dtype)
    if outside.any():
        world=w[outside].double()
        low=torch.as_tensor(g.roi[0],dtype=torch.float64)
        high=torch.as_tensor(g.roi[1],dtype=torch.float64)
        q=torch.maximum(torch.minimum(world,high),low)
        qb=(q-torch.as_tensor(g.origin,dtype=torch.float64))@torch.as_tensor(g.rotation,dtype=torch.float64)
        trace=strict(qb,"known_boundary")
        delta=(world-q).norm(dim=-1)
        values[outside]=(trace-delta).to(points.dtype)
        deltas[outside]=delta.to(points.dtype)
    return dict(training_distance=values.reshape(shape),known_mask=(~outside).reshape(shape),
        outside_mask=outside.reshape(shape),outside_distance_m=deltas.reshape(shape),
        label="training_proxy; not actual SDF or Hard clearance")

def A_report(N):
    g=geometry();b=torch.tensor([.15,.25,.35],dtype=torch.float64)
    inside=b+torch.tensor([1e-6,0,0],dtype=torch.float64)
    outside=(b-torch.tensor([1e-6,0,0],dtype=torch.float64)).requires_grad_()
    actual=g.sdf(inside);proxy=old.roi_only_candidate(outside,g)
    grad=torch.autograd.grad(scalar_cost(proxy,N),outside)[0]
    err=None
    try:g.sdf(outside)
    except ValueError as ex:err=str(ex)
    return dict(N=N,shape=[1,6,N],inside_distance=float(actual),outside_roi_only_proxy=float(proxy),
        inside_C_viol=float(scalar_cost(actual,N)),outside_C_viol=float(scalar_cost(proxy,N)),
        inside_C_shape=float(scalar_cost(actual,N,"C_shape")),outside_C_shape=float(scalar_cost(proxy,N,"C_shape")),
        gradient=grad.tolist(),strict_outside_error=err,boundary_trace=float(g.sdf(b)),
        other_same_roi_trace=float(old.fixture(False).sdf(b)))

def floor_report(batch,N=192):
    g=batch.geometry[0];q=np.array([g.origin[0],g.origin[1],g.roi[0,2]])
    a=torch.tensor((q+np.array([0,0,1e-6])-g.origin)@g.rotation,dtype=torch.float64)
    b=torch.tensor((q-np.array([0,0,1e-6])-g.origin)@g.rotation,dtype=torch.float64,requires_grad=True)
    actual=g.sdf(a);proxy=old.roi_only_candidate(b,g)
    grad=torch.autograd.grad(scalar_cost(proxy,N),b)[0]
    return dict(N=N,inside_actual_distance=float(actual),outside_roi_only_proxy=float(proxy),
        inside_C_viol=float(scalar_cost(actual,N)),outside_C_viol=float(scalar_cost(proxy,N)),
        inside_C_shape=float(scalar_cost(actual,N,"C_shape")),outside_C_shape=float(scalar_cost(proxy,N,"C_shape")),
        outside_gradient_B0=grad.tolist(),floor=g.floor,roi=g.roi.tolist(),model_forward_count=0)

@pytest.mark.parametrize("N",[2,192])
def test_A_fixed_shape_normalization_and_query_axis(N):
    d=torch.linspace(-.1,.1,6*N,dtype=torch.float64).reshape(1,6,N).requires_grad_()
    out,alpha=packet(d)
    assert alpha.shape==d.shape and out["C_viol"].shape==(1,6)
    assert torch.allclose(alpha.sum(-1),torch.ones((1,6),dtype=d.dtype),atol=1e-15,rtol=0)
    expected=(alpha*((.15-d).clamp_min(0)/.15).square()).sum(-1)
    assert torch.equal(out["C_viol"],expected)
    grad=torch.autograd.grad(out["C_viol"][0,3],d)[0]
    assert not grad[0,:3].any() and not grad[0,4:].any() and grad[0,3].abs().sum()>0

@pytest.mark.parametrize("N",[2,192])
def test_A_original_boundary_counterexample_cost_and_gradient(N):
    r=A_report(N)
    assert r["inside_C_viol"]>r["outside_C_viol"]+.5
    assert r["inside_C_shape"]>r["outside_C_shape"]
    assert r["gradient"][0]<0 and np.isfinite(r["gradient"]).all()
    assert r["boundary_trace"]<-.049 and abs(r["other_same_roi_trace"])<1e-12

@pytest.mark.parametrize("N",[2,192])
def test_A_strict_unknown_error(N):
    assert A_report(N)["strict_outside_error"]=="UNKNOWN_OUTSIDE_CERTIFIED_ROI"

@pytest.mark.parametrize("N",[2,192])
def test_A_rotation_world_B0_gradient(N):
    g=geometry(rotation=[[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]],origin=[1.,-2.,.1])
    world=np.array([.14,.25,.35])
    p=torch.tensor((world-g.origin)@g.rotation,dtype=torch.float64,requires_grad=True)
    d=old.roi_only_candidate(p,g)
    grad=torch.autograd.grad(scalar_cost(d,N),p)[0]
    moved=query_world(g,p.detach()-1e-6*grad)
    assert abs(float(d)+.01)<1e-12 and float(moved[0])>world[0]

@pytest.fixture(scope="module")
def real_batch():
    from query_minco.training.physics_only_v1_2_gate1.e1_dataset_binding import load_e1_bound
    root=BASE/"research_outputs/physics_only_v1_2_p0b_roi_proxy_v1/_preparation/p0b_seed20261104_001/TRAIN13_AND_EXCLUSIONS.json"
    r=__import__("json").loads(root.read_text())["Train13"][0]
    return load_e1_bound(r["source_manifest_path"],r["source_manifest_sha256"])

@pytest.mark.parametrize("N",[2,192])
def test_A_real_source_floor_cost_counterexample(real_batch,N):
    r=floor_report(real_batch,N)
    assert r["inside_C_viol"]>r["outside_C_viol"]+.5
    assert r["inside_C_shape"]>r["outside_C_shape"]
    assert np.isfinite(r["outside_gradient_B0"]).all()

@pytest.mark.parametrize("dtype",[torch.float32,torch.float64])
def test_B_inside_values_and_gradients_exact(dtype):
    g=geometry()
    p=torch.tensor([[.3,.4,.4],[.17,.25,.35],[.45,.5,.5]],dtype=dtype,requires_grad=True)
    original=g.sdf(p);new=projected_candidate(p,g)
    assert torch.equal(new["training_distance"],original) and new["known_mask"].all()
    a=torch.autograd.grad(original.sum(),p,retain_graph=True)[0]
    b=torch.autograd.grad(new["training_distance"].sum(),p)[0]
    assert torch.equal(a,b)

@pytest.mark.parametrize("floor",[False,True])
def test_B_boundary_continuity_occupied_and_floor(floor):
    g=geometry(floor_intersection=floor)
    b=torch.tensor([.4,.4,.08] if floor else [.15,.25,.35],dtype=torch.float64)
    n=torch.tensor([0.,0.,-1.] if floor else [-1.,0.,0.],dtype=torch.float64)
    trace=g.sdf(b)
    assert float(trace)<0
    for eps in [1e-4,1e-6,1e-8]:
        inside=g.sdf(b-eps*n)
        outside=projected_candidate(b+eps*n,g)["training_distance"]
        assert abs(float(outside-trace)+eps)<2e-12
        assert abs(float(outside-inside))<=2*eps+2e-12

@pytest.mark.parametrize("dtype",[torch.float32,torch.float64])
def test_B_six_faces_edges_corners_sign_radial_gradient_and_FD(dtype):
    g=geometry(floor_intersection=True)
    lo=g.roi[0];hi=g.roi[1];center=(lo+hi)/2
    for axes in [[0],[1],[2],[0,1],[0,1,2]]:
        for sign in [-1,1]:
            q=center.copy();n=np.zeros(3)
            for a in axes:q[a]=(lo if sign<0 else hi)[a];n[a]=sign
            n/=np.linalg.norm(n)
            p=torch.tensor(q+.02*n,dtype=dtype,requires_grad=True)
            records=[];out=projected_candidate(p,g,records);v=out["training_distance"]
            assert out["outside_mask"] and all(r["legal"] for r in records)
            assert float(v)<=-float(out["outside_distance_m"])+1e-7
            grad=torch.autograd.grad(v,p,retain_graph=True)[0]
            radial=float(torch.dot(grad,torch.tensor(n,dtype=dtype)))
            assert abs(radial+1)<2e-6
            cost=scalar_cost(v)
            cg=torch.autograd.grad(cost,p)[0]
            assert float(torch.dot(-cg,torch.tensor(n,dtype=dtype)))<0
            assert torch.isfinite(cg).all()
            if dtype==torch.float64:
                h=1e-6
                a=projected_candidate(p.detach()+h*torch.tensor(n),g)["training_distance"]
                b=projected_candidate(p.detach()-h*torch.tensor(n),g)["training_distance"]
                assert abs(float((a-b)/(2*h))+1)<1e-8

@pytest.mark.parametrize("outside_count",[147,192])
def test_B_high_unknown_fraction_and_query_gradient(outside_count):
    g=geometry()
    p=torch.tensor([.3,.4,.4],dtype=torch.float32).expand(1,6,192,3).clone()
    p[:,:,-outside_count:]=torch.tensor([.14,.25,.35])
    p.requires_grad_();records=[]
    result=projected_candidate(p,g,records);values=result["training_distance"]
    assert result["outside_mask"].sum()==6*outside_count
    assert all(r["legal"] for r in records)
    costs,_=packet(values)
    grad=torch.autograd.grad((costs["C_viol"]+.1*costs["C_shape"]).mean(),p)[0]
    assert torch.isfinite(grad).all() and (grad[:,:,-outside_count:,0]<0).all()
    assert (values[:,:,-outside_count:]<0).all()

@pytest.mark.parametrize("bad",[float("nan"),float("inf"),-float("inf")])
def test_B_nonfinite_fail_closed(bad):
    p=torch.tensor([bad,.3,.3],dtype=torch.float32)
    with pytest.raises(ValueError,match="P0B_POINT_TYPE_OR_FINITE"):projected_candidate(p,geometry())

@pytest.mark.parametrize("dtype",[torch.float32,torch.float64])
def test_B_general_rotation_projection_queries_are_known(dtype):
    # Ordinary finite points must not be silently nudged or sent outside strict ROI.
    failures=[]
    for angle in [.37,np.pi/7,np.pi/4]:
        c=np.cos(angle);s=np.sin(angle)
        R=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
        g=geometry(rotation=R.tolist(),origin=[1.1,-2.3,.15])
        for axis in range(3):
            for side in [-1,1]:
                w=(g.roi[0]+g.roi[1])/2
                w[axis]=(g.roi[0]-.02 if side<0 else g.roi[1]+.02)[axis]
                p=torch.tensor((w-g.origin)@g.rotation,dtype=dtype,requires_grad=True)
                seen=[]
                try:
                    v=projected_candidate(p,g,seen)
                    gradient=torch.autograd.grad(v["training_distance"],p)[0]
                    assert torch.isfinite(gradient).all()
                except ValueError as ex:
                    failures.append(dict(angle=angle,axis=axis,side=side,error=str(ex),queries=seen))
    assert not failures, "P0B_MATH_STOP_PROJECTED_ROUNDTRIP "+__import__("json").dumps(failures)
