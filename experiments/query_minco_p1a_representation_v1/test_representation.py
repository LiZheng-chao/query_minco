"""P1-A lightweight mathematical qualification; no network or Teacher."""
from pathlib import Path
import importlib.util
import numpy as np
import pytest
import torch
from query_minco.training.physics_only_v1_2_gate1.e1_dataset_binding import load_e1_bound
from query_minco.training.physics_only_v1.decoder import FixedTimeDecoder
from query_minco.trajectory.minco_adapter import MincoAdapter
from query_minco.evaluation.validation import evaluate,numerical_reference,bernstein_controls
from experiments.query_minco_p1a_representation_v1.representation import (
 ARMS,SPEC,D_VALUES,reference,bias,theta_math,PrototypeDecoder,raw_slots,physical_theta,objective,hard_one,samples_vector,independent_samples)
BASE=Path(__file__).resolve().parents[2]
def state(goal=2.4,v=.8,a=.1):
    return torch.tensor([[v,0.,0.,a,0.,0.,goal,0.,0.,0.,0.,-1.]],dtype=torch.float32)
@pytest.fixture(scope="module")
def batch():
    root=BASE/"research_outputs/physics_only_v1_2_p0c_world_boundary_v1/_preparation/bindings_p0c_001/BINDING_INDEX.json"
    r=__import__("json").loads(root.read_text())[0]
    return load_e1_bound(r["p0c_manifest_path"],r["p0c_manifest_sha256"])

def test_actual_A_raw_dimensions():
    u=torch.linspace(-.2,.2,15);raw=raw_slots(u);p=PrototypeDecoder("A")(raw,state())
    exact=FixedTimeDecoder(SPEC)(raw,state())
    assert torch.equal(physical_theta(p),physical_theta(exact))
    torch.testing.assert_close(physical_theta(p)[0,0].double(),theta_math(u.double(),state(),"A"),rtol=1e-6,atol=1e-6)
    changed=raw.clone();changed[...,15:]=10
    q=PrototypeDecoder("A")(changed,state())
    assert torch.equal(physical_theta(p),physical_theta(q)) and torch.equal(p.durations,q.durations)

@pytest.mark.parametrize("arm",ARMS)
@pytest.mark.parametrize("goal",[2.4,6.])
def test_head_tail_independent_reconstruction_and_AD_FD(arm,goal):
    st=state(goal,v=.8 if goal==2.4 else 1.2,a=.1 if goal==2.4 else 0.)
    u=torch.zeros(15,requires_grad=True);p=PrototypeDecoder(arm)(raw_slots(u),st)
    tr=MincoAdapter()(p.head_pva,p.tail_pva,p.inner_pts,p.durations)
    theta=physical_theta(p)[0,0].detach().double().numpy()
    ref=numerical_reference(p.head_pva[0,0].detach().double().numpy(),p.tail_pva[0,0].detach().double().numpy(),p.inner_pts[0,0].detach().double().numpy(),np.full(3,2/3))
    for order,tol in [(0,.002),(1,.01),(2,.05)]:
        err=max(np.max(np.abs(evaluate(c,np.linspace(0,2/3,65),order)-evaluate(r,np.linspace(0,2/3,65),order))) for c,r in zip(tr.coeffs[0,0].detach().numpy(),ref))
        assert err<tol
    J=torch.autograd.functional.jacobian(lambda x:samples_vector(x,st,arm),u,vectorize=True).detach().double().numpy()
    h=1e-5;fd=np.stack([(independent_samples(np.eye(15)[i]*h,st,arm)-independent_samples(-np.eye(15)[i]*h,st,arm))/(2*h) for i in range(15)],axis=-1)
    assert np.all(np.abs(J-fd)<=.003+.003*np.abs(fd))
    assert np.linalg.matrix_rank(J,tol=1e-7)==15

@pytest.mark.parametrize("arm",["B0","B1"])
def test_near_STOP_far_no_forced_STOP(arm):
    near=reference(state(),arm);far=reference(state(6,1.2,0),arm)
    assert near[9:12].norm()==0 and float(far[9:12].norm())>1.
    assert torch.allclose(near[6:9],torch.tensor([2.4,0,0],dtype=torch.float64),atol=1e-6,rtol=0)
    assert float(far[6])<6

def test_B0_B1_capacity_and_matched_initialization():
    st=state();u=torch.linspace(-.1,.1,15)
    v=(u.double()+bias(st,"B0")-bias(st,"B1")).float()
    a=theta_math(u,st,"B0");b=theta_math(v,st,"B1")
    torch.testing.assert_close(a,b,atol=3e-6,rtol=0)
    assert torch.allclose(reference(state(6,1.2,0),"B0"),reference(state(6,1.2,0),"B1"),atol=2e-7,rtol=0)

def test_shared_support_contains_any_speed_bounded_waypoints():
    D=np.array(D_VALUES);np.testing.assert_allclose(D[:9].reshape(3,3),np.array([2,4,6])[:,None]*np.ones((1,3)))
    assert np.all(D[9:12]==3) and np.all(D[12:]==8)

@pytest.mark.parametrize("arm",ARMS)
def test_saturated_logits_lose_active_rank(arm):
    st=state();u=torch.full((15,),50.,requires_grad=True)
    J=torch.autograd.functional.jacobian(lambda x:physical_theta(PrototypeDecoder(arm)(raw_slots(x),st))[0,0],u)
    assert not J.any()

@pytest.mark.parametrize("arm",ARMS)
def test_analytic_initial_speed_boundary_infeasible_is_rejected(batch,arm):
    # Synthetic initial condition only; never relabel any real dataset task.
    st=state(v=3.,a=1.)
    p=PrototypeDecoder(arm)(raw_slots(torch.zeros(15)),st)
    tr=MincoAdapter()(p.head_pva,p.tail_pva,p.inner_pts,p.durations)
    from query_minco.training.physics_only_v1_2_gate1.hard_validator import validate_polynomial
    hard=validate_polynomial(tr.coeffs[0,0].detach().numpy(),tr.durations[0,0].detach().numpy(),batch.geometry[0],st[0,6:9].numpy())
    assert not hard["passed"] and "SPEED" in hard["reasons"]

def test_far_exact_position_unreachable_does_not_mean_far_task_infeasible(batch):
    st=state(goal=20,v=1.2,a=0.)
    p=PrototypeDecoder("B1")(raw_slots(torch.zeros(15)),st)
    tr=MincoAdapter()(p.head_pva,p.tail_pva,p.inner_pts,p.durations)
    from query_minco.training.physics_only_v1_2_gate1.hard_validator import validate_polynomial
    hard=validate_polynomial(tr.coeffs[0,0].detach().numpy(),tr.durations[0,0].detach().numpy(),batch.geometry[0],st[0,6:9].numpy())
    assert float(st[0,6])>3*2 and hard["goal"]["passed"]
    assert float(p.tail_pva[0,0,1].norm())>1

def test_small_residual_cannot_cross_offset_slit_full_curve_bound():
    # This is a synthetic counterexample to tiny residuals, not tuning actual D.
    from query_minco.training.physics_only_v1_2_gate1.e1_r3a_v4 import sensitivity_controls
    M=sensitivity_controls();D=np.array(D_VALUES)
    bound=.001*np.max((np.abs(M[:,:,1,:])*D).sum(-1))
    assert bound<.5

def test_explicit_prototype_objective_keeps_P0c_Hard_and_losses(batch):
    for arm in ARMS:
        u=torch.zeros(15,requires_grad=True);r=objective(arm)(raw_slots(u),batch)
        grad=torch.autograd.grad(r.loss,u)[0]
        assert torch.isfinite(grad).all() and grad.norm()>0
        assert r.training_geometry_records[0]["label"].startswith("training_proxy")
        assert r.trajectory.coeffs.shape==(1,6,3,3,6)

def test_native_channel_metadata_is_evaluation_only():
    import json
    from experiments.query_minco_p1a_representation_v1.representation import summarize
    rows=json.loads((BASE/"research_outputs/physics_only_v1_2_p0c_world_boundary_v1/_preparation/bindings_p0c_001/BINDING_INDEX.json").read_text())
    for name in ["train_single_slit_1104_p0_near","train_two_portals_1103_p0_far"]:
        row=next(r for r in rows if r["sample_id"]==name)
        b=load_e1_bound(row["p0c_manifest_path"],row["p0c_manifest_sha256"])
        u=torch.zeros(15,requires_grad=True);r=objective("B0")(raw_slots(u),b)
        record=summarize(r,b,u,arm="B0")
        assert record["channel_coverage"]["applicable"]
        assert record["channel_coverage"]["gate_x_world_m"]==1.2
