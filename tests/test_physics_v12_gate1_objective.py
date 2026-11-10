from pathlib import Path
import json
import numpy as np
import pytest
import torch
from query_minco.training.physics_only_v1_2_gate1.objective import (
    collision_math,feasibility_math,BoundPhysicsV12Objective,collision_direction_probe)
from query_minco.training.physics_only_v1_2_gate1.binding import load_bound,sha
from query_minco.training.physics_only_v1_2_gate1.hard_validator import piece_certificate
from query_minco.training.physics_only_v1.costs import SyntheticGeometry
from query_minco.training.physics_only_v1_2.costs_v12 import collision_terms,feasibility_terms
from query_minco.training.losses.set_physical_risk import box_sdf
from query_minco.training.physics_only_v1_2.spec import PhysicsV12Spec

MANIFEST=Path("research_outputs/physics_only_v1_2_gate1_forest_v1/runs/bind_posecert_001/BOUND_MANIFEST.json")

def test_identical_collision_feasibility_math_and_gradients():
    # Separate synthetic numerical unit fixture, never a production qualification.
    spec=PhysicsV12Spec()
    boxes=torch.tensor([[[0.,0.,0.],[1.,1.,1.]]],dtype=torch.float64)
    domain=torch.tensor([[0.,-4.,-2.],[10.,4.,2.]],dtype=torch.float64)
    g=SyntheticGeometry(boxes,domain,False)
    p=torch.zeros(1,6,4,3,dtype=torch.float64);p[...,0]=1.07;p[...,1:]=.5;p.requires_grad_()
    w=torch.full((1,6,4),.25,dtype=torch.float64)
    old=collision_terms(p,w,[g],spec);new=collision_math(box_sdf(p,boxes),w,spec)
    for name in new:
        torch.testing.assert_close(new[name],old[name],rtol=1e-12,atol=1e-12)
    for name in ("C_viol","C_shape"):
        a=torch.autograd.grad(old[name].sum(),p,retain_graph=True)[0]
        b=torch.autograd.grad(new[name].sum(),p,retain_graph=True)[0]
        torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)
    p2=p.detach().clone();p2[...,0]=-.05;p2.requires_grad_()
    ana=dict(p=p2,weights=w,v=p2*80,a=p2*100,jerk=p2*500)
    old=feasibility_terms(ana,[g],spec);new=feasibility_math(ana,[domain],spec)
    for name in old:torch.testing.assert_close(old[name],new[name],rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(torch.autograd.grad(old["F"].sum(),p2,retain_graph=True)[0],
                               torch.autograd.grad(new["F"].sum(),p2)[0])

def test_production_rejects_synthetic_and_wrong_shapes():
    obj=BoundPhysicsV12Objective()
    with pytest.raises(ValueError,match="TYPED"):obj(torch.zeros(1,6,21),[])
    batch=load_bound(MANIFEST,sha(MANIFEST))
    with pytest.raises(ValueError,match="CPU_FP32_SHAPE"):obj(torch.zeros(1,6,21,dtype=torch.float64),batch)
    assert collision_direction_probe(batch)["passed"]

def test_real_ground_grazing_discrete_zero_is_not_hardpass():
    batch=load_bound(MANIFEST,sha(MANIFEST));g=batch.geometry[0]
    T=2/3
    # Parabola approaches the actual renderer ground by 0.5 micrometres too much.
    c=np.zeros((3,6));c[:,0]=g.origin
    c[2,0]=g.floor+.15-5e-7+.1*(T/2)**2
    c[2,1]=-.1*T;c[2,2]=.1
    t=np.linspace(0,T,64)
    from query_minco.evaluation.validation import evaluate
    p=torch.tensor(evaluate(c,t)-g.origin,dtype=torch.float64)
    d=g.sdf(p)
    assert ((.15-d).clamp_min(0)==0).all()
    assert not piece_certificate(c,T,g)["passed"]
