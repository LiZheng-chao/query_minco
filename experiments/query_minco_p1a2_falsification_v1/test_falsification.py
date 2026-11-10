"""Small preflight tests; no optimizer, no production mutation."""
import inspect
import numpy as np
import pytest
import torch
from experiments.query_minco_p1a2_falsification_v1 import falsification as f

@pytest.mark.parametrize("sample",f.p1.PILOT)
@pytest.mark.parametrize("which",["B0","B1"])
def test_A_inverse_actual_reference(sample,which):
    batch,_=f.load_source(sample);theta=f.actual_theta(torch.zeros(15),batch.state,which)
    u=f.inverse_A(theta.double())
    assert torch.max(torch.abs(f.actual_theta(u,batch.state,"A")-theta))<=3e-6

def test_A_inverse_rejects_boundary():
    with pytest.raises(ValueError,match="A_OPEN_SUPPORT"):f.inverse_A(torch.zeros(15,dtype=torch.float64))

def test_initial_does_not_consume_witness():
    src=inspect.getsource(f.initial)
    assert "qualification" not in src and "witness" not in src
    b,_=f.load_source(f.p1.PILOT[0])
    assert torch.equal(f.initial("A_nominal",b.state),torch.zeros(15))

def test_Far_aliases():
    b,_=f.load_source(f.p1.PILOT[1]);controls,aliases=f.canonical_controls(b.state)
    assert len(controls)==3 and len(aliases)==2

def test_impossible_speed_bound():
    # Frozen v<=3 and fixed float32 durations: 6.5 m cannot be traversed.
    assert 3*float(torch.full((3,),2/3,dtype=torch.float32).double().sum())<6.5

@pytest.mark.parametrize("case",[c for c in f.CASE_SOURCES if c.startswith("native")])
def test_prewall_Hard_is_not_TaskSuccess(case):
    b,_=f.load_source(f.CASE_SOURCES[case]);contract=f.native_contract(b,case)
    r=f.p1.objective("B0")(f.p1.raw_slots(torch.zeros(15)),b)
    out=f.crossing_success(r,b,contract)
    assert out["OriginalHardAll"] and not out["TaskSuccess"] and not out["endpoint_post_wall"]

def test_point_distance_rejects_unknown():
    b,_=f.load_source(f.p1.PILOT[0]);g=b.geometry[0]
    assert f.point_distance(g,g.roi[1]+1) is None

@pytest.mark.parametrize("case",[c for c in f.CASE_SOURCES if c.startswith("native")])
def test_original_Hard_failure_cannot_be_crossing_success(case):
    from experiments.query_minco_p1a2_falsification_v1.qualification import x_reference
    b,_=f.load_source(f.CASE_SOURCES[case]);contract=f.native_contract(b,case)
    x=x_reference(b.state).tolist()
    theta=[x[0],.15,0,x[1],.15,0,x[2],.15,0,1.2,0,0,0,0,0]
    r=f.result_for_theta(theta,b.state)
    out=f.crossing_success(r,b,contract,{"passed":False})
    assert not out["TaskSuccess"] and not out["certified_gate_crossing"]

def test_matched_B_translation_Jacobian_and_gradient():
    b,_=f.load_source(f.p1.PILOT[0])
    u0=torch.linspace(-.02,.02,15)
    u1=(u0.double()+f.p1.bias(b.state,"B0")-f.p1.bias(b.state,"B1")).float()
    u0.requires_grad_();u1.requires_grad_()
    r0=f.p1.objective("B0")(f.p1.raw_slots(u0),b);r1=f.p1.objective("B1")(f.p1.raw_slots(u1),b)
    g0=torch.autograd.grad(r0.loss,u0)[0];g1=torch.autograd.grad(r1.loss,u1)[0]
    assert torch.allclose(g0,g1,rtol=.003,atol=.003)
    J0=torch.autograd.functional.jacobian(lambda u:f.actual_theta(u,b.state,"B0"),u0)
    J1=torch.autograd.functional.jacobian(lambda u:f.actual_theta(u,b.state,"B1"),u1)
    assert torch.allclose(J0,J1,rtol=1e-5,atol=3e-6)
