"""Support audit and design counterexamples; no residual decoder implementation."""
import numpy as np
import torch
from query_minco.training.physics_only_v1_2_gate1.e1_r3a_v3 import polynomial_exit,vector_outputs,decode
def fixture():
    raw=torch.zeros(1,6,21,requires_grad=True);state=torch.zeros(1,12);state[:,6]=2.4;state[:,-1]=-1
    return raw,state
def test_exact_exit_not_sampled_index():
    c=np.zeros((1,3,6));c[0,0,1]=2
    e=polynomial_exit(c,[1.],np.array([-1.,-1.,-1.]),np.array([1.,1.,1.]))
    assert abs(e["first_exit_s"]-.5)<1e-12
def test_tangent_is_not_open_outside_interval():
    c=np.zeros((1,3,6));c[0,0,:3]=[0,4,-4]
    assert polynomial_exit(c,[1],np.array([-1.,-1.,-1.]),np.ones(3))["first_exit_s"] is None
def test_waypoints_inside_does_not_certify_curve():
    c=np.zeros((1,3,6));c[0,0,1:3]=[8,-8]
    assert np.allclose([0,0], [np.polynomial.polynomial.polyval(t,c[0,0]) for t in [0,1]])
    assert polynomial_exit(c,[1],-np.ones(3),np.ones(3))["first_exit_s"] is not None
def test_decoder_path_shapes_and_unused_physical_dimensions():
    raw,state=fixture();v=vector_outputs(raw,state);assert v.shape==(1,6,86)
    jac=torch.autograd.functional.jacobian(lambda r:vector_outputs(r,state),raw,vectorize=True)
    assert torch.count_nonzero(jac[:,:,:84,...,15:])==0
    assert torch.count_nonzero(jac[:,:,84:,...,19:])>0
    assert float(v[0,0,0])==5.
def test_terminal_velocity_acceleration_are_active():
    raw,state=fixture();v=vector_outputs(raw,state)
    for index,dim in [(21,9),(24,12)]:
        g=torch.autograd.grad(v[0,0,index],raw,retain_graph=True)[0]
        assert g[0,0,dim]>0
def test_time_is_fixed_not_a_detach_of_active_trajectory():
    raw,state=fixture();p,tr=decode(raw,state)
    assert torch.all(p.durations==torch.tensor(2/3,dtype=torch.float32))
    g=torch.autograd.grad(tr.coeffs.sum(),raw)[0]
    assert g[...,:15].abs().sum()>0 and g[...,15:].abs().sum()==0
def test_bounded_residual_bernstein_sufficiency_toy():
    ref=np.array([.2,.3,.4]);M=np.array([[.1,.2],[-.1,.1],[.05,-.05]])
    lo=ref-np.abs(M).sum(1);hi=ref+np.abs(M).sum(1)
    for x in [-1,1]:
        for y in [-1,1]:
            b=ref+M@np.array([x,y]);assert (b>=lo-1e-15).all() and (b<=hi+1e-15).all()
def test_too_small_residual_cannot_reach_required_detour():
    residual_y_limit=.1;required_clearance_offset=.3
    assert residual_y_limit<required_clearance_offset
def test_tanh_saturation_and_query_collapse_remain_counterexamples():
    z=torch.tensor([0.,50.],requires_grad=True);g=torch.autograd.grad(torch.tanh(z).sum(),z)[0]
    assert g[0]==1 and g[1]==0
    raw,state=fixture();v=vector_outputs(raw,state)
    assert torch.equal(v[:,0],v[:,1])
def test_cubic_reference_initial_and_terminal_pva_math():
    v0=np.array([.2,0,0]);a0=np.array([.1,0,0]);j=np.array([.3,.1,0]);T=2.
    p=lambda t:v0*t+.5*a0*t*t+j*t**3/6
    eps=1e-4
    assert np.allclose((p(T+eps)-p(T-eps))/(2*eps),v0+a0*T+.5*j*T*T,atol=1e-8)
    assert np.allclose(p(0),0)
