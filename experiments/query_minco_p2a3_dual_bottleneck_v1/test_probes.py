import inspect
import pytest
import torch
from experiments.query_minco_p2a3_dual_bottleneck_v1.initialization import initial,FAMILIES,SEEDS,Y_INDEX,AMPLITUDE_M,f
from experiments.query_minco_p2a3_dual_bottleneck_v1.probes import jobs
def state():
    s=torch.zeros(1,12);s[0,0]=1.2;s[0,6]=6;s[0,11]=-1;return s
def test_independent_probe_budgets():
    b=list(jobs("slit"));c=list(jobs("portal"))
    assert len(b)==18 and len(c)==48
    assert sum(j["lr"]==.003 for j in b)==6
    assert all(j["lr"]==.01 for j in c)
    assert all(j["family"]=="original_B0" for j in b)
    assert set(j["lossarm"] for j in c)=={"L0","L1"}
def test_initializer_inputs_exclude_map_and_witness():
    assert tuple(inspect.signature(initial).parameters)==("state","family","seed")
    src=inspect.getsource(initial)
    assert not any(x in src for x in ("geometry","manifest","witness","gate_x","Hard"))
    assert AMPLITUDE_M==1.0
@pytest.mark.parametrize("seed",SEEDS)
def test_signed_symmetry_common_nonlateral_and_open_support(seed):
    s=state();ts={k:f.actual_theta(initial(s,k,seed),s,"B0") for k in FAMILIES}
    y=list(Y_INDEX);other=[i for i in range(15) if i not in y]
    assert float((ts["symmetric_plus"][y]+ts["symmetric_minus"][y]).abs().max())<3e-6
    for a in FAMILIES:
        assert bool((ts[a].abs()<ts[a].new_tensor(f.p1.D_VALUES)).all())
        assert float((ts[a][other]-ts["zero_lateral"][other]).abs().max())<3e-6
    assert not ts["zero_lateral"][y].any()
    assert not ts["symmetric_plus"][[7,10,13]].any()
    assert not ts["symmetric_minus"][[7,10,13]].any()
def test_curve_antisymmetry_and_PVA():
    s=state();up=initial(s,"symmetric_plus",None).requires_grad_();um=initial(s,"symmetric_minus",None)
    p=f.p1.samples_vector(up,s,"B0").reshape(-1,3);m=f.p1.samples_vector(um,s,"B0").reshape(-1,3)
    assert float((p[:,1]+m[:,1]).abs().max())<2e-4
    assert float((p[:,[0,2]]-m[:,[0,2]]).abs().max())<2e-4
    g=torch.autograd.grad(p.square().sum(),up)[0]
    assert torch.isfinite(g).all() and bool(g.abs().sum()>0)
    params=f.p1.PrototypeDecoder("B0")(f.p1.raw_slots(up),s)
    assert not params.head_pva[...,1].any()
    assert not params.tail_pva[...,1].any()
def test_random_control_matches_original_B0():
    s=state()
    for seed in SEEDS:
        u=initial(s,"random",seed);old=f.initial("B0_nominal",s,seed)
        assert float((f.actual_theta(u,s,"B0")-f.actual_theta(old,s,"B0")).abs().max())<3e-6
def test_invalid_family_rejects():
    with pytest.raises(ValueError,match="FAMILY"):initial(state(),"unknown",None)
