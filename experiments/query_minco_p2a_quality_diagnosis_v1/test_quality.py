import pytest
import torch
from experiments.query_minco_p2a_quality_diagnosis_v1 import quality as q
from query_minco.training.physics_only_v1_2.set_objective_v12 import aggregate_terms
from types import SimpleNamespace
@pytest.mark.parametrize("arm",q.ARMS)
def test_alpha1_equal_frozen_phase_A(arm):
    terms={k:torch.rand(1,6,requires_grad=True) for k in ("C_viol","F","G_task_train","C_shape","P","S")}
    loss,_=aggregate_terms(terms,q.f.p1.SPEC)
    r=SimpleNamespace(loss=loss,metrics=terms)
    torch.testing.assert_close(q.scalar(r,arm,1),loss)
def test_static_progress_scaling_only():
    terms={k:torch.ones(1,6,requires_grad=True) for k in ("C_viol","F","G_task_train","C_shape","P","S")}
    loss,_=aggregate_terms(terms,q.f.p1.SPEC);r=SimpleNamespace(loss=loss,metrics=terms)
    torch.testing.assert_close(q.scalar(r,"L1",4)-loss,torch.tensor(.075))
    for k in terms:
        g=torch.autograd.grad(q.scalar(r,"L1",4),terms[k],retain_graph=True)[0]
        target=(.1 if k in ("C_shape","S","P") else 1.)/6
        torch.testing.assert_close(g,torch.full_like(g,target))
@pytest.mark.parametrize("hard,q_new,accepted",[(True,.9,True),(False,.9,False),(True,1.1,False),(True,1.,True)])
def test_L2_feasible_state_filter(hard,q_new,accepted):
    ok,_=q.accept("L2",True,hard,1.,q_new,0)
    assert ok==accepted
def test_L2_is_not_restoration_guarantee():
    assert q.accept("L2",False,False,1.,2.,0)[0]
def test_unknown_cannot_be_certified():
    with pytest.raises(ValueError,match="HARD_UNKNOWN"):
        q.accept("L2",True,True,1.,0.,1)
    assert not q.accept("L2",True,False,1.,0.,1)[0]
def test_filter_rejection_can_stagnate_and_momentum_is_not_claimed_optimal():
    assert not any(q.accept("L2",True,False,1.,.5,0)[0] for _ in range(4))
def test_logged_Adam_first_step():
    replay=q.LoggedAdamReplay();u=torch.zeros(15);g=torch.ones(15)
    out,clipped=replay.predict(u,g,float(g.norm()))
    torch.testing.assert_close(out,torch.full((15,),-.01),rtol=1e-6,atol=1e-8)
    assert torch.equal(clipped,g)
def test_nonfinite_filter_rejected():
    with pytest.raises(ValueError,match="NONFINITE"):
        q.accept("L2",True,True,1.,float("nan"),0)
