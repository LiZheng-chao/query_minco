import inspect
import torch
import pytest
from experiments.query_minco_p1a3_validation_v1 import alignment as a
from query_minco.training.physics_only_v1_2.set_objective_v12 import aggregate_terms

def test_Phase_A_actual_contributions_and_J_are_distinct():
    spec=a.f.p1.SPEC
    terms={k:torch.full((1,6),float(i+1),requires_grad=True) for i,k in enumerate(a.TERMS)}
    loss,diag=aggregate_terms(terms,spec)
    expected=sum(a.FACTORS[k]*terms[k].mean() for k in a.TERMS)
    torch.testing.assert_close(loss,expected)
    assert not torch.allclose(loss,diag["J"].mean())
    for k in a.TERMS:
        grad=torch.autograd.grad(loss,terms[k],retain_graph=True)[0]
        torch.testing.assert_close(grad,torch.full_like(grad,a.FACTORS[k]/6))

def test_zero_gradient_and_cosine_semantics():
    x=torch.ones(15,requires_grad=True)
    assert torch.equal(a.gradient_or_zero(torch.tensor(0.),x),torch.zeros(15))
    assert a.cosine(torch.zeros(15),x) is None
    assert abs(a.cosine(x,-x)+1)<1e-12

def test_evaluation_path_has_no_optimizer():
    src=inspect.getsource(a)
    assert "torch.optim" not in src and ".step(" not in src
    assert a.FACTORS["P"]==.025 and a.FACTORS["C_shape"]==.1

@pytest.mark.parametrize("shape",[(1,15),(15,1)])
def test_evaluation_leaf_shape_rejects(shape):
    with pytest.raises(ValueError,match="EVALUATION_LEAF"):
        a.evaluate_theta(torch.zeros(shape,requires_grad=True),None)
