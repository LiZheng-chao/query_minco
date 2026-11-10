"""V4 numerical/semantic gates and read-only design impossibility tests."""
import ast,inspect,json
from pathlib import Path
import numpy as np
import pytest
import torch
from query_minco.training.physics_only_v1_2_gate1 import e1_r3a_v4 as v4
class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__();self.raw=torch.nn.Parameter(torch.zeros(1,6,21))
    def forward(self,depth,state):
        assert torch.is_grad_enabled() and not torch.is_inference_mode_enabled()
        return self.raw+0
def batch():
    from types import SimpleNamespace
    state=torch.zeros(1,12);state[:,6]=2.4;state[:,-1]=-1
    return SimpleNamespace(depth=torch.zeros(1,2,128,160),state=state)
def test_capture_retains_original_forward_autograd():
    torch.set_num_threads(1)
    record,raw,p,tr,ana=v4.capture(Toy().eval(),batch())
    assert raw.requires_grad and record["execution"]["grad_enabled"]
    assert tr.coeffs.requires_grad and ana["p"].requires_grad
    assert np.asarray(record["tail_PVA"]).shape==(1,6,3,3)
def test_capture_rejects_disabled_grad_before_model_forward():
    torch.set_num_threads(1)
    with torch.no_grad(),pytest.raises(ValueError,match="FORWARD_SEMANTICS"):
        v4.capture(Toy().eval(),batch())
def test_capture_has_no_grad_disabling_context():
    source=inspect.getsource(v4.capture)
    assert "no_grad" not in source and "inference_mode(" not in source
    tree=ast.parse(inspect.getsource(v4.support))
    assert not any(isinstance(n,ast.Name) and n.id=="BoundPhysicsV12Objective" for n in ast.walk(tree))
def test_exact_comparison_records_small_difference_without_tolerance():
    names=["raw_logits","waypoints_B0","head_PVA","tail_PVA","coefficients_B0","durations","sampled_p_B0"]
    a={n:[0.] for n in names};b={n:[0.] for n in names};b["raw_logits"]=[1e-8]
    out=v4.differences(a,b)
    assert not out["raw_logits"]["exact_numeric_equal"]
    assert out["raw_logits"]["max_abs_difference"]==1e-8
def test_readonly_reference_exposes_near_terminal_stop_impossibility():
    state=np.array([.8,0,0,.1,0,0,2.4,0,0,0,0,-1])
    c,d,r=v4.reference_polynomials(state)
    assert np.allclose(r["endpoint"],[1.95,0,0])
    assert np.allclose(r["terminal_velocity"],[1.225,0,0])
    for alpha in [1.,.5,.25,.125]:
        bound=v4.endpoint_box_lower_bounds(r,state[6:9],alpha)
        assert bound["min_possible_terminal_speed_mps"]>=.475-1e-12>.25
def test_fixed_time_bernstein_sensitivity_shape_and_nonzero_terminal_axes():
    M=v4.sensitivity_controls()
    assert M.shape==(3,6,3,15)
    assert all(np.linalg.norm(M[...,i])>0 for i in range(15))
def test_reference_initial_conditions_and_segment_continuity():
    st=np.array([.8,.1,0,.1,0,0,2.4,0,0,0,0,-1])
    c,d,r=v4.reference_polynomials(st)
    from query_minco.evaluation.validation import evaluate
    assert np.allclose(evaluate(c[0],0),0)
    assert np.allclose(evaluate(c[0],0,1),st[:3])
    assert np.allclose(evaluate(c[0],0,2),st[3:6])
    for k in range(5):
        for i in range(2):assert np.allclose(evaluate(c[i],d[i],k),evaluate(c[i+1],0,k),atol=1e-12)
def test_emit_valid_json(capsys):
    v4.emit(dict(status="fixture",outside_queries=6))
    assert json.loads(capsys.readouterr().out)["outside_queries"]==6
