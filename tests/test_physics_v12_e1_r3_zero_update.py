"""R3a diagnostics: numerical precision, scene isolation and ROI fail-closed."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from query_minco.training.physics_only_v1_2_gate1.e1_r3_zero_update import (
    norm_stats,phase_spec,PHASE_COEFFICIENTS,module_groups,roi_diagnostics,scene_equal_summary,packet)
def test_explicit_phase_a():
    s=phase_spec()
    assert s.phase=="A" and s.aggregation=="foundation"
    assert s.beta_A*s.w_p==.025==PHASE_COEFFICIENTS["P"]
    assert s.lambda_div==0 and not s.selector_enabled
@pytest.mark.parametrize("dtype",[torch.float32,torch.float64])
def test_fp64_statistics(dtype):
    x=torch.tensor([1e20,3.,-4.],dtype=dtype)
    a=norm_stats(x)
    assert a["norm"]==float(x.flatten().double().norm())
    assert a["rms"]==float(x.flatten().double().square().mean().sqrt())
    assert a["finite"]
def test_nonfinite_rejected():
    with pytest.raises(ValueError,match="NONFINITE"):norm_stats(torch.tensor([float("nan")]))
def test_groups_and_original_initial_identity():
    from query_minco.training.physics_only_v1_2_gate1.short_train import initialize,INITIAL_SHA
    from query_minco.training.physics_only_v1.smoke import parameter_hash
    m=initialize();g=module_groups(list(m.named_parameters()))
    assert parameter_hash(m)==INITIAL_SHA
    assert not set(g["shared"])&set(g["TrajectoryHead"]+g["LearnableQueries"])
    assert set(g["all_network"])==set().union(*(set(g[k]) for k in ("CNN","StateEncoder","Transformer","TrajectoryHead","LearnableQueries")))
def test_packet_actual_path_zero_and_fp64():
    names=["depth_encoder.w","state_encoder.w","query_decoder.layers.w","trajectory_head.w","query_decoder.learned_queries"]
    named=[(n,torch.nn.Parameter(torch.ones(1))) for n in names]
    raw=sum(p for n,p in named)*torch.ones(1,6,21)
    active=raw[...,:15].square().mean()
    got=packet({"P":.025*active,"zero":active*0},raw,named,module_groups(named))
    assert got["stats"]["P"]["Raw"]["norm"]>0
    assert got["cosine"]["Raw"]["zero"]["P"] is None
    assert np.count_nonzero(np.asarray(got["raw_gradients"]["P"])[...,15:])==0
def test_continuous_roi_cannot_hide_between_samples():
    g=SimpleNamespace(origin=np.zeros(3),rotation=np.eye(3),roi=np.array([[-1.,-1.,-1.],[.8,1.,1.]]))
    c=np.zeros((6,3,3,6));c[:,:,0,1]=4;c[:,:,0,2]=-4
    points=torch.zeros(6,2,3)
    r=roi_diagnostics(points,g,c,np.ones((6,3)))
    assert not r["any_sample_outside"] and r["any_continuous_center_outside"]
def test_sample_unknown_is_not_clamped():
    g=SimpleNamespace(origin=np.zeros(3),rotation=np.eye(3),roi=np.array([[-1.,-1.,-1.],[1.,1.,1.]]))
    p=torch.zeros(6,2,3);p[0,1,0]=2
    r=roi_diagnostics(p,g,np.zeros((6,3,3,6)),np.ones((6,3)))
    assert r["any_sample_outside"] and r["queries"][0]["first_world_point"]==[2.,0.,0.]
def test_scene_equal_and_incomplete_denominator():
    rows=[dict(sample_id="a%d_near"%i,scene_id="a") for i in range(4)]+[dict(sample_id="b0_near",scene_id="b")]
    records=[dict(**r,status="PASS",gradients=dict(gram_mean_per_parameter={"Raw":{"P":{"P":1. if r["scene_id"]=="a" else 9.}}})) for r in rows]
    full=scene_equal_summary(records,rows)["scopes"]["all"]
    assert full["status"]=="COMPLETE"
    assert full["available_scene_summary"]["Raw"]["RMS"]["P"]==np.sqrt(5.)
    partial=scene_equal_summary(records[:-1],rows)["scopes"]["all"]
    assert partial["status"]=="BLOCKED_INCOMPLETE_CORPUS"
    assert partial["completed_scene_tasks"]["b"]==0 and partial["no_missing_zero_imputation"]
