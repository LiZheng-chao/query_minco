"""State-machine and independent mathematical evidence contracts."""
from types import SimpleNamespace
import copy
import numpy as np
import pytest
import torch
from query_minco.training.physics_only_v1_2_gate1 import e1_reachability_v2 as r
from query_minco.training.physics_only_v1_2_gate1.e1_r2c_audit import actual_gate_geometry

def geometry(box=None):
    boxes=np.array([[[9,2,1],[9.1,2.1,1.1]]] if box is None else [box],float)
    return SimpleNamespace(origin=np.zeros(3),rotation=np.eye(3),roi=np.array([[-2,-4,-2],[12,4,4]],float),
        boxes=boxes,floor=-1.9,domain=torch.tensor([[0,-4,-2],[10,4,2]],dtype=torch.float32),identity={"fixture":"not-production"},validate=lambda:None)
def state(goal=2.4):
    return np.array([.8,0,0,.1,0,0,goal,0,0,0,0,-1],np.float32)

def test_witness_failure_is_unknown_not_infeasible():
    d=r.classify(True,[dict(qualified_witness=False)],state()[6:9],geometry())
    assert d["source"]==r.SOURCE_QUALIFIED and d["reachability"]==r.UNKNOWN and not d["data_eligible"]

def test_source_and_reachability_axes_are_independent():
    d=r.classify(False,[dict(qualified_witness=True)],state()[6:9],geometry())
    assert d["source"]==r.SOURCE_PENDING and d["reachability"]==r.FEASIBLE and not d["data_eligible"]

def test_true_infeasibility_proof_is_not_search_failure():
    g=geometry([[2.3,-.1,-.1],[2.5,.1,.1]]);goal=np.array([2.4,0,0])
    proof=r.endpoint_exclusion_proof(goal,g)
    assert proof and r.verify_exclusion(proof,goal,g)
    d=r.classify(True,[],goal,g,proof)
    assert d["reachability"]==r.INFEASIBLE and not d["data_eligible"]

def test_proof_lipschitz_bound_covers_entire_near_ball():
    g=geometry([[2.3,-.1,-.1],[2.5,.1,.1]]);goal=np.array([2.4,0,0]);proof=r.endpoint_exclusion_proof(goal,g)
    rng=np.random.RandomState(20261104);directions=rng.normal(size=(256,3));directions/=np.linalg.norm(directions,axis=1)[:,None]
    points=goal+directions*.15
    distances=np.array([r.signed_box_distance(x,g.boxes[0]) for x in points])
    assert distances.max()<=proof["upper_distance_over_entire_goal_ball_m"]+1e-12
    assert proof["upper_distance_over_entire_goal_ball_m"]<.15

@pytest.mark.parametrize("fault",["goal","box","upper","geometry"])
def test_forged_or_wrong_task_proof_rejected(fault):
    g=geometry([[2.3,-.1,-.1],[2.5,.1,.1]]);goal=np.array([2.4,0,0]);p=r.endpoint_exclusion_proof(goal,g);p=copy.deepcopy(p)
    if fault=="goal":p["goal_B0"][0]+=.1
    if fault=="box":p["box_world"][0][0]-=.01
    if fault=="upper":p["upper_distance_over_entire_goal_ball_m"]-=.01
    if fault=="geometry":p["geometry_identity"]["fixture"]="wrong"
    with pytest.raises(ValueError,match="INVALID_INFEASIBILITY"):r.classify(True,[],goal,g,p)

def test_conflicting_proofs_are_global_stop():
    g=geometry([[2.3,-.1,-.1],[2.5,.1,.1]]);goal=np.array([2.4,0,0]);p=r.endpoint_exclusion_proof(goal,g)
    with pytest.raises(ValueError,match="CONTRADICTORY"):r.classify(True,[dict(qualified_witness=True)],goal,g,p)

def test_far_or_unknown_roi_not_falsely_proven_infeasible():
    g=geometry([[2.3,-.1,-.1],[2.5,.1,.1]])
    assert r.endpoint_exclusion_proof(np.array([6,0,0]),g) is None
    g.roi[1,0]=2.45
    assert r.endpoint_exclusion_proof(np.array([2.4,0,0]),g) is None

def test_frozen_kinematic_reference_passes_unchanged_Hard():
    s=state();g=geometry();c,d=r.make_candidate(s,2.4,0,0);v=r.evaluate_witness(c,d,s,g)
    assert v["qualified_witness"] and v["hard"]["passed"]
    assert max(v["C0_C4_errors"])<1e-7 and v["initial_PVA_error"]<1e-7

def test_discontinuous_or_wrong_initial_witness_not_certified():
    s=state();g=geometry();c,d=r.make_candidate(s,2.4,0,0)
    c[1,0,0]+=.01
    assert not r.evaluate_witness(c,d,s,g)["qualified_witness"]
    c,d=r.make_candidate(s,2.4,0,0);c[0,0,1]+=.01
    assert not r.evaluate_witness(c,d,s,g)["qualified_witness"]

@pytest.mark.parametrize("fault",["nan","time","shape"])
def test_invalid_numeric_candidates_raise_global_error(fault):
    s=state();g=geometry();c,d=r.make_candidate(s,2.4,0,0)
    if fault=="nan":c[0,0,0]=np.nan
    if fault=="time":d[0]+=.01
    if fault=="shape":c=c[:2]
    with pytest.raises(ValueError):r.evaluate_witness(c,d,s,g)

def test_registry_budget_and_local_failure_semantics():
    config=r.protocol();assert config["near_max_candidates"]==45 and config["far_max_candidates"]==180
    assert config["maximum_28_task_calls"]==14*(45+180)
    config["per_task_wall_s"]=-1 # Fault injection: no candidate budget; never INFEASIBLE.
    result=r.search_task(state(),geometry(),True,config)
    assert result["reachability"]==r.UNKNOWN and result["attempts_executed"]==0
    assert result["local_budget_exhausted"]

@pytest.mark.parametrize("event",["witness_failed","reachability_unknown","infeasible_certified","no_certified_pose"])
def test_task_local_isolation_allows_independent_work(event):
    assert r.failure_action(event).startswith("ISOLATE")

@pytest.mark.parametrize("event",["source_drift","scene_split_collision","frame_mismatch","monitor_failure","resource_budget","contradictory_proofs"])
def test_integrity_and_resource_errors_are_global(event):
    assert r.failure_action(event)=="GLOBAL_STOP"

def controlled(slit=True):
    c=np.zeros((51,81,51),np.int32);c[:,:,0]=1
    if slit:c[10:14,1:80,1:42]=1;c[10:14,40:44,1:42]=0
    else:c[10:14,35:45,1:42]=1
    m=dict(grid_origin=[0,-4,0],resolution_m=.1)
    g=geometry();g.origin=np.array([.25,0,1.8]);g.floor=.10001;g.roi=np.array([[0,-4,0],[5.1,4.1,5.1]])
    layout=dict(slab_box_m=[[1,-3.9,0],[1.4,4,4.2]],remove_y_interval_m=[0,.4],gate_x_m=1.2) if slit else dict(block_box_m=[[1,-.5,0],[1.4,.5,4.2]],gate_x_m=1.2)
    return m,c,layout,g

def test_actual_slit_width_includes_reserve_not_nominal_only():
    m,c,layout,g=controlled();result=actual_gate_geometry(m,c,layout,g)
    expected=4*float(np.float32(.1))-2e-5
    assert abs(result["gap_m"]-expected)<1e-12
    assert result["center_corridor_width_m"]<.1
    assert result["geometry_only_not_dynamic_certificate"]
    c[13,41,18]=1
    assert actual_gate_geometry(m,c,layout,g)["gap_m"]==0

def test_portal_vertical_separator_and_intervals_checked():
    m,c,layout,g=controlled(False);result=actual_gate_geometry(m,c,layout,g)
    assert result["both_geometric_intervals_positive"] and result["no_over_under"]
    c[:,:,31:]=0
    assert not actual_gate_geometry(m,c,layout,g)["no_over_under"]
