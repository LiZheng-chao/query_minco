"""CPU offline reachability evidence only. No teacher/network/loss/optimizer use."""
from dataclasses import dataclass
import hashlib
import itertools
import json
import time
import numpy as np
from query_minco.evaluation.goal_config import CONFIG as GOAL
from query_minco.evaluation.validation import numerical_reference,evaluate
from .hard_validator import validate_polynomial

SOURCE_QUALIFIED="SOURCE_QUALIFIED"
SOURCE_PENDING="SOURCE_PENDING_BINDING"
FEASIBLE="FEASIBLE_CERTIFIED"
INFEASIBLE="INFEASIBLE_CERTIFIED"
UNKNOWN="REACHABILITY_UNKNOWN"

def protocol():
    return dict(id="e1_reachability_offline_v2",duration_s=2.,segments=[2/3]*3,
        y_offsets_m=[0.,-.2,.2,-.4,.4,-.6,.6,-.8,.8],z_offsets_m=[0.,-.2,.2,-.4,.4],
        far_endpoint_progress_m=[3.6,1.2,1.8,2.4],near_endpoint="unchanged goal exactly",
        inner_x_fractions=[1/3,2/3],same_y_z_offset_at_both_inner_points=True,
        terminal_near_velocity=[0,0,0],terminal_far_velocity=[1.2,0,0],terminal_acceleration=[0,0,0],
        near_max_candidates=45,far_max_candidates=180,per_task_wall_s=20.,maximum_28_task_calls=3150,
        independent_validator="unchanged hard_validator.validate_polynomial",
        boundary_continuity_atol=1e-7,teacher_labels=False,network_input=False,
        failure="finite candidate failures/budget exhaustion imply UNKNOWN only",
        global_stop=["source/scene/hash/frame contradiction","resource monitor or budget failure","conflicting verified feasible/infeasible proofs","numerical/nonfinite implementation error"])

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

def signed_box_distance(point,box):
    q=np.maximum(np.maximum(box[0]-point,point-box[1]),0)
    outside=float(np.linalg.norm(q))
    if outside>0:return outside
    return -float(np.min(np.minimum(point-box[0],box[1]-point)))

def endpoint_exclusion_proof(goal,g):
    """One occupied AABB's 1-Lipschitz distance upper bound over entire near-goal ball."""
    g.validate();goal=np.asarray(goal,float)
    if np.linalg.norm(goal)>GOAL["near_distance_threshold_m"]:return None
    w=g.origin+g.rotation@goal;tol=GOAL["near_endpoint_tolerance_m"]
    if (w-tol<g.roi[0]).any() or (w+tol>g.roi[1]).any():return None
    bounds=np.array([signed_box_distance(w,b)+tol for b in g.boxes])
    if not len(bounds):return None
    i=int(np.argmin(bounds));upper=float(bounds[i])
    if upper>=.15-1e-9:return None
    return dict(kind="near_endpoint_ball_excluded_by_known_occupied_AABB_v1",goal_B0=goal.tolist(),
        goal_tolerance_m=tol,required_distance_m=.15,box_index=i,box_world=g.boxes[i].tolist(),
        upper_distance_over_entire_goal_ball_m=upper,geometry_identity=dict(g.identity),
        proof="signed box distance is 1-Lipschitz; d_box(goal)+goal_tolerance < .15 excludes every permissible terminal position",
        scope="fixed finite known geometry and unchanged near task; not a general search failure")

def verify_exclusion(proof,goal,g):
    actual=endpoint_exclusion_proof(goal,g)
    return actual is not None and proof==actual

def classify(source_qualified,attempts,goal,g,proof=None):
    feasible=any(a["qualified_witness"] for a in attempts)
    infeasible=proof is not None and verify_exclusion(proof,goal,g)
    if proof is not None and not infeasible:raise ValueError("INVALID_INFEASIBILITY_PROOF")
    if feasible and infeasible:raise ValueError("CONTRADICTORY_PROOFS_GLOBAL_STOP")
    reach=FEASIBLE if feasible else INFEASIBLE if infeasible else UNKNOWN
    return dict(source=SOURCE_QUALIFIED if source_qualified else SOURCE_PENDING,reachability=reach,
                data_eligible=bool(source_qualified and reach==FEASIBLE),proof=proof,
                old_record_mutated=False,offline_only=True)

def make_candidate(state,endpoint_x,y,z):
    state=np.asarray(state,float);goal=state[6:9];near=np.linalg.norm(goal)<=GOAL["near_distance_threshold_m"]
    endpoint=goal.copy() if near else goal/np.linalg.norm(goal)*endpoint_x
    head=np.stack([np.zeros(3),state[:3],state[3:6]])
    tail=np.stack([endpoint,np.zeros(3) if near else np.array([1.2,0,0]),np.zeros(3)])
    inner=np.stack([endpoint/3,endpoint*2/3]);inner[:,1]+=y;inner[:,2]+=z
    d=np.full(3,2/3)
    return numerical_reference(head,tail,inner,d),d

def evaluate_witness(coeffs,durations,state,g):
    c=np.asarray(coeffs,float);d=np.asarray(durations,float);s=np.asarray(state,float)
    if c.shape!=(3,3,6) or d.shape!=(3,) or s.shape!=(12,) or not np.isfinite(np.r_[c.ravel(),d,s]).all():raise ValueError("NONFINITE_OR_SHAPE_GLOBAL_STOP")
    if not np.allclose(d,np.full(3,2/3),atol=1e-12,rtol=0):raise ValueError("FROZEN_DURATION")
    expected=np.stack([np.zeros(3),s[:3],s[3:6]])
    initial=max(float(np.abs(evaluate(c[0],0,k)-expected[k]).max()) for k in range(3))
    continuity=[max(float(np.abs(evaluate(c[i],d[i],k)-evaluate(c[i+1],0,k)).max()) for i in range(2)) for k in range(5)]
    hard=validate_polynomial(c,d,g,s[6:9])
    qualified=bool(hard["passed"] and initial<=1e-7 and max(continuity)<=1e-7)
    return dict(qualified_witness=qualified,hard=hard,initial_PVA_error=initial,C0_C4_errors=continuity)

def gate_sides(coeffs,d,g,gate):
    if gate is None:return []
    crossed=[]
    for c,t in zip(coeffs,d):
        world=np.einsum("ij,jk->ik",g.rotation,c);world[:,0]+=g.origin
        poly=world[0].copy();poly[0]-=gate["x"]
        for root in np.polynomial.polynomial.polyroots(poly):
            if abs(root.imag)<1e-7 and 0<=root.real<=t:
                p=evaluate(world,float(root.real))
                if p[1]<gate["block_y_low"]-.150001:crossed.append("negative_y")
                if p[1]>gate["block_y_high"]+.150001:crossed.append("positive_y")
    end=evaluate(coeffs[-1],d[-1])@g.rotation.T+g.origin
    if end[0]<=gate["block_x_high"]+.150001:return []
    return sorted(set(crossed))

def search_task(state,g,source_qualified,config,gate=None):
    state=np.asarray(state,float)
    near=np.linalg.norm(state[6:9])<=GOAL["near_distance_threshold_m"]
    endpoints=[float(np.linalg.norm(state[6:9]))] if near else config["far_endpoint_progress_m"]
    registry=list(itertools.product(endpoints,config["y_offsets_m"],config["z_offsets_m"]))
    limit=config["near_max_candidates"] if near else config["far_max_candidates"]
    if len(registry)!=limit:raise ValueError("PREREGISTRY_SIZE")
    proof=endpoint_exclusion_proof(state[6:9],g);attempts=[];witnesses={};sides=set();started=time.monotonic();budget_hit=False
    if proof is None:
        for i,(endpoint,y,z) in enumerate(registry):
            if time.monotonic()-started>config["per_task_wall_s"]:budget_hit=True;break
            c,d=make_candidate(state,endpoint,y,z);row=evaluate_witness(c,d,state,g)
            row.update(index=i,endpoint_progress_m=endpoint,y_offset_m=y,z_offset_m=z,gate_sides=gate_sides(c,d,g,gate) if row["qualified_witness"] else [])
            attempts.append(row)
            if row["qualified_witness"]:
                witnesses[str(i)]=dict(coeffs_B0=c.tolist(),durations_s=d.tolist(),offline_audit_only=True)
                sides.update(row["gate_sides"])
                if gate is None or near or sides=={"negative_y","positive_y"}:break
    result=classify(source_qualified,attempts,state[6:9],g,proof)
    result.update(attempts=attempts,witnesses=witnesses,registry_sha256=digest(registry),
        fixed_registry_size=len(registry),attempts_executed=len(attempts),elapsed_s=time.monotonic()-started,
        local_budget_exhausted=budget_hit,portal_sides=sorted(sides),
        portal_coverage="PASS" if sides=={"negative_y","positive_y"} else "UNKNOWN" if gate is not None else "NOT_APPLICABLE",
        task_state=state.tolist(),geometry_identity=dict(g.identity),infeasibility_not_from_search=True)
    return result

def failure_action(event):
    if event in ("witness_failed","reachability_unknown","infeasible_certified"):
        return "ISOLATE_TASK_CONTINUE_INDEPENDENT_TASKS"
    if event=="no_certified_pose":return "ISOLATE_POSE_NO_RENDER_CONTINUE_INDEPENDENT_SCENES"
    if event in ("source_drift","scene_split_collision","frame_mismatch","monitor_failure","resource_budget","contradictory_proofs"):
        return "GLOBAL_STOP"
    raise ValueError("UNCLASSIFIED_FAILURE_GLOBAL_STOP")
