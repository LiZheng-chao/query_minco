"""Conservative GT-free diagnostic selector. No ROS and no controller publication."""
import hashlib,importlib.util
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import numpy as np
import torch
from query_minco.inference.planner_adapter import InferenceAdapter
from query_minco.evaluation.validation import evaluate,norm_max
class Observability(str,Enum):
    CERTIFIED_FREE='CERTIFIED_FREE'
    BLOCKED='BLOCKED'
    UNKNOWN='UNKNOWN'
@dataclass(frozen=True)
class Limits:
    age_s:float=.25
    speed:float=3.
    acceleration:float=8.
    jerk:float=100.
OLD=Path('/home/lizheng/reach_planner/query_minco/p2c2b2b/selector.py')
OLD_SHA='4c66a50dfb45446eeedcbb9b8fd27862e556d3a5a75dd355c53b34c2cf4dab23'
def lawful_features(depth,state,out):
    if hashlib.sha256(OLD.read_bytes()).hexdigest()!=OLD_SHA:raise ValueError('OLD_FEATURE_SOURCE_DRIFT')
    spec=importlib.util.spec_from_file_location('read_only_s1_features',OLD)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    names=['uniform_p','uniform_v','uniform_a','uniform_jerk','uniform_times']
    arrays=[torch.from_numpy(np.array(out[k],copy=True))[None] for k in names]
    with torch.inference_mode():
        return m.point_features(torch.from_numpy(depth.copy())[None],torch.from_numpy(state.copy())[None],*arrays).numpy()[0]
def select(depth,state,out,age_s,limits=Limits()):
    """Ranking is diagnostic only. There is deliberately no certificate injection API.
    Numerical extrema use existing polynomial roots, not interval arithmetic.
    A valid centre ray or finite point sample cannot certify a swept body volume.
    """
    rejected=[]
    try:
        InferenceAdapter.inputs(depth[None],state[None],None)
        if not np.isfinite(age_s) or not 0<=age_s<=limits.age_s:raise ValueError('STALE_OR_FUTURE')
        c=np.asarray(out['coefficients']);d=np.asarray(out['durations'])
        if c.shape!=(6,3,3,6) or d.shape!=(6,3) or not np.isfinite(c).all() or not np.isfinite(d).all():raise ValueError('COEFFICIENT_SHAPE_FINITE')
        if not np.asarray(out['solve_valid']).all():raise ValueError('SOLVE_INVALID')
        features=lawful_features(depth,state,out)
    except (ValueError,KeyError,IndexError) as e:
        return dict(selected_query=None,certified_count=0,input_gate='BLOCKED',reason=str(e),
            candidates=[dict(query=i,observability='UNKNOWN',execution_gate='BLOCKED') for i in range(6)],diagnostic_ranking=[])
    rows=[]
    for qi,(coeffs,times) in enumerate(zip(c,d)):
        reasons=[];maxima=[None]*3
        if np.min(times)<.12-1e-6 or not .8-1e-5<=sum(times)<=2.2+1e-5:reasons.append('TIME_CONTRACT')
        if not reasons:
            head=np.stack([evaluate(coeffs[0],0,k) for k in range(3)])
            expected=np.stack([np.zeros(3),state[:3],state[3:6]])
            if np.max(np.abs(head-expected))>.01:reasons.append('HEAD_PVA')
            for k,tol in enumerate([.0002,.002,.01]):
                if max(np.max(np.abs(evaluate(coeffs[i],times[i],k)-evaluate(coeffs[i+1],0,k))) for i in range(2))>tol:reasons.append('PVA_CONTINUITY_'+str(k))
            maxima=[max(norm_max(a,float(t),k) for a,t in zip(coeffs,times)) for k in (1,2,3)]
            for name,value,limit in zip(['SPEED','ACCELERATION','JERK'],maxima,[limits.speed,limits.acceleration,limits.jerk]):
                if value>limit:reasons.append(name+'_LIMIT')
        endpoint=evaluate(coeffs[-1],float(times[-1]),0) if (times>0).all() else np.zeros(3)
        goal_error=float(np.linalg.norm(endpoint-state[6:9]))
        # Sampled depth feature is retained ONLY as an observability diagnostic.
        # Behind a depth return is occluded/UNKNOWN, not proved occupied.
        rows.append(dict(query=qi,observability=Observability.UNKNOWN.value,
            execution_gate='BLOCKED' if reasons else 'WAITING_FOR_BODY_CERTIFICATE',
            reasons=reasons,polynomial_extrema=maxima,point_unknown_fraction=float(features[qi,:,25].mean()),
            point_occluded_fraction=float(features[qi,:,26].mean()),goal_error_m=goal_error,
            duration_s=float(times.sum())))
    ranking=sorted(range(6),key=lambda i:(bool(rows[i]['reasons']),rows[i]['goal_error_m'],rows[i]['duration_s'],i))
    return dict(selected_query=None,certified_count=0,input_gate='VALID_NUMERICAL_INPUT',
        candidates=rows,diagnostic_ranking=ranking,ranking_is_control_permission=False,
        reason='NO_CONTINUOUS_OBSERVED_BODY_VOLUME_CERTIFICATE')
