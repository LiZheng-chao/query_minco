"""SI quality uses the existing exact positive-Gauss implementation, never a new loss."""
import math
from types import SimpleNamespace as NS
import numpy as np
import torch
from query_minco.training.io import require
from query_minco.training.pg1_v2_r1.losses import dynamic_quality_loss
ID='pg3_next_quality_offline_v1'
SCOPE='OFFLINE_GT_SELECTOR_UPPER_BOUND'
def quality(c,d,solve_valid=None):
    c=torch.as_tensor(c,dtype=torch.float64);d=torch.as_tensor(d,dtype=torch.float64)
    if c.ndim==3:c=c[None,None].expand(1,6,-1,-1,-1).clone();d=d[None,None].expand(1,6,-1).clone()
    elif c.ndim==4:c=c[None];d=d[None]
    valid=torch.ones(c.shape[:2],dtype=torch.bool) if solve_valid is None else torch.as_tensor(solve_valid,dtype=torch.bool).reshape(c.shape[:2])
    z=dynamic_quality_loss(NS(coeffs=c,durations=d,solve_valid=valid),a_ref=8.,j_ref=100.,time_ref=2.2)
    return dict(E_a_SI=z['integrals_si']['acceleration'],E_j_SI=z['integrals_si']['jerk'],QA=z['unmasked_terms']['acceleration'],QJ=z['unmasked_terms']['jerk'],QT=z['unmasked_terms']['time'],duration_s=z['total_time_s'])
def select(rows,key='J_quality'):
    require(key in ('J_quality','QA','QJ','duration_s'),'NEXT_SELECTOR_KEY')
    candidates=[]
    for r in rows:
        if not all(r.get(k) is True for k in ('solve_valid','hard_physics_pass','goal_pass')):continue
        if not all(isinstance(r.get(k),(int,float)) and math.isfinite(r[k]) for k in ('QA','QJ','QT','duration_s')):continue
        require(type(r['query_id']) is int and 0<=r['query_id']<6,'NEXT_QUERY_ID')
        score=r['QA']+r['QJ']+.2*r['QT'] if key=='J_quality' else r[key]
        candidates.append((score,r['query_id']))
    return (min(candidates)[1],'SELECTED_OFFLINE_GT') if candidates else (None,'NO_CERTIFIED_CANDIDATE')
def path_length(c,d):
    from scipy.integrate import quad
    from query_minco.evaluation.validation import evaluate
    return float(sum(quad(lambda t:np.linalg.norm(evaluate(cc,t,1)),0,float(T),epsabs=1e-8,epsrel=1e-8,limit=100)[0] for cc,T in zip(c,d)))
