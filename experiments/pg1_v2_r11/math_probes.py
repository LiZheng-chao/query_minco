"""R1.1 diagnostic math only. No recipe, optimizer, or certificate."""
import torch
from query_minco.training.pg1_v2_r1.losses import weighted_lse,weighted_upper_tail
def aggregate(risk,alpha,rho,tau=.025,q=.1):
    """rho detached. New rho variants use differentiable Z=sum(alpha*rho).
    All new outputs have metres as units and mass exactly normalized.
    Legacy E2 normalizer was detached: same values, not necessarily time VJP.
    """
    if risk.shape!=alpha.shape or rho.shape!=risk.shape:raise ValueError('SHAPE')
    if not all(torch.isfinite(x).all() for x in (risk,alpha,rho)):raise ValueError('FINITE')
    if not (alpha>=0).all() or not (rho>0).all():raise ValueError('MASS')
    a=alpha/alpha.sum(-1,keepdim=True)
    m=a*rho.detach();m=m/m.sum(-1,keepdim=True)
    return dict(mean_plain=(a*risk).sum(-1),mean_rho=(m*risk).sum(-1),
      lse=weighted_lse(risk,a,tau),lse_rho=weighted_lse(risk,m,tau),
      tail=weighted_upper_tail(risk,a,q),tail_rho=weighted_upper_tail(risk,m,q))
def current_time_quality(qa,qj,total):
    if not torch.isfinite(total).all() or not (total>0).all():raise ValueError('TIME')
    return dict(QA_current=qa*2.2/total,QJ_current=qj*2.2/total)
def project_quality(gq,gs):
    """Euclidean parameter gradient projection against OLD obstacle only.
    Descent -gq has nonpositive first-order old proxy change after projection.
    No hard-clearance guarantee. Not a training strategy.
    """
    den=gs.double().dot(gs.double());dot=gq.double().dot(gs.double())
    if den<=0 or dot>=0:return gq.clone()
    return gq-(dot/den).to(gq)*gs
def unique_task_rows(rows):
    """Stable metadata dedup, preserves all exposure logs; never select by loss."""
    result=[];seen={}
    for row in rows:
        key=(row['model'],row['sample_id'])
        payload={k:v for k,v in row.items() if k not in ('batch','first_visit')}
        if key in seen:
            if payload!=seen[key]:raise ValueError('DUPLICATE_TASK_PAYLOAD_MISMATCH')
        else:
            seen[key]=payload;result.append(row)
    return result
def mark_first_visits(ids,seen):
    flags=[]
    for sid in ids:
        flags.append(sid not in seen)
        seen.add(sid)
    return flags
