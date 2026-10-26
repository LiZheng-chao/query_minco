"""F3 wrapper reusing immutable F2 numerical gate; only certified candidates rank."""
import numpy as np
from query_minco.inference.online_selector import select as numerical_select,Limits
from query_minco.inference.observed_space import Scope
def world_coefficients(coeffs,origin,rotation):
    origin=np.asarray(origin,float);rotation=np.asarray(rotation,float)
    if origin.shape!=(3,) or rotation.shape!=(3,3) or not np.isfinite(origin).all() or not np.isfinite(rotation).all() or not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-7) or abs(np.linalg.det(rotation)-1)>1e-7:
        raise ValueError('WORLD_FRAME_POSE')
    c=np.einsum('ij,...jk->...ik',np.asarray(rotation,float),np.asarray(coeffs,float))
    c[...,0]+=np.asarray(origin,float);return c
def select_with_safety(depth,state,out,age_s,history,now,origin,rotation,limits=Limits()):
    # No GT, Oracle, raw19/20 probability or old S1 weights.
    numerical=numerical_select(depth,state,out,age_s,limits)
    volume=history.volume(now);rows=[];eligible=[]
    for row in numerical['candidates']:
        row=dict(row);qi=row['query']
        if numerical['input_gate']=='BLOCKED' or row.get('reasons'):
            row.update(status='BLOCKED',safety_reason='NUMERICAL_OR_KINEMATIC_GATE',certificate=None)
        else:
            c=world_coefficients(out['coefficients'][qi],origin,rotation)
            cert=volume.certify(c,out['durations'][qi],now)
            # Analytic fixtures / an empty-room supervisor can never impersonate online sensor certification.
            online_scope=volume.scope==Scope.NATIVE_RAYS
            status=cert['status'] if online_scope else 'UNKNOWN'
            row.update(status=status,safety_reason=cert['reason'],certificate=cert)
            if status=='CERTIFIED_FREE':eligible.append(qi)
        rows.append(row)
    ranking=sorted(eligible,key=lambda i:(rows[i]['goal_error_m'],rows[i]['duration_s'],i))
    return dict(candidates=rows,eligible_ranking=ranking,selected_query=ranking[0] if ranking else None,
        certified_count=len(ranking),GT_used=False,raw_scores_used=False,control_permission=False)
