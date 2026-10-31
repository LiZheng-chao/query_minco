"""Euclidean Adam-displacement cone projection, solved only in constraint space."""
import time
import numpy as np
from scipy.optimize import minimize
from query_minco.training.io import require
ID='pg3_next_query_cone_qp_v1'
ALPHAS=(1.,.5,.25,.125,.0625,.03125)
KKT_TOL=1e-7

def project(adam,G):
    start=time.perf_counter();a=np.asarray(adam,dtype=np.float64);G=np.asarray(G,dtype=np.float64)
    require(a.ndim==1 and G.ndim==2 and G.shape[1]==len(a),'NEXT_QP_SHAPE')
    require(np.isfinite(a).all() and np.isfinite(G).all(),'NEXT_QP_NONFINITE')
    scale=float(np.linalg.norm(a));require(scale>0,'NEXT_ZERO_ADAM_PROPOSAL')
    if not len(G):return a.copy(),dict(status='PASS_NO_CONSTRAINT',kkt_max=0.,runtime_s=time.perf_counter()-start,relative_norm=1.)
    norms=np.linalg.norm(G,axis=1);require((norms>1e-12).all(),'NEXT_ZERO_PROTECTION_GRADIENT')
    A=G/norms[:,None];unit=a/scale;b=A@unit;H=A@A.T
    if (b>=0).all():lam=np.zeros(len(A));success=True;message='Adam already feasible';iterations=0
    else:
        r=minimize(lambda x:.5*x@H@x+b@x,np.zeros(len(A)),jac=lambda x:H@x+b,bounds=[(0,None)]*len(A),method='SLSQP',options=dict(ftol=1e-13,maxiter=2000))
        lam=r.x;success=bool(r.success);message=str(r.message);iterations=int(r.nit)
    # Deterministic dual coordinate polishing solves the SAME nonnegative QP;
    # it tightens convergence, never changes epsilon, weights or KKT acceptance.
    residual=H@lam+b;polish_sweeps=0;polish_converged=False
    for sweep in range(5000):
        if np.max(np.abs(lam-np.maximum(lam-residual,0.)))<=1e-10:polish_converged=True;break
        for i in range(len(lam)):
            nxt=max(0.,lam[i]-residual[i]/H[i,i]);change=nxt-lam[i]
            if change:lam[i]=nxt;residual+=change*H[:,i]
        polish_sweeps=sweep+1
    success=success or polish_converged
    v=unit+A.T@lam;primal=A@v
    station=float(np.linalg.norm(v-unit-A.T@lam));infeasible=float(max(0.,-primal.min()));dual=float(max(0.,-lam.min()));complement=float(np.max(np.abs(lam*primal)))
    gap=float(abs(.5*np.linalg.norm(v-unit)**2+.5*lam@H@lam+b@lam))
    kkt=max(station,infeasible,dual,complement,gap)
    info=dict(status='PASS' if success and kkt<=KKT_TOL else 'FAILED_QP',success=success,message=message,iterations=iterations,polish_sweeps=polish_sweeps,polish_converged=polish_converged,kkt_max=kkt,primal_violation=infeasible,dual_violation=dual,stationarity=station,complementarity=complement,duality_gap=gap,normalized_predicted_margins=primal.tolist(),multipliers=lam.tolist(),runtime_s=time.perf_counter()-start,relative_norm=float(np.linalg.norm(v)),gram_rank=int(np.linalg.matrix_rank(H)),constraints=len(A),parameter_dimension=len(a),hessian_dimension=len(A),epsilon_i=0.)
    require(info['status']=='PASS','NEXT_QP_KKT_OR_CONVERGENCE_FAILED:'+str(info))
    return scale*v,info

def search(direction,evaluate,*,baseline_mode,baseline_geom=None):
    trials=[]
    for alpha in ALPHAS:
        row=evaluate(alpha*direction);row=dict(row,alpha=alpha);trials.append(row)
        if row.get('protected_pass') is True and row.get('all_numerical_valid') is True and row.get('protected_goal_pass') is True:
            return dict(status='ACCEPTED',chosen_alpha=alpha,trials=trials,chosen=row,progress_not_zero=bool(row.get('relative_update_norm',0)>1e-6))
    return dict(status='REJECTED',chosen_alpha=None,trials=trials,chosen=None,progress_not_zero=False)


def numerical_attempt(evaluate,delta):
    """A failed native MINCO solve rejects this alpha, never the remaining grid."""
    from query_minco.trajectory.minco_adapter import NumericalSolveError
    tick=time.perf_counter()
    try:return evaluate(delta)
    except NumericalSolveError as error:
        return dict(protected_pass=False,protected_goal_pass=False,all_numerical_valid=False,pre_safe_pass_to_fail=None,pre_fail_to_pass=None,protected_min_clearance_change_m=None,relative_update_norm=0.,cosine_to_Adam=None,Mode_loss_delta=None,Geom_loss_delta=None,validator_runtime_s=time.perf_counter()-tick,total_runtime_s=time.perf_counter()-tick,numerical_rejection=str(error),formal_optimizer_state_equivalent=False)
