"""Qualification-only analytic witnesses. Never imported by initialization functions."""
import itertools
import numpy as np
import torch
from experiments.query_minco_p1a2_falsification_v1.falsification import result_for_theta,crossing_success,p1

def x_reference(state):
    st=state[0].double();T=2.
    M=st.new_tensor([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,2,0,0,0],
        [1,T,T*T,T**3,T**4,T**5],[0,1,2*T,3*T*T,4*T**3,5*T**4],[0,0,2,6*T,12*T*T,20*T**3]])
    c=torch.linalg.solve(M,st.new_tensor([0,float(st[0]),float(st[3]),4.,1.2,0.]))
    ts=st.new_tensor([2/3,4/3,2.])
    return (ts[:,None]**torch.arange(6,dtype=st.dtype))@c
def qualification_candidates(batch,contract):
    x=x_reference(batch.state).detach().numpy()
    grids=[("slit",[.1,.15,.2],[.15,.2],.15)] if contract["kind"]=="slit" else [("negative_y",[-.3,-.4],[-.7,-.8],-.7),("positive_y",[.4,.5],[1.,1.1],.7)]
    records=[]
    for target,p1s,p2s,tail in grids:
        for a,b in itertools.product(p1s,p2s):
            theta=[x[0],a,0,x[1],b,0,x[2],tail,0,1.2,0,0,0,0,0]
            result=result_for_theta(theta,batch.state);hard=p1.hard_one(result,batch)
            task=crossing_success(result,batch,contract,hard)
            records.append(dict(target_channel=target,theta=theta,coefficients=result.trajectory.coeffs[0,0].detach().tolist(),
                durations=result.trajectory.durations[0,0].detach().tolist(),hard=hard,task=task,
                qualification_only=True,Teacher_label=False,used_as_optimizer_initialization=False))
    need=["slit"] if contract["kind"]=="slit" else ["negative_y","positive_y"]
    qualified={s:any(r["task"]["TaskSuccess"] and s in r["task"]["successful_channels"] for r in records) for s in need}
    return dict(status="FEASIBLE_CROSSING_CERTIFIED" if all(qualified.values()) else "TASK_NOT_QUALIFIED",
        side_certification=qualified,candidates=records,max_candidates=6 if contract["kind"]=="slit" else 8,network_updates=0,optimizer_steps=0)
