import torch
from query_minco.trajectory import MincoTrajectory,TrajectorySampler
from .base_imitation import sample_cost_matrix
from query_minco.training.teacher.sampling import teacher_evaluate
CONFIG={"trajectory_cost_points":32}

def cost_matrix(output,sample,b=0):
    tr=MincoTrajectory(output.trajectory.coeffs[b:b+1],output.trajectory.durations[b:b+1],
                       output.trajectory.solve_valid[b:b+1],{})
    costs,pos=sample_cost_matrix(tr,sample,points=CONFIG["trajectory_cost_points"])
    T=tr.durations[0].sum(-1)
    endpoint=TrajectorySampler()(tr,times=T[None,:,None])
    extra=[]
    for i in range(len(sample.teacher_durations)):
        tc=sample.teacher_coeffs[i].to(T.device);td=sample.teacher_durations[i].to(T.device)
        end_v=teacher_evaluate(tc,td,td.sum().expand(6,1),1)[:,0]
        end_a=teacher_evaluate(tc,td,td.sum().expand(6,1),2)[:,0]
        extra.append(.1*(endpoint.v[0,:,0]-end_v).square().mean(-1)+
                     .01*(endpoint.a[0,:,0]-end_a).square().mean(-1))
    return costs+torch.stack(extra,1),pos
