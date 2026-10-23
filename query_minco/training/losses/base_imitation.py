import torch
from query_minco.trajectory import TrajectorySampler
from query_minco.training.teacher.sampling import teacher_evaluate

class NoTeacherError(ValueError):
    pass

def sample_cost_matrix(trajectory,sample,points=32):
    if sample.teacher_coeffs.shape[0]==0:
        raise NoTeacherError(sample.sample_id+": no teacher")
    device=trajectory.coeffs.device
    tc=sample.teacher_coeffs.to(device);td=sample.teacher_durations.to(device)
    predicted_total=trajectory.durations[0].sum(-1)
    u=torch.linspace(0,1,points,device=device)
    end_times=predicted_total[None,:,None]
    pred_end=TrajectorySampler()(trajectory,times=end_times)
    costs=[];position_errors=[]
    for m in range(len(td)):
        total=td[m].sum()
        # Both curves are evaluated at identical physical seconds, up to their common domain.
        times=torch.minimum(predicted_total,total)[...,None]*u
        pred=TrajectorySampler()(trajectory,times=times[None])
        target_p=teacher_evaluate(tc[m],td[m],times,0)
        target_v=teacher_evaluate(tc[m],td[m],times,1)
        target_end=teacher_evaluate(tc[m],td[m],total.expand(6,1),0)[:,0]
        pos=(pred.p[0]-target_p).square().mean(dim=(-2,-1))
        vel=(pred.v[0]-target_v).square().mean(dim=(-2,-1))
        end=(pred_end.p[0,:,0]-target_end).square().mean(-1)
        duration=(predicted_total-total).square()
        costs.append(pos+.1*vel+.5*end+.2*duration)
        position_errors.append(pos)
    return torch.stack(costs,dim=1),torch.stack(position_errors,dim=1)
