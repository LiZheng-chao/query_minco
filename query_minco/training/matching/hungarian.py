import torch
from scipy.optimize import linear_sum_assignment
from query_minco.training.losses.base_imitation import NoTeacherError

def match(cost,arm,teacher_index=None):
    if cost.ndim!=2 or cost.shape[0]!=6:raise ValueError("cost [6,M] required")
    M=cost.shape[1]
    if M==0:raise NoTeacherError("M=0: diagnosis only, no pseudo teacher")
    if M>6 or not bool(torch.isfinite(cost).all()):raise ValueError("invalid teacher count/nonfinite cost")
    if arm=="B_set":
        q,t=linear_sum_assignment(cost.detach().cpu().numpy())
        q=torch.as_tensor(q,device=cost.device,dtype=torch.long)
        t=torch.as_tensor(t,device=cost.device,dtype=torch.long)
        assert len(q)==M and q.unique().numel()==M
    elif arm=="A_single_best6":
        if type(teacher_index) is not int or not 0<=teacher_index<M:raise ValueError("single teacher index required")
        t=torch.tensor([teacher_index],device=cost.device);q=cost[:,teacher_index].detach().argmin().reshape(1)
    else:raise ValueError("unknown arm")
    return cost[q,t].mean(),q,t
