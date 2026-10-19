"""Reproducible FP32 directional differences at multiple physical-logit steps."""
import torch
from query_minco.tests.trajectory_common import trajectory_loss

GROUPS = {"q1":(0,3),"q2":(3,6),"q3":(6,9),"vT":(9,12),"aT":(12,15),
          "u_total":(15,16),"time_proportions":(16,19)}
STEPS = (.03,.01,.003)


def directional_checks(raw,state):
    x=raw.clone().requires_grad_()
    loss=trajectory_loss(x,state)
    grad=torch.autograd.grad(loss,x)[0]
    generator=torch.Generator(device=raw.device).manual_seed(808)
    records=[]
    for name,(lo,hi) in GROUPS.items():
        direction=torch.zeros_like(raw)
        direction[...,lo:hi]=torch.randn(raw[...,lo:hi].shape,device=raw.device,generator=generator)
        if name=="time_proportions":
            direction[...,lo:hi]-=direction[...,lo:hi].mean(-1,keepdim=True)
        direction/=direction.norm()
        analytic=float((grad*direction).sum())
        for step in STEPS:
            with torch.no_grad():
                finite_difference=float((trajectory_loss(raw+step*direction,state)-
                                         trajectory_loss(raw-step*direction,state))/(2*step))
            absolute=abs(finite_difference-analytic)
            records.append({"group":name,"step":step,"autograd":analytic,
                            "finite_difference":finite_difference,"absolute_error":absolute,
                            "scaled_relative_error":absolute/max(abs(analytic),1e-3)})
    return grad,records
