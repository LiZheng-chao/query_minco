"""Optional shared-head coordination and detached functional AdamW simulation."""
import copy,math
import torch
from torch.optim._functional import adamw
from query_minco.training.io import require
EPS=1e-12
def masks(named_parameters,device):
 head=[];score=[]
 for name,p in named_parameters:
  h=torch.full((p.numel(),),name.startswith("trajectory_head."),dtype=torch.bool,device=device);s=torch.zeros(p.shape,dtype=torch.bool,device=device)
  if name.startswith("trajectory_head.") and len(p.shape)>0 and p.shape[0]==21:s[19:21]=True
  head.append(h);score.append(s.flatten())
 require(bool(head),"HEAD_PARAMETERS_EMPTY")
 return torch.cat(head),torch.cat(score)
def coordinate(total,imitation,repair,named_parameters,enabled=True,eps=EPS):
 require(type(enabled) is bool and eps==EPS,"HEAD_ROUTING_CONFIG")
 require(total.ndim==1 and total.shape==imitation.shape==repair.shape and total.dtype==imitation.dtype==repair.dtype and total.device==imitation.device==repair.device,"HEAD_ROUTING_SHAPE")
 require(all(bool(torch.isfinite(x).all()) for x in (total,imitation,repair)),"HEAD_ROUTING_NONFINITE")
 head,score=masks(named_parameters,total.device);require(len(total)==len(head) and bool(head.any()),"HEAD_LAYOUT")
 require(all(not torch.count_nonzero(x[score]) for x in (total,imitation,repair)),"HEAD_SCORE_GRADIENT")
 gi=imitation[head].detach().double();gr=repair[head].detach().double();dot=float(gi@gr);normsq=float(gr@gr)
 coefficient=min(dot,0.)/(normsq+eps) if enabled else 0.
 result=total.detach().clone()
 if coefficient:
  new_im=(gi-coefficient*gr).to(total.dtype);result[head]=total[head]+(new_im-imitation[head])
 require(torch.equal(result[~head],total[~head]) and not torch.count_nonzero(result[score]) and bool(torch.isfinite(result).all()),"HEAD_ROUTING_SCOPE")
 actual_new_im=gi+(result[head]-total[head]).double()
 return result,dict(enabled=enabled,conflict=dot<0,applied=coefficient!=0.,eps=eps,coefficient=coefficient,imitation_repair_dot_before=dot,imitation_repair_dot_after=float(actual_new_im@gr),repair_head_norm=math.sqrt(normsq),correction_norm=float((result-total).norm()),outside_head_bitwise_unchanged=True)
def actual_repair_vector(z,vectors):
 return sum(z["weights"][k]*vectors[k] for k in ("mode","bottleneck","convex_corridor"))
def functional_candidate(named_parameters,optimizer_state,total_gradient,lr,clip=10.):
 """Mutates detached temporary tensors only; never creates/calls an Optimizer."""
 require(type(lr) is float and math.isfinite(lr) and lr>0 and clip==10.,"FUNCTIONAL_ADAM_CONFIG")
 named_parameters=list(named_parameters);require(len(optimizer_state["param_groups"])==1,"FUNCTIONAL_ADAM_GROUPS")
 state=copy.deepcopy(optimizer_state);group=state["param_groups"][0];ids=group["params"]
 require(len(ids)==len(named_parameters) and not group["amsgrad"] and not group["capturable"] and not group["differentiable"] and not group["maximize"] and group["weight_decay"]==0.,"FUNCTIONAL_ADAM_FLAGS")
 pp=[];gg=[];mm=[];vv=[];steps=[];off=0;norms={}
 for pid,(name,p) in zip(ids,named_parameters):
  q=p.detach().clone();g=total_gradient[off:off+p.numel()].reshape_as(p).detach().clone();off+=p.numel();q.grad=g
  pp.append(q);gg.append(q.grad);norms[name]=float(g.norm())
  if pid not in state["state"]:state["state"][pid]=dict(step=torch.tensor(0.),exp_avg=torch.zeros_like(q),exp_avg_sq=torch.zeros_like(q))
  s=state["state"][pid];mm.append(s["exp_avg"]);vv.append(s["exp_avg_sq"]);steps.append(s["step"])
 require(off==len(total_gradient),"FUNCTIONAL_ADAM_LAYOUT")
 norm=float(torch.nn.utils.clip_grad_norm_(pp,clip))
 with torch.no_grad():
  adamw(pp,gg,mm,vv,[],steps,foreach=group.get("foreach"),capturable=False,differentiable=False,fused=group.get("fused"),grad_scale=None,found_inf=None,has_complex=False,amsgrad=False,beta1=group["betas"][0],beta2=group["betas"][1],lr=lr,weight_decay=group["weight_decay"],eps=group["eps"],maximize=False)
 require(all(bool(torch.isfinite(x).all()) for x in pp+mm+vv),"FUNCTIONAL_ADAM_NONFINITE")
 state["param_groups"][0]["lr"]=lr
 result={name:q.detach().clone() for (name,p),q in zip(named_parameters,pp)}
 delta=torch.cat([(q-p.detach()).flatten() for (name,p),q in zip(named_parameters,pp)])
 return result,state,delta,dict(gradient_norm_before_clip=norm,clip_threshold=clip,per_parameter_gradient_norm=norms,clipped=norm>clip,optimizer_object_created=False,optimizer_step_called=False)
