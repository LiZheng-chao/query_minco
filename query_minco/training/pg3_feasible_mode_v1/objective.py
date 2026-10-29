"""Independent PG3 augmentation of the unchanged PG2 Safe computation."""
import torch
from query_minco.training.io import require
from query_minco.training.pg2_guided_v2.objective import forward as pg2_forward
from query_minco.training.pg2_guided_v2.spec import PREFIX as PG2_PREFIX
from .spec import EXTRA,effective,recipe
from .teacher_labels import TeacherBank
from .assignment import assign
from .bottleneck import bottleneck_loss
from .corridor import penalty
from .losses import mode_query,conditional_control,diversity

def augment(z,samples,bank,method,weights,step=0,preferred=None):
 cfg=bank.cfg;raw=z["raw"];out=z["output"];par=out.parameters;tr=out.trajectory
 zero=(raw.sum()+tr.coeffs.sum()+par.durations.sum())*0
 banks=[bank.get(s,f) for s,f in zip(samples,z["fields"])]
 assignments=[];mode_batches=[];way_batches=[];bneck_batches=[];corr_batches=[];route_records=[];qmode=[];qbn=[];qcor=[]
 for b,teachers in enumerate(banks):
  ass=assign(par.inner_pts[b],par.tail_pva[b,:,0],z["base"]["matches"][b],z["labels"]["passed"][b],teachers["teachers"],step,cfg["capacity"],preferred.get(b) if preferred else None)
  assignments.append(ass);ml=[];wl=[];bl=[];cl=[];mq=[zero]*6;bq=[zero]*6;cq=[zero]*6;record=[]
  for q,m in enumerate(ass["route_teacher_id"]):
   row=dict(query=q,teacher=m,reason=ass["reasons"][q])
   target=teachers["teachers"][m] if m>=0 else None
   if q in ass["selected"]:
    value,wp=mode_query(par.inner_pts[b,q],par.tail_pva[b,q],par.durations[b,q],target,cfg)
    ml.append(value);wl.append(wp);mq[q]=value
   eligible=target is not None and target["representable"] and not bool(z["labels"]["passed"][b,q])
   if eligible:
    value,bd=bottleneck_loss(tr.coeffs[b,q],par.durations[b,q],target["bottlenecks"],cfg["sigma_perp"],cfg["sigma_parallel"],cfg["beta_parallel"])
    cv,cd=penalty(tr.coeffs[b,q],par.durations[b,q],target["corridor"],teachers["scene"],teachers["empty_certified"],cfg["sigma_plane"])
    if bd["count"]:bl.append(value)
    if cd["terms"]:cl.append(cv)
    bq[q]=value;cq[q]=cv;row.update(bottleneck=bd,corridor=cd,mode_label_source=target["mode_label_source"],mode_signature=target["mode_signature"])
   else:row["route_loss_reason"]="SAFE_QUERY_PRESERVED_OR_NO_REPRESENTABLE_ASSIGNMENT"
   record.append(row)
  mean=lambda xs:torch.stack(xs).mean() if xs else zero
  mode_batches.append(mean(ml));way_batches.append(mean(wl));bneck_batches.append(mean(bl));corr_batches.append(mean(cl))
  qmode.append(torch.stack(mq));qbn.append(torch.stack(bq));qcor.append(torch.stack(cq));route_records.append(record)
 signed=[]
 for i,s in enumerate(z["safe"]["signed_samples"]):
  signed.append(s.detach().amin(-1) if s is not None else raw.new_full((6,),1.))
 quality,qd=conditional_control(tr,z["labels"]["passed"],z["goal_labels"],torch.stack(signed),cfg)
 div,dd=diversity(tr,assignments,banks,cfg["diversity_fraction"])
 terms=dict(mode=torch.stack(mode_batches).mean(),bottleneck=torch.stack(bneck_batches).mean(),convex_corridor=torch.stack(corr_batches).mean(),**quality,diversity=div)
 require(set(terms)==set(EXTRA),"PG3_TERMS")
 ew=effective(weights,method)
 total=z["loss"]+sum(ew[k]*terms[k] for k in EXTRA)
 require(bool(torch.isfinite(total)),"PG3_LOSS_NONFINITE")
 return dict(loss=total,terms=terms,base=z,raw=raw,output=out,matched=z["matched"],labels=z["labels"],goal_labels=z["goal_labels"],
  assignments=assignments,banks=banks,records=route_records,quality=qd,diversity_records=dd,waypoint_diagnostic=torch.stack(way_batches).mean(),
  per_query=dict(mode=torch.stack(qmode),bottleneck=torch.stack(qbn),convex_corridor=torch.stack(qcor),**qd["per_query"]),
  weights=ew,method=method,recipe=cfg)

def forward(model,samples,geo,method,pg2_weights,weights,bank=None,context=None,step=0):
 z=pg2_forward(model,samples,geo,PG2_PREFIX+"safe_v1",context,pg2_weights)
 return augment(z,samples,bank or TeacherBank(recipe()),method,weights,step)
