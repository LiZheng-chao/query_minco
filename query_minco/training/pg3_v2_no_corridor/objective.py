"""PG3 V2 composition; unchanged PG2 Safe and Hungarian primitives."""
import math
import torch
from query_minco.training.io import require
from query_minco.training.pg2_guided_v2.objective import forward as pg2_forward
from query_minco.training.pg3_feasible_mode_v1.spec import recipe
from query_minco.training.pg3_feasible_mode_v1.teacher_labels import TeacherBank
from query_minco.training.pg3_feasible_mode_v1.losses import mode_query
from query_minco.training.pg3_feasible_mode_v1.bottleneck import bottleneck_loss,at_progress
from .revision import PerSampleFairAssignment, FrozenSafeReference, section_side_proxy, full_stopping_risk, query_collapse_diagnostic
METHOD_ID="query_minco_pg3_v2_no_corridor_v1"
KEYS=("mode","bottleneck","diversity","section","safe_keep","full_stop")
PILOT=dict(section=.10,safe_keep=.05,full_stop=.01)

def reliable_target(t):
    if not t["certified"] or not t["representable"]:return dict(t,bottlenecks=[],mode_signature=[])
    labels=[x for x in t["bottlenecks"] if x["valid"] and x.get("side") is not None
            and x.get("event_status")=="UNIQUE_FORWARD_CROSSING"]
    return dict(t,bottlenecks=labels,mode_signature=[(x["box"],*x["side"]) for x in labels])

def diversity(trajectory,assignments,banks,fraction):
    """Teacher-directed separation on verified distinct faces; nonzero at collapse."""
    values=[];records=[]
    for b,a in enumerate(assignments):
        active=a["selected"]
        for i,q in enumerate(active):
            for r in active[i+1:]:
                tm,tn=[banks[b]["teachers"][a["route_teacher_id"][j]] for j in (q,r)]
                if not tm["mode_signature"] or not tn["mode_signature"] or tm["mode_signature"]==tn["mode_signature"]:continue
                pairs=[(x,y) for x in tm["bottlenecks"] for y in tn["bottlenecks"] if x["box"]==y["box"] and x["side"]!=y["side"]]
                if not pairs:continue
                x,y=pairs[0];c=trajectory.coeffs[b];d=trajectory.durations[b]
                normal=c.new_tensor(x["normal"]);target=c.new_tensor(x["position"])-c.new_tensor(y["position"])
                target=target-(target*normal).sum()*normal
                if float(target.norm())<1e-8:continue
                delta=at_progress(c[q],d[q],[x["progress"]])[0]-at_progress(c[r],d[r],[y["progress"]])[0]
                separation=(delta*target/target.norm()).sum()
                values.append((fraction*target.norm()-separation).relu().square())
                records.append(dict(batch=b,q=q,r=r,box=x["box"],source="VERIFIED_DISTINCT_AABB_FACES_NOT_HOMOTOPY"))
    zero=(trajectory.coeffs.sum()+trajectory.durations.sum())*0
    return (torch.stack(values).mean() if values else zero),records

class Objective:
    def __init__(self, initial_model, expected_sha, weights):
        require(set(weights)==set(KEYS) and all(math.isfinite(v) and v>=0 for v in weights.values()),"V2_WEIGHTS")
        self.weights=dict(weights)
        self.assignment=PerSampleFairAssignment()
        self.reference=FrozenSafeReference(initial_model,expected_sha=expected_sha)
        self.bank=TeacherBank(dict(recipe(),build_corridor=False))

    def __call__(self, model, samples, geo, pg2_weights, context=None, step=0):
        z=pg2_forward(model,samples,geo,"query_minco_pg2_guided_v2_safe_v1",context,pg2_weights)
        return self.augment(z,samples,step)

    def augment(self,z,samples,step=0):
        cfg=self.bank.cfg; o=z["output"]; tr=o.trajectory; par=o.parameters
        zero=z["raw"].sum()*0
        # Exact value AND gradient equivalence; no new Teacher/reference work.
        if not any(self.weights.values()):
            return dict(z,base=z,terms={k:zero for k in KEYS},weights=self.weights,assignments=[],banks=[],
                        method=METHOD_ID,diagnostics={"disabled":True})
        banks=[self.bank.get(s,f) for s,f in zip(samples,z["fields"])]
        assignments=[]; mode=[]; bneck=[]; section=[]; section_info=[]
        for b,bank in enumerate(banks):
            a=self.assignment.choose(samples[b].sample_id,par.inner_pts[b],par.tail_pva[b,:,0],
                 z["base"]["matches"][b],z["labels"]["passed"][b],bank["teachers"],
                 tr.coeffs[b],par.durations[b],None,step,2)
            assignments.append(a); ml=[]; bl=[]; sl=[]
            for q,m in enumerate(a["route_teacher_id"]):
                if m<0:continue
                t=bank["teachers"][m]
                if q in a["selected"]:
                    ml.append(mode_query(par.inner_pts[b,q],par.tail_pva[b,q],par.durations[b,q],t,cfg)[0])
                    value,info=section_side_proxy(tr.coeffs[b,q],par.durations[b,q],reliable_target(t),bank["scene"])
                    if info["count"]:sl.append(value)
                    section_info.append(dict(batch=b,query=q,teacher=m,**info))
                if t["certified"] and t["representable"] and not bool(z["labels"]["passed"][b,q]):
                    value,info=bottleneck_loss(tr.coeffs[b,q],par.durations[b,q],t["bottlenecks"])
                    if info["count"]:bl.append(value)
            mean=lambda xs:torch.stack(xs).mean() if xs else zero
            mode.append(mean(ml)); bneck.append(mean(bl)); section.append(mean(sl))
        filtered=[dict(b,teachers=[reliable_target(t) for t in b["teachers"]]) for b in banks]
        div,div_info=diversity(tr,assignments,filtered,cfg["diversity_fraction"])
        keep,keep_info=self.reference.loss(z,samples,z["fields"]) if self.weights["safe_keep"] else (zero,{"disabled":True})
        base=z["base"]
        stop=full_stopping_risk(base["positions"],base["velocities"],base["time_weights"],z["fields"]) if self.weights["full_stop"] else zero
        terms=dict(mode=torch.stack(mode).mean(),bottleneck=torch.stack(bneck).mean(),diversity=div,
                   section=torch.stack(section).mean(),safe_keep=keep,full_stop=stop)
        total=z["loss"]+sum(self.weights[k]*terms[k] for k in KEYS if self.weights[k])
        require(bool(torch.isfinite(total)),"V2_TOTAL_NONFINITE")
        return dict(loss=total,base=z,terms=terms,weights=dict(self.weights),raw=z["raw"],output=o,
            matched=z["matched"],labels=z["labels"],goal_labels=z["goal_labels"],assignments=assignments,banks=banks,
            method=METHOD_ID,diagnostics=dict(section=section_info,keep=keep_info,diversity=div_info,
                collapse=query_collapse_diagnostic(tr),convex_corridor=0,teacher_corridor_constructed=False))
