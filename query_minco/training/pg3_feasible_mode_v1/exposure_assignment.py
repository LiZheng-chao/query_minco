"""Transactional per-sample history ordering; original slot/Teacher solver is reused."""
import copy
import torch
from query_minco.training.io import require, digest
from .assignment import assign

METHOD_ID = "query_minco_pg3_feasible_mode_v1_mode_exposure_v1"
SCHEMA = "pg3_sample_exposure_history_v1"

def blank():
    return dict(exposures=0, count=[0]*6, eligible=[0]*6,
                wait=[0]*6, longest=[0]*6, last_selected=[-1]*6,
                matched=[0]*6)

class ExposureRotation:
    """choose is read-only on committed state; commit runs after successful update."""
    def __init__(self):
        self.state = {}
        self.last_committed_step = -1
        self.pending = {}
        self.pending_step = None

    def state_dict(self):
        require(not self.pending, "EXPOSURE_PENDING_NOT_CHECKPOINTABLE")
        return copy.deepcopy(dict(schema=SCHEMA, last_committed_step=self.last_committed_step,
                                  samples=self.state))

    def load_state_dict(self, value):
        require(not self.pending and value.get("schema")==SCHEMA, "EXPOSURE_STATE_SCHEMA")
        require(type(value.get("last_committed_step")) is int and value["last_committed_step"]>=-1,
                "EXPOSURE_STATE_STEP")
        for sid, h in value["samples"].items():
            require(isinstance(sid,str) and bool(sid) and set(h)==set(blank()), "EXPOSURE_STATE_FIELDS")
            require(type(h["exposures"]) is int and h["exposures"]>=0, "EXPOSURE_STATE_COUNT")
            for key in ("count","eligible","wait","longest","last_selected","matched"):
                require(len(h[key])==6 and all(type(x) is int and x>=(-1 if key=="last_selected" else 0)
                                              for x in h[key]), "EXPOSURE_STATE_VALUES")
            require(all(c<=e<=h["exposures"] and w<=l<=e for c,e,w,l in
                        zip(h["count"],h["eligible"],h["wait"],h["longest"])), "EXPOSURE_STATE_INVARIANT")
            require(all(x<=value["last_committed_step"] for x in h["last_selected"]), "EXPOSURE_STATE_FUTURE")
        self.state=copy.deepcopy(value["samples"])
        self.last_committed_step=value["last_committed_step"]

    def choose(self,sid,inner,tail,matches,hard_pass,teachers,coeffs,durations,signed,step,capacity):
        require(type(step) is int and step==self.last_committed_step+1, "EXPOSURE_STEP_SEQUENCE")
        require(self.pending_step is None or step==self.pending_step, "EXPOSURE_PENDING_STEP")
        self.pending_step=step
        h=self.state.get(sid,blank())
        # No risk, Teacher distance or development metric affects Query ordering.
        priority=sorted(range(6),key=lambda q:(h["count"][q],(q-h["exposures"])%6))
        out=assign(inner,tail,matches,hard_pass,teachers,step,capacity,fixed_priority=priority)
        matched=out["matched"]
        eligible=[not matched[q] and not bool(hard_pass[q].detach()) for q in range(6)]
        slots=sum(max(0,capacity-sum(int(m)==j for m in matches["teachers"]))
                  for j,t in enumerate(teachers) if t["certified"] and t["representable"])
        fingerprint=digest(dict(matches={k:[int(x) for x in matches[k]] for k in ("queries","teachers")},
            passed=hard_pass.detach().tolist(),inner=inner.detach().tolist(),tail=tail.detach().tolist(),
            teachers=[dict(certified=t["certified"],representable=t["representable"],
                           inner=t["inner"],tail=t["tail"]) for t in teachers]))
        details=[]
        for q in range(6):
            reason=out["reasons"][q]
            if eligible[q] and not slots:
                reason="NO_CERTIFIED_REPRESENTABLE_TEACHER" if not any(
                    t["certified"] and t["representable"] for t in teachers) else "HUNGARIAN_CAPACITY_EXHAUSTED"
            elif reason=="CAPACITY_DEFERRED":
                reason="EXPOSURE_CAPACITY_DEFERRED"
            elif q in out["selected"]:
                reason="EXPOSURE_LEAST_SUPERVISED_SLOT"
            out["reasons"][q]=reason
            details.append(dict(sample_id=sid,query=q,matched=matched[q],
                hard_pass=bool(hard_pass[q].detach()),eligible=eligible[q],selected=q in out["selected"],
                teacher=out["route_teacher_id"][q],count_before=h["count"][q],
                eligible_before=h["eligible"][q],last_selected_before=h["last_selected"][q],
                exposure_before=h["exposures"],wait_before=h["wait"][q],reason=reason,
                priority_rank=priority.index(q),available_extra_slots=slots,
                inner_waypoints=inner[q].detach().tolist(),
                signed_clearance_sampled_m=float(signed[q].detach().min()) if signed is not None else None))
        out["exposure"]=details
        if sid in self.pending:
            require(self.pending[sid]["fingerprint"]==fingerprint and self.pending[sid]["assignment"]==out,
                    "EXPOSURE_DUPLICATE_STATE")
        else:
            self.pending[sid]=dict(fingerprint=fingerprint,assignment=copy.deepcopy(out))
        return out

    def abort(self):
        self.pending={};self.pending_step=None

    def commit(self, step):
        require(type(step) is int and step==self.last_committed_step+1 and
                (self.pending_step is None or self.pending_step==step), "EXPOSURE_COMMIT_STEP")
        for sid, p in self.pending.items():
            row=p["assignment"];h=self.state.setdefault(sid,blank());h["exposures"]+=1
            for q,d in enumerate(row["exposure"]):
                if d["matched"]:h["matched"][q]+=1
                if d["eligible"]:
                    h["eligible"][q]+=1
                    if d["selected"]:
                        h["count"][q]+=1;h["last_selected"][q]=step;h["wait"][q]=0
                    else:h["wait"][q]+=1
                    h["longest"][q]=max(h["longest"][q],h["wait"][q])
                else:h["wait"][q]=0
        self.last_committed_step=step
        self.abort()
        return self.state_dict()
