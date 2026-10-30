"""Fixed sequential component ablations. Existing loss implementations are untouched."""
import copy
import torch
from query_minco.training.io import require,digest
from query_minco.inference.identity import state_sha
from .executor_v21 import Engine,validate_protocol
from .objective import ObjectiveV21,KEYS_V21
from .revision import UniqueSampleFairAssignment,full_stopping_risk_v21
FROZEN=dict(stop=.000307330688447153,mode=.010859947391365976,geom=.00010354880977059005,keep=.012369432948913293,div=0.)
ACTIVE=dict(S=("stop",),M=("stop","mode"),G=("stop","mode","geom"),K=("stop","mode","geom","keep"))
METHOD={a:"query_minco_pg3_v21_ablation_"+a.lower()+"_v1" for a in ACTIVE}
def weights_for(arm):
    require(arm in ACTIVE,"ABL_UNKNOWN_ARM")
    return {k:FROZEN[k] if k in ACTIVE[arm] else 0. for k in KEYS_V21}
def mapped_spec(spec):
    s=copy.deepcopy(spec);s["arm"]="D";s["execution"]["purpose"]="PG3_V21_SHORT4" if s["execution"]["limit"]==4 else "PG3_V21_PAIRED128"
    return s

def validate_ablation_spec(s):
    require(s["arm"] in ACTIVE and s["method_id"]==METHOD[s["arm"]],"ABL_METHOD")
    require(s["weights"]==weights_for(s["arm"]) and s["ablation"]==dict(active=list(ACTIVE[s["arm"]]),frozen_weights=FROZEN,semantics="sequential_conditional_increment_v1"),"ABL_FROZEN_SWITCHES")
    require(s["execution"]["purpose"]==("PG3_V21_ABLATION_SHORT4" if s["execution"]["limit"]==4 else "PG3_V21_ABLATION128"),"ABL_SCOPE")
    validate_protocol(mapped_spec(s))

def authorize_ablation(s,a):
    validate_ablation_spec(s)
    require(a.get("approved") is True and a.get("spec_sha256")==digest(s) and a.get("scope")==s["execution"]["purpose"]
        and a.get("max_updates")==s["execution"]["limit"] and bool(a.get("authorized_by")),"ABL_AUTHORIZATION")

class NoDirectExposure(UniqueSampleFairAssignment):
    """S commits an optimizer cursor only; never invents eligible/direct supervision."""
    def commit_after_optimizer_step(self,step,*,optimizer_succeeded=False,finite=False,scores_unchanged=False):
        require(optimizer_succeeded is True and finite is True and scores_unchanged is True,"V21_OPTIMIZER_NOT_CONFIRMED")
        require(self.pending_step is None and not self.pending and not self.history and step==self.last_step+1,"ABL_S_EXPOSURE")
        self.last_step=step
        return dict(step=step,direct_supervision=False,batch_occurrences=0,unique_samples=0,decisions={},state_sha256=digest(self.state_dict()))

class AblationObjective(ObjectiveV21):
    def __init__(self,model,expected_sha,arm):
        super().__init__(model,expected_sha,weights_for(arm));self.arm=arm
        if arm=="S":self.assignment=NoDirectExposure()
    def augment(self,z,samples,step=0):
        if self.arm!="S":
            result=super().augment(z,samples,step)
            # Only enabled existing terms enter the graph/audit. Assignment is unchanged for M/G/K.
            zero=(z["raw"].sum()+z["output"].trajectory.coeffs.sum()+z["output"].parameters.durations.sum())*0
            for k in KEYS_V21:
                if k not in ACTIVE[self.arm]:result["terms"][k]=zero
            if "geom" not in ACTIVE[self.arm]:
                result["branch_terms"]={k:zero for k in result["branch_terms"]};result["geometry"]=[]
            if "keep" not in ACTIVE[self.arm]:result["diagnostics"]["keep"]=dict(disabled=True)
            result["loss"]=result["base_loss"]+sum(self.weights[k]*result["terms"][k] for k in KEYS_V21)
        else:
            tr=z["output"].trajectory;p=z["output"].parameters;zero=(z["raw"].sum()+tr.coeffs.sum()+p.durations.sum())*0
            old=z["base"];stop=full_stopping_risk_v21(old["positions"],old["velocities"],old["time_weights"],z["fields"])
            base=z["loss"]-z["weights"]["braking"]*z["terms"]["braking"]
            terms={k:stop if k=="stop" else zero for k in KEYS_V21}
            result=dict(pg2=z,base_loss=base,loss=base+self.weights["stop"]*stop,raw=z["raw"],output=z["output"],matched=z["matched"],labels=z["labels"],goal_labels=z["goal_labels"],weights=dict(self.weights),stop_mode="replace_full_stop",terms=terms,branch_terms={k:zero for k in ("SECTION","BNECK_MATCHED","BNECK_FALLBACK")},assignments=[],banks=[],geometry=[],diagnostics=dict(keep=dict(disabled=True),convex_corridor_constructed=False))
        result.update(method=METHOD[self.arm],ablation_active=list(ACTIVE[self.arm]))
        return result

class AblationEngine(Engine):
    def __init__(self,model,spec,config):
        validate_ablation_spec(spec);super().__init__(model,mapped_spec(spec),config)
        self.spec=copy.deepcopy(spec);self.objective=AblationObjective(model,state_sha(model),spec["arm"])
        self.exposure=self.objective.assignment;self.reference=self.objective.reference
