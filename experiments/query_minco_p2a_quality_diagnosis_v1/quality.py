"""P2-A local experimental scalars/filter; production objective is immutable."""
import torch
from experiments.query_minco_p1a2_falsification_v1 import falsification as f
ARMS=("L0","L1","L2")
CALIBRATION_ALPHA=(1.,2.,4.,8.)
PILOT_CASES=("free_near","native_slit_cross_far_v1","native_two_portals_cross_far_v1")
PILOT_SEEDS=(None,20261109)
def parts(result,alpha=1.):
    m=result.metrics
    V=(m["C_viol"]+m["F"]+m["G_task_train"]).mean()
    Q=(.1*m["C_shape"]+.025*alpha*m["P"]+.1*m["S"]).mean()
    return V,Q
def scalar(result,arm,alpha):
    if arm not in ARMS:raise ValueError("ARM")
    if arm=="L0":return result.loss
    v,q=parts(result,alpha);return v+q
def accept(arm,current_hard,proposed_hard,q_current,q_proposed,unknown_count):
    if proposed_hard and unknown_count:raise ValueError("HARD_UNKNOWN_CONFLICT")
    if not all(torch.isfinite(torch.tensor(x)) for x in [q_current,q_proposed]):raise ValueError("NONFINITE_FILTER")
    if arm!="L2":return True,"ordinary_AdamW"
    if not current_hard:return True,"infeasible_restoration_L1"
    ok=bool(proposed_hard and unknown_count==0 and q_proposed<=q_current)
    return ok,"feasible_quality_accepted" if ok else "rejected_hard_or_quality"
class LoggedAdamReplay:
    """No optimizer construction/step; reconstruct CPU foreach=False WD0 formula."""
    def __init__(self):self.m=torch.zeros(15);self.v=torch.zeros(15);self.t=0
    def predict(self,u,gradient,preclip):
        self.t+=1
        coefficient=(10./(torch.tensor(preclip,dtype=torch.float32)+1e-6)).clamp(max=1.)
        g=gradient.clone()*coefficient
        self.m.lerp_(g,.1);self.v.mul_(.999).addcmul_(g,g,value=1-.999)
        denom=(self.v.sqrt()/(1-.999**self.t)**.5).add_(1e-8)
        predicted=u.clone().mul_(1.)
        predicted.addcdiv_(self.m,denom,value=-.01/(1-.9**self.t))
        return predicted,g
