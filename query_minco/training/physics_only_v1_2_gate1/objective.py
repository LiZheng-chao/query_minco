"""Production-only orchestration of frozen V1.2 math; no synthetic retagging."""
import torch
import torch.nn.functional as functional
from query_minco.trajectory.minco_adapter import MincoAdapter
from query_minco.training.physics_only_v1.decoder import FixedTimeDecoder
from query_minco.training.physics_only_v1_2.spec import PhysicsV12Spec
from query_minco.training.physics_only_v1_2.goal_v12 import finite_tensor,validate_goal_batch,goal_terms,phi2
from query_minco.training.physics_only_v1_2.costs_v12 import sample_weights,smoothness_terms
from query_minco.training.pg1.losses import analytic_samples
from query_minco.training.pg1_v2_r1.losses import weighted_lse
from query_minco.training.physics_only_v1_2.set_objective_v12 import aggregate_terms,V12Result
from .binding import BoundBatch


def collision_math(distance,alpha,spec):
    safe=spec.radius_m+spec.clearance_m
    violation=(safe-distance).clamp_min(0)/safe
    risk=(spec.collision_epsilon_m/safe)*functional.softplus((safe-distance)/spec.collision_epsilon_m)
    return dict(C_viol=(alpha*violation.square()).sum(-1),
                C_shape=weighted_lse(risk,alpha,spec.collision_tau),
                min_sdf_m=distance.amin(-1),
                clearance_violation_peak_m=(safe-distance).clamp_min(0).amax(-1))


def feasibility_math(ana,domains,spec):
    p=ana["p"];alpha=sample_weights(p,ana["weights"]);terms={};peaks={}
    for field,name,limit in (("v","speed",3.),("a","acceleration",8.),("jerk","jerk",100.)):
        x=ana[field];finite_tensor(x,field)
        if x.shape!=p.shape or x.dtype!=p.dtype or x.device!=p.device:raise ValueError("DERIVATIVE_SHAPE")
        norms=x.norm(dim=-1)
        terms["F_"+name]=(alpha*(norms/limit-1).clamp_min(0).square()).sum(-1)
        peaks[name+"_sample_peak"]=norms.amax(-1)
    ds=[];excursions=[]
    for b,domain in enumerate(domains):
        low=(domain[0]-p[b]).clamp_min(0);high=(p[b]-domain[1]).clamp_min(0)
        if spec.domain_mode!="physical_0p15":raise ValueError("PRODUCTION_REQUIRES_FIXED_V12")
        ds.append((alpha[b]*(phi2(low/spec.domain_ref_m)+phi2(high/spec.domain_ref_m)).sum(-1)).sum(-1))
        excursions.append(torch.maximum(low,high).amax((-1,-2)))
    dom=torch.stack(ds);dyn=sum(terms.values())
    return dict(F=dyn+spec.eta_domain*dom,F_dyn=dyn,F_domain=dom,
                domain_sample_peak_m=torch.stack(excursions),**terms,**peaks)


class BoundPhysicsV12Objective:
    def __init__(self,spec=None,*,training_geometry=None,num_candidates=6):
        if training_geometry is not None:
            from .roi_safe_training import RoiSafeTrainingGeometry
            if type(training_geometry) is not RoiSafeTrainingGeometry:raise ValueError("EXPLICIT_TRAINING_GEOMETRY_REQUIRED")
        self.training_geometry=training_geometry
        self.spec=spec or PhysicsV12Spec()
        self.decoder=FixedTimeDecoder(self.spec,num_candidates=num_candidates);self.adapter=MincoAdapter(num_candidates=num_candidates)
        self.num_candidates=num_candidates

    def __call__(self,raw,batch):
        if type(batch) is not BoundBatch:raise ValueError("TYPED_BOUND_BATCH_REQUIRED")
        batch.validate()
        finite_tensor(raw,"raw")
        state=batch.state
        if raw.dtype!=torch.float32 or raw.device.type!="cpu" or raw.shape!=(1,self.num_candidates,21):
            raise ValueError("BOUND_RAW_CPU_FP32_SHAPE")
        enabled=torch.ones(1,dtype=torch.bool);tasks=("normal",)
        validate_goal_batch(state[:,6:9],enabled,tasks)
        parameters=self.decoder(raw,state)
        trajectory=self.adapter(parameters.head_pva,parameters.tail_pva,parameters.inner_pts,parameters.durations)
        ana=analytic_samples(trajectory,self.spec.points_per_segment)
        alpha=sample_weights(ana["p"],ana["weights"])
        if self.training_geometry is None:
            distance=torch.stack([g.sdf(ana["p"][i]) for i,g in enumerate(batch.geometry)])
        else:
            self.training_geometry.records=[self.training_geometry.query(ana["p"][i],g) for i,g in enumerate(batch.geometry)]
            distance=torch.stack([r["training_distance"] for r in self.training_geometry.records])
        terms={**collision_math(distance,alpha,self.spec),
               **feasibility_math(ana,[g.domain for g in batch.geometry],self.spec),
               **smoothness_terms(trajectory,self.spec),
               **goal_terms(ana["endpoint"],ana["terminal_velocity"],state[:,6:9],enabled,tasks)}
        loss,diagnostics=aggregate_terms(terms,self.spec)
        return V12Result(loss,{**terms,**diagnostics},parameters,trajectory)


def collision_direction_probe(batch):
    """Real occupied cell local perturbation; gradient descent must increase clearance."""
    import numpy as np
    g=batch.geometry[0];origin=g.origin
    closest=np.minimum(np.maximum(origin,g.boxes[:,0]),g.boxes[:,1])
    distances=np.linalg.norm(origin-closest,axis=1);i=int(np.argmin(distances))
    n=(origin-closest[i])/distances[i]
    probe=closest[i]+.075*n
    p=torch.tensor((probe-origin)@g.rotation,dtype=torch.float64,requires_grad=True)
    d=g.sdf(p);cost=((.15-d).clamp_min(0)/.15).square()
    gradient=torch.autograd.grad(cost,p)[0]
    if not (0<float(d)<.15 and bool(torch.isfinite(gradient).all()) and float(gradient.norm())>0):
        raise ValueError("INVALID_REAL_COLLISION_PROBE")
    moved=p.detach()-.001*gradient/gradient.norm()
    after=g.sdf(moved)
    if not float(after)>float(d):raise ValueError("COLLISION_GRADIENT_WRONG_DIRECTION")
    return dict(point_B0_m=p.detach().tolist(),nearest_voxel_world_m=g.boxes[i].tolist(),
                sdf_before_m=float(d),sdf_after_m=float(after),step_m=.001,
                C_viol_before=float(cost),gradient_B0=gradient.tolist(),passed=True)
