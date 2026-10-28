"""Differentiable math prototypes, NOT registered training methods or certificates.

Quality: exact polynomial acceleration/jerk integrals by positive 4-point
Gauss-Legendre quadrature (degree six/four), normalized with FIXED SI references.
Safety: normalized physical-time LSE and weighted upper-tail of softplus risk.
Both safety aggregators remain discrete proxies, never continuous free-space proof.
"""
import math
import torch
import torch.nn.functional as F
from query_minco.training.io import require
from query_minco.training.pg1.losses import finite,polynomial_values
from query_minco.training.losses.set_physical_risk import box_sdf

COMPONENT_ID='pg1_v2_r1_math_prototype_v1'
QUALITY_TERMS=('acceleration','jerk','time')
def _positive(value,name):
    require(type(value) in (float,int) and math.isfinite(value) and value>0,name)
def _trajectory(tr):
    c,d=tr.coeffs,tr.durations
    finite(c,'R1_COEFFS');finite(d,'R1_DURATION')
    require(c.ndim==5 and c.shape[1:]==(6,3,3,6) and len(c)>0 and d.shape==c.shape[:3],'R1_TRAJECTORY_SHAPE')
    require(c.dtype==d.dtype and c.device==d.device,'R1_TRAJECTORY_TYPE')
    require(tr.solve_valid.shape==c.shape[:2] and tr.solve_valid.dtype==torch.bool and bool(tr.solve_valid.all()),'R1_SOLVE_VALID')
    total=d.sum(-1)
    require(bool((d>=.12-1e-6).all()) and bool((total>=.8-1e-6).all()) and bool((total<=2.2+1e-6).all()),'R1_EXISTING_TIME_CONTRACT')
    return c,d,total

def dynamic_quality_loss(trajectory,*,a_ref=8.,j_ref=100.,time_ref=2.2,enabled=QUALITY_TERMS):
    """Returns separate dimensionless [B,6] terms; no combined/final weights.
    acceleration = integral ||a||² dt / (a_ref²*time_ref);
    jerk = integral ||j||² dt / (j_ref²*time_ref); time = T/time_ref.
    Coefficients and local node times/duration weights ALL retain autograd.
    """
    for x,n in [(a_ref,'R1_A_REF'),(j_ref,'R1_J_REF'),(time_ref,'R1_T_REF')]:_positive(x,n)
    require(isinstance(enabled,(tuple,list)) and len(set(enabled))==len(enabled) and set(enabled)<=set(QUALITY_TERMS),'R1_ENABLED')
    c,d,total=_trajectory(trajectory)
    # Nodes and weights on [0,1], exact for polynomials of degree <=7.
    nodes=c.new_tensor([.06943184420297371,.33000947820757187,.6699905217924281,.9305681557970262])
    weights=c.new_tensor([.17392742256872692,.32607257743127307,.32607257743127307,.17392742256872692])
    times=d[...,None]*nodes
    integrals={}
    for name,order in [('acceleration',2),('jerk',3)]:
        v=polynomial_values(c,times,order)
        integrals[name]=(v.square().sum(-1)*d[...,None]*weights).sum((-1,-2))
    unmasked=dict(acceleration=integrals['acceleration']/(a_ref*a_ref*time_ref),
                  jerk=integrals['jerk']/(j_ref*j_ref*time_ref),time=total/time_ref)
    terms={k:v if k in enabled else v*0 for k,v in unmasked.items()}
    require(all(bool(torch.isfinite(x).all()) for x in terms.values()),'R1_QUALITY_FINITE')
    return dict(terms=terms,unmasked_terms=unmasked,integrals_si=integrals,total_time_s=total,
      references=dict(a_ref_mps2=a_ref,j_ref_mps3=j_ref,time_ref_s=time_ref))

def _distribution(values,weights):
    finite(values,'R1_RISK');finite(weights,'R1_TIME_WEIGHTS')
    require(values.ndim==3 and values.shape[1]==6 and len(values)>0 and values.shape[-1]>=2 and weights.shape==values.shape,'R1_RISK_SHAPE')
    require(values.device==weights.device and values.dtype==weights.dtype and bool((weights>=0).all()) and bool((weights.sum(-1)>0).all()),'R1_TIME_MASS')
    return weights/weights.sum(-1,keepdim=True)

def weighted_lse(values,weights,temperature):
    """Temperature has the SAME unit as values. Not a count-weighted top-k."""
    _positive(temperature,'R1_LSE_TEMPERATURE');alpha=_distribution(values,weights)
    tiny=torch.finfo(alpha.dtype).tiny
    logmass=torch.where(alpha>0,alpha.clamp_min(tiny).log(),alpha.new_full((),-float('inf')))
    return temperature*torch.logsumexp(values/temperature+logmass,dim=-1)

def weighted_upper_tail(values,weights,fraction):
    """Exact upper fraction of a discrete physical-time mass distribution.
    Sorting is stable; boundary mass may be partial. Piecewise differentiable;
    at ties/quantile changes the chosen deterministic subgradient is not unique.
    """
    _positive(fraction,'R1_TAIL_FRACTION');require(fraction<=1,'R1_TAIL_FRACTION')
    alpha=_distribution(values,weights)
    ordered,index=torch.sort(values,dim=-1,descending=True,stable=True)
    mass=alpha.gather(-1,index);before=mass.cumsum(-1)-mass
    used=torch.minimum(mass,(fraction-before).clamp_min(0))
    return (used*ordered).sum(-1)/fraction

def bottleneck_safety_loss(positions,time_weights,boxes,*,geometry_bound,certified_empty,
                           temperature,tail_fraction,radius=.12,margin=.03,eps=.05):
    """Separate [B,6] mean/LSE/tail of eps*softplus(-(SDF-r-margin)/eps).
    geometry_bound is a caller assertion, not provenance. Empty geometry ALSO
    needs explicit certified_empty=True; missing/unknown geometry always fails.
    Radius/margin are fixed for R1 comparisons, not tunable safety thresholds.
    """
    finite(positions,'R1_POSITIONS');finite(time_weights,'R1_TIME_WEIGHTS')
    require(positions.ndim==4 and positions.shape[1]==6 and positions.shape[-1]==3 and len(positions)>0,'R1_POSITION_SHAPE')
    alpha=_distribution(positions[...,0]*0,time_weights)
    require(type(radius) in (int,float) and radius==.12 and type(margin) in (int,float) and margin==.03,'R1_FROZEN_SAFETY_ENVELOPE')
    _positive(eps,'R1_SOFTPLUS_EPS');_positive(temperature,'R1_LSE_TEMPERATURE');_positive(tail_fraction,'R1_TAIL_FRACTION');require(tail_fraction<=1,'R1_TAIL_FRACTION')
    require(boxes is not None and geometry_bound is not None and certified_empty is not None and
      len(boxes)==len(geometry_bound)==len(certified_empty)==len(positions),'R1_GEOMETRY_BINDING')
    require(all(type(v) is bool and v for v in geometry_bound) and all(type(v) is bool for v in certified_empty),'R1_GEOMETRY_AUTH')
    rows=[];surface=[]
    for b,field in enumerate(boxes):
        finite(field,'R1_BOXES')
        require(field.ndim==3 and field.shape[1:]==(2,3) and field.device==positions.device and field.dtype==positions.dtype and not field.requires_grad,'R1_BOX_TYPE')
        if not len(field):
            require(certified_empty[b] is True,'R1_EMPTY_NOT_CERTIFIED')
            rows.append(positions[b,...,0]*0);surface.append(None)
        else:
            require(certified_empty[b] is False,'R1_EMPTY_CONTRADICTION')
            sdf=box_sdf(positions[b],field);clearance=sdf-radius
            surface.append(clearance)
            rows.append(eps*F.softplus(-(clearance-margin)/eps))
    risk=torch.stack(rows)
    losses=dict(mean=(alpha*risk).sum(-1),lse=weighted_lse(risk,alpha,temperature),
                tail=weighted_upper_tail(risk,alpha,tail_fraction))
    # Make certified-empty result exactly zero, not floating logsumexp roundoff.
    empty=torch.tensor(certified_empty,dtype=torch.bool,device=positions.device)[:,None]
    for k,v in losses.items():losses[k]=torch.where(empty,positions[...,0].sum(-1)*0,v)
    require(all(bool(torch.isfinite(v).all()) for v in losses.values()),'R1_SAFETY_FINITE')
    return dict(losses=losses,point_risk=risk,normalized_time_mass=alpha,
       surface_clearance=surface,certified_empty=list(certified_empty),
       certificate=False,component_id=COMPONENT_ID)
