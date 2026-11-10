"""Independent world polynomial checks; no use of differentiable distance or samples."""
import numpy as np
from query_minco.evaluation.validation import (
    evaluate,norm_max,axis_range,bernstein_controls,certify_piece,numerical_reference)
from query_minco.evaluation.goal import task_metrics

COMPONENT_ID="physics_v12_independent_hard_positive_duration_v2"
LEGACY_SOURCE_SHA256="f55159ffc879b3b1d0062638bcde42690a40719cfcd3c23b34a9f8ec11cbeacf"


def transform_coefficients(coeffs,origin,rotation):
    world=np.einsum("ij,sjk->sik",rotation,np.asarray(coeffs,dtype=np.float64))
    world[:,:,0]+=origin
    return world


def piece_certificate(c,T,geometry):
    bounds=axis_range(c,T)
    # Entire reference sphere must remain in the explicitly known finite region.
    if (bounds[:,0]-geometry.roi[0]<.150001).any() or (geometry.roi[1]-bounds[:,1]<.150001).any():
        return dict(passed=False,reason="UNKNOWN_OR_ROI_MARGIN",axis_ranges_world_m=bounds.tolist())
    controls=bernstein_controls(c,T);low=controls.min(0);high=controls.max(0)
    # Independent conservative broad phase: AABB of Bernstein convex hull.
    gap=np.maximum(np.maximum(geometry.boxes[:,0]-high,low-geometry.boxes[:,1]),0)
    use=np.linalg.norm(gap,axis=1)<=.150002
    boxes=[dict(min=b[0],max=b[1]) for b in geometry.boxes[use]]
    boxes.append(dict(min=geometry.roi[0],max=[geometry.roi[1,0],geometry.roi[1,1],geometry.floor]))
    cert=certify_piece(c,T,boxes)
    cert["broadphase_boxes"]=int(use.sum())
    return cert


def validate_polynomial(coeffs,durations,geometry,goal):
    geometry.validate()
    c=np.asarray(coeffs,dtype=np.float64)
    if c.shape!=(3,3,6) or not np.isfinite(c).all():
        return dict(passed=False,reasons=["INVALID_POLYNOMIAL"])
    try:
        d=np.asarray(durations,dtype=np.float64)
    except (TypeError,ValueError,OverflowError,RuntimeError):
        return dict(passed=False,reasons=["INVALID_DURATION_TYPE"])
    if d.shape!=(c.shape[0],):
        return dict(passed=False,reasons=["INVALID_DURATION_SHAPE"])
    if not np.isfinite(d).all():
        return dict(passed=False,reasons=["INVALID_DURATION_NONFINITE"])
    if (d<=0).any():
        return dict(passed=False,reasons=["NONPOSITIVE_DURATION"])
    world=transform_coefficients(c,geometry.origin,geometry.rotation)
    certificates=[piece_certificate(a,t,geometry) for a,t in zip(world,d)]
    limits=np.array([3.,8.,100.])
    maxima=[max(norm_max(a,t,k) for a,t in zip(c,d)) for k in (1,2,3)]
    ranges=np.stack([axis_range(a,t) for a,t in zip(c,d)])
    domain=geometry.domain.numpy()
    domain_ok=not ((ranges[:,:,0]<domain[0]-2e-4).any() or (ranges[:,:,1]>domain[1]+2e-4).any())
    task=task_metrics(evaluate(c[-1],d[-1]),evaluate(c[-1],d[-1],1),goal)
    reasons=[]
    if not all(x["passed"] for x in certificates):reasons.append("COLLISION_CLEARANCE_OR_UNKNOWN")
    if not domain_ok:reasons.append("CENTER_DOMAIN")
    reasons += [name for value,limit,name in zip(maxima,limits,("SPEED","ACCELERATION","JERK")) if value>limit]
    if not task["passed"]:reasons.append("GOAL_TASK")
    return dict(passed=not reasons,reasons=reasons,certificates=certificates,
                dynamic_polynomial_maxima=maxima,axis_ranges_B0_m=ranges.tolist(),
                domain_passed=domain_ok,goal=task,reference_radius_m=.12,margin_m=.03,
                actual_airframe_certified=False)


def validate_result(result,batch):
    rows=[]
    tr=result.trajectory;p=result.parameters
    for k in range(6):
        c=tr.coeffs[0,k].detach().numpy();d=tr.durations[0,k].detach().numpy()
        row=validate_polynomial(c,d,batch.geometry[0],batch.state[0,6:9].numpy())
        head=p.head_pva[0,k].detach().numpy();tail=p.tail_pva[0,k].detach().numpy()
        inner=p.inner_pts[0,k].detach().numpy()
        ref=numerical_reference(head,tail,inner,d)
        errors=[max(float(np.abs(evaluate(a,np.linspace(0,t,65),order)-evaluate(b,np.linspace(0,t,65),order)).max())
                    for a,b,t in zip(c,ref,d)) for order in (0,1,2)]
        row["independent_minco_max_abs_pva"]=errors
        if any(e>tol for e,tol in zip(errors,(.002,.01,.05))):
            row["passed"]=False;row["reasons"].append("INDEPENDENT_MINCO")
        rows.append(row)
    return rows
