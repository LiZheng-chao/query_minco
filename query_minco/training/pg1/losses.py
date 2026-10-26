"""PG1 pure analytic proxies; discrete quadrature is not a safety certificate."""
import math
import torch
from query_minco.evaluation.goal_config import CONFIG
from query_minco.training.io import require

def finite(x, name):
    require(isinstance(x, torch.Tensor) and x.is_floating_point() and bool(torch.isfinite(x).all()), name)

def polynomial_values(coefficients, times, derivative=0):
    """Independent Horner derivative: [...,3,6] x [...,N] -> [...,N,3]."""
    finite(coefficients, 'PG1_COEFFICIENTS'); finite(times, 'PG1_TIMES')
    require(type(derivative) is int and 0 <= derivative <= 5, 'PG1_DERIVATIVE')
    require(coefficients.shape[-2:] == (3,6) and times.shape[:-1] == coefficients.shape[:-2], 'PG1_POLY_SHAPE')
    require(times.dtype == coefficients.dtype and times.device == coefficients.device, 'PG1_POLY_TYPE')
    y = torch.zeros_like(times[...,None] * coefficients[...,0,0,None,None])
    for k in range(5, derivative-1, -1):
        y = y * times[...,None] + coefficients[..., :, k].unsqueeze(-2) * (math.factorial(k)/math.factorial(k-derivative))
    finite(y, 'PG1_POLY_OUTPUT')
    return y

def analytic_samples(trajectory, points_per_segment=64):
    """New p/v/a/jerk helper; old segment_samples is never changed or relabelled."""
    c, dt = trajectory.coeffs, trajectory.durations
    finite(c, 'PG1_COEFFICIENTS'); finite(dt, 'PG1_DURATIONS')
    require(c.ndim == 5 and c.shape[1:] == (6,3,3,6) and dt.shape == c.shape[:3], 'PG1_TRAJECTORY_SHAPE')
    require(c.dtype == dt.dtype and c.device == dt.device, 'PG1_TRAJECTORY_TYPE')
    require(trajectory.solve_valid.shape == c.shape[:2] and trajectory.solve_valid.dtype == torch.bool and
            bool(trajectory.solve_valid.all()), 'PG1_INVALID_SOLVE')
    require(type(points_per_segment) is int and points_per_segment >= 2, 'PG1_SAMPLE_COUNT')
    total = dt.sum(-1)
    require(bool((dt >= .12-1e-6).all()) and bool((total >= .8-1e-6).all()) and
            bool((total <= 2.2+1e-6).all()), 'PG1_TIME_CONTRACT')
    u = torch.linspace(0,1,points_per_segment,dtype=c.dtype,device=c.device)
    t = dt[...,None] * u
    values = {name: polynomial_values(c,t,k).flatten(2,3) for k,name in enumerate(('p','v','a','jerk'))}
    trap = torch.ones_like(u); trap[0] = .5; trap[-1] = .5
    values['weights'] = (dt[...,None]*trap/(points_per_segment-1)/total[...,None,None]).flatten(2,3)
    values['endpoint'] = polynomial_values(c[:,:,-1],dt[:,:,-1,None],0).squeeze(-2)
    values['terminal_velocity'] = polynomial_values(c[:,:,-1],dt[:,:,-1,None],1).squeeze(-2)
    values['total_time'] = total
    return values

def goal_task_loss(endpoint, terminal_velocity, goals, goal_enabled=None):
    """Per-query normalized squared hinges, including masked rows in denominator.
    enabled=True means supervised UNKNOWN/normal, not proven reachable.
    Branch selection matches original metrics on the stored state promoted to64.
    """
    for x,n in [(endpoint,'PG1_ENDPOINT'),(terminal_velocity,'PG1_TERMINAL_V'),(goals,'PG1_GOALS')]: finite(x,n)
    require(endpoint.ndim == 3 and endpoint.shape[1:] == (6,3) and terminal_velocity.shape == endpoint.shape and
            goals.shape == (len(endpoint),3), 'PG1_GOAL_SHAPE')
    require(all(x.dtype == endpoint.dtype and x.device == endpoint.device for x in (terminal_velocity,goals)), 'PG1_GOAL_TYPE')
    d = goals.norm(dim=-1)
    require(bool((d>0).all()), 'PG1_ZERO_GOAL')
    unit = goals/d[:,None]
    remaining = (endpoint-goals[:,None]).norm(dim=-1)
    speed = terminal_velocity.norm(dim=-1)
    near = goals.detach().double().norm(dim=-1) <= CONFIG['near_distance_threshold_m']
    near_loss = ((remaining-CONFIG['near_endpoint_tolerance_m']).clamp_min(0)/CONFIG['near_endpoint_tolerance_m']).square()
    near_loss = near_loss + ((speed-CONFIG['near_terminal_speed_max']).clamp_min(0)/CONFIG['near_terminal_speed_max']).square()
    progress = (endpoint*unit[:,None]).sum(-1)
    reduction = d[:,None]-remaining
    lateral = (endpoint-progress[...,None]*unit[:,None]).norm(dim=-1)
    cosine = progress/endpoint.norm(dim=-1).clamp_min(1e-12)
    far_loss = ((CONFIG['far_min_distance_reduction_m']-reduction).clamp_min(0)/CONFIG['far_min_distance_reduction_m']).square()
    far_loss = far_loss + ((CONFIG['far_min_projected_progress_m']-progress).clamp_min(0)/CONFIG['far_min_projected_progress_m']).square()
    far_loss = far_loss + ((CONFIG['far_min_direction_cosine']-cosine).clamp_min(0)/CONFIG['far_min_direction_cosine']).square()
    far_loss = far_loss + ((lateral-CONFIG['far_max_terminal_ray_offset_m']).clamp_min(0)/CONFIG['far_max_terminal_ray_offset_m']).square()
    unmasked = torch.where(near[:,None],near_loss,far_loss)
    if goal_enabled is None: goal_enabled = torch.ones(len(endpoint),dtype=torch.bool,device=endpoint.device)
    require(goal_enabled.shape == (len(endpoint),) and goal_enabled.dtype == torch.bool and goal_enabled.device == endpoint.device, 'PG1_GOAL_MASK')
    value = unmasked*goal_enabled[:,None]
    finite(value, 'PG1_GOAL_LOSS')
    return value, dict(unmasked=unmasked,near=near)

def _samples(x, weights):
    finite(x, 'PG1_SAMPLES'); finite(weights, 'PG1_WEIGHTS')
    require(x.ndim == 4 and x.shape[1] == 6 and x.shape[-1] == 3 and weights.shape == x.shape[:-1], 'PG1_SAMPLE_SHAPE')
    require(x.dtype == weights.dtype and x.device == weights.device and bool((weights>=0).all()) and
            bool((weights.sum(-1)>0).all()), 'PG1_QUADRATURE')
    return weights/weights.sum(-1,keepdim=True)

def dynamic_limit_loss(velocity, acceleration, jerk, weights, limits=(3.,8.,100.)):
    alpha = _samples(velocity,weights)
    require(len(limits)==3 and all(math.isfinite(x) and x>0 for x in limits), 'PG1_LIMITS')
    terms = {}
    for x,limit,name in zip((velocity,acceleration,jerk),limits,('speed','acceleration','jerk')):
        _samples(x,weights);require(x.shape==velocity.shape,'PG1_DYNAMIC_SHAPE')
        terms[name] = (alpha*(x.norm(dim=-1)/limit-1).clamp_min(0).square()).sum(-1)
    return sum(terms.values()), terms

def center_domain_loss(positions, weights, bounds):
    alpha = _samples(positions,weights);finite(bounds,'PG1_DOMAIN')
    require(bounds.shape == (len(positions),2,3) and bounds.dtype == positions.dtype and bounds.device == positions.device and
            not bounds.requires_grad and bool((bounds[:,1]>bounds[:,0]).all()), 'PG1_DOMAIN_CONTRACT')
    lo, hi = bounds[:,0,None,None], bounds[:,1,None,None]
    violation = ((lo-positions).clamp_min(0)/(hi-lo)).square()+((positions-hi).clamp_min(0)/(hi-lo)).square()
    return (alpha*violation.sum(-1)).sum(-1)
