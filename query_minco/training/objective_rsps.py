"""Explicit RSPS method bindings and a CPU-only, zero-update validation runner.

This is an extension seam, not a second trainer implementation. It reuses the
reference forward and never imports an optimizer, checkpoint writer or control.
"""
from dataclasses import dataclass
import torch
from query_minco.inference.identity import state_sha
from . import objective
from .io import require, digest, artifact, exact
from .run_spec import source_files, verify_code
from .losses.set_physical_risk import segment_samples, physical_loss, reduce_set

METHODS = {('query_minco_rsps_e%d_cpu_dry_v1' % arm): arm for arm in [1, 2, 3]}


def recipe(training_method_id):
    require(training_method_id in METHODS, 'UNKNOWN_RSPS_METHOD')
    arm = METHODS[training_method_id]
    return dict(schema='rsps_cpu_dry_v1', training_method_id=training_method_id,
                arm=arm, sampler='fixed_segment_trapezoid_v1', points_per_segment=64,
                radius=.12, margin=.03, eps=.05, beta=0. if arm == 1 else 1.,
                v_ref=1., w_max=3., physical_scale=1.,
                matched_strength=1., unmatched_strength=2. if arm == 3 else 1.,
                optimizer_updates=0, device='cpu')


def forward(model, samples, sidecar, training_method_id):
    cfg = recipe(training_method_id)
    require(next(model.parameters()).device.type == 'cpu', 'CPU_DRY_RUN_ONLY')
    fields = sidecar.load(samples)  # Authenticate before invoking the network.
    imitation, matches, raw, output, costs = objective.forward(model, samples, 'B_set', 'cpu')
    p, v, weights = segment_samples(output.trajectory, cfg['points_per_segment'])
    per_query, diagnostics = physical_loss(p, v, weights, [f.boxes for f in fields],
        geometry_bound=[True] * len(fields), radius=cfg['radius'], margin=cfg['margin'],
        eps=cfg['eps'], beta=cfg['beta'], v_ref=cfg['v_ref'], w_max=cfg['w_max'])
    mask = torch.zeros(raw.shape[:2], dtype=torch.bool, device=raw.device)
    for b, record in enumerate(matches):
        mask[b, record['queries']] = True
    physical = reduce_set(per_query, mask, cfg['matched_strength'], cfg['unmatched_strength'])
    return dict(loss=imitation + cfg['physical_scale'] * physical, imitation=imitation,
                physical=physical, per_query=per_query, matches=matches, mask=mask,
                raw=raw, output=output, costs=costs, positions=p, velocities=v,
                time_weights=weights, diagnostics=diagnostics, recipe=cfg)


def cpu_dry_run(model, samples, sidecar, training_method_id, run_spec):
    """Forward/backward once, preserve parameters, return auditable tensors."""
    exact(run_spec, ['schema','training_method_id','seed','batch_index','checkpoint','weights_sha256','sidecar','code','recipe_sha256'], 'RSPS_DRY_SPEC')
    require(run_spec['schema']=='rsps_cpu_zero_update_v1' and run_spec['training_method_id']==training_method_id and
            run_spec['recipe_sha256']==digest(recipe(training_method_id)) and run_spec['sidecar']==sidecar.binding, 'DRY_SPEC_BINDING')
    verify_code(run_spec['code'])
    checkpoint = torch.load(str(artifact(run_spec['checkpoint'])), map_location='cpu', weights_only=True)
    sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    current = model.state_dict()
    require(set(sd)==set(current) and all(torch.equal(sd[k],current[k].cpu()) for k in sd),
            'CHECKPOINT_MODEL_CONTENT')
    require(run_spec['seed'] in sidecar.orders and type(run_spec['batch_index']) is int and
            0<=run_spec['batch_index']<len(sidecar.orders[run_spec['seed']].ids), 'DRY_ORDER_INDEX')
    require([s.sample_id for s in samples]==sidecar.orders[run_spec['seed']].batch(run_spec['batch_index'])[0], 'DRY_ORDER')
    require(state_sha(model)==run_spec['weights_sha256'], 'DRY_INITIALIZATION_SHA')
    require(model.training, 'EXPLICIT_TRAIN_MODE_REQUIRED')
    model.zero_grad(set_to_none=True)
    before = state_sha(model)
    z = forward(model, samples, sidecar, training_method_id)
    raw, output = z['raw'], z['output']
    output.parameters.durations.retain_grad()
    imitation_grad = torch.autograd.grad(z['imitation'], raw, retain_graph=True)[0]
    physical_grad = torch.autograd.grad(z['physical'], raw, retain_graph=True)[0]
    raw.grad = None
    output.parameters.durations.grad = None
    z['loss'].backward()
    require(raw.grad is not None and torch.isfinite(raw.grad).all(), 'RSPS_RAW_GRADIENT')
    require(torch.count_nonzero(raw.grad[..., 19:21]) == 0, 'UNSUPERVISED_RAW_GRADIENT')
    require(output.parameters.durations.grad is not None and
            torch.isfinite(output.parameters.durations.grad).all(), 'MINCO_DURATION_GRADIENT')
    require(torch.count_nonzero(imitation_grad[~z['mask']]) == 0, 'E0_UNMATCHED_GRADIENT')
    require(torch.count_nonzero(physical_grad[...,19:21]) == 0, 'PHYSICAL_UNSUPERVISED_GRADIENT')
    for name, parameter in model.named_parameters():
        require(parameter.grad is not None and torch.isfinite(parameter.grad).all(), 'MODEL_GRADIENT')
        if name.startswith('trajectory_head.') and parameter.shape[0] == 21:
            require(torch.count_nonzero(parameter.grad[19:21]) == 0, 'UNSUPERVISED_HEAD_GRADIENT')
    require(state_sha(model) == before, 'PARAMETER_UPDATE_FORBIDDEN')
    # No unconditional nonzero requirement for free/tied/saturated trajectories.
    arrays = {k: z[k].detach().cpu().numpy() for k in ['raw','positions','velocities','time_weights','per_query','mask']}
    arrays.update(raw_grad=raw.grad.detach().numpy(), imitation_raw_grad=imitation_grad.detach().numpy(),
                  physical_raw_grad=physical_grad.detach().numpy(),
                  coeffs=output.trajectory.coeffs.detach().numpy(),
                  durations=output.parameters.durations.detach().numpy(),
                  duration_grad=output.parameters.durations.grad.detach().numpy())
    arrays.update({'cost_%d' % b: c.detach().numpy() for b,c in enumerate(z['costs'])})
    z['arrays'] = arrays
    z['record'] = dict(schema='rsps_dry_record_v1', recipe=z['recipe'], recipe_sha256=digest(z['recipe']),
        ids=[s.sample_id for s in samples], matches=z['matches'],
        run_spec=run_spec, run_spec_sha256=digest(run_spec),
        loss=float(z['loss'].detach()), imitation=float(z['imitation'].detach()), physical=float(z['physical'].detach()),
        raw_gradient_norms=raw.grad.detach().norm(dim=-1).tolist(),
        physical_gradient_norms=physical_grad.detach().norm(dim=-1).tolist(),
        imitation_gradient_norms=imitation_grad.detach().norm(dim=-1).tolist(),
        parameter_sha_before=before, parameter_sha_after=state_sha(model),
        package_sources=source_files(), optimizer_updates=0)
    return z
