"""Independent NumPy scalar reconstruction for CPU RSPS records. E0 audit is untouched."""
import numpy as np
from types import SimpleNamespace
from query_minco.inference.identity import state_sha
import torch
from scipy.optimize import linear_sum_assignment
from .io import require, digest, artifact
from .objective_rsps import recipe
from .run_spec import source_files
from .losses.imitation import cost_matrix
from query_minco.trajectory import TrajectoryPostprocessor
from .losses.set_physical_risk import segment_samples
from .datasets.geometry_sidecar import validate_semantics


def numpy_physical(p, v, alpha, boxes, cfg):
    """Independent signed AABB distance, analytic normal, risk and quadrature."""
    qloss = []
    alpha = alpha.astype(np.float64)
    alpha = alpha / alpha.sum(-1, keepdims=True)
    for b, field in enumerate(boxes):
        if not len(field):
            qloss.append(np.zeros(6));continue
        field = np.asarray(field, np.float64)
        delta = p[b, ..., None, :] - field.mean(1)
        q = np.abs(delta) - (field[:,1]-field[:,0])/2
        outside = np.maximum(q,0)
        norm = np.linalg.norm(outside,axis=-1)
        dist = norm + np.minimum(q.max(-1),0)
        index = dist.argmin(-1)
        normal = np.sign(delta)*outside/np.maximum(norm[...,None],1e-300)
        inner_axis = q.argmax(-1)
        inner_normal = np.eye(3)[inner_axis]*np.sign(delta)
        normal = np.where((norm==0)[...,None],inner_normal,normal)
        chosen = np.take_along_axis(normal,index[...,None,None],axis=-2)[...,0,:]
        d = np.take_along_axis(dist,index[...,None],axis=-1)[...,0]-cfg['radius']-cfg['margin']
        rate = (chosen*v[b]).sum(-1)
        risk = np.clip(1+cfg['beta']*np.maximum(-rate/cfg['v_ref'],0),1,cfg['w_max'])
        risk = risk/(alpha[b]*risk).sum(-1,keepdims=True)
        qloss.append((alpha[b]*risk*cfg['eps']*np.logaddexp(0,-d/cfg['eps'])).sum(-1))
    return np.stack(qloss)


def audit_record(record, arrays, samples, sidecar, seed, batch_index, expected_method_id, expected_spec):
    require(record['schema']=='rsps_dry_record_v1' and record['optimizer_updates']==0, 'AUDIT_MODE')
    require(record['run_spec']==expected_spec and record['run_spec_sha256']==digest(expected_spec), 'DRY_SPEC_SHA')
    require(expected_spec['training_method_id']==expected_method_id and expected_spec['seed']==seed and expected_spec['batch_index']==batch_index and expected_spec['sidecar']==sidecar.binding, 'EXPECTED_SPEC_BINDING')
    checkpoint=torch.load(str(artifact(expected_spec['checkpoint'])),map_location='cpu',weights_only=True)
    sd=checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    require(state_sha(SimpleNamespace(state_dict=lambda: sd))==expected_spec['weights_sha256'],
            'CHECKPOINT_WEIGHTS_CONTENT')
    require(record['parameter_sha_before']==expected_spec['weights_sha256'], 'INITIALIZATION_BINDING')
    cfg=recipe(expected_method_id)
    require(record['recipe']==cfg and record['recipe_sha256']==digest(cfg), 'METHOD_RECIPE_BINDING')
    require(record['package_sources']==source_files(), 'AUDIT_SOURCE_SHA')
    require(record['parameter_sha_before']==record['parameter_sha_after'], 'MODEL_CHANGED')
    ids,_=sidecar.orders[seed].batch(batch_index)
    require(record['ids']==ids==[s.sample_id for s in samples], 'AUDIT_ORDER')
    fields=sidecar.load(samples)
    require(len(fields)==len(samples),'GEOMETRY_BATCH')
    for field in fields:
        validate_semantics(field.row)
        require(field.boxes.shape==(len(field.row['obstacles']),2,3),'GEOMETRY_BOX_BINDING')
    require(all(np.isfinite(v).all() for v in arrays.values()), 'AUDIT_ARRAY_FINITE')
    raw=torch.from_numpy(arrays['raw'])
    require(raw.dtype==torch.float32 and raw.shape==(len(samples),6,21), 'AUDIT_RAW')
    with torch.no_grad():
        out=TrajectoryPostprocessor()(raw,torch.stack([s.state for s in samples]),num_samples=32)
        p,v,a=segment_samples(out.trajectory,cfg['points_per_segment'])
        for key,actual in [('positions',p),('velocities',v),('time_weights',a),
                           ('coeffs',out.trajectory.coeffs),('durations',out.parameters.durations)]:
            require(np.array_equal(actual.numpy(),arrays[key]), 'RECONSTRUCTED_'+key)
        im=[];mask=np.zeros((len(samples),6),bool)
        for b,sample in enumerate(samples):
            cost,_=cost_matrix(out,sample,b)
            require(np.array_equal(cost.numpy(),arrays['cost_%d'%b]), 'IMITATION_COST_ARRAY')
            q,t=linear_sum_assignment(cost.numpy())
            logged=record['matches'][b]
            require(logged['sample_id']==sample.sample_id and q.tolist()==logged['queries'] and
                    t.tolist()==logged['teachers'], 'HUNGARIAN_RECONSTRUCTION')
            im.append(cost[q,t].mean().item());mask[b,q]=True
    require(np.array_equal(mask,arrays['mask']), 'MATCH_MASK')
    per_query=numpy_physical(arrays['positions'],arrays['velocities'],arrays['time_weights'],
                            [f.boxes.numpy() for f in fields],cfg)
    require(np.allclose(per_query,arrays['per_query'],rtol=2e-5,atol=2e-7), 'PHYSICAL_NUMPY_RECONSTRUCTION')
    weights=np.where(mask,cfg['matched_strength'],cfg['unmatched_strength'])
    phys=(np.sum(per_query*weights,-1)/weights.sum(-1)).mean()
    imitation=np.mean(im);total=imitation+cfg['physical_scale']*phys
    for key,value in [('imitation',imitation),('physical',phys),('loss',total)]:
        require(np.isclose(record[key],value,rtol=2e-5,atol=2e-7), 'LOSS_RECONSTRUCTION_'+key)
    g=arrays['raw_grad'];pg=arrays['physical_raw_grad'];ig=arrays['imitation_raw_grad']
    require(np.allclose(g,ig+cfg['physical_scale']*pg,rtol=1e-4,atol=2e-6), 'GRADIENT_SUM')
    require(not np.count_nonzero(g[...,19:21]) and not np.count_nonzero(pg[...,19:21]) and
            not np.count_nonzero(ig[~mask]), 'GRADIENT_CONTRACT')
    for key,grad in [('raw_gradient_norms',g),('physical_gradient_norms',pg),('imitation_gradient_norms',ig)]:
        require(np.allclose(record[key],np.linalg.norm(grad,axis=-1),rtol=1e-5,atol=1e-8), 'GRADIENT_NORM_LOG')
    require(arrays['duration_grad'].shape==(len(samples),6,3), 'DURATION_GRADIENT_SHAPE')
    pg_norm=np.linalg.norm(pg,axis=-1)
    clear=[];domain_out=[]
    for b,field in enumerate(fields):
        box=field.boxes.numpy()
        if field.row['empty_certified']:
            clear.append(np.full(arrays['positions'].shape[1],np.inf))
        else:
            delta=np.abs(arrays['positions'][b,...,None,:]-box.mean(1))-(box[:,1]-box[:,0])/2
            sdf=np.linalg.norm(np.maximum(delta,0),axis=-1)+np.minimum(delta.max(-1),0)
            clear.append(sdf.min(-1).min(-1)-cfg['radius'])
        domain=np.asarray(field.row['flight_center_domain_m'])
        domain_out.append(((arrays['positions'][b]<domain[0])|(arrays['positions'][b]>domain[1])).any((-1,-2)).tolist())
    clear=np.asarray(clear)
    distinct=[]
    for paths in arrays['positions']:
        accepted=[]
        for path in paths:
            if all(np.sqrt(np.mean((path-other)**2))>.25 for other in accepted):
                accepted.append(path)
        distinct.append(len(accepted))
    cosine=float(np.sum(pg*ig)/max(np.linalg.norm(pg)*np.linalg.norm(ig),1e-30))
    return dict(passed=True,imitation=float(imitation),physical=float(phys),total=float(total),
                matched_physical_raw_grad_norm=float(np.linalg.norm(pg[mask])),
                unmatched_physical_raw_grad_norm=float(np.linalg.norm(pg[~mask])),
                unmatched_nonzero_queries=int(np.count_nonzero(pg_norm[~mask])),
                duration_gradient_max=float(np.abs(arrays['duration_grad']).max()),
                per_query_physical=per_query.tolist(),
                sampled_sphere_collision_fraction=float(np.mean(clear<0)),
                sampled_surface_margin_violation_fraction=float(np.mean(clear<cfg['margin'])),
                minimum_sampled_surface_clearance=[[None if fields[b].row['empty_certified'] else float(x) for x in row] for b,row in enumerate(clear)],
                certified_empty=[f.row['empty_certified'] for f in fields],
                sampled_collision=(clear<0).tolist(),sampled_margin_violation=(clear<cfg['margin']).tolist(),
                sampled_domain_out=domain_out,
                distinct_sampled_paths_rms025=distinct, physical_to_imitation_gradient_ratio=float(np.linalg.norm(pg)/max(np.linalg.norm(ig),1e-30)),
                physical_imitation_gradient_cosine=cosine, stationary_physical_queries=int(np.count_nonzero(pg_norm<=1e-12)),
                )
