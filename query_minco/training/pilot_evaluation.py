"""Frozen open-dev evaluation, independent of sampled training loss."""
from types import SimpleNamespace
import numpy as np
import torch
from .io import artifact,read,require,digest
from query_minco.trajectory import TrajectoryPostprocessor
from query_minco.evaluation.validation import validate_teacher,evaluate,sample_curve
from query_minco.evaluation.goal import task_metrics
from query_minco.evaluation.signed_geometry import numpy_box_sdf
from .matching.hungarian import match
from .losses.imitation import cost_matrix

def _certified_empty(scene):
    require(isinstance(scene['obstacles'],list),'DEV_OBSTACLE_LIST')
    empty=not scene['obstacles']
    if empty:
        require(scene.get('certified_empty') is True and scene.get('kind')=='free','DEV_EMPTY_CERTIFICATE')
    else:
        require(scene.get('certified_empty') is not True,'DEV_EMPTY_CONTRADICTION')
    return empty

def qualify_development(binding,train_ids,train_scenes):
    p=artifact(binding);m=read(p);require(m['schema']=='rsps_open_dev8_v1' and len(m['rows'])==8,'DEV_SCHEMA')
    original=read(artifact(m['source']));original={r['id']:r for r in original['rows']}
    paths=[p,artifact(m['source'])]
    require(len({x['id'] for x in m['rows']})==8,'DEV_DUPLICATE')
    for row in m['rows']:
        src=original[row['id']]
        require(row['id'] not in train_ids and src['split']=='development' and src['M']==row['M'] and
            row['scene_sha256'] not in train_scenes,'DEV_TRAIN_LEAKAGE')
        for key,src_key in [('tensors','npz'),('metadata','metadata'),('certificates','checks')]:
            require(row[key]['sha256']==src[src_key+'_sha256'] and
                artifact(row[key]).name==src[src_key].split('/')[-1],'DEV_ORIGINAL_SHA')
            paths.append(artifact(row[key]))
        meta=read(row['metadata']['path'])
        require(meta['id']==row['id'] and meta['split']=='development' and
            digest(dict(obstacles=meta['scene']['obstacles'],domain=meta['scene']['flight_center_domain_m']))==row['scene_sha256'],'DEV_SCENE')
        _certified_empty(meta['scene'])
    return paths

def load_rows(binding):
    m=read(artifact(binding));result=[];empty_provenance_checked=False
    for row in m['rows']:
        meta=read(artifact(row['metadata']))
        if _certified_empty(meta['scene']) and not empty_provenance_checked:
            # Standalone evaluation must also verify the original dev-source anchor.
            # Train leakage remains checked with actual train IDs by pilot_spec.qualify.
            qualify_development(binding,set(),set())
            empty_provenance_checked=True
        with np.load(artifact(row['tensors']),allow_pickle=False) as z:
            fields={k:torch.from_numpy(z[k].copy()) for k in ['depth_input','state','teacher_coeffs','teacher_durations']}
        require(all(x.dtype==torch.float32 and torch.isfinite(x).all() for x in fields.values()),'DEV_ARRAYS')
        result.append((SimpleNamespace(sample_id=row['id'],**fields),meta))
    return result

def evaluate_model(model,binding):
    before=model.training;model.eval();rows=[];device=next(model.parameters()).device
    with torch.no_grad():
        for sample,meta in load_rows(binding):
            raw=model(depth_input=sample.depth_input[None].to(device),state=sample.state[None].to(device))
            out=TrajectoryPostprocessor()(raw,sample.state[None].to(device),num_samples=32)
            # Evaluation is deliberately CPU NumPy with original conservative verifier.
            cost,_=cost_matrix(out,sample);_,mq,_=match(cost,'B_set')
            p=out.parameters;c=out.trajectory.coeffs[0].cpu().numpy();dt=p.durations[0].cpu().numpy()
            goals=[];physics=[];modes=set();collisions=[];false_negatives=[];domain=[];clearances=[];sampled=[];sampled_collisions=[]
            for q in range(6):
                scene=meta['scene']
                check=validate_teacher(c[q],dt[q],p.head_pva[0,q].cpu().numpy(),p.tail_pva[0,q].cpu().numpy(),p.inner_pts[0,q].cpu().numpy(),scene)
                goal=task_metrics(evaluate(c[q,-1],dt[q,-1]),evaluate(c[q,-1],dt[q,-1],1),sample.state.numpy()[6:9])
                # Independent collision/margin certificates; unproven is a conservative failure.
                collision=not all(x['passed'] for x in check['certificates'])
                xyz=np.concatenate([evaluate(a,np.linspace(0,t,64)) for a,t in zip(c[q],dt[q])])
                if _certified_empty(scene):
                    sampled_violation=False;clearances.append(None);sampled_collisions.append(False)
                else:
                    d=np.stack([numpy_box_sdf(xyz,b) for b in scene['obstacles']]).min(0)
                    sampled_violation=bool((d<.15).any());clearances.append(float(d.min()))
                    sampled_collisions.append(bool((d<.12).any()))
                sampled.append(sampled_violation)
                false_negatives.append(not sampled_violation and collision)
                goals.append(goal['passed']);physics.append(check['passed']);collisions.append(collision)
                domain.append('flight_domain' in check['reasons'])
                if check['passed'] and goal['passed']:
                    _,curve=sample_curve(c[q],dt[q],1025);i=int(np.argmin(np.abs(curve[:,0]-scene['crossing_x_m'])))
                    mode='direct'
                    if scene['kind']!='free':
                        y,z=curve[i,1:];mode=('left' if y>=0 else 'right') if abs(y)>=abs(z) else ('up' if z>=0 else 'down')
                    if mode in meta['teacher_modes']:modes.add(mode)
            unmatched=sorted(set(range(6))-set(mq.tolist()))
            rows.append(dict(id=sample.sample_id,certified_empty=_certified_empty(meta['scene']),
                minimum_obstacle_clearance=clearances,sampled_collision=sampled_collisions,sampled_margin_violation=sampled,
                goal=goals,physics=physics,complete_oracle=bool(np.any(np.asarray(goals)&physics)),
                known_feasible_modes=sorted(modes),unmatched_collision=[collisions[q] for q in unmatched],
                conservative_collision=collisions,domain_violation=domain,sampled_false_negatives=false_negatives))
    model.train(before)
    return dict(schema='rsps_open_dev8_result_v1',rows=rows,goal_rate=float(np.mean([x['goal'] for x in rows])),
        physics_rate=float(np.mean([x['physics'] for x in rows])),oracle_rate=float(np.mean([x['complete_oracle'] for x in rows])),
        feasible_mode_count=sum(len(x['known_feasible_modes']) for x in rows),selector='N/A',
        collision_semantics='original Bernstein numerical bound; unproven counts as violation',
        thin_sampling_false_negatives=sum(sum(x['sampled_false_negatives']) for x in rows))

def metric_alarm(history,current,baseline,policy):
    bad=(baseline['goal_rate']-current['goal_rate']>.05 or baseline['oracle_rate']-current['oracle_rate']>.05 or
         (baseline['feasible_mode_count']>0 and current['feasible_mode_count']<.8*baseline['feasible_mode_count']))
    history.append(bool(bad))
    require(not(len(history)>=2 and all(history[-2:])),'DEV_TWO_CHECKPOINT_DEGRADATION')
    return bad
