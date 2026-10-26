"""Trusted task gating from rehashed training source, never scores or labels."""
import copy
import numpy as np
import torch
from query_minco.evaluation.goal import separating_slab_proof
from query_minco.training.io import artifact,read,require,digest
from query_minco.training.datasets.geometry_sidecar import validate_semantics,geometry_digest

def authenticated_task_gates(samples, sidecar):
    fields = sidecar.load(samples)  # Original source/anchor/tensor authentication first.
    require(len(fields)==len(samples),'PG1_GEOMETRY_BATCH')
    rows=[]
    for sample,field in zip(samples,fields):
        row=field.row;validate_semantics(row)
        require(field.boxes.shape==(len(row['obstacles']),2,3) and
                field.boxes.dtype==torch.float32 and not field.boxes.requires_grad and
                bool(torch.isfinite(field.boxes).all()),'PG1_BOX_BINDING')
        source=sidecar._read(row['scene_source'])
        scene=source;parts=row['scene_pointer'].strip('/').split('/')
        try:
            for part in parts: scene=scene[int(part)] if isinstance(scene,list) else scene[part]
            entry=source[int(parts[0])] if isinstance(source,list) else source
        except (IndexError,KeyError,TypeError,ValueError) as exc:
            raise ValueError('PG1_SCENE_POINTER') from exc
        require(scene['obstacles']==row['obstacles'] and scene['flight_center_domain_m']==row['flight_center_domain_m'] and
                scene['frame']==row['frame']=='fixed B0 FLU' and geometry_digest(scene)==row['scene_id'],
                'PG1_SOURCE_GEOMETRY')
        state=sample.state.detach().cpu().numpy()
        require(state.shape==(12,) and state.dtype==np.float32 and np.isfinite(state).all() and
                np.array_equal(state,np.asarray(entry['state'],np.float32)) and
                abs(np.linalg.norm(state[9:12])-1)<1e-4,'PG1_STATE_BINDING')
        require(isinstance(scene.get('kind'),str),'PG1_SCENE_KIND')
        if scene['kind']=='blocked':
            require(sum(b['name']=='separating_slab' for b in scene['obstacles'])==1,'PG1_SLAB_SOURCE')
        proof=separating_slab_proof(copy.deepcopy(scene),state)
        proven=proof is not None and proof['proven_infeasible'] is True
        # Hash binds kind, state and proof, beyond the obstacle/domain-only scene ID.
        rows.append(dict(sample_id=sample.sample_id,goal_enabled=not proven,
            status='PROVEN_INFEASIBLE' if proven else 'UNKNOWN_NOT_PROVEN_INFEASIBLE',
            reason=proof['reason'] if proven else 'No trusted separating-slab proof; normal Near/Far supervision retained',
            proof=proof,task_source_sha256=row['scene_source']['sha256'],
            task_state_scene_digest=digest(dict(state=state.tolist(),scene=scene)),
            certified_empty=row['empty_certified']))
    return fields,rows
