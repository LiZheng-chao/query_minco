"""Versioned scientific split binding over qualified forest and channel source loaders."""
from pathlib import Path
import json
import numpy as np,torch
from query_minco.evaluation.forest_adapter_v1 import ForestAdapter,geometry,verify_files
from query_minco.evaluation.avoidance_families_v1 import load_family_batch
from query_minco.training.physics_only_v1_2_gate1.binding import BoundBatch
from query_minco.evaluation.mirror_holdout_v1 import sha
COMPONENT_ID='k1_forest_and_mirror_scientific_split_binding_v1'
def load(row,required_split):
 if row['split']!=required_split:raise PermissionError('TRAIN_DEV_BOUNDARY')
 if row['kind']=='mirror':return load_family_batch(row['manifest'],row['manifest_sha256'],required_split)
 if row['kind']!='forest':raise ValueError('DATA_KIND')
 # Verify the unchanged generic source adapter's full native source/camera contract first.
 original=ForestAdapter.load_snapshot(row['manifest'],row['manifest_sha256'])
 m=original.manifest if hasattr(original,'manifest') else json.loads(Path(row['manifest']).read_text())
 m=json.loads(Path(row['manifest']).read_text())
 if m['seed']!=row['map_seed'] or m['classification']!='CERTIFIED_FEASIBLE' or m['seed'] in [3101,3102,3201,3202,3301,3302]:raise ValueError('FOREST_ROLE_OR_TEST_LEAKAGE')
 folder=Path(m['map_folder']);pd=Path(m['pose_folder']);meta=json.loads((folder/'COMPACT_GRID.json').read_text());pose=json.loads((pd/'POSE_CERTIFICATE.json').read_text())
 verify=original.validate
 counts=np.fromfile(folder/'COMPACT_COUNTS.i32','<i4').reshape(meta['grid_shape'])
 g=geometry(counts,meta,pose,dict(component=COMPONENT_ID,map_id=m['map_id'],seed=m['seed'],split=required_split),verify)
 # The old generic source loader uses an internal forest_test tag; scientific role is explicitly rebound,never source-hash edited.
 manifest=dict(component=COMPONENT_ID,split=required_split,source_manifest_sha256=row['manifest_sha256'],map_seed=m['seed'])
 b=BoundBatch(original.depth,original.state,(g,),manifest,verify);b._depth=original.depth.numpy().tobytes();b._state=original.state.numpy().tobytes();b.validate()
 return b
