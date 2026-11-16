"""Real frozen forest image qualification;GT Hard is offline evaluation only."""
from pathlib import Path
import sys,json,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from query_minco.integration.yopo_execution_v4.trajectory import build
from query_minco.integration.yopo_execution_v4.sensor_space import ObservedSpace
from query_minco.evaluation.forest_adapter_v1 import geometry
from query_minco.training.physics_only_v1_2_gate1.hard_validator import validate_polynomial,COMPONENT_ID
from query_minco.evaluation.mirror_holdout_v1 import sha,write
R=Path.cwd();O=R/'research_outputs/g4cp2_execution_20261111_v1'
folder=R/'research_outputs/g3d_yopo_forest_bridge_20261110_v1/dataset/forest_sparse_3101'
m=json.loads((folder/'MAP_MANIFEST.json').read_text())
for name,h in m['files_sha256'].items():
 if sha(name)!=h:raise ValueError('SOURCE_SHA '+name)
pose=json.loads((folder/'pose0/POSE_CERTIFICATE.json').read_text());meta=json.loads((folder/'COMPACT_GRID.json').read_text());counts=np.fromfile(folder/'COMPACT_COUNTS.i32','<i4').reshape(meta['grid_shape'])
input_path=folder/'pose0/depth_input_2x128x160.f32';inp=np.fromfile(input_path,'<f4').reshape(2,128,160)
depth=np.where(inp[1,19:109]>0,inp[0,19:109]*20,np.float32(20)).astype(np.float32);p=np.array(pose['position_world']);plan=build(p,np.zeros(3),np.zeros(3),[0,0,0,1],p+[.4,0,0],100.)
space=ObservedSpace(depth,dict(p=p,q=[0,0,0,1]),100.);failure=None
try:space.certify(plan)
except ValueError as e:failure=str(e)
# Immutable arrays were populated from SHA-verified bytes;revalidate backing metadata and SHA before Hard evaluation.
def verify():
 for name in ['COMPACT_GRID.json','COMPACT_COUNTS.i32']:
  path=folder/name
  if sha(path)!=m['files_sha256'][str(path)]:raise ValueError('HARD_DATA_SHA')
g=geometry(counts,meta,pose,dict(source='frozenG3D,map3101,offline only'),verify)
hard=validate_polynomial(plan.c,plan.d,g,[.4,0,0])
write(O/'FOREST_SENSOR_ROUTE_QUALIFICATION.json',dict(status='FAIL' if failure else 'PASS',reason=failure,map_seed=3101,depth_source=str(input_path),depth_sha256=sha(input_path),
 offline_GT_initial_clearance_m=pose['clearance_lower_bound_m'],offline_GT_hard=hard,validator=COMPONENT_ID,GT_used_online=False,
 interpretation='UNKNOWN sensor tube/bootstrap qualification failure,not proof of task infeasibility',E_manual_arm_permitted=not bool(failure),planned_maxima=plan.maxima,solver_errors=plan.errors))
np.savez_compressed(O/'STATIC_REFERENCE_CURVE.npz',coefficients=plan.c,durations=plan.d,origin=plan.origin,R=plan.R)
print('FOREST_SENSOR_QUALIFICATION','FAIL' if failure else 'PASS','GT_HARD',hard['passed'],'REASON',failure)
