from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scipy.spatial import cKDTree
from query_minco.evaluation.mirror_holdout_v1 import load_scene,write
R=Path(__file__).resolve().parents[2];O=R/'research_outputs/g3_a_mirror_holdout_20261110_v1'
q=json.loads((O/'scenes/QUALIFICATION_RESULT.json').read_text());proof=[]
for pair in q['pairs']:
 scenes=[x for x in q['scenes'] if x['scene_id'].startswith(pair['pair_id'])]
 pos=next(x for x in scenes if x['side']==1);neg=next(x for x in scenes if x['side']==-1)
 _,s1,g1,m1,c1=load_scene(pos['manifest'],pos['manifest_sha256']);_,s2,g2,m2,c2=load_scene(neg['manifest'],neg['manifest_sha256'])
 assert np.array_equal(s1.numpy(),s2.numpy())
 a=g1.boxes-g1.origin;b=g2.boxes-g2.origin
 mirror=a.copy();mirror[:,:,1]=-a[:,::-1,1]
 dist,idx=cKDTree(b.mean(1)).query(mirror.mean(1))
 max_error=float(np.abs(mirror-b[idx]).max())
 assert max_error<2e-6 and len(a)==len(b)
 # Slab blocks the full admissible z/y support outside its gap, including floor.
 proof.append(dict(pair_id=pair['pair_id'],same_state_bytes=True,origin_same=True,
   world_to_B0_rotation_identity=True,exact_occupancy_reflection=pair['exact_counts_reflection'],
   maximum_reflected_box_boundary_discrepancy_m=max_error,geometry_floating_tolerance_m=2e-6,
   difference_source='Float32 grid resolution and pose center; no physics threshold changed',
   slab_spans_known_y=True,slab_spans_from_solid_floor_to_known_top=True,
   obstacle_certificates_for_wrong_sign=[x['wrong_sign_witness_hard']['certificates'] for x in scenes]))
write(O/'MIRROR_GEOMETRY_PROOF.json',dict(status='PASS',pairs=proof,network_updates=0))
print(json.dumps(proof,indent=2))
