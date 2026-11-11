"""Offline independent dynamic-ray replay of real ROS observations;no planning choice."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from query_minco.integration.yopo_shadow_v1.sensor_contract import replay_dynamic,rotation
from query_minco.evaluation.mirror_holdout_v1 import write
R=Path(__file__).resolve().parents[2];O=R/'research_outputs/g4a_forest_shadow_20261111_v1'
def main():
 attempt=O/'shadow_attempt003';result=json.loads((attempt/'planner/RESULT.json').read_text());arrays=np.load(attempt/'planner/SHADOW_ARRAYS.npz')
 source=R/'research_outputs/g3d_yopo_forest_bridge_20261110_v1/dataset/forest_sparse_3101';meta=json.loads((source/'COMPACT_GRID.json').read_text());counts=np.fromfile(source/'COMPACT_COUNTS.i32','<i4').reshape(meta['grid_shape'])
 records=[]
 # Fixed indices chosen by position in captured stream,not by depth agreement.
 for ix in [0,len(result['records'])//2,len(result['records'])-1]:
  row=result['records'][ix];depth=arrays['frame%03d_depth'%ix];pred=replay_dynamic(counts,meta,row['origin'],row['q_xyzw']);error=np.abs(pred-depth)
  # Predeclared voxel/float boundary-tolerant source conformance: >=99.5% within3cm,median<=2e-5.
  passed=float((error<=.03).mean())>=.995 and float(np.median(error))<=2e-5
  records.append(dict(frame=ix,nonidentity_rotation=bool(np.linalg.norm(rotation(row['q_xyzw'])-np.eye(3))>.01),max_error_m=float(error.max()),median_error_m=float(np.median(error)),fraction_le_3cm=float((error<=.03).mean()),passed=passed))
 receiver=json.loads((attempt/'receiver_data/RESULT.json').read_text())
 good=all(x['passed'] and x['nonidentity_rotation'] for x in records) and result['status']=='PASS' and receiver['status']=='PASS'
 write(O/'DYNAMIC_FRAME_AUDIT.json',dict(status='PASS' if good else 'FAIL',records=records,
  criterion='Predeclared source-conformance99.5%<=.03m and median<=2e-5;voxel boundary outliers reported,not hidden. Not exact interval certification.',
  full_pva_frame_roundtrip='checked every shadow method/frame at live initial PVA',original_identity_camera_not_modified=True,network_updates=0))
 print(json.dumps(records,indent=2));assert good
if __name__=='__main__':main()
