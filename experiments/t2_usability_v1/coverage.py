import json,math,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from query_minco.integration.yopo_shadow_v1.sensor_contract import OPTICAL_TO_BODY
O=Path('research_outputs/t2_usability_20261116_v1')
rows=json.loads((O/'OBSERVABILITY_RECORDS.json').read_text())
R=.6;r=.33;a=math.sqrt(R*R-r*r)+.01;result=[]
for row in rows:
 if row['mode']!='single':continue
 state=row['camera_pose'];rot=Rotation.from_quat(state['q']).as_matrix()
 point=np.asarray(state['p'])+rot@np.array([a,0,r])
 opt=(point-np.asarray(state['p']))@rot@OPTICAL_TO_BODY
 u=80+80*opt[0]/opt[2];v=45+80*opt[1]/opt[2]
 result.append(dict(frame=row['frame'],world_point=point.tolist(),camera_optical=opt.tolist(),pixel=[float(u),float(v)],
     outside_birth=bool(np.linalg.norm(point-np.asarray(row['birth']['center']))>R),outside_current_FOV=not(0<=u<160 and 0<=v<90),query_ball_center_offset=a,query_radius=r))
summary=dict(scope='STRAIGHT_FORWARD_QUERY_TUBE_COUNTEREXAMPLE_NOT_GLOBAL_PHYSICAL_INFEASIBILITY',
 half_vertical_degrees=math.degrees(math.atan(45/80)),
 required_ideal_half_angle_degrees=math.degrees(math.asin(r/R)),
 minimum_ideal_birth_radius_for_this_straight_connection=r/math.sin(math.atan(45/80)),rows=result)
(O/'FOV_COUNTEREXAMPLE.json').write_text(json.dumps(summary,indent=2))
print('FOV',summary['half_vertical_degrees'],'NEEDED',summary['required_ideal_half_angle_degrees'],'RMIN',summary['minimum_ideal_birth_radius_for_this_straight_connection'],'OUTSIDE',sum(x['outside_birth'] and x['outside_current_FOV'] for x in result))
