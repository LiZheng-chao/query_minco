"""External measured start-zone evidence validator. It never writes FREE voxels."""
import json,math,hashlib
from pathlib import Path
import numpy as np
def validate(path,now,scene_id,world_ball,radius=.30):
 d=json.loads(Path(path).read_text())
 required=['device_id','calibration_sha256','measurement_files','center_world','measured_radius_m','error_bound_m','valid_from','valid_until','scene_id','reviewer','scope']
 if any(k not in d for k in required):raise ValueError('MEASUREMENT_SCHEMA')
 if d['scope']!='INITIAL_BALL_ONLY' or d['scene_id']!=scene_id:raise ValueError('MEASUREMENT_SCOPE')
 if not d['reviewer'] or not d['device_id']:raise ValueError('MEASUREMENT_SOURCE')
 if not d['valid_from']<=now<=d['valid_until'] or not 0<d['valid_until']-d['valid_from']<=60:raise ValueError('MEASUREMENT_EXPIRED')
 if len(d['calibration_sha256'])!=64:raise ValueError('CALIBRATION_ID')
 if not d['measurement_files']:raise ValueError('NO_MEASUREMENT')
 for record in d['measurement_files']:
  if hashlib.sha256(Path(record['path']).read_bytes()).hexdigest()!=record['sha256']:raise ValueError('MEASUREMENT_SHA')
 center=np.asarray(d['center_world'],float);p=np.asarray(world_ball,float)
 values=[d['measured_radius_m'],d['error_bound_m'],radius]
 if center.shape!=(3,) or p.shape!=(3,) or not np.isfinite(center).all() or not np.isfinite(p).all() or not all(math.isfinite(x) for x in values):raise ValueError('MEASUREMENT_FINITE')
 if d['error_bound_m']<0 or radius<.30 or d['measured_radius_m']>1.:raise ValueError('MEASUREMENT_BOUND')
 if np.linalg.norm(p-center)+radius>d['measured_radius_m']-d['error_bound_m']:raise ValueError('OUTSIDE_MEASURED_VOLUME')
 # Hashes authenticate bytes,not truthful physical measurement. Separate reviewer qualification is mandatory.
 return dict(status='SCHEMA_AND_CONTAINMENT_ONLY',motion_authority=False,center=center.tolist(),expires=d['valid_until'])
