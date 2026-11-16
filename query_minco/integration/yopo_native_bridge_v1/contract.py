"""Native YOPO bridge contracts;no alternative controller or solver."""
import numpy as np
from query_minco.integration.yopo_shadow_v1.sensor_contract import observation
from query_minco.integration.yopo_execution_v4.trajectory import build
def acceleration(previous,current):
 dt=current['stamp']-previous['stamp']
 if not .005<=dt<=.05:raise ValueError('ODOM_DIFFERENCE_TIME')
 a=(np.asarray(current['v'])-previous['v'])/dt
 if not np.isfinite(a).all() or np.linalg.norm(a)>8:raise ValueError('ACCELERATION_UNQUALIFIED')
 return a
def inputs(depth,pose,a,goal,K,stamp,now):
 if stamp!=pose['stamp'] or not 0<=now-stamp<=.2:raise ValueError('EXACT_SOURCE_TIME_REQUIRED')
 return observation(depth,pose['q'],pose['v'],a,pose['p'],goal,K)
def reference(pose,a,goal):
 return build(np.asarray(pose['p']),np.asarray(pose['v']),np.asarray(a),pose['q'],np.asarray(goal),pose['stamp'])
def imu_fresh(stamp,now):
 if stamp<=0 or not 0<=now-stamp<=.2:raise ValueError('IMU_SOURCE_STAMP_INVALID')
