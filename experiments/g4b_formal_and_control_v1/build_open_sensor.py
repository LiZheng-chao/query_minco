"""Compile native sensor with a new empty-floor config;no physics execution."""
from pathlib import Path
import sys,json,os
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,yaml
from query_minco.audit.run_provenance_v1 import run_logged
from query_minco.evaluation.mirror_holdout_v1 import sha,write
R=Path(__file__).resolve().parents[2];O=R/'research_outputs/g4b_formal128_and_control_20261111_v1'
def main():
 b=O/'open_sensor_build';b.mkdir();old=R/'research_outputs/g4a_forest_shadow_20261111_v1/ros_build'
 x,y=np.meshgrid(np.linspace(-10,10,201,dtype=np.float32),np.linspace(-10,10,201,dtype=np.float32))
 pts=np.concatenate([np.stack([x.ravel(),y.ravel(),np.zeros(x.size)],1),np.array([[-10,-10,10],[-10,10,10],[10,-10,10],[10,10,10]])]).astype('<f4')
 cloud=b/'floor_only.ply'
 with cloud.open('xb') as f:f.write(('ply\nformat binary_little_endian 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\nend_header\n'%len(pts)).encode()+pts.tobytes())
 cfg=yaml.safe_load((R.parent/'YOPO/Simulator/src/config/config.yaml').read_text());cfg.update(random_map=False,ply_file=str(cloud),render_lidar=False,render_stereo=False,depth_fps=20,
  odom_topic='/g4b/native/odom',depth_topic='/g4b/native/depth',camera_info_topic='/g4b/native/camera_info')
 config=b/'config.yaml'
 with config.open('x') as f:yaml.safe_dump(cfg,f)
 compile=json.loads((old/'compile/PROCESS_START.json').read_text())['argv']
 compile=[('-DCONFIG_FILE_PATH="'+str(config)+'"') if t.startswith('-DCONFIG_FILE_PATH=') else str(b/'sensor.o') if t==str(old/'native_sensor.o') else t for t in compile]
 link=json.loads((old/'link/PROCESS_START.json').read_text())['argv']
 link=[str(b/'sensor.o') if t==str(old/'native_sensor.o') else str(b/'native_open_sensor') if t==str(old/'native_sensor_shadow') else t for t in link]
 env=dict(os.environ,LD_LIBRARY_PATH='/usr/local/cuda-11.8/lib64:/opt/ros/noetic/lib:'+str(R.parent/'YOPO/Simulator/devel/lib'))
 for name,argv,cwd in [('compile',compile,R),('link',link,R.parent/'YOPO/Simulator/build')]:
  result=run_logged(argv,b/name,cwd=cwd,env=env,timeout_s=180)
  if result['exit_code']:raise RuntimeError(name)
 write(b/'BUILD_IDENTITY.json',dict(binary=str(b/'native_open_sensor'),sha256=sha(b/'native_open_sensor'),config=str(config),config_sha256=sha(config),
  cloud=str(cloud),cloud_sha256=sha(cloud),floor_only=True,no_physics_run=True,source_config_sha256=sha(R.parent/'YOPO/Simulator/src/config/config.yaml')))
 print('OPEN_SENSOR_COMPILE_PASS')
if __name__=='__main__':main()
