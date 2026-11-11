"""Compile native current ROS sensor with an isolated config; no YOPO writes."""
from pathlib import Path
import sys,json,os,shlex
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,yaml
from query_minco.audit.run_provenance_v1 import run_logged
from query_minco.evaluation.mirror_holdout_v1 import sha,write
R=Path(__file__).resolve().parents[2];O=R/'research_outputs/g4a_forest_shadow_20261111_v1';Y=R.parent/'YOPO'
def main():
 b=O/'ros_build';b.mkdir();old=R/'research_outputs/g3d_yopo_forest_bridge_20261110_v1';mp=old/'dataset/forest_sparse_3101'
 pts=np.fromfile(mp/'compact_cloud.f32','<f4').reshape(-1,3);cloud=b/'shadow_cloud.ply'
 with cloud.open('xb') as f:
  f.write(('ply\nformat binary_little_endian 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\nend_header\n'%len(pts)).encode());f.write(pts.tobytes())
 cfg=yaml.safe_load((Y/'Simulator/src/config/config.yaml').read_text())
 cfg.update(random_map=False,ply_file=str(cloud),render_lidar=False,render_stereo=False,depth_fps=10,odom_topic='/g4a/shadow/odom_input',depth_topic='/g4a/shadow/depth',camera_info_topic='/g4a/shadow/camera_info')
 cfgpath=b/'native_config.yaml'
 with cfgpath.open('x') as f:yaml.safe_dump(cfg,f)
 env=dict(os.environ,LD_LIBRARY_PATH='/usr/local/cuda-11.8/lib64:/opt/ros/noetic/lib:'+str(Y/'Simulator/devel/lib'),PYTHONDONTWRITEBYTECODE='1')
 src=Y/'Simulator/src/src/test_simulator_cuda.cpp';obj=b/'native_sensor.o';binary=b/'native_sensor_shadow'
 inc=['-I'+str(Y/'Simulator/src/include'),'-I'+str(Y/'Simulator/src/src'),'-I/usr/include/opencv4','-I/usr/include/pcl-1.10','-I/usr/include/eigen3','-I/opt/ros/noetic/include','-I/usr/local/cuda-11.8/include']
 cmd=['/usr/bin/g++-9','-std=c++17','-O2']+inc+['-DCONFIG_FILE_PATH="'+str(cfgpath)+'"','-c',str(src),'-o',str(obj)]
 p=run_logged(cmd,b/'compile',cwd=R,env=env,runspec=O/'PREREGISTRATION.json',timeout_s=180)
 if p['exit_code']:raise RuntimeError('NATIVE_HOST_COMPILE')
 link=shlex.split((Y/'Simulator/build/CMakeFiles/sensor_simulator_cuda.dir/link.txt').read_text())
 for i,t in enumerate(link):
  if t.endswith('test_simulator_cuda.cpp.o'):link[i]=str(obj)
  elif t.endswith('maps.cpp.o'):link[i]=str(old/'build/source1.o')
  elif t.endswith('perlinnoise.cpp.o'):link[i]=str(old/'build/source2.o')
 link[link.index('-o')+1]=str(binary)
 # Native CMake link file contains a relative -lraycast_cuda; resolve from original build.
 p=run_logged(link,b/'link',cwd=Y/'Simulator/build',env=env,runspec=O/'PREREGISTRATION.json',timeout_s=120)
 if p['exit_code']:raise RuntimeError('NATIVE_HOST_LINK')
 identity=dict(binary=str(binary),sha256=sha(binary),config=str(cfgpath),config_sha256=sha(cfgpath),cloud=str(cloud),cloud_sha256=sha(cloud),
 sources={str(p):sha(p) for p in [src,Y/'Simulator/src/src/maps.cpp',Y/'Simulator/src/src/perlinnoise.cpp',Y/'Simulator/src/src/sensor_simulator.cu',Y/'Simulator/devel/lib/libraycast_cuda.so',old/'build/source1.o',old/'build/source2.o']},
 native_cuda_library='read-only installed binary SHA,not claimed newly recompiled',no_yopo_write=True,control_publishers=False)
 write(b/'BUILD_IDENTITY.json',identity);print('NATIVE_SENSOR_BUILD_PASS',str(binary))
if __name__=='__main__':main()
