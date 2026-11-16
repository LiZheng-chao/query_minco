"""Compile native sensor with a new fixed forest config;never execute native simulation."""
from pathlib import Path
import sys,json,os,numpy as np,yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from query_minco.audit.run_provenance_v1 import run_logged
from query_minco.evaluation.mirror_holdout_v1 import sha,write
R=Path.cwd();O=R/'research_outputs/g4cp2_execution_20261111_v1'
def main():
 old=R/'research_outputs/g4b_formal128_and_control_20261111_v1/open_sensor_build';map_folder=R/'research_outputs/g3d_yopo_forest_bridge_20261110_v1/dataset/forest_sparse_3101'
 manifest=json.loads((map_folder/'MAP_MANIFEST.json').read_text())
 for p,h in manifest['files_sha256'].items():
  if sha(p)!=h:raise ValueError('FOREST_ASSET_SHA '+p)
 b=O/'forest_sensor_build';b.mkdir();points=np.fromfile(map_folder/'compact_cloud.f32','<f4').reshape(-1,3);cloud=b/'forest.ply'
 with cloud.open('xb') as f:f.write(('ply\nformat binary_little_endian 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\nend_header\n'%len(points)).encode()+points.tobytes())
 cfg=yaml.safe_load((old/'config.yaml').read_text());cfg.update(random_map=False,ply_file=str(cloud));config=b/'config.yaml'
 with config.open('x') as f:yaml.safe_dump(cfg,f)
 compile=json.loads((old/'compile/PROCESS_START.json').read_text())['argv'];compile=[('-DCONFIG_FILE_PATH="'+str(config)+'"') if t.startswith('-DCONFIG_FILE_PATH=') else str(b/'sensor.o') if t==str(old/'sensor.o') else t for t in compile]
 link=json.loads((old/'link/PROCESS_START.json').read_text())['argv'];link=[str(b/'sensor.o') if t==str(old/'sensor.o') else str(b/'native_forest_sensor') if t==str(old/'native_open_sensor') else t for t in link]
 env=dict(os.environ,LD_LIBRARY_PATH='/usr/local/cuda-11.8/lib64:/opt/ros/noetic/lib:/home/lizheng/YOPO/Simulator/devel/lib')
 for name,argv,cwd in [('compile',compile,R),('link',link,R.parent/'YOPO/Simulator/build')]:
  result=run_logged(argv,b/name,cwd=cwd,env=env,timeout_s=180);assert result['exit_code']==0
 pose=json.loads((map_folder/'pose0/POSE_CERTIFICATE.json').read_text());assert pose['passed']
 plant=R/'research_outputs/g4cp1_startup_control_20261111_v1/build_equilibrium_v3/gated_host'
 build=json.loads((plant.parent/'BUILD_IDENTITY.json').read_text());assert sha(plant)==build['binary_sha256']
 scenes=dict(open=dict(sensor_binary=str(old/'native_open_sensor'),plant_binary=str(plant),spawn=[0.,0.,2.]),
 forest=dict(sensor_binary=str(b/'native_forest_sensor'),plant_binary=str(plant),spawn=pose['position_world'],map_folder=str(map_folder),map_seed=3101,exposure='previously used G3D test map,engineering integration only'))
 write(O/'SCENES.json',scenes)
 write(b/'BUILD_IDENTITY.json',dict(binary_sha256=sha(b/'native_forest_sensor'),cloud_sha256=sha(cloud),compact_source_sha256=sha(map_folder/'compact_cloud.f32'),config_sha256=sha(config),point_count=len(points),native_sensor_executed=False,plant_reused_sha256=sha(plant)))
 print('FIXED_FOREST_SENSOR_COMPILE_PASS_NO_EXECUTION',len(points))
if __name__=='__main__':main()
