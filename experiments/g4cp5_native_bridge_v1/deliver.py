from pathlib import Path
import json,hashlib,subprocess,xml.etree.ElementTree as ET
r=Path.cwd();y=Path('/home/lizheng/YOPO');o=r/'research_outputs/g4cp5_native_bridge_20261112_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
e=json.loads((o/'ENTRY.json').read_text());assert not [p for p,h in e['source'].items() if sha(p)!=h]
after={}
for root,original in e['git'].items():
 after[root]={k:subprocess.check_output(['git','-C',root]+a,text=True) for k,a in [('head',['rev-parse','HEAD']),('status',['status','--short']),('diff',['diff']),('cached',['diff','--cached'])]}
 for k in ['head','diff','cached']:assert after[root][k]==original[k]
 if root==str(y):assert after[root]['status']==original['status']
assets={}
for base in [y/'Controller/devel/lib/so3_quadrotor_simulator',y/'Simulator/devel/lib/sensor_simulator',y/'Controller/src',y/'Simulator/src/config']:
 for p in base.rglob('*'):
  if p.is_file() and (p.suffix in ['.yaml','.launch','.msg'] or p.parent.name in ['so3_quadrotor_simulator','sensor_simulator']):assets[str(p)]=sha(p)
for p in (y/'Controller/src').rglob('*.launch'):ET.parse(p)
for p in (r/'ros/g4cp5_native_bridge_v1').glob('*.py'):compile(p.read_text(),str(p),'exec')
(o/'NATIVE_ASSET_SHA.json').write_text(json.dumps(assets,indent=2))
(o/'HISTORICAL_PROTECTION.json').write_text(json.dumps(dict(status='PASS',protected_source_count=len(e['source']),git_after=after,scope='preexisting mainline/YOPO source hashes and Git identity;all historical output/checkpoint rehash NOT_RUN;no writes issued there'),indent=2))
reports={
'YOPO_SIMULATOR_INVENTORY.md':'''# Read-only actual inventory
YOPO HEAD d5e1e60f83cc991602f142d9b6b96204a24d6b9f.
Query HEAD4a2d962f599550d0e48f1b9dbfb127565d2c1594.
Untracked YOPO .catkin_workspace and __pycache__ existed before audit;not removed.
Controller launches:
Controller/src/so3_quadrotor_simulator/launch/simulator_attitude_control.launch (README policy default);
simulator_position_control.launch (native PVA nodelet);
simulator_attitude_control uses network_control_node,READY branch primarily acceleration feedforward plus observer,not generic PVA PID.
Position nodelet consumes p/v/a,yaw/yaw_dot with existing gains and mass0.98. No gains modified.
Simulator/devel/lib/sensor_simulator/sensor_simulator_cuda is built;native dynamics binary is Controller/devel/lib/so3_quadrotor_simulator/quadrotor_simulator_so3.
Binary/source exact build correspondence is UNVERIFIED;asset hashes recorded.

Native physics:Quadrotor.cpp dynamics,quadrotor_simulator_so3.cpp steps at1000Hz default;odom100Hz;motor RPM and SO3 force/orientation consumption in original source.
Sensor config Simulator/src/config/config.yaml:seed3,maze_type5,tree_dist4,resolution0.1,tree.ply,60x60x15,random_map=true.
No generator/dataset collector run;no old map overwritten.
Depth33Hz,/depth_image sensor_msgs/Image 32FC1 160x90 optical-Z meters.
Odom /sim/odom nav_msgs/Odometry world;IMU /sim/imu sensor_msgs/Imu world.
CameraInfo /camera_info sensor_msgs/CameraInfo,K80/80/80/45,distortionzero.
Command /so3_control/pos_cmd quadrotor_msgs/PositionCommand.
Final /so3_cmd quadrotor_msgs/SO3Command ->native Plant ~cmd.
Cloud /mock_map and /lidar_points PointCloud2;cloud is geometry/evaluation only,not route oracle.
Goal /move_base_simple/goal geometry_msgs/PoseStamped.
Depth/CameraInfo stamp derives actual odom stamp. Native IMU lacks stamp assignment:reject zero,do not silently restamp.
Physical renderer uses odom pos and quat*quat_bc;display TF t_bc is not actual renderer translation.
Original policy control timer0.02s=50Hz,not100Hz. Depth callbacks trigger planning;15 candidates,two MINCO pieces,velocity6 config differs from K1.
Native nodelet reuses old command on odom;timeout hover remains TODO. No qualified collision-to-stop/reset service identified in examined native physics/sensor path.
ROS shutdown is not a certified collision response;monitor/owned Plant stop requires integration.
''',
'ORIGINAL_YOPO_CLOSED_LOOP_BASELINE.md':'''# Original baseline
S0 actual forest physical closed-loop:NOT_RUN this round.
README supplies intended dynamics+sensor+policy launch;built binaries and pretrained weight are present by source tree,not evidence of completed physical feedback run.
No native policy inference or physical process was started. Do not claim baseline success from docs,Path or binary existence.
Run original baseline separately before comparing algorithms;retain original6m/s policy semantics rather than calling it equivalent to1m/s Query execution.
Record actual goal,physics odom,depth,PositionCommand,SO3,collision monitoring and process exit.
No exact per-command native Plant ACK exists in unmodified baseline;source wiring plus observed motion would support consumption,but is currently NOT_RUN.
''',
'QUERY_MINCO_ROS_BRIDGE_AUDIT.md':'''# Implemented native bridge
ros/g4cp5_native_bridge_v1/bridge.py consumes actual native Depth/Odom/CameraInfo/IMU/SO3.
Exact depth/pose time binding;finite fresh world state,measured quaternion,physical camera K and frame,32FC1 shape/endian checked.
Uses original observation conversion to2x128x160 and State12,original build/Plan solver/sampler/extrema,three pieces,T2.
Deterministic goal is actual pose+.4 world X;candidate replanning5Hz,typed candidate sampling100Hz.
Real odom velocity finite difference supplies explicitly labeled SHADOW acceleration because native IMU stamp0 cannot pass freshness.
This is not fake odom and not qualified executable acceleration.
Publishes ONLY /query_minco/p5/position_candidate quadrotor_msgs/PositionCommand,never /so3_control/pos_cmd.
No second Simulator/Controller or learned model.
No executable motion flag is exposed:existing owned-process/watchdog/truth utilities cannot silently be assumed to support uninstrumented native Plant.
No online GT read/free corridor. This version does not yet run sensor-space certificate,truth collision monitor or external owned-stop supervisor.
Thus native candidate Bridge is implemented;executable S1 bridge is INCOMPLETE.
''',
'MINCO_PHYSICS_INTEGRATION_TESTS.md':'''# Tests
logs/tests real argv/environment/stdout/stderr/exit0.
5 STATIC_PASS:source-time mismatch,zero IMU stamp rejection,original three-piece MINCO boundary and continuous speed check,real odom derivative,noncontrol publication guard.
ROS Python compile and original launch XML parse PASS.
Native C++ rebuild NOT_RUN;reuse untouched built binaries,source/binary equivalence UNVERIFIED.
ROS message live-type/MD5 interrogation NOT_RUN. Message declarations traced in source and launch.
Actual physics,SO3 consumption,tracking,truth collision,watchdog stop,replanning continuity:NOT_RUN.
No Path or replay counted as physical PASS.
''',
'FOREST_CLOSED_LOOP_RUN_PLAN.md':'''# Gates
S0:manually run original baseline,collect native real physics/control feedback;not claimed already passing.
S1:use existing position-control launch for deterministic PVA;retain original controller/gains/Plant.
Before S1 movement:external owned-process watchdog and independent collision monitor must be wired and physically stop native Plant on failure;source-zero IMU requires qualified actual-state estimator or new external sensor path,not forged stamp.
Bridge currently shadow-only;do not connect candidate topic to controller manually.
P5 engineering birth premise is accepted as an explicit fixed collision-free simulation initial condition. It does NOT require P4 proof to be relabeled PASS,and it does NOT authorize free space along the path.
P4 strict initial observability remains FAIL. Engineering motion also still requires its own verified birth and runtime safety.
Reject subsequent unknown swept region;never expand the birth premise into a GT-filled corridor.
S2 afterS1:5Hz first,>=20 valid replan/switch,>=2m,<=1m/s;record continuous limits and actual collisions/stop.
S3:frozen K1 shadow only until same input/math/safety-selection qualification;no weight updates.
No new network/Anchor experiment.
''',
'MANUAL_SIMULATION_COMMANDS.md':'''# Manual commands,not executed
These are actual original README/code interfaces. S0 starts physical simulation immediately;there is no HELD handshake in this baseline.
Use an isolated researcher-managed run and retain terminal logs. No autonomous goals sent automatically.

S0 terminal1 original baseline:
  cd /home/lizheng/YOPO/Controller
  source devel/setup.bash
  roslaunch so3_quadrotor_simulator simulator_attitude_control.launch
S0 terminal2 sensor (cwd required for relative config/tree assets):
  cd /home/lizheng/YOPO/Simulator
  source devel/setup.bash
  rosrun sensor_simulator sensor_simulator_cuda
S0 terminal3 original planner:
  cd /home/lizheng/YOPO/YOPO
  source /home/lizheng/YOPO/Controller/devel/setup.bash
  /home/lizheng/anaconda3/envs/yopo/bin/python -B test_yopo_ros.py --trial=1 --epoch=50
S0 visualization:
  cd /home/lizheng/YOPO/YOPO
  rviz -d yopo.rviz
Choose goal manually only under researcher monitoring;not a certified1m/s safety setup. ROS native failures must be recorded,not hidden by retries.

S1 NONCONTROLLING native bridge:
With native physics/sensors running and actual /sim/odom,/depth_image,/camera_info:
  cd /home/lizheng/query_minco_mainline
  source /opt/ros/noetic/setup.bash
  source /home/lizheng/YOPO/Controller/devel/setup.bash
  PYTHONDONTWRITEBYTECODE=1 /home/lizheng/anaconda3/envs/yopo/bin/python -B ros/g4cp5_native_bridge_v1/bridge.py --duration 10 --output "research_outputs/p5_native_shadow_$(date +%Y%m%d_%H%M%S)_$$"
Inspect output RESULT/EVENTS. Candidate messages are not flight commands.
Do not remap candidate to /so3_control/pos_cmd.
For future S1 execution the correct existing native controller launch is simulator_position_control.launch;no executable Bridge command released until external stop/truth monitor and state-timing qualification pass.
This is a documented incomplete S1 motion delivery,not a fabricated ready launch.
''',
'FINAL_GO_NO_GO.md':'''# Final
NATIVE_INVENTORY=SOURCE_AUDITED
ORIGINAL_FOREST_CLOSED_LOOP=NOT_RUN
NATIVE_CANDIDATE_BRIDGE=IMPLEMENTED
STATIC_TESTS=PASS5
ROS_NATIVE_STREAM=NOT_RUN
S1_EXECUTION_BRIDGE=INCOMPLETE
NATIVE_WATCHDOG_COLLISION_STOP=NOT_QUALIFIED
S1_MOTION=NO_GO
S2_S3_EXECUTION=NOT_RUN
P4_STRICT_BOOTSTRAP=FAIL_UNCHANGED
P5_FIXED_BIRTH_ENGINEERING_PREMISE=EXPLICITLY_ALLOWED_NOT_YET_RUN
NEW_NETWORK_UPDATES=0
NEW_PHYSICS_RUNS=0
YOPO_SOURCE_PROTECTION=PASS
Use original Simulator/Controller;do not build replacements.
Next code task:wire existing owned-process stop and independent truth observer to actual native process identity,qualify native acceleration timing and publication admission,then researcher-run S1.
The missing executable safety integration is not concealed by shadow output or old V4 PASS.
'''
}
for name,body in reports.items():(o/name).write_text(body)
files=[p for a in e['allowed']+[str(o.relative_to(r))] for p in (r/a).rglob('*') if p.is_file()]
(o/'FILES_AND_SHA.json').write_text(json.dumps({str(p.relative_to(r)):sha(p) for p in files},indent=2))
print('DELIVERY',len(reports),'reports;YOPO unchanged;5 static tests;native motion NOT_RUN')
