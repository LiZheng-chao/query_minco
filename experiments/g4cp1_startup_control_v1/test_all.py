"""No ROS master,no native simulator,no optimizer. Real child contracts are labelled."""
from pathlib import Path
import sys,json,unittest,importlib.util,subprocess,ast,math,io,os
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from query_minco.evaluation.mirror_holdout_v1 import write,sha
from query_minco.integration.yopo_startup_v2.readiness import Readiness as Frozen
R=Path.cwd();O=R/'research_outputs/g4cp1_startup_control_20261111_v1/final_tests';O.mkdir()
# Reproduce frozen Python callback-count defect before asserting the new version.
r=Frozen(True)
payload=dict(odom=dict(p=[0,0,2],v=[0,0,0]),imu={},depth={},request=[[0,0,2],[0,0,0],[0,0,0]],raw=dict(force=[0,0,9.6138]),
 plant=dict(phase='HELD_BEFORE_INTEGRATION',integration_steps=0,final_command_stamp=0,command_valid=False,mass=.98,gravity=9.81))
for _ in range(5):
 for k,v in payload.items():r.observe(k,10.,10.,v)
assert r.tick(10.)=='PREPARE_HOLD'
write(O/'FROZEN_V2_PYTHON_FAILURE_EVIDENCE.json',dict(status='FAIL',expected_defect_reproduced=True,mode=r.mode,counts=r.counts,unique_stamps_per_stream=1,source_sha256=sha(R/'query_minco/integration/yopo_startup_v2/readiness.py')))
suite=unittest.TestSuite()
for folder in ['tests/g4cp1_startup_control_v1','tests/g3_a_mirror_holdout_v1']:
 for p in (R/folder).glob('test*.py'):
  spec=importlib.util.spec_from_file_location('p1_'+p.parent.name+'_'+p.stem,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(m))
result=unittest.TextTestRunner(verbosity=2).run(suite);assert result.wasSuccessful()
b=O/'header_tests';b.mkdir()
for name,source in [('old_repro','tests/g4cp1_startup_control_v1/old_gate_repro.cpp'),('new_contract','tests/g4cp1_startup_control_v1/gate_test.cpp')]:
 cmds=[['g++','-std=c++14','-Iquery_minco/integration/yopo_startup_v3',source,'-o',str(b/name)],[str(b/name)]]
 for i,cmd in enumerate(cmds):
  p=subprocess.run(cmd,capture_output=True,text=True);write(b/(name+str(i)+'.json'),dict(argv=cmd,stdout=p.stdout,stderr=p.stderr,exit_code=p.returncode,actual_dynamics=False));assert p.returncode==0
# Installed ROS message serialization is tested without creating publishers or a ROS master.
sys.path[:0]=['/opt/ros/noetic/lib/python3/dist-packages','/home/lizheng/YOPO/Controller/devel/lib/python3/dist-packages']
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu,Image
from quadrotor_msgs.msg import PositionCommand,SO3Command
messages=[]
for cls,frame in [(Odometry,'world'),(Imu,'world'),(Image,'camera_link'),(PositionCommand,'world'),(SO3Command,'/quadrotor')]:
 m=cls();m.header.stamp.secs=100;m.header.stamp.nsecs=123456789;m.header.frame_id=frame
 stream=io.BytesIO();m.serialize(stream);restored=cls();restored.deserialize(stream.getvalue());assert restored.header==m.header
 messages.append(dict(type=m._type,frame=frame,stamp_exact=True))
write(O/'ROS_SERIALIZATION_CONTRACT.json',dict(status='PASS',messages=messages,ros_processes='NOT_RUN'))
# SIGSTOP/SIGKILL contract on a plain sleeper,never a physics process.
from query_minco.integration.yopo_control_supervisor_v1.lease import OwnedProcessStop
proc=subprocess.Popen([sys.executable,'-B','-c','import time;time.sleep(60)'],start_new_session=True)
stop=OwnedProcessStop(proc).stop('SYNTHETIC_OWNER_FAILURE_TEST');assert stop['stop_state_observed'] and stop['terminal_state']=='OWNED_PROCESS_TERMINATED'
write(O/'OWNED_PROCESS_STOP_CONTRACT.json',dict(status='PASS',process='plain Python sleeper,no dynamics',record=stop))
for base in ['query_minco/integration/yopo_startup_v3','experiments/g4cp1_startup_control_v1','ros/g4cp1_startup_v3','tests/g4cp1_startup_control_v1']:
 for p in (R/base).rglob('*.py'):ast.parse(p.read_text(),filename=str(p))
# Analytical motor lag is a source-derived prediction,not simulated physical evidence.
m=.98;g=9.81;kf=8.98132e-9;tau=1/30;rpm=math.sqrt(m*g/(4*kf))
write(O/'MOTOR_STARTUP_ANALYTIC_CHECK.json',dict(status='DESIGN_ONLY',hover_force_N=m*g,hover_rpm_native=rpm,initial_motor_rpm=0.,tau_s=tau,
 initial_vertical_acceleration_mps2=-g,uncompensated_asymptotic_v_drop_mps=-1.5*g*tau,hard_acceleration_limit=8.,startup_violation_risk=True,
 assumptions='constant gravity-compensating motor input,no feedback or drag;does not replace native dynamics test'))
write(O/'STATIC_TEST_RESULT.json',dict(status='PASS',python_tests=result.testsRun,cpp_header_tests='PASS',ros_serialization='PASS',kernel_owner_stop='PASS',
 native_simulator='NOT_RUN',physics_feedback='NOT_RUN',optimizer_updates=0))
cli=[]
base=[sys.executable,'-B','experiments/g4cp1_startup_control_v1/manual_startup.py']
for args,code in [(['--help'],0),(['--output','research_outputs/SHOULD_NOT_CREATE_P1','--allow-held-processes','NO'],1),
 (['--stage','hover','--output','research_outputs/SHOULD_NOT_CREATE_P1','--allow-held-processes','RESEARCHER_MANUAL_HELD_ONLY'],1),
 (['--output','research_outputs/SHOULD_NOT_CREATE_P1','--allow-held-processes','RESEARCHER_MANUAL_HELD_ONLY','--motor-initial-state','equilibrium_hover'],1)]:
 cmd=base+args;p=subprocess.run(cmd,capture_output=True,text=True);assert p.returncode==code
 cli.append(dict(argv=cmd,exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr,ros_started=False))
assert not (R/'research_outputs/SHOULD_NOT_CREATE_P1').exists()
write(O/'CLI_AUTHORIZATION_TESTS.json',dict(status='PASS',records=cli,actual_physics='NOT_RUN'))
print('P1_STATIC_CONTRACTS_PASS',result.testsRun)

