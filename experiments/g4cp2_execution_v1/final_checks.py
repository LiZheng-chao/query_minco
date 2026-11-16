"""Read-only final syntax/interface/admission checks;no ROS init or physics."""
from pathlib import Path
import sys,ast,json,subprocess,importlib.util,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from query_minco.evaluation.mirror_holdout_v1 import write
R=Path.cwd();O=R/'research_outputs/g4cp2_execution_20261111_v1'
for base in ['query_minco/integration/yopo_execution_v4','ros/g4cp2_execution_v4','experiments/g4cp2_execution_v1','tests/g4cp2_execution_v1']:
 for p in (R/base).rglob('*.py'):ast.parse(p.read_text(),filename=str(p))
spec=importlib.util.spec_from_file_location('final_runtime_tests',R/'tests/g4cp2_execution_v1/test_runtime.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
suite=unittest.TestSuite([m.Runtime('test_owner_launch_exception_is_fail'),m.Runtime('test_owned_stop_failure_has_explicit_fallback')]);result=unittest.TextTestRunner(verbosity=2).run(suite);assert result.wasSuccessful()
sys.path[:0]=['/opt/ros/noetic/lib/python3/dist-packages','/home/lizheng/YOPO/Controller/devel/lib/python3/dist-packages']
import rospy,io,numpy as np
from quadrotor_msgs.msg import PositionCommand
from query_minco.integration.yopo_shadow_v1.command_contract import position_command
msg=position_command(PositionCommand,np.array([[.4,0,2],[0,0,0],[0,0,0]]),rospy.Time.from_sec(100.),0.,1)
s=io.BytesIO();msg.serialize(s);copy=PositionCommand();copy.deserialize(s.getvalue());assert copy.position.x==.4 and copy.header.frame_id=='world'
records=[]
commands=[
 [sys.executable,'-B','experiments/g4cp2_execution_v1/manual.py','--stage','C','--allow-physics','RESEARCHER_MANUAL_C_ONLY','--output','research_outputs/NOT_CREATED_BY_CHECK_ONLY','--check-only'],
 [sys.executable,'-B','experiments/g4cp2_execution_v1/manual.py','--stage','E','--allow-physics','RESEARCHER_MANUAL_E_ONLY','--output','research_outputs/NOT_CREATED_BY_CHECK_ONLY','--check-only']]
for i,cmd in enumerate(commands):
 p=subprocess.run(cmd,capture_output=True,text=True)
 assert p.returncode==(0 if i==0 else 1)
 if i==1:assert 'E_INITIAL_OBSERVED_ROUTE_NOT_QUALIFIED' in p.stderr
 records.append(dict(argv=cmd,stdout=p.stdout,stderr=p.stderr,exit_code=p.returncode,physics_executed=False))
assert not (R/'research_outputs/NOT_CREATED_BY_CHECK_ONLY').exists()
write(O/'FINAL_INTERFACE_CHECKS.json',dict(status='PASS',records=records,installed_ROS_PositionCommand_serialization='PASS',ros_init=False,native_execution=False))
print('FINAL_INTERFACE_PASS_C_PREREQUISITES_E_DENIED_NO_ARM')
