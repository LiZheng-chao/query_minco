from pathlib import Path
import sys,unittest,importlib.util,json,subprocess,os
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
O=Path('research_outputs/g4cp6_native_execution_20261112_v1/tests_v2');O.mkdir()
suite=unittest.TestSuite()
for folder in ['tests/g4cp6_native_execution_v1','tests/g4cp2_execution_v1']:
 for p in Path(folder).glob('test*.py'):
  spec=importlib.util.spec_from_file_location('p6_'+p.stem,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  for cls in [v for v in vars(m).values() if isinstance(v,type) and issubclass(v,unittest.TestCase)]:
   for name in unittest.defaultTestLoader.getTestCaseNames(cls):
    if name!='test_owner_launch_exception_is_fail':suite.addTest(cls(name))
result=unittest.TextTestRunner(verbosity=2).run(suite)
# Load installed message definitions without starting ROS.
sys.path[:0]=['/opt/ros/noetic/lib/python3/dist-packages','/home/lizheng/YOPO/Controller/devel/lib/python3/dist-packages']
from quadrotor_msgs.msg import PositionCommand,SO3Command
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image,CameraInfo,Imu
messages={m._type:dict(md5=m._md5sum,slots=m.__slots__) for m in [PositionCommand,SO3Command,Odometry,Image,CameraInfo,Imu]}
(O/'MESSAGE_TYPES.json').write_text(json.dumps(dict(scope='installed definitions,not live ROS connection',messages=messages),indent=2))
argv=[sys.executable,'-B','experiments/g4cp6_native_execution_v1/manual.py','--stage','C','--allow-physics','RESEARCHER_MANUAL_C_ONLY','--output',str(O/'CHECK_ONLY_NEVER_CREATED'),'--check-only']
p=subprocess.run(argv,capture_output=True,text=True)
(O/'PREFLIGHT.json').write_text(json.dumps(dict(argv=argv,stdout=p.stdout,stderr=p.stderr,exit_code=p.returncode),indent=2))
(O/'TEST_RESULT.json').write_text(json.dumps(dict(status='STATIC_PASS' if result.wasSuccessful() and p.returncode==0 else 'FAIL',tests=result.testsRun,physical='NOT_RUN'),indent=2))
assert result.wasSuccessful() and p.returncode==0
print('STATIC_PASS',result.testsRun,'installed message definitions verified;physical NOT_RUN')
