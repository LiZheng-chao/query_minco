"""Pure unit/contract tests and header-only plant gate;no simulator is executed."""
from pathlib import Path
import sys,subprocess,unittest,json,os,importlib.util
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
R=Path.cwd();O=R/'research_outputs/g4c_startup_and_dynamics_20261111_v1'
suite=unittest.TestSuite()
for directory in ['tests/g4c_startup_and_dynamics_v1','tests/g3_a_mirror_holdout_v1']:
 for file in Path(directory).glob('test*.py'):
  name='g4c_suite_'+file.parent.name+'_'+file.stem
  spec=importlib.util.spec_from_file_location(name,file);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
res=unittest.TextTestRunner(verbosity=2).run(suite)
assert res.wasSuccessful()
build=O/'header_only_tests_final';build.mkdir()
cmd=['g++','-std=c++14','-Iquery_minco/integration/yopo_startup_v2','tests/g4c_startup_and_dynamics_v1/plant_gate_test.cpp','-o',str(build/'plant_gate_test')]
p=subprocess.run(cmd,capture_output=True,text=True)
(build/'compile.json').write_text(json.dumps(dict(argv=cmd,stdout=p.stdout,stderr=p.stderr,exit_code=p.returncode),indent=2))
assert p.returncode==0
p=subprocess.run([str(build/'plant_gate_test')],capture_output=True,text=True)
(build/'execution.json').write_text(json.dumps(dict(argv=[str(build/'plant_gate_test')],stdout=p.stdout,stderr=p.stderr,exit_code=p.returncode,contains_quadrotor=False),indent=2))
assert p.returncode==0
(O/'UNIT_TEST_RESULT_FINAL.json').write_text(json.dumps(dict(status='PASS',python_tests=res.testsRun,header_gate_executable='PASS',actual_simulator_execution='NOT_RUN',optimizer_steps=0),indent=2))
print('UNIT_AND_HARD_REGRESSION_PASS',res.testsRun)
