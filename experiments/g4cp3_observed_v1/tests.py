from pathlib import Path
import sys,json,unittest,importlib.util,ast,time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from unittest.mock import patch
from query_minco.evaluation.mirror_holdout_v1 import write
R=Path.cwd();O=R/'research_outputs/g4cp3_observed_20261111_v1/final_static_tests_v4';O.mkdir();torch.set_num_threads(1)
suite=unittest.TestSuite()
for folder in ['tests/g4cp3_observed_v1','tests/g4cp2_execution_v1']:
 for p in (R/folder).glob('test*.py'):
  # P2 tests contain a synthetic output writer in the old directory;do not rerun that historical-output test.
  spec=importlib.util.spec_from_file_location('p3_'+p.parent.name+'_'+p.stem,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  for cls in [v for v in vars(m).values() if isinstance(v,type) and issubclass(v,unittest.TestCase)]:
   for name in unittest.defaultTestLoader.getTestCaseNames(cls):
    if name=='test_owner_launch_exception_is_fail':continue
    suite.addTest(cls(name))
with patch.object(torch.optim.Optimizer,'__init__',side_effect=AssertionError('NO_OPTIMIZER')):
 result=unittest.TextTestRunner(verbosity=2).run(suite)
write(O/'STATIC_TEST_RESULT.json',dict(status='STATIC_PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),network_updates=0,native_execution=0))
assert result.wasSuccessful()
print('P3_STATIC_PASS',result.testsRun)
