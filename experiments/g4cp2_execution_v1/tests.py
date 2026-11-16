from pathlib import Path
import sys,json,unittest,importlib.util,ast,subprocess,os
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from unittest.mock import patch
from query_minco.evaluation.mirror_holdout_v1 import write
R=Path.cwd();O=R/'research_outputs/g4cp2_execution_20261111_v1/audit_tests';O.mkdir()
torch.set_num_threads(1);suite=unittest.TestSuite()
for folder in ['tests/g4cp2_execution_v1','tests/g3_a_mirror_holdout_v1']:
 for p in (R/folder).glob('test*.py'):
  spec=importlib.util.spec_from_file_location('p2_'+p.parent.name+'_'+p.stem,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(m))
with patch.object(torch.optim.Optimizer,'__init__',side_effect=AssertionError('NO_OPTIMIZER')):
 result=unittest.TextTestRunner(verbosity=2).run(suite)
assert result.wasSuccessful()
for folder in ['query_minco/integration/yopo_execution_v4','ros/g4cp2_execution_v4','experiments/g4cp2_execution_v1']:
 for p in (R/folder).rglob('*.py'):ast.parse(p.read_text(),filename=str(p))
cli=[]
for args,expected in [(['--help'],0),(['--stage','C','--output','research_outputs/NEVER_CREATED_P2'],1),(['--stage','D','--output','research_outputs/NEVER_CREATED_P2'],1)]:
 cmd=[sys.executable,'-B','experiments/g4cp2_execution_v1/manual.py']+args;p=subprocess.run(cmd,capture_output=True,text=True);assert p.returncode==expected
 cli.append(dict(argv=cmd,exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr))
assert not (R/'research_outputs/NEVER_CREATED_P2').exists()
write(O/'STATIC_TEST_RESULT.json',dict(status='PASS',tests=result.testsRun,optimizer_steps=0,ros_nodes_launched=0,physics_steps=0,cli=cli))
print('P2_STATIC_QUALIFIED',result.testsRun)
