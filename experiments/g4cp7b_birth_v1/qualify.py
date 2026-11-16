from pathlib import Path
import sys,json,unittest,importlib.util,subprocess
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
O=Path('research_outputs/g4cp7b_birth_execution_20261112_v1/tests_v3');O.mkdir()
spec=importlib.util.spec_from_file_location('t','tests/g4cp7b_birth_v1/test_birth.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(m))
for p in Path('ros/g4cp7b_birth_v1').glob('*.py'):compile(p.read_text(),str(p),'exec')
cmd=[sys.executable,'-B','experiments/g4cp7b_birth_v1/manual.py','--stage','C','--allow-physics','RESEARCHER_MANUAL_C_ONLY','--output',str(O/'UNUSED_CHECK_ONLY'),'--check-only']
p=subprocess.run(cmd,capture_output=True,text=True)
(O/'PREFLIGHT.json').write_text(json.dumps(dict(argv=cmd,exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr),indent=2))
(O/'TEST_RESULT.json').write_text(json.dumps(dict(status='STATIC_PASS' if r.wasSuccessful() and p.returncode==0 else 'FAIL',tests=r.testsRun,physics='NOT_RUN')))
assert r.wasSuccessful() and p.returncode==0
