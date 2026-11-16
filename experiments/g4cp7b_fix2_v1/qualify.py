from pathlib import Path
import sys,unittest,importlib.util,subprocess,json
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
O=Path('research_outputs/g4cp7b_fix2_20261112_v1/tests_v2');O.mkdir()
s=importlib.util.spec_from_file_location('fix2','tests/g4cp7b_fix2_v1/test_fix2.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(m))
for p in Path('ros/g4cp7b_fix2_v1').glob('*.py'):compile(p.read_text(),str(p),'exec')
cmd=[sys.executable,'-B','experiments/g4cp7b_fix2_v1/manual.py','--stage','S','--allow-physics','RESEARCHER_MANUAL_S_ONLY','--output',str(O/'UNUSED_PREFLIGHT'),'--check-only']
p=subprocess.run(cmd,capture_output=True,text=True)
(O/'PREFLIGHT.json').write_text(json.dumps(dict(argv=cmd,exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr),indent=2))
(O/'TEST_RESULT.json').write_text(json.dumps(dict(status='STATIC_PASS' if r.wasSuccessful() and p.returncode==0 else 'FAIL',tests=r.testsRun,physical='NOT_RUN')))
assert r.wasSuccessful() and p.returncode==0
