from pathlib import Path
import sys,json,subprocess,unittest,importlib.util
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
O=Path('research_outputs/stop_latency_fix_20261116_v1/tests_v2');O.mkdir()
s=importlib.util.spec_from_file_location('tests_stop','tests/stop_latency_fix_v1/test_stop.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(m))
cmd=[sys.executable,'-B','experiments/stop_latency_fix_v1/manual.py','--stage','S','--allow-physics','RESEARCHER_MANUAL_S_ONLY','--output',str(O/'UNUSED_CHECK_ONLY'),'--check-only']
p=subprocess.run(cmd,capture_output=True,text=True)
(O/'S_PREFLIGHT.json').write_text(json.dumps(dict(argv=cmd,stdout=p.stdout,stderr=p.stderr,exit_code=p.returncode),indent=2))
(O/'TEST_RESULT.json').write_text(json.dumps(dict(status='STATIC_PASS' if r.wasSuccessful() and p.returncode==0 else 'FAIL',tests=r.testsRun,physics='NOT_RUN')))
assert r.wasSuccessful() and p.returncode==0
