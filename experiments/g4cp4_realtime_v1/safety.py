from pathlib import Path
import sys,unittest,importlib.util,json
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
spec=importlib.util.spec_from_file_location('safety','tests/g4cp4_realtime_v1/test_safety.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(m))
Path('research_outputs/g4cp4_realtime_20261112_v1/SAFETY.json').write_text(json.dumps(dict(tests=r.testsRun,status='STATIC_PASS' if r.wasSuccessful() else 'FAIL')))
assert r.wasSuccessful()
