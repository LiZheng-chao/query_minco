from pathlib import Path
import sys,json,unittest,importlib.util,subprocess,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
O=Path('research_outputs/g4cp7b_fix_20261112_v1/tests_v2');O.mkdir()
s=importlib.util.spec_from_file_location('gridtests','tests/g4cp7b_fix_v1/test_grid.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(m))
for p in Path('ros/g4cp7b_fix_v1').glob('*.py'):compile(p.read_text(),str(p),'exec')
cmd=[sys.executable,'-B','experiments/g4cp7b_fix_v1/manual.py','--stage','S','--allow-physics','RESEARCHER_MANUAL_S_ONLY','--output',str(O/'UNUSED_CHECK'),'--check-only']
p=subprocess.run(cmd,capture_output=True,text=True)
(O/'PREFLIGHT.json').write_text(json.dumps(dict(argv=cmd,exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr),indent=2))
raw=Path('research_outputs/p7b_native_safety_20261112_122048_479/run/request_timeout')
rows=[json.loads(x) for x in (raw/'supervisor_data/EVENTS.jsonl').read_text().splitlines()]
ds=[x for x in rows if x.get('kind')=='depth' and x.get('accepted')]
record=dict(accepted_depth=len(ds),source_gaps_s=np.diff([x['stamp'] for x in ds]).tolist(),receipt_age_s=[x['ros_now']-x['stamp'] for x in ds],last_depth=ds[-1],fault=[x for x in rows if x.get('kind')=='fault'],source_render_and_transport_split='UNVERIFIED_NO_SOURCE_TIMING_IN_OLD_RUN',new_real_sensor_run='NOT_RUN')
(O/'FAILURE_TIMELINE.json').write_text(json.dumps(record,indent=2))
(O/'TEST_RESULT.json').write_text(json.dumps(dict(status='STATIC_PASS' if r.wasSuccessful() and p.returncode==0 else 'FAIL',tests=r.testsRun,physics='NOT_RUN')))
assert r.wasSuccessful() and p.returncode==0
