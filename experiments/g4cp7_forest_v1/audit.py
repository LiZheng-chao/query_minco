from pathlib import Path
import sys,json,hashlib,unittest,importlib.util,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from query_minco.integration.yopo_execution_v4.evidence import predecessor
O=Path('research_outputs/g4cp7_forest_integration_20261112_v1');R=Path.cwd()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
a=R/'research_outputs/p6_manual_A_20261112_111259_479/run/RESULT.json';b=R/'research_outputs/p6_manual_B_20261112_111401_479/run/RESULT.json'
d=json.loads(b.read_text());binding=d['source_binding'];assert all(sha(p)==h for p,h in binding.items())
predecessor(a,'C_CAMPAIGN',binding);predecessor(b,'D',binding)
rows=[json.loads(x) for x in (b.parent/'none/supervisor_data/EVENTS.jsonl').read_text().splitlines()]
assert sha(b.parent/'none/supervisor_data/EVENTS.jsonl')==d['events_sha256']
final={x['stamp']:x for x in rows if x.get('kind')=='final_so3'};acks=[]
for x in rows:
 if x.get('kind')=='plant':
  payload=x['payload'];sent=final.get(payload.get('final_command_stamp'))
  if sent:
   assert np.max(np.abs(np.array(sent['payload']['force_world'])-payload['force_world']))<2e-5
   acks.append(x['monotonic']-sent['monotonic'])
summary=dict(status='PHYSICAL_PASS_REVIEW',A_sha256=sha(a),B_sha256=sha(b),motion=d['motion'],runtime=d['runtime_status'],cleanup=d['cleanup_status'],
 exact_final_ACK_matches=len(acks),ack_receipt_latency_s=dict(median=float(np.median(acks)),max=float(max(acks))),maxima=d['full_physical_window']['maxima'],streams=d['healthy_window']['intervals'],
 source_binding_matches=True,scope='existing P6 only;P7 real physics NOT_RUN')
(O/'P6_RECEIPT.json').write_text(json.dumps(summary,indent=2))
spec=importlib.util.spec_from_file_location('pair','tests/g4cp7_forest_v1/test_pairing.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
t=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(mod));assert t.wasSuccessful()
for p in Path('ros/g4cp7_forest_v1').glob('*.py'):compile(p.read_text(),str(p),'exec')
(O/'TEST_RESULT.json').write_text(json.dumps(dict(status='STATIC_PASS',tests=t.testsRun,physical='NOT_RUN'),indent=2));print(json.dumps(summary,indent=2))
