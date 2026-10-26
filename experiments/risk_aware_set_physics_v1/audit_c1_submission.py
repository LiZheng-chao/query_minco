"""CPU-only transport audit. NEVER invokes ROS mode or grants authority."""
import dataclasses,hashlib,json,pickle,subprocess
from pathlib import Path
import numpy as np
from query_minco.inference.committed_reference import FrozenPolynomial
from experiments.risk_aware_set_physics_v1.export_native_reference import export
def shifted(p):
 c=np.frombuffer(p.coefficients,dtype='<f8').reshape(-1,3,6).copy();c[:,0,0]+=100.
 return FrozenPolynomial(c.astype('<f8').tobytes(),p.durations)
def main():
 root=Path('/home/lizheng/query_minco_mainline');out=root/'research_outputs/rsps_f3_low_speed_v1/f3_execution006'
 with (out/'shadow_preparation/TRANSPORT_BUNDLES.pkl').open('rb') as f:b=pickle.load(f)
 with (out/'shadow_inputs001/c1/banks_006.pkl').open('rb') as f:q=pickle.load(f)[5]
 valid={0:b[0],1:b[1],101:q};export(valid,out/'runtime/C1_CPU_TRANSPORT_ONLY.bin')
 # Counterexample: same PVA continuity, but geometrically outside the room,
 # with stale hashes/proofs. This is an explicitly invalid input fixture.
 forged=dataclasses.replace(q,prefix=shifted(q.prefix),stops=tuple(dataclasses.replace(s,polynomial=shifted(s.polynomial)) for s in q.stops))
 export({0:b[0],1:b[1],101:forged},out/'runtime/INVALID_GEOMETRY_CPU_FIXTURE.bin')
 exe=out/'runtime/native_reference_shadow';rows=[]
 for file in ('C1_CPU_TRANSPORT_ONLY.bin','INVALID_GEOMETRY_CPU_FIXTURE.bin'):
  r=subprocess.run([str(exe),'--cpu-test',str(out/'runtime'/file)],capture_output=True,text=True,timeout=5)
  rows.append(dict(file=file,command=[str(exe),'--cpu-test',str(out/'runtime'/file)],returncode=r.returncode,stdout=r.stdout,stderr=r.stderr))
 source=(root/'query_minco/inference/native_reference_publisher.cpp').read_text()
 report=dict(mode='CPU_ONLY_NOT_ROS',cases=rows,polynomial_transport_compatible=rows[0]['returncode']==0,
 invalid_geometry_fixture_passes_mechanical_CPU_PVA_checker=rows[1]['returncode']==0,
 inference='CPU transport checks are PVA/terminal-state tests, not geometry, model identity, authenticity or dynamic authorization. Runtime static-file out-of-band seal remains necessary. No dynamic Query subscriber/service exists.',
 implemented_dynamic_C1_submission=False,control_permission=False,ready_for_Query_takeover=False,
 minimum_receiver_contract=['New independent authorization token bound to run/model SHA/worker version/scene/error-budget and exclusive owner epoch',
 'Bounded immutable envelope with source frame/query, measured-state receipt, exact prefix and all-stop digests, inherited evidence expiry in both clocks, activation ROS/mono and parent command version',
 'Off-publication-thread verify all proof digests and scoped continuous full-body evidence, source/causal-state freshness, hard extrema, command PVA continuity, complete stop coverage and remaining lifetime; atomically latch verified version only',
 'Reject duplicate/reordered/late/future/stale/tampered/wrong-model/wrong-epoch proposals; preserve existing committed reference+backup on rejection',
 'Actual activation gate and single-publisher arbitration in existing C++ tick, never run Python/planning/IO in it'])
 (out/'DYNAMIC_SUBMISSION_AUDIT.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
 assert report['polynomial_transport_compatible'] and report['invalid_geometry_fixture_passes_mechanical_CPU_PVA_checker']
if __name__=='__main__':main()
