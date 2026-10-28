"""Finalize verified CPU0 and emit NOT-approved paired specs; never trains."""
import argparse,json
from pathlib import Path
from query_minco.training.io import read,write,sha,digest,require
from query_minco.training.run_spec import verify_code,source_files
from query_minco.training.pg1_v2_r12.spec import R0,RTAIL,make_spec,authorization,cpu0_contract,runtime_identity,validate_cpu0_seal
from query_minco.training.pg1.spec import qualify_bindings
p=argparse.ArgumentParser();p.add_argument('--cpu0',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--run-index',type=int,default=2);a=p.parse_args()
require(1<=a.run_index<100000,'R12_RUN_INDEX')
r=a.cpu0.resolve();out=a.output
result=read(r/'audit/CPU0_RESULT.json');require(result['passed'] and result['optimizer_created'] is False and result['optimizer_steps']==0,'R12_CPU0_RESULT')
require(result.get('schema')=='pg1_v2_r12_cpu0_v2' and result.get('source_bundle_clean') is True and result['code']['allow_uncommitted_sources'] is False,'R12_CLEAN_CPU0_V2_REQUIRED')
require(result.get('contract')==cpu0_contract() and result.get('runtime')==runtime_identity(),'R12_CPU0_CONTRACT_RUNTIME')
verify_code(result['code'])
ds,geo,init,_=qualify_bindings(result['bindings'],42)
rows=[json.loads(l) for l in (r/'runs/BATCHES.jsonl').read_text().splitlines()]
for method in [R0,RTAIL]:
 rr=[v for v in rows if v['method']==method];require(len(rr)==96 and len(set(sid for v in rr for sid in v['ids']))==384,'R12_COMPLETE_RECORDS')
 require(all(v['weights_before']==v['weights_after']==init['weights_sha256'] and v['ratio_alarm_streak']<2 for v in rr),'R12_RECORD_GATES')
 require(all(v['numpy_reconstruction'] and v['raw19_21_zero'] and v['original_loss_and_dynamic_regression'] for v in rr),'R12_RECORD_NUMERICS')
seal=r/'CPU0_SEAL.json'
if not seal.exists():
 files={str(p.relative_to(r)):sha(p) for p in r.rglob('*') if p.is_file()}
 write(seal,dict(schema='pg1_v2_r12_cpu0_seal_v2',passed=True,optimizer_created=False,optimizer_steps=0,
  code=result['code'],bindings=result['bindings'],contract=result['contract'],runtime=result['runtime'],model_weights_before=result['model_weights_before'],model_weights_after=result['model_weights_after'],lambda_tail=.0025,external_sources=result['external_sources'],files=files))
else:require(all(sha(r/n)==h for n,h in read(seal)['files'].items()),'R12_CPU0_RESEAL_REFUSED')
validate_cpu0_seal(dict(code=result['code'],bindings=result['bindings']),seal,init['weights_sha256'])
coverage=read(r/'_preparation/COVERAGE.json')
require(all(coverage['batches'][i]==geo.orders[42].batch(i)[0] for i in range(96)),'R12_ACTUAL_FROZEN_ORDER')
out.mkdir(parents=True,exist_ok=False)
for method,name in [(R0,'R0'),(RTAIL,'Rtail')]:
 s=make_spec(result['bindings'],method,'A42_'+name+'_R12_CPU_Pilot32_run%03d'%a.run_index)
 s['cpu0']=dict(path=str(seal),sha256=sha(seal))
 write(out/(name+'_RUN_SPEC.json'),s);write(out/(name+'_AUTHORIZATION_TEMPLATE.json'),authorization(s))
write(out/'READINESS.json',dict(CPU0_passed=True,training_authorized=False,source_clean=result['source_bundle_clean'],
  status='AWAITING_SEPARATE_APPROVAL' if result['source_bundle_clean'] else 'BLOCKED_SOURCE_FREEZE_AND_CLEAN_CPU0_AND_APPROVAL',
  hardware='CPU_ONLY',optimizer_updates=0,historical_R0='review HISTORICAL_R0.json; default fresh paired R0 for same runtime bundle'))
print('R12_PLANS_WRITTEN_NOT_AUTHORIZED',out)
