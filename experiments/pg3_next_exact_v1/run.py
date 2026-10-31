"""No formal128/ROS/control dispatch. Authorized scope is disposable diagnostics and qualification."""
import argparse,copy,io,os,shutil,sys,traceback,unittest
from pathlib import Path
import torch
from query_minco.training.io import read,write,artifact,require
from query_minco.training.initialization import configure
from query_minco.training.run_spec import ROOT
from query_minco.inference.identity import state_sha
from query_minco.training.pg3_v2_no_corridor.objective import ObjectiveV21
from query_minco.training.pg3_v2_no_corridor.audit import audit_v21
from query_minco.training.pg3_v2_no_corridor.ablation_v21 import FROZEN
from query_minco.training.pg2_guided_v1.geometry import TeacherGuidance
from experiments.pg3_feasible_mode_v1.common import model_from
from experiments.pg3_v2_no_corridor.qualification_v21 import ReportResult,ref
from .common import DEFAULT,prepare_context,verify_history,runpath

def legacy_evidence(out,context,code):
    folder=out/'REGRESSION_EVIDENCE';folder.mkdir();origin=runpath('A').parent.parent/'_preparation/head002'
    for name in ('GRADIENT_AUDIT_V2_1.json','SHADOW_OPTIMIZER_PROBE.json'):shutil.copyfile(origin/name,folder/name)
    ds,geo,init,q,train,dev=context;m=model_from(init);before=state_sha(m);obj=ObjectiveV21(m,before,FROZEN);ids,_=geo.orders[42].batch(0);s=read(runpath('K')/'RUN_SPEC.json');z=obj(m,ds.load(ids),geo,s['pg2_weights'],TeacherGuidance(),0);audit,_,_=audit_v21(m,z,False);obj.assignment.rollback(0)
    require(before==state_sha(m),'NEXT_LEGACY_CPU0_MUTATION')
    write(folder/'CPU0_V2_1.json',dict(status='PASS' if audit['ratios']['R_D']<=10 else 'FAILED',threshold=10,ratios=audit['ratios'],weights_before=before,weights_after=state_sha(m),optimizer_steps=0,exposure_unchanged=True,code=code,detail=audit,recalibrated=False))
    write(folder/'ORIGIN.json',dict(historical_files={name:ref(origin/name) for name in ('GRADIENT_AUDIT_V2_1.json','SHADOW_OPTIMIZER_PROBE.json')},new_CPU0='actually recomputed frozen V2.1 weights on current source; no new calibration'))

def tests(out):
    os.environ['PG3_NEXT_EVIDENCE']=str(out);os.environ['PG3_V21_EVIDENCE']=str(out/'REGRESSION_EVIDENCE')
    suite=unittest.TestSuite()
    for pattern in ('test_pg2*.py','test_pg3*.py'):suite.addTests(unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern=pattern))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromName('experiments.pg3_feasible_mode_v1.test_fulltrain128'))
    stream=io.StringIO();r=unittest.TextTestRunner(stream=stream,verbosity=2,resultclass=ReportResult).run(suite)
    rows=r.rows;report=dict(total=r.testsRun,passed=sum(x['status']=='PASS' for x in rows.values()),failed=sum(x['status']=='FAIL' for x in rows.values()),not_run=sum(x['status']=='NOT_RUN' for x in rows.values()),tests=rows,log=stream.getvalue(),T01_T12={k.split('.test_',1)[1]:v for k,v in rows.items() if 'ExactContract' in k})
    write(out/'TEST_REPORT.json',report);print('NEXT_TEST_REPORT',report['total'],report['passed'],report['failed'],report['not_run'],flush=True)
    minimal=read(out/'MINIMAL/QUALIFICATION.json');minimal.update(status='QUALIFIED' if r.wasSuccessful() and not report['not_run'] else 'BLOCKED_FINAL_TESTS',tests=ref(out/'TEST_REPORT.json'))
    write(out/'MINIMAL/FINAL_QUALIFICATION.json',minimal)
    return report

def diagnose(out):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','NEXT_CPU_ONLY');configure(dict(seed=42,threads=1,device='cpu'))
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False);verify_history(out);prep,context,code=prepare_context(out);config=read(artifact(prep['configuration']))
    write(out/'RUN_SCOPE.json',dict(scope='PG3_NEXT_EXACT_DIAGNOSTICS_ONLY',code=code,formal_updates=0,controlled_training_method_implemented=False,forest_control=False))
    legacy_evidence(out,context,code)
    from .minimal_qualification import run as qualify
    qualify(out,context,prep,code,config)
    from .counterfactual import run as cf
    cf(out,context,config)
    from .quality_audit import run as quality
    quality(out,context)
    from .readiness import run as ready
    ready(out,context)
    tests(out)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['diagnose','tests']);p.add_argument('--output',default=str(DEFAULT/'evidence001'));a=p.parse_args()
    if a.action=='diagnose':diagnose(a.output)
    else:tests(Path(a.output).resolve())
