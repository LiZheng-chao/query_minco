"""Commit-bound qualification and executor short tests; never invokes formal128."""
import argparse,copy,io,os,unittest
from pathlib import Path
from query_minco.training.io import read,write,artifact,digest,require
from query_minco.training.run_spec import ROOT
from experiments.pg3_next_exact_v1.common import prepare_context,verify_history,ref,snapshot
from experiments.pg3_next_exact_v1.run import legacy_evidence
from experiments.pg3_next_exact_v1.minimal_qualification import run as minimal_qualify
from experiments.pg3_v2_no_corridor.qualification_v21 import ReportResult
from .executor import run,compare,SCOPE,validate

def prepare(out):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False);verify_history(out)
    prep,context,code=prepare_context(out);legacy_evidence(out,context,code)
    minimal_qualify(out,context,prep,code,read(artifact(prep['configuration'])))
    (out/'RUN_SPECS').mkdir();formal={}
    for arm in ('A','MINIMAL'):
        s=read(out/'MINIMAL/RUN_SPECS'/(arm+'_FORMAL_PLAN.json'))
        s.update(schema='pg3_minimal_pair_executor_v1',formal_scope=SCOPE,minimal_qualification=str(out/'FINAL_QUALIFICATION.json'),output_root=str(out/'runs'),run_id='A42_'+arm+'_MINPAIR128_run001',cpu0=ref(out/'MINIMAL/CPU0.json'))
        s.pop('next_exact_qualification',None);validate(s);formal[arm]=s;write(out/'RUN_SPECS'/(arm+'.json'),s)
        write(out/'RUN_SPECS'/(arm+'_AUTHORIZATION_PENDING.json'),dict(approved=False,spec_sha256=digest(s),scope=SCOPE,max_updates=128,resources=s['execution'],authorized_by=''))
        for suffix in ('u','r'):
            ss=copy.deepcopy(s);ss.update(run_id='A42_'+arm+'_'+suffix+'_run001',output_root=str(out/'short'));ss['execution'].update(limit=4,purpose='PG3_V21_SHORT4')
            sp=out/'RUN_SPECS'/(arm+'_'+suffix+'.json');ap=out/'RUN_SPECS'/(arm+'_'+suffix+'_AUTH.json');write(sp,ss)
            write(ap,dict(approved=True,spec_sha256=digest(ss),scope='PG3_MINIMAL_SHORT4_V1',max_updates=4,resources=ss['execution'],authorized_by='user request: short exact restore diagnostics only'))
            if suffix=='u':run(sp,ap)
            else:run(sp,ap,pause_after=2);run(sp,ap,resume=True)
    compare(out)
    for k in ('initialization','bindings','order','configuration','pg2_weights','seed','threads','batch_size','updates','optimizer','schedule','precision','device'):
        require(formal['A'][k]==formal['MINIMAL'][k],'MINPAIR_PAIR:'+k)
    os.environ['PG3_NEXT_EVIDENCE']='/home/lizheng/query_minco_mainline/research_outputs/pg3_next_exact_v1/evidence003'
    os.environ['PG3_V21_EVIDENCE']=str(out/'REGRESSION_EVIDENCE');os.environ['PG3_MINIMAL_PAIR_EVIDENCE']=str(out)
    suite=unittest.TestSuite()
    for pattern in ('test_pg2*.py','test_pg3*.py'):suite.addTests(unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern=pattern))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromName('experiments.pg3_feasible_mode_v1.test_fulltrain128'))
    stream=io.StringIO();r=unittest.TextTestRunner(stream=stream,verbosity=2,resultclass=ReportResult).run(suite)
    rows=r.rows;t=dict(total=r.testsRun,passed=sum(x['status']=='PASS' for x in rows.values()),failed=sum(x['status']=='FAIL' for x in rows.values()),not_run=sum(x['status']=='NOT_RUN' for x in rows.values()),tests=rows,log=stream.getvalue(),code_commit=code['git_commit'],historical_replay_evidence_origin=os.environ['PG3_NEXT_EVIDENCE'])
    write(out/'TEST_REPORT.json',t);q=read(out/'MINIMAL/QUALIFICATION.json');q.update(status='QUALIFIED' if r.wasSuccessful() and t['not_run']==0 else 'BLOCKED',sources=snapshot(),tests=ref(out/'TEST_REPORT.json'),executor_short=ref(out/'SHORT_PAIR_RESULT.json'),formal_spec_sha256={arm:digest(s) for arm,s in formal.items()},formal_training='NOT_RUN',formal_authorized=False)
    write(out/'FINAL_QUALIFICATION.json',q)
    for arm,s in formal.items():
        if q['status']=='QUALIFIED':
            from .executor import qualified
            qualified(s)
    print('MINPAIR_FINAL',q['status'],t['total'],t['passed'],t['failed'],t['not_run'],flush=True)
    require(q['status']=='QUALIFIED','MINPAIR_QUALIFICATION_FAILED')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();prepare(a.output)
