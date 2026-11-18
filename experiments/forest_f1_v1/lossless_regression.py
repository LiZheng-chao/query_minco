"""Strict before/after comparison; native datasets are read-only."""
from pathlib import Path
import sys,json,time,types,importlib.util,argparse,subprocess,os,hashlib
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
OUT=ROOT/'research_outputs/forest_f1_lossless_20261118_v1'
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--before',action='store_true');parser.add_argument('--label');parser.add_argument('--synthetic-only',action='store_true');a=parser.parse_args()
    name=a.label or ('before_results' if a.before else 'after_results');dest=OUT/name;dest.mkdir()
    if a.before:
        for module,file in [('query_minco.integration.observed_space_v1.map','map.py'),('query_minco.integration.depth_minco_t2_v1.engine','engine.py')]:
            spec=importlib.util.spec_from_file_location(module,OUT/'before'/file);loaded=importlib.util.module_from_spec(spec);sys.modules[module]=loaded;spec.loader.exec_module(loaded)
    import torch
    torch.set_num_threads(1)
    import experiments.forest_f1_v1.blockers_experiment as experiment
    experiment.OUT=dest
    native=ROOT/'research_outputs/forest_f1_blockers_20261118_v3'
    records=json.loads((native/'INPUTS.json').read_text())['records']
    if not a.synthetic_only:experiment.geometry(records);experiment.engine_tests(records);experiment.original_realtime()
    # Static scene at native frame rate: geometry is actually feasible, never called flight qualification.
    import numpy as np
    from query_minco.integration.depth_minco_t2_v1.engine import Engine
    from query_minco.integration.depth_minco_t2_v1.space import BirthGrant
    epoch=100.;wall=time.monotonic();clock=lambda:epoch+time.monotonic()-wall
    e=Engine(lambda n:BirthGrant((0,0,0),.6,n+.2,'SYNTHETIC_TEST_NOT_FLIGHT_QUALIFICATION'),clock);e.start();samples=[];events=[];inputs=[];p=dict(p=[0,0,0],v=[0,0,0],q=[0,0,0,1]);next_frame=0.
    while clock()<102.:
        n=clock()
        if n-epoch>=next_frame:
            try:
                e.observe(np.full((90,160),5.,np.float32),p,n,n);e.set_state(dict(p,a=[0,0,0]),n);e.set_goal([.28,0,0],n)
                inputs.append(dict(stamp=n,queue=len(e.pending),accepted=True))
            except Exception as ex:inputs.append(dict(stamp=n,queue=len(e.pending),accepted=False,reason=str(ex)))
            next_frame+=.03
        c=e.command(clock())
        if c is not None:e.published(c)
        samples.append(dict(time=clock(),valid=c is not None));events.extend(e.take_events());time.sleep(.001)
    e.close();events.extend(e.take_events())
    gaps=[];began=None
    for x in samples:
        if not x['valid'] and began is None:began=x['time']
        if x['valid'] and began is not None:gaps.append(x['time']-began);began=None
    if began is not None:gaps.append(samples[-1]['time']-began)
    (dest/'SYNTHETIC_CONTINUITY.json').write_text(json.dumps(dict(scope='SYNTHETIC_TEST_NOT_FLIGHT_QUALIFICATION',samples=samples,inputs=inputs,events=events,valid_fraction=sum(x['valid'] for x in samples)/len(samples),longest_invalid_s=max(gaps or [0]),overflow=e.evidence_overflow),indent=2)+'\n')
    print(name,'completed')
if __name__=='__main__':main()
