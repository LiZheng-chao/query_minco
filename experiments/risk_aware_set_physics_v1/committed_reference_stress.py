"""Real host-clock CPU scheduling test, NOT ROS, NOT real measured PVA."""
import json,time,multiprocessing as mp,threading,os
from pathlib import Path
import numpy as np
from query_minco.inference.committed_reference import *
from query_minco.inference.observed_space import FreeVolume,box_cell,Scope

def computation_worker(start,ready,done,lock,output):
    ready.set();start.wait();time.sleep(.2)
    t=time.monotonic()
    # GIL-bound work in a separate PROCESS, including the measured49.418ms case.
    end=t+.049418
    while time.monotonic()<end:sum(range(1000))
    lock_wait=[]
    def competitor():
        began=time.monotonic()
        with lock:lock_wait.append(time.monotonic()-began)
    with lock:
        th=threading.Thread(target=competitor);th.start()
        time.sleep(.12)
    th.join()
    # Evidence IO is owned by this worker, never by tick().
    with (Path(output)/('logs/WORKER_IO_%d.log'%os.getpid())).open('x') as f:f.write('EVIDENCE_TEST\n'*10000)
    done.put(dict(computation_block_s=time.monotonic()-t,lock_competitor_wait_s=lock_wait))

def ticker(bundle,start,ready,result):
    k=ReferenceKernel();ready.set();start.wait()
    rm=time.monotonic();rr=time.time();activation_m=rm+.1;activation_r=rr+.1
    k.submit(bundle,activation_r,activation_m)
    stamps=[];cost=[];n=0
    while time.monotonic()-rm<5.:
        due=rm+n*.02
        while time.monotonic()<due:time.sleep(max(0,min(.001,due-time.monotonic())))
        mono=time.monotonic();ros=rr+(mono-rm)
        expected=bundle.prefix.pva(0) if k.active is None else k._reference(mono)
        state=StateSnapshot(ros,mono,tuple(map(tuple,expected)),topology_received_mono=mono)
        t=time.perf_counter()
        k.tick(ros,mono,state,watchdog=n==15)
        cost.append((time.perf_counter()-t)*1000);stamps.append(mono);n+=1
    result.put(dict(stamps=stamps,tick_ms=cost,events=list(k.events),qualification_failed=k.qualification_failed))

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out)
    c=np.zeros((1,3,6));c[0,:,0]=[0,0,2];c[0,0,1]=.5;c[0,0,2]=.025
    now=time.time();mono=time.monotonic()
    v=FreeVolume((box_cell([-2.8,-2.8,.2],[6.8,2.8,3.8],now+120,'CPU-ONLY'),),Scope.EXTERNAL_SUPERVISOR)
    bundle=prepare_reference(Program(c,[.4]),v,now,mono)
    ctx=mp.get_context('spawn');start=ctx.Event();wr=ctx.Event();tr=ctx.Event();lock=ctx.Lock();wq=ctx.Queue();tq=ctx.Queue()
    worker=ctx.Process(target=computation_worker,args=(start,wr,wq,lock,str(out)))
    timer=ctx.Process(target=ticker,args=(bundle,start,tr,tq))
    worker.start();timer.start()
    assert wr.wait(20) and tr.wait(20),'READY_TIMEOUT'
    start.set();w=wq.get(timeout=15);t=tq.get(timeout=15)
    worker.join(5);timer.join(5);assert worker.exitcode==timer.exitcode==0
    gaps=np.diff(t['stamps'])*1000;lat=[e['lateness']*1000 for e in t['events'] if e['reason']=='ACTIVATED']
    result=dict(scope='CPU_HOST_CLOCK_PROCESS_ISOLATION_IDEAL_STATE_NOT_ROS',samples=len(t['stamps']),
        worker=w,gap_p50_p95_max_ms=np.quantile(gaps,[.5,.95,1]).tolist(),
        tick_compute_p50_p95_max_ms=np.quantile(t['tick_ms'],[.5,.95,1]).tolist(),activation_lateness_ms=lat,
        events=t['events'],qualification_failed=t['qualification_failed'],
        passed=bool(gaps.max()<=40 and lat and max(lat)<=10 and not t['qualification_failed']))
    with (out/'CPU_PROCESS_STRESS.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2));assert result['passed'],'CPU_STRESS_FAILED'
if __name__=='__main__':main()
