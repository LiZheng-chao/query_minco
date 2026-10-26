"""CPU planning worker. Runs independently before physical motion."""
import argparse,json,pickle,time
from pathlib import Path
import numpy as np
from query_minco.inference.trajectory_execution import Program,quintic,concatenate
from query_minco.inference.execution_clock_v3 import cruise_segment
from query_minco.inference.committed_reference import prepare_reference
from query_minco.inference.observed_space import FreeVolume,box_cell,Scope

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--job');a=p.parse_args();out=Path(a.out)
    if a.job:
        job=json.loads(Path(a.job).read_text())
        if job['kind']=='stress':
            from query_minco.inference.execution_clock_v3 import soft_brake
            start=time.monotonic()
            soft_brake(np.array([[2.6382471374,0,2],[-.5902274469,0,0],[-.4893392048,0,0]]),soft_jerk=1.)
            while time.monotonic()-start<.049418:sum(range(1000))
            import threading
            lock=threading.Lock()
            with lock:time.sleep(.12)
            with (out/'STRESS_COMPLETE.json').open('x') as f:json.dump(dict(start_mono=start,end_mono=time.monotonic(),scope='extra planning work, no publisher locks'),f)
            return
        head=np.asarray(job['head'],float)
        tail=head.copy();tail[0]+=np.array([2,0,0])
        from query_minco.inference.trajectory_execution import trim
        prog=trim(quintic(head,tail,8.),0,2.8)
        now=time.time();mono=time.monotonic()
        vol=FreeVolume((box_cell([-2.8,-2.8,.2],[6.8,2.8,3.8],now+120,'VERIFIED_EMPTY_ROOM'),),Scope.EXTERNAL_SUPERVISOR)
        bundle=prepare_reference(prog,vol,now,mono)
        with (out/'ABORT_REFERENCE.pkl').open('xb') as f:pickle.dump(bundle,f,protocol=4)
        return
    h=np.array([[-1.6,0,2],[0,0,0],[0,0,0]],float)
    static=cruise_segment(h,.4)
    back=h.copy();back[0,0]=-1.96
    pre=quintic(h,back,4.01)
    top=back.copy();top[0,0]=-.76;top[1,0]=.8
    pre=concatenate(pre,quintic(back,top,3.))
    pre=concatenate(pre,cruise_segment(top,.95))
    mid=np.array([[0,0,2],[.8,0,0],[0,0,0]],float)
    goal=np.array([[3,0,2],[0,0,0],[0,0,0]],float)
    nominal=concatenate(pre,quintic(mid,goal,7.5))
    # Deliberate heartbeat timeout during this real nonzero-v/a source reference.
    watchdog=quintic(goal,np.array([[0,0,2],[0,0,0],[0,0,0]],float),8.3)
    from query_minco.inference.trajectory_execution import trim
    watchdog=trim(watchdog,0,2.8)
    now=time.time();mono=time.monotonic()
    vol=FreeVolume((box_cell([-2.8,-2.8,.2],[6.8,2.8,3.8],now+120,'VERIFIED_EMPTY_ROOM'),),Scope.EXTERNAL_SUPERVISOR)
    result={};rows=[]
    for name,prog in [('STATIC',static),('NOMINAL',nominal),('WATCHDOG',watchdog)]:
        start=time.perf_counter()
        result[name]=prepare_reference(prog,vol,now,mono)
        rows.append(dict(name=name,prepare_ms=(time.perf_counter()-start)*1000,
            stops=len(result[name].stops),prefix_duration=prog.duration,prefix_extrema=prog.extrema()))
    with (out/'PREPARED_REFERENCES.pkl').open('xb') as f:pickle.dump(result,f,protocol=4)
    with (out/'PREPARED_REFERENCE_SUMMARY.json').open('x') as f:json.dump(dict(prepared_ros=now,prepared_mono=mono,valid_for=120,programs=rows,
        scope='CPU_ALL_PREFIX_PHASES_HAVE_FULL_PRECERTIFIED_DELAYED_STOP; physical tracking remains untested',
        nominal_handover_phase_s=7.96),f,indent=2)
    print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
