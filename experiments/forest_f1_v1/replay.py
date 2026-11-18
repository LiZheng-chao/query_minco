"""Recorded native F0 replay. No ROS, GT geometry, simulator or actuator imports."""
from pathlib import Path
import sys,json,time,hashlib,collections,os
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from query_minco.integration.depth_minco_t2_v1.engine import Engine
from query_minco.integration.depth_minco_t2_v1.space import BirthGrant,Space,authorized_snapshot
from query_minco.integration.observed_space_v2.core import ObservedMap,Prepared,sample_batch
from query_minco.integration.observed_space_v1.map import Config,KNOWN_FREE,UNKNOWN,OCCUPIED
from query_minco.integration.forest_fix2_v1.depth_planner import DepthPlanner
RUN=ROOT/'research_outputs/f0_controller_consumption_20261118_v1/runs/F0_20261118_104715_1163/none'
OUT=ROOT/'research_outputs/forest_f1_20261118_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def load():
    events=[json.loads(x) for x in (RUN/'supervisor_data/EVENTS.jsonl').read_text().splitlines()]
    odom={e['stamp']:e for e in events if e.get('kind')=='odom' and e.get('accepted')}
    imu={e['stamp']:e for e in events if e.get('kind')=='imu' and e.get('accepted')}
    depths={e['stamp']:e for e in events if e.get('kind')=='depth' and e.get('accepted')}
    chains={e['source_stamp']:e for e in events if e.get('kind')=='depth_chain' and e.get('accepted')}
    pairs=[];missing=[]
    for p in sorted((RUN/'reference_data').glob('depth_*.npz')):
        a=np.load(p);s=float(a['stamp'])
        if s not in odom or s not in imu or s not in depths or s not in chains:
            missing.append(dict(path=str(p),stamp=s,reason='EXACT_MATCH_MISSING'));continue
        pairs.append(dict(stamp=s,path=str(p),sha256=sha(p),pose=odom[s]['payload'],imu=imu[s]['payload'],
            receipt_ros=chains[s]['receipt_ros'],receipt_monotonic=chains[s]['receipt_monotonic'],
            odom_receipt_ros=odom[s]['ros_now'],imu_receipt_ros=imu[s]['ros_now']))
    pairs.sort(key=lambda x:x['receipt_ros']);return events,pairs,missing

def run():
    OUT.mkdir();(OUT/'frames').mkdir();torch.set_num_threads(1)
    events,pairs,missing=load();write(OUT/'PAIRS.json',dict(pairs=pairs,missing=missing,camera_K=[80,0,80,0,80,45,0,0,1],camera_source='frozen native camera contract; raw CameraInfo not recorded',source= str(RUN)))
    if not pairs:write(OUT/'F1_REPLAY_RESULT.json',dict(status='INSUFFICIENT_OBSERVED_SPACE',reason='NO_EXACT_NATIVE_PAIRS'));return
    birth=json.loads((RUN/'BIRTH_CERTIFICATE.json').read_text());selected=json.loads((RUN/'BIRTH_SELECTION.json').read_text())
    truth=[json.loads(x) for x in (RUN/'truth_data/EVENTS.jsonl').read_text().splitlines()];truth=[e for e in truth if e.get('kind')=='truth']
    argv=json.loads((RUN/'plant/PROCESS_START.json').read_text())['argv'];raw=(RUN/'BIRTH_CERTIFICATE.json').read_bytes()
    def grant(now):
        rows=[x for x in truth if x['stamp']<=now]
        if not rows or not birth['issued']<=now<=birth['expires']:return None
        t=rows[-1];mono=t['monotonic']+(now-t['stamp'])
        return BirthGrant.from_records(raw,t,argv,selected,now,mono)
    clock=[pairs[0]['stamp']];engine=Engine(grant,lambda:clock[0]);engine.start()
    stream=[]
    for e in events:
        if e.get('accepted') and e.get('kind') in ['odom','imu']:stream.append((e['ros_now'],e['kind'],e))
    for p in pairs:stream.append((p['receipt_ros'],'depth',p))
    stream.sort(key=lambda x:x[0]);start=stream[0][0];wall=time.monotonic();last={};samples=[];audit=[];gaps=[];gap_start=None;paired_by_stamp={p['stamp']:p for p in pairs}
    # Strict real-rate replay; processing delay contributes to source age.
    next_tick=start;total_revocations=0
    for receipt,kind,row in stream:
        while True:
            now=start+time.monotonic()-wall
            if now>=receipt:break
            clock[0]=now
            if now>=next_tick:
                cmd=engine.command(now);samples.append(dict(time=now,valid=cmd is not None,revision=engine.revision));next_tick+=.01
                if cmd is not None:engine.published(cmd)
            time.sleep(min(.001,receipt-now))
        clock[0]=start+time.monotonic()-wall
        try:
            if kind in ['odom','imu']:
                last[kind]=row
                if 'odom' in last and 'imu' in last and last['odom']['stamp']==last['imu']['stamp']:
                    state=dict(last['odom']['payload'],a=last['imu']['payload']['a_world']);engine.set_state(state,row['stamp']);engine.set_goal(np.asarray(selected['selected']['position'])+[3,0,0],row['stamp'])
            else:
                d=np.load(row['path'])['depth'];before=engine.committed is not None;rev=engine.revision
                engine.observe(d,row['pose'],row['stamp'],row['stamp'])
                if before and engine.committed is None:total_revocations+=1
                audit.append(dict(event='OBSERVE',source_stamp=row['stamp'],now=clock[0],late_s=clock[0]-receipt,revision_before=rev,revision_after=engine.revision,revoked=before and engine.committed is None))
        except Exception as ex:audit.append(dict(event='INPUT_REJECTED',source_stamp=row.get('stamp'),reason=str(ex)))
        audit.extend(engine.take_events())
    engine.close();audit.extend(engine.take_events());write(OUT/'ENGINE_EVENTS.json',audit);write(OUT/'COMMAND_AVAILABILITY.json',samples)
    for sample in samples:
        if not sample['valid'] and gap_start is None:gap_start=sample['time']
        if sample['valid'] and gap_start is not None:gaps.append(sample['time']-gap_start);gap_start=None
    if gap_start is not None:gaps.append(samples[-1]['time']-gap_start)
    counts=collections.Counter(x.get('event') for x in audit)
    # Independent per-frame latency/geometry probes, at recorded poses only.
    probes=[];ledger=ObservedMap(Config(shape=(32,24,16)));chosen=pairs[::max(1,len(pairs)//20)]
    from experiments.forest_f1_v1.native_shadow import evaluate_frame
    for index,p in enumerate(chosen):
        d=np.load(p['path'])['depth'];tick=time.perf_counter();now=p['stamp'];clock2=lambda:now+(time.perf_counter()-tick)
        result=dict(stamp=now,receipt=p['receipt_ros'],source=p['path']);snap=None
        try:
            t=time.perf_counter();ledger.add(d,p['pose'],now,now,now);sensor=ledger.snapshot(now);snap=authorized_snapshot(sensor,grant(now),now,sensor.epoch,lambda:sensor.epoch);result['map_s']=time.perf_counter()-t
            result['cells']={name:int((snap.states==value).sum()) for name,value in [('KNOWN_FREE',KNOWN_FREE),('UNKNOWN',UNKNOWN),('OCCUPIED',OCCUPIED)]}
            planner=DepthPlanner(clock2,space=Space(snap,clock2));state=dict(p['pose'],a=p['imu']['a_world'])
            # Instrument existing methods without changing algorithm or budgets.
            search=planner.space.search_goal
            def timed_search(*args):
                t=time.perf_counter()
                try:return search(*args)
                finally:result['astar_s']=time.perf_counter()-t
            planner.space.search_goal=timed_search
            t=time.perf_counter();plan,proof=planner.propose(state,np.asarray(selected['selected']['position'])+[3,0,0],now)
            result.update(status='CERTIFIED',plan=plan.record,proof=proof,stage_times=plan._planning_profile,
                outside_birth=bool((np.linalg.norm(sample_batch(plan,np.linspace(0,2,129))[:,0]-np.asarray(birth['center_world']),axis=1)>.6).any()))
        except Exception as ex:result.update(status='REJECTED',reason=str(ex),details=getattr(ex,'record',None))
        result['total_s']=time.perf_counter()-tick;probes.append(result);write(OUT/'frames'/('frame_%03d.json'%index),result)
    write(OUT/'DEPTH_MINCO_PROBES.json',probes)
    certified=[x for x in probes if x.get('status')=='CERTIFIED'];outside=[x for x in certified if x.get('outside_birth')]
    summary=dict(status='OBSERVED_OUTSIDE_BIRTH_PLAN_FOUND' if outside else 'INSUFFICIENT_OBSERVED_SPACE',
        scope='OFFLINE_REPLAY_NOT_ONLINE_QUALIFICATION',paired_frames=len(pairs),unpaired_frames=len(missing),recorded_duration_s=pairs[-1]['stamp']-pairs[0]['stamp'],
        engine_counts=dict(counts),observe_revoked_committed=total_revocations,command_samples=len(samples),valid_command_fraction=sum(x['valid'] for x in samples)/max(1,len(samples)),
        longest_no_valid_command_s=max(gaps or [0]),final_map_revision=engine.revision,probes=len(probes),certified_plans=len(certified),certified_outside_birth=len(outside),
        rejection_counts=dict(collections.Counter(x.get('reason') for x in probes if x['status']=='REJECTED')),GT_used_for_geometry=False,
        limitation='Only the recorded 0.28m viewpoint history; offline birth grant restricted to original certificate and recorded truth freshness')
    write(OUT/'F1_REPLAY_RESULT.json',summary);print(json.dumps(summary,indent=2))
    from experiments.forest_f1_v1.native_shadow import run_shadow
    run_shadow(chosen,probes,selected,grant)
if __name__=='__main__':run()
