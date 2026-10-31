"""Read-only evidence analysis. No controller publications, training or weight changes."""
from pathlib import Path
import json,csv,collections,hashlib,time
import numpy as np,torch
from query_minco.training.io import read,write,sha,require
from query_minco.models import QueryMincoPlanner
from query_minco.trajectory import TrajectoryPostprocessor
from query_minco.inference.identity import state_sha
from query_minco.integration.stagec_execution_v2.renderer_volume import RendererHistory
from query_minco.integration.pg2_stagec_v1.reference import Reference
BASE=Path('/home/lizheng/query_minco_mainline/research_outputs/pg2_stagec_execution_v2');ROOT=BASE.parents[1]
def rows(path):return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
def peaks(odom,start_ns):
    r=[x for x in odom if x['wall_ns']>=start_ns];r=[x for i,x in enumerate(r) if i==0 or x['stamp']>r[i-1]['stamp']]
    if len(r)<4:return dict(status='NOT_AVAILABLE')
    t=np.array([x['stamp'] for x in r]);v=np.array([x['v'] for x in r]);dt=np.diff(t);a=np.diff(v,axis=0)/dt[:,None];valid=(dt>=.005)&(dt<=.03);j=np.diff(a,axis=0)/((dt[1:]+dt[:-1])/2)[:,None];jvalid=valid[1:]&valid[:-1]
    return dict(vmax_mps=float(np.linalg.norm(v,axis=1).max()),amax_causal_mps2=float(np.linalg.norm(a[valid],axis=1).max()) if valid.any() else None,jmax_causal_mps3=float(np.linalg.norm(j[jvalid],axis=1).max()) if jvalid.any() else None,scope='finite-difference odometry statistics; no interval/worst-case or rotor proof',source_samples=len(r))
def control():
    out=BASE/'CONTROL_ANALYSIS001';out.mkdir(exist_ok=False);allrows=[]
    for case in sorted((BASE/'SYSTEM').iterdir()):
        if not case.is_dir():continue
        result_path=case/('NATIVE_WATCHDOG_RESULT.json' if case.name.startswith('native_watchdog') else 'GATEWAY_RESULT.json')
        result=read(result_path) if result_path.exists() else dict(status='BLOCKED_INCOMPLETE')
        run=read(case/'RUN.json') if (case/'RUN.json').exists() else dict(status='NOT_RUN');entry=read(case/'GATEWAY_READY.json') if (case/'GATEWAY_READY.json').exists() else None
        odom=rows(case/('native_ODOM.jsonl' if case.name.startswith('native_watchdog') else 'gateway_ODOM.jsonl'));cmd=rows(case/'gateway_COMMAND.jsonl');att=rows(case/('native_ATTITUDE.jsonl' if case.name.startswith('native_watchdog') else 'gateway_ATTITUDE.jsonl'))
        rec=dict(case=case.name,run=run['status'],result=result,cleanup=read(case/'CLEANUP.json'),entry=entry,plant_peaks_after_qualified_entry=peaks(odom,entry['wall_ns']) if entry else dict(status='NOT_RUN_NO_QUALIFIED_ENTRY'),normalized_thrust_range=[min(x['thrust'] for x in att),max(x['thrust'] for x in att)] if att else None,thrust_scope='measured normalized command only; no RPM/worst-case bound',Query_navigation=False)
        if (case/'INJECTION.json').exists() and result.get('trigger'):
            inj=read(case/'INJECTION.json');rec['fault_detection_after_injection_s']=(result['trigger']['wall_ns']-inj['wall_ns'])*1e-9
            fallback=[x for x in cmd if x['fault'] is not None];rec['first_fallback_command_after_injection_s']=(fallback[0]['wall_ns']-inj['wall_ns'])*1e-9 if fallback else None
        # Actual P/V packet tracking, no time-shift optimization. Old logs may only
        # have send-log time; new exact header/sample fields are distinguished.
        matched=[]
        if odom:
            t=np.array([x['stamp'] for x in odom]);p=np.array([x['p'] for x in odom]);v=np.array([x['v'] for x in odom])
            for x in cmd:
                stamp=x.get('header_stamp',x['ros'])
                if not t[0]<=stamp<=t[-1]:continue
                actual_p=np.array([np.interp(stamp,t,p[:,k]) for k in range(3)]);actual_v=np.array([np.interp(stamp,t,v[:,k]) for k in range(3)]);want=np.array(x['pva']);matched.append(dict(p=float(np.linalg.norm(actual_p-want[0])),v=float(np.linalg.norm(actual_v-want[1])),fault=x['fault'],stamp=stamp))
        if matched:
            errors=np.array([x['p'] for x in matched]);rec['tracking']=dict(position_RMS_m=float(np.sqrt(np.mean(errors**2))),position_P95_m=float(np.percentile(errors,95)),position_peak_m=float(errors.max()),velocity_peak_mps=max(x['v'] for x in matched),samples=len(matched),alignment='exact command header' if cmd and 'header_stamp' in cmd[0] else 'same-host send-log time, not separately instrumented header; approximate packet alignment',scope='synthetic rig reference only; NOT Query tracking or a guaranteed inflation bound')
        if case.name.startswith('semantics') and matched and (case/'INJECTION.json').exists():
            end=read(case/'INJECTION.json')['ros'];steady=[x for x in matched if end-1<=x['stamp']<end];rec['semantics_position_error_last_second_m']=float(np.mean([x['p'] for x in steady])) if steady else None
        allrows.append(rec)
    write(out/'RESULT.json',dict(cases=allrows,FALLBACK_VERIFIED=False,Query_control='NOT_RUN',notes='Initial failures and default-disabled negative control are retained, not counted as safety passes.'))
    return allrows

def shadow():
    out=BASE/'SHADOW_ANALYSIS001';out.mkdir(exist_ok=False);case=BASE/'SYSTEM/domain_shadow_r3';frames=sorted(case.glob('shadow_frame*.npz'));model=QueryMincoPlanner();cp=ROOT/'research_outputs/pg3_v21_paired128_v1/runs/A42_A_PG3V21_formal_run001/commits/step0128/checkpoint.pt';model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True)['model']);model.eval().requires_grad_(False);torch.set_num_threads(1);identity=state_sha(model);require(identity=='268c4d80308e5bbcc3e8677ca2fdc67ccdcc3da84d664341727a3eb3984d8e36','FROZEN_PG2_ID')
    nn={int(np.load(f)['source_stamp_ns']):f for f in frames};events={r['stamp_ns']:r for r in rows(case/'MOVING_SHADOW.jsonl') if r['status']=='RECORDED_NO_CONTROL'};h=RendererHistory();joined=[];grid=collections.Counter();grid_rows=[]
    for f in sorted((case/'FULL_NATIVE_INPUTS').glob('*.npz')):
        x=np.load(f);ns=int(x['source_stamp_ns']);stamp=ns*1e-9;h.add(stamp,x['depth_m'],x['origin'],x['rotation'])
        # Read-only grid probe, explicitly posthoc diagnostic, never a selector.
        if ns in nn:
            qdata=np.load(nn[ns]);event=events[ns];now=stamp+event['output_age_s'];c0=np.zeros((3,3,6))
            for distance in [.6,.8,1.,1.2]:
                c0[:,:,0]=x['origin']+x['rotation']@np.array([distance,0,0]);z=h.certify(c0,np.ones(3)*.02,now);grid[(distance,z['status'])]+=1;grid_rows.append(dict(stamp_ns=ns,distance_m=distance,status=z['status'],reason=z['reason'],scope=z['scope'],online_permission=False))
            with torch.inference_mode():raw=model(torch.from_numpy(qdata['depth_input'][None]),torch.from_numpy(qdata['state'][None]));pred=TrajectoryPostprocessor()(raw,torch.from_numpy(qdata['state'][None]),num_samples=65)
            require(np.array_equal(raw.numpy()[0],qdata['raw']) and np.array_equal(pred.trajectory.coeffs.numpy()[0],qdata['coefficients']) and np.array_equal(pred.parameters.durations.numpy()[0],qdata['durations']),'FROZEN_RAW_COEFFICIENT_REBUILD')
            for q in range(6):
                ref=Reference(qdata['coefficients'][q],qdata['durations'][q],qdata['position_W'],qdata['R_WB0'],stamp,q);certificate=h.certify(ref.world_coefficients(),ref.durations,now);require(not certificate['online_permission'] and not certificate['selected_permission'],'NO_EXECUTION_AUTHORITY')
                joined.append(dict(stamp_ns=ns,query=q,frame=nn[ns].name,raw_coefficients_times_bitwise=True,c2_residuals=ref.seam_residuals(),output_age_s=event['output_age_s'],certificate=certificate))
    before=identity;require(state_sha(model)==before,'FROZEN_PG2_MODIFIED')
    write(out/'RESULT.json',dict(frames_saved=len(frames),full_native_joined_frames=len({r['stamp_ns'] for r in joined}),candidates=len(joined),raw_coefficient_rebuild='PASS',body_results=dict(collections.Counter(r['certificate']['status'] for r in joined)),forward_grid_posthoc=[dict(distance_m=k[0],status=k[1],count=v) for k,v in sorted(grid.items())],posthoc_scope='read-only geometry probes; not independent calibration or executable trajectories',model_sha256=identity,checkpoint_sha256=sha(cp),Query_tracking='NOT_RUN',online_releases=0,original_selector_all_none=all(e['selector']['selected_query'] is None for e in events.values())))
    write(out/'CANDIDATE_CERTIFICATES.json',dict(rows=joined));write(out/'FORWARD_GRID_DIAGNOSTIC.json',dict(rows=grid_rows))
    return joined

def independent():
    out=BASE/'PG3_ANALYSIS001';out.mkdir(exist_ok=False);result=read(BASE/'PG3_INDEPENDENT001/RESULT.json');cases=result['cases'];methods={}
    for name in ['ADAMW','GLOBAL_BACKTRACK','EXISTING_QP','FROZEN_QUERY_LOCAL']:
        rr=[c['methods'][name] for c in cases];chosen=[r['chosen'] for r in rr if r['chosen']];methods[name]=dict(accepted=len(chosen),case_count=len(cases),hard_repairs=sum(r['pre_fail_to_pass'] for r in chosen),safe_losses=sum(r['pre_safe_pass_to_fail'] for r in chosen),matched_safe_degraded=sum(r['matched_safe_degraded'] for r in chosen),oracle_before=sum(r['oracle_before'] for r in chosen),oracle_after=sum(r['oracle_after'] for r in chosen),known_mode_count_before=sum(r['known_modes_before'] for r in chosen),known_mode_count_after=sum(r['known_modes_after'] for r in chosen),cpu_s=sum(r['total_cpu_s'] for r in rr),actual_method_extra_cpu_s=sum(r['extra_cpu_s'] for r in rr))
    # Check matched-budget decisions and actual cost domination, not equal padding.
    increments=[]
    for c in cases:
        baseline=c['methods']['GLOBAL_BACKTRACK']['chosen']
        for name in ['EXISTING_QP','FROZEN_QUERY_LOCAL']:
            r=c['methods'][name]['chosen']
            if baseline and r and r['pre_fail_to_pass']>baseline['pre_fail_to_pass'] and r['pre_safe_pass_to_fail']<=baseline['pre_safe_pass_to_fail'] and c['methods'][name]['total_cpu_s']<=c['methods']['GLOBAL_BACKTRACK']['total_cpu_s']:increments.append(dict(case=c['case'],method=name))
    write(out/'RESULT.json',dict(methods=methods,matched_cost_increment_cases=increments,decision='NO_GO' if not increments else 'PILOT_ONLY_NOT_FORMALLY_QUALIFIED',case_count=len(cases),data_protocol=result['case_protocol'],formal_updates=0,Dev_used=False))
    return methods
if __name__=='__main__':
    print('CONTROL',len(control()),flush=True);print('SHADOW',len(shadow()),flush=True);print('PG3',independent(),flush=True)
