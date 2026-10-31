"""Frozen students/Teacher quality and offline GT-first selector, no optimizer."""
import collections,json,time
from pathlib import Path
import numpy as np
import torch
from query_minco.training.io import read,write,require,artifact
from query_minco.training.pg3_next_exact_v1.quality import quality,select,path_length,SCOPE
from query_minco.training.pg3_feasible_mode_v1.teacher_labels import TeacherBank
from query_minco.training.pg3_feasible_mode_v1.spec import recipe
from query_minco.trajectory import PhysicalParameterDecoder,TrajectoryPostprocessor
from query_minco.training.pg3_feasible_mode_v1.teacher_labels import inverse_raw
from query_minco.trajectory.minco_adapter import constraint_residuals
from query_minco.evaluation.validation import evaluate,norm_max
from query_minco.evaluation.goal import task_metrics
from query_minco.evaluation.signed_geometry import fine_geometry
from .common import ARMS,runpath,table,csvwrite

def run(out,context):
    folder=out/'QUALITY';folder.mkdir(exist_ok=False);effective={}
    for arm in ARMS:
        effective[arm]={k:sorted({read(runpath(arm)/('commits/step%04d/UPDATE.json'%i))['audit']['components']['pg2/'+k]['effective_weight'] for i in range(1,129)}) for k in ('lateral','acceleration','jerk','time')}
        require(all(v==[0.] for v in effective[arm].values()),'NEXT_PG2_QUALITY_WEIGHT_NONZERO')
    write(folder/'EFFECTIVE_WEIGHTS.json',dict(status='VERIFIED',actual128_updates=effective,all_quality_weights_zero=True))
    ds,geo,init,q,train,dev=context;students=[];teachers=[];excluded=[];comparisons=[];selectors=[];bank=TeacherBank(dict(recipe(),build_corridor=False))
    for split,items in (('training384',train),('development192',dev)):
        student_groups={}
        for arm in ('C1',)+ARMS:
            step=0 if arm=='C1' else 128;run=runpath('A' if arm=='C1' else arm);base=run/'evaluation'/('%s_step%04d'%(split,step));csv=table(str(base)+'.csv')
            with np.load(str(base)+'.npz') as f:arrays={k:f[k] for k in f.files}
            byid={(r['sample_id'],int(r['query'])):r for r in csv}
            for j,(sample,field,meta) in enumerate(items):
                require(str(arrays['ids'][j])==sample.sample_id,'NEXT_QUALITY_ID_ORDER')
                c=arrays['coeffs'][j];d=arrays['durations'][j];raw=torch.from_numpy(arrays['raw'][j:j+1]);p=PhysicalParameterDecoder()(raw,sample.state[None]);_,valid=constraint_residuals(torch.from_numpy(c[None]),torch.from_numpy(d[None]),p.head_pva,p.tail_pva,p.inner_pts)
                qq=quality(c,d,valid);group=[]
                for k in range(6):
                    old=byid[sample.sample_id,k];row=dict(split=split,arm=arm,sample_id=sample.sample_id,kind=meta['scene']['kind'],query_id=k,solve_valid=bool(valid[0,k]),hard_physics_pass=old['hard_pass']=='True',goal_pass=old['goal_pass']=='True',**{name:float(value[0,k]) for name,value in qq.items()},path_length_m=path_length(c[k],d[k]),max_speed_mps=float(old['vmax']),max_acc_mps2=float(old['amax']),max_jerk_mps3=float(old['jmax']),min_surface_clearance_m=float(old['signed_clearance']) if old['signed_clearance'] else None,quality_source='EXISTING_DYNAMIC_QUALITY_LOSS_4_POINT_POSITIVE_GAUSS_SI',tracking_error='NOT_AVAILABLE')
                    require(abs(row['E_a_SI']-float(old['qa']))<1e-6 and abs(row['E_j_SI']-float(old['qj']))<1e-5,'NEXT_EXISTING_ENERGY_CROSSCHECK');group.append(row);students.append(row)
                student_groups[arm,sample.sample_id]=group
                rng=np.random.RandomState(42+100000*j+(0 if split=='training384' else 100000000));random_query=int(rng.randint(6))
                eligible={r['query_id'] for r in group if r['solve_valid'] and r['hard_physics_pass'] and r['goal_pass'] and all(np.isfinite(r[k]) for k in ('QA','QJ','QT'))}
                policies=dict(query0=0 if 0 in eligible else None,random_seed42=random_query if random_query in eligible else None)
                for key in ('J_quality','QA','QJ','duration_s'):policies['safe_'+key]=select(group,key)[0]
                oracle=any(r['solve_valid'] and r['hard_physics_pass'] and r['goal_pass'] and all(np.isfinite(r[k]) for k in ('QA','QJ','QT')) for r in group)
                for policy,k in policies.items():
                    r=group[k] if k is not None else None;selectors.append(dict(split=split,arm=arm,sample_id=sample.sample_id,policy=policy,selected_query=k,status='SELECTED' if r else 'NO_CERTIFIED_CANDIDATE',selected_hard_pass=None if r is None else r['hard_physics_pass'],selected_goal_pass=None if r is None else r['goal_pass'],QA=None if r is None else r['QA'],QJ=None if r is None else r['QJ'],QT=None if r is None else r['QT'],oracle_at6=oracle,scope=SCOPE,execution_permission=False))
            print('QUALITY_STUDENTS',split,arm,len(items),flush=True)
        for sample,field,meta in items:
            tb=bank.get(sample,field)
            for t in tb['teachers']:
                flags=t['representability'];strict=t['certified'] and all(flags.values())
                if not strict:
                    excluded.append(dict(split=split,sample_id=sample.sample_id,teacher_id=t['teacher'],certified=t['certified'],existing_mode_representable=t['representable'],strict_full_decoder_representable=strict,flags=json.dumps(flags)));continue
                c=np.array(t['coeffs']);d=np.array(t['durations']);decoded=TrajectoryPostprocessor()(torch.tensor(inverse_raw(t['inner'],t['tail'],t['durations']),dtype=torch.float32)[None,None].repeat(1,6,1),sample.state[None]);require(bool(decoded.trajectory.solve_valid.all()),'NEXT_TEACHER_DECODER_SOLVE');qq=quality(c,d);signed=fine_geometry(c,d,meta['scene']);goal=task_metrics(evaluate(c[-1],d[-1]),evaluate(c[-1],d[-1],1),sample.state[6:9].numpy())
                r=dict(split=split,sample_id=sample.sample_id,kind=meta['scene']['kind'],teacher_id=t['teacher'],certified=True,strict_full_decoder_representable=True,existing_mode_representable=t['representable'],representation_flags=json.dumps(flags),solve_valid=True,hard_physics_pass=t['check']['passed'],goal_pass=goal['passed'],**{name:float(v[0,0]) for name,v in qq.items()},path_length_m=path_length(c,d),max_speed_mps=max(norm_max(cc,T,1) for cc,T in zip(c,d)),max_acc_mps2=max(norm_max(cc,T,2) for cc,T in zip(c,d)),max_jerk_mps3=max(norm_max(cc,T,3) for cc,T in zip(c,d)),min_surface_clearance_m=signed['minimum_signed_surface_clearance_m'],route_signature=json.dumps(t['mode_signature']),bank_identity=tb['identity'],tracking_error='NOT_AVAILABLE')
                teachers.append(r)
                for arm in ('C1',)+ARMS:
                    sg=student_groups[arm,sample.sample_id];safe=[x for x in sg if x['hard_physics_pass'] and x['goal_pass'] and x['solve_valid']]
                    comparisons.append(dict(split=split,sample_id=sample.sample_id,arm=arm,teacher_id=t['teacher'],teacher_QA=r['QA'],teacher_QJ=r['QJ'],teacher_T=r['duration_s'],student_QA_min=min(x['QA'] for x in sg),student_QJ_min=min(x['QJ'] for x in sg),safe_student_QA_min=min([x['QA'] for x in safe],default=None),safe_student_QJ_min=min([x['QJ'] for x in safe],default=None),student_oracle=bool(safe)))
            if len(teachers)%128==0:print('QUALITY_TEACHERS',split,len(teachers),flush=True)
    csvwrite(folder/'STUDENT_QUALITY.csv',students);csvwrite(folder/'TEACHER_QUALITY.csv',teachers);csvwrite(folder/'TEACHER_STUDENT_COMPARISON.csv',comparisons);csvwrite(folder/'OFFLINE_SELECTION.csv',selectors)
    write(folder/'EXCLUDED_TEACHERS.json',excluded)
    def stats(rr):
        return {k:dict(mean=float(np.mean([x[k] for x in rr])),quantiles=np.quantile([x[k] for x in rr],[0,.1,.5,.9,1]).tolist()) for k in ('QA','QJ','QT','E_a_SI','E_j_SI','duration_s','min_surface_clearance_m') if all(x[k] is not None for x in rr)}
    summary={}
    for split in ('training384','development192'):
        for kind in sorted({r['kind'] for r in teachers if r['split']==split}):summary[split+'/'+kind]=stats([r for r in teachers if r['split']==split and r['kind']==kind])
    selected={}
    for split in ('training384','development192'):
        for arm in ('C1',)+ARMS:
            for policy in sorted({r['policy'] for r in selectors}):
                rr=[r for r in selectors if r['split']==split and r['arm']==arm and r['policy']==policy];chosen=[r for r in rr if r['selected_query'] is not None]
                selected[split+'/'+arm+'/'+policy]=dict(tasks=len(rr),selected=len(chosen),selected_hard_and_goal=sum(r['selected_hard_pass'] and r['selected_goal_pass'] for r in chosen),oracle_at6=sum(r['oracle_at6'] for r in rr),QA_mean=float(np.mean([r['QA'] for r in chosen])) if chosen else None,QJ_mean=float(np.mean([r['QJ'] for r in chosen])) if chosen else None,scope=SCOPE)
    write(folder/'RESULT.json',dict(status='VERIFIED',students=len(students),certified_representable_teachers=len(teachers),excluded=len(excluded),teacher_strata=summary,selector_summary=selected,quality_formula=dict(QA='Ea/(8^2*2.2)',QJ='Ej/(100^2*2.2)',QT='T/2.2'),legacy_warning='Previous evaluation CSV qa/qj were raw SI integrals, not these normalized QA/QJ',control_tracking_metrics='NOT_AVAILABLE',ranking_weights=[1,1,.2],weights_used_for_training=False))
