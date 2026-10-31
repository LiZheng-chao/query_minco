"""Posthoc Dev attribution ONLY; no proposal, protection, acceptance or tuning access."""
from pathlib import Path
import json,csv,time
import numpy as np
import torch
from query_minco.training.io import read,write,sha,require
from query_minco.trajectory import TrajectoryPostprocessor
from query_minco.evaluation.validation import validate_teacher,evaluate
from query_minco.evaluation.goal import task_metrics
from query_minco.training.losses.imitation import cost_matrix
from query_minco.training.matching.hungarian import match
from experiments.pg3_feasible_mode_v1.common import model_from
from experiments.pg3_feasible_mode_v1.fulltrain128_evaluation import signature,integral
from .counterfactual import context,BASE,RUN,QUAL
LOST=('f02_right_only_g4_v0','f02_right_only_g4_v1','f03_narrow3d_g5_v1')
def run():
    torch.set_num_interop_threads(1)
    out=BASE/'ATTRIBUTION001';out.mkdir(exist_ok=False);spec,ctx,cfg=context(out);ds,geo,init,_,train,dev=ctx
    items=[item for item in dev if item[0].sample_id in LOST];require(len(items)==3,'LOST_DEV_IDS');m=model_from(init);m.eval();rows=[];ids=[s.sample_id for s,f,meta in items];teacher=[];checkpoints={};start=time.monotonic()
    updates=[read(RUN/('commits/step%04d/UPDATE.json'%step)) for step in range(1,129)]
    require(all(not set(u['ids'])&set(ids) for u in updates),'DEV_WAS_DIRECTLY_SUPERVISED')
    for s,f,meta in items:
        for i,t in enumerate(meta['teachers']):
            c=s.teacher_coeffs[i].numpy();d=s.teacher_durations[i].numpy();teacher.append(dict(sample_id=s.sample_id,teacher=i,qa=integral(c,d,2),qj=integral(c,d,3),time=float(d.sum()),route=signature(c,d,meta['scene'])))
    for step in range(129):
        cp=RUN/('commits/step%04d/checkpoint.pt'%step);blob=torch.load(cp,map_location='cpu',weights_only=True);m.load_state_dict(blob['model']);checkpoints[str(step)]=sha(cp)
        with torch.no_grad():
            for s,f,meta in items:
                raw=m(s.depth_input[None],s.state[None]);o=TrajectoryPostprocessor()(raw,s.state[None]);cost,_=cost_matrix(o,s);_,qq,tt=match(cost,'B_set');matches={int(q):int(t) for q,t in zip(qq,tt)};group=[]
                for q in range(6):
                    c=o.trajectory.coeffs[0,q].numpy();d=o.parameters.durations[0,q].numpy();hard=validate_teacher(c,d,o.parameters.head_pva[0,q].numpy(),o.parameters.tail_pva[0,q].numpy(),o.parameters.inner_pts[0,q].numpy(),meta['scene']);goal=task_metrics(evaluate(c[-1],d[-1]),evaluate(c[-1],d[-1],1),s.state[6:9].numpy());complete=bool(hard['passed'] and goal['passed'])
                    group.append(dict(query=q,hard=bool(hard['passed']),goal=bool(goal['passed']),complete=complete,reasons=hard['reasons'],matched=q in matches,matched_teacher=matches.get(q),signature=signature(c,d,meta['scene']) if complete else None,qa=integral(c,d,2),qj=integral(c,d,3),time=float(d.sum()),raw19=raw[0,q,:19].tolist(),vmax=hard['max_speed_mps'],amax=hard['max_acceleration_mps2'],jmax=hard['max_jerk_mps3']))
                rows.append(dict(step=step,sample_id=s.sample_id,oracle=any(r['complete'] for r in group),safe_queries=[r['query'] for r in group if r['complete']],queries=group))
        if step%16==0:print('ATTRIBUTION',step,flush=True)
    findings=[]
    for sid in ids:
        timeline=[r for r in rows if r['sample_id']==sid];changes=[]
        for a,b in zip(timeline,timeline[1:]):
            if a['safe_queries']!=b['safe_queries']:changes.append(dict(step=b['step'],before=a['safe_queries'],after=b['safe_queries'],oracle_lost=a['oracle'] and not b['oracle'],before_queries=a['queries'],after_queries=b['queries']))
        findings.append(dict(sample_id=sid,initial_safe_queries=timeline[0]['safe_queries'],final_safe_queries=timeline[-1]['safe_queries'],first_oracle_loss=next((r['step'] for r in changes if r['oracle_lost']),None),changes=changes,training_direct_supervision=False,causal_status='NOT_IDENTIFIED: shared update exposure is structural; frozen observational timeline does not isolate Mode causality'))
    write(out/'TIMELINE.json',dict(rows=rows,checkpoint_sha256=checkpoints,development_use='POSTHOC_ONLY',no_protection_no_alpha_selection=True))
    write(out/'RESULT.json',dict(status='VERIFIED',tasks=findings,teacher_quality=teacher,direct_dev_supervision=False,network_coupling='shared encoder/attention/head; query embeddings participate in self-attention, so local rows are not independent',formal_training=0,wall_s=time.monotonic()-start))
if __name__=='__main__':run()
