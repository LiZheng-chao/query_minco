"""Two independent manual-only offline probes; no production algorithm edits."""
import json,time,torch
from pathlib import Path
from experiments.query_minco_p2a_quality_diagnosis_v1.quality import f,scalar,parts,accept
from .initialization import initial,FAMILIES,SEEDS
from .audit import choose
CASES=dict(slit="native_slit_cross_far_v1",portal="native_two_portals_cross_far_v1")
def initial_audit(dest,write,csvout):
    torch.set_num_threads(1)
    batch,source=f.load_source(f.CASE_SOURCES[CASES["portal"]]);contract=f.native_contract(batch,CASES["portal"])
    obj=f.p1.objective("B0");rows=[];details=[]
    for family in FAMILIES:
        for seed in SEEDS:
            u=initial(batch.state,family,seed).requires_grad_()
            r=obj(f.p1.raw_slots(u),batch);g=torch.autograd.grad(r.loss,u,retain_graph=True)[0]
            g1=torch.autograd.grad(scalar(r,"L1",8),u)[0]
            if not torch.isfinite(g).all() or not torch.isfinite(g1).all():raise ValueError("INITIAL_NONFINITE")
            rec=f.p1.summarize(r,batch,u,g,arm="B0");task=f.crossing_success(r,batch,contract,rec["hard"])
            details.append(dict(family=family,seed=seed,source=source,record=rec,Task=task,L1_gradient=g1.detach().tolist()))
            rows.append(dict(family=family,seed=seed,Hard=rec["hard"]["passed"],TaskSuccess=task["TaskSuccess"],UNKNOWN=rec["UNKNOWN_count"],theta=rec["theta"],gradient_norm=rec["gradient_norm_FP64"],L1_gradient_norm=float(g1.double().norm()),v_a_j=rec["hard"]["dynamic_polynomial_maxima"],reasons=rec["hard"]["reasons"]))
            if rec["UNKNOWN_count"]:raise ValueError("INITIAL_OUTSIDE_CERTIFIED_ROI")
    write(dest/"INITIAL_DETAILS.json",details);csvout(dest/"INITIAL_QUALIFICATION.csv",rows)
    write(dest/"RESULT.json",dict(status="PASS",initializations=24,loss_or_Hard_changes=0,optimizer_updates=0,network_updates=0,note="candidate Hard/Task failure, if any, is recorded rather than relabelled"))
def jobs(stage):
    if stage=="slit":
        for seed in SEEDS:
            for strategy,lr,lossarm in (("L1_lr001",.01,"L1"),("L1_lr0003",.003,"L1"),("L2_filter",.01,"L2")):
                yield dict(case=CASES[stage],seed=seed,strategy=strategy,lr=lr,lossarm=lossarm,family="original_B0")
    elif stage=="portal":
        for seed in SEEDS:
            for family in FAMILIES:
                for lossarm in ("L0","L1"):
                    yield dict(case=CASES[stage],seed=seed,strategy=family+"__"+lossarm,lr=.01,lossarm=lossarm,family=family,physical_initial_alias="zero_lateral__None" if family=="random" and seed is None else None)
    else:raise ValueError("PROBE")
def optimize(stage,dest,write,csvout,guard):
    torch.set_num_threads(1);rows=[];proposals=0;current=None
    try:
        for job in jobs(stage):
            guard();current=job["strategy"]+"__"+str(job["seed"]);folder=dest/current;folder.mkdir()
            batch,source=f.load_source(f.CASE_SOURCES[job["case"]]);contract=f.native_contract(batch,job["case"])
            init=f.initial("B0_nominal",batch.state,job["seed"]) if stage=="slit" else initial(batch.state,job["family"],job["seed"])
            u=init.clone().requires_grad_();obj=f.p1.objective("B0")
            opt=torch.optim.AdamW([u],lr=job["lr"],betas=(.9,.999),eps=1e-8,weight_decay=0,amsgrad=False,foreach=False)
            write(folder/"IDENTITY.json",dict(**job,alpha=8,initial_u=init.tolist(),source=source,head_state=batch.state.tolist(),steps=64,decoder="B0",no_witness=True,no_network=True))
            history=[];nonzero=0;rejects=0;started=time.monotonic()
            with (folder/"STEPS.jsonl").open("x") as log:
                for step in range(65):
                    r=obj(f.p1.raw_slots(u),batch);loss=scalar(r,job["lossarm"],8)
                    gradient=torch.autograd.grad(loss,u,retain_graph=True)[0]
                    if not torch.isfinite(loss) or not torch.isfinite(gradient).all():raise ValueError("NONFINITE")
                    rec=f.p1.summarize(r,batch,u,gradient,arm="B0")
                    task=f.crossing_success(r,batch,contract,rec["hard"]);_,q=parts(r,8)
                    rec.update(step=step,strategy=job["strategy"],lossarm=job["lossarm"],lr=job["lr"],family=job["family"],optimization_scalar=float(loss),q_alpha=float(q),Task=task)
                    rec["weighted_raw_component_gradients"]={}
                    for k,w in (("C_viol",1),("F",1),("G_task_train",1),("C_shape",.1),("P",.025 if job["lossarm"]=="L0" else .2),("S",.1)):
                        rec["weighted_raw_component_gradients"][k]=torch.autograd.grad(w*r.metrics[k].mean(),u,retain_graph=True)[0].detach().tolist()
                    history.append(rec)
                    if step in (0,8,16,32,64):write(folder/("STATE_"+str(step)+".json"),rec)
                    if step<64:
                        opt.zero_grad(set_to_none=True);u.grad=gradient
                        clip=torch.nn.utils.clip_grad_norm_([u],10,error_if_nonfinite=True)
                        before=u.detach().clone();opt.step();proposals+=1
                        proposed_u=u.detach().clone();pr=obj(f.p1.raw_slots(u),batch)
                        proposed=f.p1.summarize(pr,batch,u,arm="B0");pt=f.crossing_success(pr,batch,contract,proposed["hard"]);_,pq=parts(pr,8)
                        ok,reason=accept(job["lossarm"],rec["hard"]["passed"],proposed["hard"]["passed"],float(q),float(pq),proposed["UNKNOWN_count"])
                        if not ok:
                            with torch.no_grad():u.copy_(before)
                            rejects+=1
                        du=u.detach()-before;nonzero+=bool(du.any())
                        rec.update(accepted=ok,filter_reason=reason,preclip_norm=float(clip),proposal_u=proposed_u.tolist(),proposed_du=(proposed_u-before).tolist(),actual_du=du.tolist(),actual_update_norm=float(du.double().norm()),proposal_record=proposed,proposed_Task=pt,moment_state_retained_on_rejection=True,Adam_m=opt.state[u]["exp_avg"].tolist(),Adam_v=opt.state[u]["exp_avg_sq"].tolist())
                    log.write(json.dumps(rec,allow_nan=False)+"\n");log.flush()
            final=history[64]
            def summarize_readout(label,selected):
                firstH=next((x["step"] for x in history if x["hard"]["passed"]),None)
                firstT=next((x["step"] for x in history if x["Task"]["TaskSuccess"]),None)
                return dict(**job,readout=label,selected_step=selected["step"] if selected else None,selected_Hard=selected["hard"]["passed"] if selected else False,selected_TaskSuccess=selected["Task"]["TaskSuccess"] if selected else False,
                    selected_L0=selected["loss"] if selected else None,selected_q_alpha=selected["q_alpha"] if selected else None,selected_progress=selected["hard"]["goal"]["distance_reduction_m"] if selected else None,selected_S=selected["metrics"]["S"] if selected else None,selected_path_length=selected["path_length_m"] if selected else None,selected_v_a_j=selected["hard"]["dynamic_polynomial_maxima"] if selected else None,selected_clearance=selected["known_sample_clearance_m"] if selected else None,selected_UNKNOWN=selected["UNKNOWN_count"] if selected else None,
                    first_Hard_step=firstH,ever_Hard=firstH is not None,first_TaskSuccess_step=firstT,ever_TaskSuccess=firstT is not None,first_certified_gate_crossing=next((x["step"] for x in history if x["Task"]["certified_gate_crossing"]),None),final_Hard=final["hard"]["passed"],final_TaskSuccess=final["Task"]["TaskSuccess"],nonzero_moves=nonzero,rejects=rejects,
                    successful_channels=sorted({c for x in history if x["Task"]["TaskSuccess"] for c in x["Task"]["successful_channels"]}),final_channels=final["Task"]["successful_channels"],quality_status="JOINT_FEASIBLE" if selected and selected["hard"]["passed"] and selected["Task"]["TaskSuccess"] else "NA",source_proposals=64,objective_forwards=129,Hard_candidate_APIs=129,objective_gradients=65,component_gradient_queries=390,extra_Hard_for_selection=0,elapsed_s=time.monotonic()-started,independent_maps=1)
            returned=[summarize_readout("final64",final)]
            if stage=="slit" and job["strategy"]=="L1_lr001":
                returned.append(summarize_readout("SAVE_min_q_Hard",choose(history,"min_q_Hard")))
            rows.extend(returned);write(folder/"RESULT.json",returned);print(json.dumps(dict(job=job,readouts=returned)),flush=True)
        write(dest/"READOUTS.json",rows);csvout(dest/"READOUTS.csv",rows)
        write(dest/"RESULT.json",dict(status="PASS",probe=stage,optimizer_trials=18 if stage=="slit" else 48,reported_readouts=len(rows),AdamW_proposals=proposals,network_updates=0,no_witness=True))
    except BaseException:
        write(dest/"PARTIAL_READOUTS.json",rows);write(dest/"RESULT.json",dict(status="STOP",current_job=current,AdamW_proposals=proposals,error=__import__("traceback").format_exc()));raise
