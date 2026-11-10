"""Read-only P1-A3 raw-step reanalysis and gradient/update decomposition."""
from pathlib import Path
import json,csv,collections,math
import numpy as np
import torch
from experiments.query_minco_p1a2_falsification_v1 import falsification as f
from experiments.query_minco_p1a3_validation_v1 import alignment as old_a
from .quality import LoggedAdamReplay
BASE=Path(__file__).resolve().parents[2]
MAIN=BASE/"research_outputs/query_minco_p1a3_validation_v1/runs/paired_main_manual_001"
CONTROL=("B0_nominal","A_matched_B0_ref","A_nominal")
def read(p):return json.loads(Path(p).read_text())
def trials():
    return sorted(p for p in MAIN.iterdir() if p.is_dir() and (p/"STEPS.jsonl").is_file())
def steps(p):return [json.loads(x) for x in (p/"STEPS.jsonl").read_text().splitlines()]
def paired_reanalysis():
    result=read(MAIN/"RESULT.json")
    if result!=dict(status="PASS",trials=96,trajectory_parameter_updates=6144,network_updates=0):raise ValueError("MAIN_IDENTITY")
    imported=list(csv.DictReader((MAIN/"P1A3_PAIRED_MAIN.csv").open()))
    if len(imported)!=192:raise ValueError("MAIN_CSV_ROWS")
    csvmap={(r["case"],r["control"],None if r["seed"]=="" else int(r["seed"]),int(r["prefix_step"])):r for r in imported}
    expected={(case,c,seed) for case,sample in f.CASE_SOURCES.items() for c in f.canonical_controls(f.load_source(sample)[0].state)[0] for seed in [None]+list(range(20261109,20261114))}
    pairs=[];events=[];counts=collections.Counter();replay_errors=[];seen=set()
    for p in trials():
        identity=read(p/"IDENTITY.json");key=(identity["case"],identity["control"],identity["seed"]);seen.add(key)
        rows=steps(p)
        if len(rows)!=65 or [r["step"] for r in rows]!=list(range(65)):raise ValueError("STEP_ORDER")
        # Model-independent exact reconstruction of actual recorded AdamW motion from gradients.
        replay=LoggedAdamReplay()
        for i in range(64):
            r=rows[i];nxt=rows[i+1]
            u=torch.tensor(r["u"]);g=torch.tensor(r["gradient"])
            predicted,clipped=replay.predict(u,g,r["preclip_norm"])
            error=float((predicted-torch.tensor(nxt["u"])).abs().max());replay_errors.append(error)
            if error>3e-7:raise ValueError("ADAM_REPLAY_MISMATCH "+str(p)+" "+str(i)+" "+str(error))
            counts["logged_updates"]+=1
        entry=dict(case=key[0],control=key[1],seed=key[2],trial_path=str(p))
        for t in (0,16,32,64):
            r=rows[t];m=r["metrics"];h=r["hard"];task=r["Task"]
            entry.update({str(t)+"_"+k:v for k,v in dict(L=r["loss"],V=m["V"],Q=m["Q"],J=m["J"],progress=h["goal"]["distance_reduction_m"],goal_error=h["goal"]["endpoint_goal_error_m"],Hard=h["passed"],TaskSuccess=task["TaskSuccess"],known=r["known_count"],UNKNOWN=r["UNKNOWN_count"],length=r["path_length_m"],max_v=h["dynamic_polynomial_maxima"][0],max_a=h["dynamic_polynomial_maxima"][1],max_j=h["dynamic_polynomial_maxima"][2],endpoint=r["endpoint"],hard_reasons=h["reasons"]).items()})
            if t in (16,64):
                c=csvmap[(*key,t)]
                for field,actual in [("loss",r["loss"]),("V",m["V"]),("Q",m["Q"]),("progress",h["goal"]["distance_reduction_m"]),("goal_error",h["goal"]["endpoint_goal_error_m"]),("path_length_m",r["path_length_m"])]:
                    if abs(float(c[field])-actual)>1e-10*(1+abs(actual)):raise ValueError("CSV_RAW_MISMATCH")
                if (c["final_Hard"]=="True")!=h["passed"] or (c["final_TaskSuccess"]=="True")!=task["TaskSuccess"]:raise ValueError("CSV_HARD_MISMATCH")
        for field in ("L","V","Q","J","progress","goal_error","known","UNKNOWN","length","max_v","max_a","max_j"):
            entry["delta_64_minus_16_"+field]=entry["64_"+field]-entry["16_"+field]
        entry.update(first_Hard_step=next((r["step"] for r in rows if r["hard"]["passed"]),None),first_TaskSuccess_step=next((r["step"] for r in rows if r["Task"]["TaskSuccess"]),None),first_certified_crossing_step=next((r["step"] for r in rows if r["Task"].get("certified_gate_crossing",False)),None))
        pairs.append(entry)
        for r in rows:
            task=r["Task"];ev=task.get("events",[])
            geometric=any(e["forward_sign_bracket"] and e["channel"] is not None and e["all_box_known_surface_distance_m"] is not None and e["all_box_known_surface_distance_m"]>=.150001 for e in ev)
            events.append(dict(**{k:entry[k] for k in ("case","control","seed")},step=r["step"],OriginalHardAll=r["hard"]["passed"],TaskSuccess=task["TaskSuccess"],Hard_without_Task=r["hard"]["passed"] and not task["TaskSuccess"],geometric_known_opening_forward_event=geometric,geometric_event_but_dynamic_failure=geometric and any(x>y for x,y in zip(r["hard"]["dynamic_polynomial_maxima"],(3,8,100))),certified_crossing=task.get("certified_gate_crossing",False),crossing_without_post_region=task.get("certified_gate_crossing",False) and not task.get("endpoint_post_wall",False),hard_reasons=r["hard"]["reasons"]))
    if seen!=expected or len(pairs)!=96 or counts["logged_updates"]!=6144:raise ValueError("TRIAL_SCHEDULE")
    quality=list(csv.DictReader((MAIN/"P1A3_TRAJECTORY_QUALITY.csv").open()))
    coverage=list(csv.DictReader((MAIN/"P1A3_CROSSING_COVERAGE.csv").open()))
    if len(quality)!=192 or len(coverage)!=12:raise ValueError("QUALITY_COVERAGE_SHAPE")
    for r in quality:
        key=(r["case"],r["control"],None if r["seed"]=="" else int(r["seed"]),int(r["prefix_step"]))
        c=csvmap[key];eligible=c["final_TaskSuccess"]=="True"
        if (r["quality_eligible"]=="True")!=eligible:raise ValueError("QUALITY_QUALIFICATION")
    for c in coverage:
        ps=[p for p in pairs if p["case"]==c["case"] and p["control"]==c["control"]]
        t=c["prefix_step"]
        if int(c["trials"])!=len(ps) or int(c["final_Hard"])!=sum(p[t+"_Hard"] for p in ps) or int(c["final_TaskSuccess"])!=sum(p[t+"_TaskSuccess"] for p in ps):raise ValueError("COVERAGE_COUNTS")
    return pairs,events,dict(status="PASS",trials=96,steps=6240,leaf_updates_verified=6144,paired_rows=192,quality_rows=len(quality),coverage_rows=len(coverage),max_AdamW_reconstruction_raw_abs=max(replay_errors),network_updates=0)

def gradient_update_records():
    cache={sample:f.load_source(sample)[0] for sample in f.CASE_SOURCES.values()}
    records=[]
    for p in trials():
        identity=read(p/"IDENTITY.json")
        if identity["control"] not in CONTROL:continue
        rows=steps(p);batch=cache[f.CASE_SOURCES[identity["case"]]];which=identity["arm"]
        contract=f.native_contract(batch,identity["case"]) if identity["case"].startswith("native") else None
        for t in (0,16,32,64):
            old=rows[t];u=torch.tensor(old["u"],dtype=torch.float32,requires_grad=True)
            r=f.p1.objective(which)(f.p1.raw_slots(u),batch)
            gu=torch.autograd.grad(r.loss,u,retain_graph=True)[0]
            if abs(float(r.loss)-old["loss"])>.003+.003*abs(old["loss"]) or not torch.allclose(gu,torch.tensor(old["gradient"]),rtol=.003,atol=.003):raise ValueError("GRADIENT_REPRODUCTION")
            if f.p1.hard_one(r,batch)["passed"]!=old["hard"]["passed"]:raise ValueError("HARD_REPRODUCTION")
            J=torch.autograd.functional.jacobian(lambda z:f.actual_theta(z,batch.state,which),u).detach()
            du=torch.tensor(rows[t+1]["u"])-u.detach() if t<64 else None
            norm=torch.tensor(old.get("preclip_norm",float(gu.norm())))
            gclip=gu.detach()*(10/(norm+1e-6)).clamp(max=1)
            component={}
            for k,factor in old_a.FACTORS.items():
                g=torch.autograd.grad(factor*r.metrics[k].mean(),u,retain_graph=True)[0].detach()
                # Diagonal original A/B mappings; report physical units and D-scaled metric too.
                diag=J.diag()
                if bool((diag.abs()<1e-8).any()):raise ValueError("SATURATED_DIAGNOSTIC_POINT")
                gt=g/diag
                component[k]=dict(scalar=float(r.metrics[k].mean()),weighted=factor*float(r.metrics[k].mean()),raw_gradient=g.tolist(),raw_norm=float(g.double().norm()),theta_gradient=gt.tolist(),theta_norm=float(gt.double().norm()),theta_D_scaled_norm=float((gt.double()*torch.tensor(f.p1.D_VALUES,dtype=torch.float64)).norm()),raw_descent_end_x=float(-(J@g)[6]),actual_update_first_order_cost=float(g.double()@du.double()) if du is not None else None)
            cosines={a+"__"+b:old_a.cosine(torch.tensor(component[a]["raw_gradient"]),torch.tensor(component[b]["raw_gradient"])) for a,b in __import__("itertools").combinations(component,2)}
            rec=dict(case=identity["case"],control=identity["control"],seed=identity["seed"],step=t,loss=old["loss"],components=component,gradient_cosines_raw=cosines,raw_negative_gradient=gu.detach().neg().tolist(),clipped_negative_gradient=gclip.neg().tolist(),actual_du=du.tolist() if du is not None else None,gradient_vs_actual_du_cosine=old_a.cosine(-gu.detach(),du) if du is not None else None,linear_endpoint_x_response=float((J@du)[6]) if du is not None else None,actual_endpoint_x_change=rows[t+1]["endpoint"][0]-old["endpoint"][0] if du is not None else None,next_loss_change=rows[t+1]["loss"]-old["loss"] if du is not None else None,next_Hard=rows[t+1]["hard"]["passed"] if du is not None else None,Hard=old["hard"]["passed"],TaskSuccess=old["Task"]["TaskSuccess"],UNKNOWN=old["UNKNOWN_count"],active=dict(C_viol=old["metrics"]["C_viol"]>0,F_speed=old["metrics"]["F_speed"]>0,F_acceleration=old["metrics"]["F_acceleration"]>0,F_jerk=old["metrics"]["F_jerk"]>0,F_domain=old["metrics"]["F_domain"]>0,G_task=old["metrics"]["G_task_train"]>0),physical_maxima=old["hard"]["dynamic_polynomial_maxima"],theta_Jacobian_diagonal=J.diag().tolist())
            records.append(rec)
    return records
