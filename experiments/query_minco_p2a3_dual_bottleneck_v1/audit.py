"""P2-A3 read-only raw-log and checkpoint audit. Selection never uses TaskSuccess."""
from pathlib import Path
import json,csv,collections
import numpy as np
import torch
from experiments.query_minco_p2a_quality_diagnosis_v1.quality import f,parts
BASE=Path(__file__).resolve().parents[2]
MAIN=BASE/"research_outputs/query_minco_p2a_quality_diagnosis_v1/runs/quality_main_manual_001"
SELECTORS=("fixed16","fixed32","fixed64","last_Hard","min_L0_Hard","min_q_Hard")
def read(p):return json.loads(Path(p).read_text())
def choose(rows,name):
    if name.startswith("fixed"):
        index=int(name[5:]);return rows[index] if index<len(rows) else None
    qualified=[r for r in rows if r["hard"]["passed"]]
    if not qualified:return None
    if name=="last_Hard":return max(qualified,key=lambda r:r["step"])
    if name=="min_L0_Hard":return min(qualified,key=lambda r:(r["loss"],r["step"]))
    if name=="min_q_Hard":return min(qualified,key=lambda r:(r["q_alpha"],r["step"]))
    raise ValueError("SELECTOR")
def collect():
    result=read(MAIN/"RESULT.json")
    assert result["status"]=="PASS" and result["trials"]==54 and result["AdamW_proposals"]==3456 and result["applied_nonzero_moves"]==2489 and result["network_updates"]==0
    data={};states=[];rejects=0;nonzero=0
    for p in sorted(MAIN.iterdir()):
        if not p.is_dir() or not (p/"STEPS.jsonl").exists():continue
        ident=read(p/"IDENTITY.json");rows=[json.loads(s) for s in (p/"STEPS.jsonl").read_text().splitlines()]
        key=(ident["case"],ident["seed"],ident["arm"])
        if key in data or len(rows)!=65 or [r["step"] for r in rows]!=list(range(65)):raise ValueError("STEP_IDENTITY")
        for i,r in enumerate(rows):
            if r["alpha"]!=8 or r["experiment_arm"]!=ident["arm"]:raise ValueError("ALPHA_ARM")
            q=.1*r["metrics"]["C_shape"]+.2*r["metrics"]["P"]+.1*r["metrics"]["S"]
            if abs(q-r["q_alpha"])>1e-6*(1+abs(q)):raise ValueError("Q_RECONSTRUCTION")
            if i<64:
                expected=np.asarray(r["proposal_u"] if r["accepted"] else r["u"],dtype=np.float32)
                if not np.array_equal(expected,np.asarray(rows[i+1]["u"],dtype=np.float32)):raise ValueError("ACTUAL_UPDATE")
                if not r["accepted"]:rejects+=1
                nonzero+=bool(any(r["actual_du"]))
                if ident["arm"]=="L2" and r["hard"]["passed"]:
                    if not rows[i+1]["hard"]["passed"] or rows[i+1]["q_alpha"]>r["q_alpha"]:raise ValueError("FILTER_CONTRACT")
            if r["hard"]["passed"] and r["UNKNOWN_count"]:raise ValueError("HARD_UNKNOWN")
            states.append(dict(case=key[0],seed=key[1],arm=key[2],step=i,Hard=r["hard"]["passed"],TaskSuccess=r["Task"]["TaskSuccess"],L0=r["loss"],optimization_scalar=r["optimization_scalar"],q_alpha=r["q_alpha"],vmax=r["hard"]["dynamic_polynomial_maxima"][0],amax=r["hard"]["dynamic_polynomial_maxima"][1],jmax=r["hard"]["dynamic_polynomial_maxima"][2],progress=r["hard"]["goal"]["distance_reduction_m"],goal_error=r["hard"]["goal"]["endpoint_goal_error_m"],UNKNOWN=r["UNKNOWN_count"],sample_clearance=r["known_sample_clearance_m"],certified_surface_clearance_lower_bound=r["certified_surface_clearance_lower_bound_m"],accepted=r.get("accepted"),rejection_reason=r.get("filter_reason"),nonzero_motion=bool(any(r.get("actual_du",[]))),reasons=r["hard"]["reasons"]))
        data[key]=dict(identity=ident,path=str(p),rows=rows)
    expected={(c,s,a) for c in ("free_near","native_slit_cross_far_v1","native_two_portals_cross_far_v1") for s in [None]+list(range(20261109,20261114)) for a in ("L0","L1","L2")}
    if set(data)!=expected or rejects!=967 or nonzero!=2489:raise ValueError("MAIN_COUNTS")
    checkpoints=list(csv.DictReader((MAIN/"P2A_MAIN_RESULTS.csv").open()))
    if len(checkpoints)!=216:raise ValueError("CHECKPOINT_CSV")
    for r in checkpoints:
        key=(r["case"],None if r["seed"]=="" else int(r["seed"]),r["arm"]);old=data[key]["rows"][int(r["step"])]
        for col,actual in [("loss0",old["loss"]),("optimization_scalar",old["optimization_scalar"]),("q_alpha",old["q_alpha"]),("progress",old["hard"]["goal"]["distance_reduction_m"]),("goal_error",old["hard"]["goal"]["endpoint_goal_error_m"])]:
            if abs(float(r[col])-actual)>1e-10*(1+abs(actual)):raise ValueError("CSV_NUMERIC")
        if (r["OriginalHardAll"]=="True")!=old["hard"]["passed"] or (r["TaskSuccess"]=="True")!=old["Task"]["TaskSuccess"]:raise ValueError("CSV_HARD_TASK")
    final=list(csv.DictReader((MAIN/"P2A_PILOT_RESULTS.csv").open()))
    if len(final)!=54:raise ValueError("FINAL_CSV")
    for r in final:
        key=(r["case"],None if r["seed"]=="" else int(r["seed"]),r["arm"]);old=data[key]["rows"][64]
        if (r["final_Hard"]=="True")!=old["hard"]["passed"] or (r["final_TaskSuccess"]=="True")!=old["Task"]["TaskSuccess"]:raise ValueError("FINAL_MISMATCH")
    quality=list(csv.DictReader((MAIN/"P2A_JOINT_QUALITY_PAIRED.csv").open()))
    if len(quality)!=216:raise ValueError("JOINT_QUALITY_CSV")
    for r in quality:
        seed=None if r["seed"]=="" else int(r["seed"]);t=int(r["step"])
        a=data[(r["case"],seed,r["arm_A"])]["rows"][t];b=data[(r["case"],seed,r["arm_B"])]["rows"][t]
        eligible=a["hard"]["passed"] and a["Task"]["TaskSuccess"] and b["hard"]["passed"] and b["Task"]["TaskSuccess"]
        if (r["both_joint_feasible"]=="True")!=eligible:raise ValueError("QUALITY_ELIGIBILITY")
    return data,states

def evaluate_selected(data):
    cache={};batches={};logical_pairs=[]
    def fresh(key,t):
        cachekey=str(key)+"__"+str(t)
        if cachekey in cache:return cache[cachekey]
        source=data[key];old=source["rows"][t];sample=f.CASE_SOURCES[key[0]]
        if sample not in batches:batches[sample]=f.load_source(sample)[0]
        batch=batches[sample];u=torch.tensor(old["u"],dtype=torch.float32,requires_grad=True)
        r=f.p1.objective("B0")(f.p1.raw_slots(u),batch);rec=f.p1.summarize(r,batch,u,arm="B0");_,q=parts(r,8.)
        contract=f.native_contract(batch,key[0]) if key[0].startswith("native") else None
        task=f.crossing_success(r,batch,contract,rec["hard"]) if contract else dict(OriginalHardAll=rec["hard"]["passed"],TaskSuccess=rec["hard"]["passed"],successful_channels=[])
        diff=float(np.abs(np.asarray(rec["coefficients"])-np.asarray(old["coefficients"])).max())
        if diff>2e-4 or abs(rec["loss"]-old["loss"])>.003+.003*abs(old["loss"]):raise ValueError("FRESH_NUMERIC_DRIFT")
        if rec["hard"]["passed"]!=old["hard"]["passed"] or rec["hard"]["reasons"]!=old["hard"]["reasons"] or task["TaskSuccess"]!=old["Task"]["TaskSuccess"]:raise ValueError("FRESH_HARD_TASK_DRIFT")
        cache[cachekey]=dict(key=list(key),step=t,original_path=source["path"],full_record=rec,Task=task,q_alpha=float(q),coefficient_max_abs_error=diff)
        return cache[cachekey]
    for case in ("free_near","native_slit_cross_far_v1","native_two_portals_cross_far_v1"):
        for seed in [None]+list(range(20261109,20261114)):
            l1key=(case,seed,"L1");l2key=(case,seed,"L2");sequence=data[l1key]["rows"]
            counter=fresh(l2key,64)
            for selector in SELECTORS:
                picked=choose(sequence,selector)
                rec=fresh(l1key,picked["step"]) if picked is not None else None
                a=rec["full_record"] if rec else None;b=counter["full_record"]
                joint=bool(rec and a["hard"]["passed"] and rec["Task"]["TaskSuccess"] and b["hard"]["passed"] and counter["Task"]["TaskSuccess"])
                logical_pairs.append(dict(case=case,seed=seed,selector=selector,selected_step=picked["step"] if picked is not None else None,
                    selected_Hard=a["hard"]["passed"] if a else False,selected_TaskSuccess=rec["Task"]["TaskSuccess"] if rec else False,
                    selected_L0=a["loss"] if a else None,selected_q_alpha=rec["q_alpha"] if rec else None,
                    selected_progress=a["hard"]["goal"]["distance_reduction_m"] if a else None,selected_goal_error=a["hard"]["goal"]["endpoint_goal_error_m"] if a else None,
                    selected_vaj=a["hard"]["dynamic_polynomial_maxima"] if a else None,selected_S=a["metrics"]["S"] if a else None,selected_path_length=a["path_length_m"] if a else None,
                    selected_sample_clearance=a["known_sample_clearance_m"] if a else None,selected_certified_surface_clearance_bound=a["certified_surface_clearance_lower_bound_m"] if a else None,
                    selected_UNKNOWN=a["UNKNOWN_count"] if a else None,selected_events=rec["Task"].get("events",[]) if rec else [],
                    L2_final_Hard=b["hard"]["passed"],L2_final_TaskSuccess=counter["Task"]["TaskSuccess"],L2_final_q_alpha=counter["q_alpha"],L2_final_progress=b["hard"]["goal"]["distance_reduction_m"],L2_final_S=b["metrics"]["S"],L2_final_path_length=b["path_length_m"],L2_final_sample_clearance=b["known_sample_clearance_m"],
                    quality_comparison="BOTH_JOINT_FEASIBLE" if joint else "NA",delta_S_selected_minus_L2=a["metrics"]["S"]-b["metrics"]["S"] if joint else None,
                    delta_progress_selected_minus_L2=a["hard"]["goal"]["distance_reduction_m"]-b["hard"]["goal"]["distance_reduction_m"] if joint else None,
                    delta_path_selected_minus_L2=a["path_length_m"]-b["path_length_m"] if joint else None,
                    selected_validation_key=str(l1key)+"__"+str(picked["step"]) if picked else None,L2_validation_key=str(l2key)+"__64",
                    selection_uses_TaskSuccess=False,privileged_Hard_offline_only=True,historical_source_proposals_each=64,historical_Hard_candidate_API_calls_each=129,cached_accepted_state_Hard_labels_each=65,extra_Hard_calls_for_selection=0,validation_shared_cache=True))
    return logical_pairs,list(cache.values())
