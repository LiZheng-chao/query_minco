"""Read-only attribution of sealed V2.1 artifacts; no optimizer or weight writes."""
import argparse,csv,json,collections,itertools
from pathlib import Path
import numpy as np
from query_minco.training.io import read,write,sha,require
from query_minco.evaluation.config import CONFIG
from query_minco.evaluation.validation import evaluate
from query_minco.training.pg3_v2_no_corridor.executor_v21 import inspect_chain
STEPS=(0,32,64,96,128)
SPLITS=("training384","development192")
def table(path):
    with Path(path).open() as f:return list(csv.DictReader(f))
def save_csv(path,rows):
    require(bool(rows),"ATTR_EMPTY_TABLE")
    with Path(path).open("x",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def truth(r,k="hard_pass"):return r[k]=="True"
def taxonomy(row):
    reasons=row["hard_reasons"].split(";") if row["hard_reasons"] else []
    result=[];clear=float(row["signed_clearance"]) if row["signed_clearance"] else None
    for reason in reasons:
        if reason=="collision_or_required_margin_violation":result.append("Collision" if clear is not None and clear<0 else "Margin")
        elif "speed" in reason or "velocity" in reason:result.append("Velocity")
        elif "acceleration" in reason:result.append("Acceleration")
        elif "jerk" in reason:result.append("Jerk")
        elif "domain" in reason or "boundary" in reason or "continuity" in reason:result.append("Domain")
        else:result.append("Other")
    return sorted(set(result))
def verify_evidence(base):
    inventory={};missing=[]
    for name in ["PAIRED128_RESULT.json","MONITOR_AGGREGATES.json","ORACLE_TASK_TRANSITIONS.json","Q3_FROZEN_EVAL_TRACE.json","EXPOSURE_PER_QUERY.csv"]:
        path=base/"summary"/name
        if not path.exists():missing.append(str(path))
        else:inventory[str(path)]=sha(path)
    for arm in ("A","D"):
        run=base/"runs"/("A42_"+arm+"_PG3V21_formal_run001")
        seal=read(run/"RUN_SEAL.json")
        for name,h in seal["files"].items():
            f=run/name
            if not f.exists():missing.append(str(f));continue
            require(sha(f)==h,"ATTR_SEALED_MEMBER_CHANGED:"+str(f));inventory[str(f)]=h
        if not missing:inspect_chain(run,read(run/"RUN_SPEC.json"))
    return dict(missing=missing,files=inventory,status="PASS" if not missing else "BLOCKED")
def summarize_gradients(records):
    components={};pairs=collections.defaultdict(list);direction=collections.defaultdict(list)
    for r in records:
        audit=r["audit"]
        for name,c in audit["components"].items():
            entry=components.setdefault(name,dict(norm=[],weighted_norm=[],cosine_imitation=[],modules=collections.defaultdict(list)))
            entry["norm"].append(c["parameter_norm"]);entry["weighted_norm"].append(c["weighted_norm"])
            if c["cosine_imitation"] is not None:entry["cosine_imitation"].append(c["cosine_imitation"])
            for module,value in c["parameter_groups"].items():entry["modules"][module].append(value)
        names=audit["component_order"]
        for i,name in enumerate(names):
            for j in range(i):
                c=audit["cosine"][i][j]
                if c is not None:pairs[name+" / "+names[j]].append(c)
        for name,v in r["delta_cosine_negative"].items():
            if v is not None:direction[name].append(v)
    def stats(v):return dict(count=len(v),mean=float(np.mean(v)),median=float(np.median(v)),min=float(min(v)),max=float(max(v)),negative=sum(x<0 for x in v)) if v else None
    return dict(components={k:dict(norm=stats(v["norm"]),weighted_norm=stats(v["weighted_norm"]),cosine_imitation=stats(v["cosine_imitation"]),modules={m:stats(vals) for m,vals in v["modules"].items()}) for k,v in components.items()},cosines={k:stats(v) for k,v in pairs.items()},actual_adam_delta_vs_negative_gradient={k:stats(v) for k,v in direction.items()},max_ratios={k:max(r["audit"]["ratios"][k] for r in records) for k in records[0]["audit"]["ratios"]},learnable_queries="Included in query_decoder aggregate in historical audit; separated only by optional frozen replay, not inferred from aggregate",causality="Norm/cosine and output drift are associations, not proof a component causes long-run harm")
def analyze(base,out):
    base=Path(base).resolve();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    evidence=verify_evidence(base);write(out/"EVIDENCE_INVENTORY.json",evidence);require(not evidence["missing"],"ATTR_MISSING_EVIDENCE")
    runs={a:base/"runs"/("A42_"+a+"_PG3V21_formal_run001") for a in ("A","D")}
    records={a:[read(runs[a]/"commits"/("step%04d"%i)/"UPDATE.json") for i in range(1,129)] for a in runs}
    occurrences={a:collections.defaultdict(list) for a in runs}
    for arm,rr in records.items():
        for r in rr:
            for b,sid in enumerate(r["ids"]):
                decision=r["exposure_receipt"].get("decisions",{}).get(sid,{}).get("decision",{})
                for q in range(6):
                    gr=[g for g in r["audit"]["geometry"] if g["batch"]==b and g["query"]==q]
                    kd=[g for g in r["audit"].get("diagnostics",{}).get("keep",{}).get("drift",[]) if g["sample_id"]==sid and g["query"]==q]
                    keep=r["audit"]["components"]["keep"];rawkeep=keep["raw_per_query"]
                    occurrences[arm][(sid,q)].append(dict(update=r["completed"],batch=b,mode_selected=q in decision.get("selected",[]),geom=gr[0]["geom_source"] if gr else "NONE",keep_active=kd[0]["active"] if kd else False,weighted_keep_raw_norm=0. if rawkeep is None else abs(keep["effective_weight"])*rawkeep[b][q],whole_batch_weighted_keep_parameter_norm=keep["weighted_norm"]))
    snapshots={};arrays={}
    for split,arm,step in itertools.product(SPLITS,runs,STEPS):
        path=runs[arm]/"evaluation"/(split+"_step%04d"%step)
        snapshots[split,arm,step]={(r["sample_id"],int(r["query"])):r for r in table(str(path)+".csv")}
        with np.load(str(path)+".npz") as archive:arrays[split,arm,step]={k:archive[k] for k in archive.files}
    transitions=[];allsets={};oracle_cases=[];exposure=[];failure=collections.Counter();q3=[]
    for split in SPLITS:
        initial=snapshots[split,"A",0];require(initial==snapshots[split,"D",0],"ATTR_INIT_CSV_MISMATCH")
        ids=arrays[split,"A",0]["ids"].tolist();index={sid:i for i,sid in enumerate(ids)}
        oracle={}
        for arm,step in itertools.product(runs,STEPS):
            oracle[arm,step]={sid:any(truth(snapshots[split,arm,step][sid,q],"complete") for q in range(6)) for sid in ids}
        sets={k:[] for k in ("D_unique_regression","D_unique_repair","common_regression","Oracle_lost_tasks")}
        for sid,q in initial:
            i=initial[sid,q];fa=snapshots[split,"A",128][sid,q];fd=snapshots[split,"D",128][sid,q]
            group="D_unique_regression" if truth(i) and truth(fa) and not truth(fd) else "D_unique_repair" if not truth(i) and not truth(fa) and truth(fd) else "common_regression" if truth(i) and not truth(fa) and not truth(fd) else "other"
            direct_final=0
            for arm,step in itertools.product(runs,STEPS):
                row=snapshots[split,arm,step][sid,q];npz=arrays[split,arm,step];j=index[sid]
                c=npz["coeffs"][j,q];d=npz["durations"][j,q];c0=arrays[split,"A",0]["coeffs"][j,q];d0=arrays[split,"A",0]["durations"][j,q]
                require(npz["ids"][j]==sid,"ATTR_ARRAY_ID")
                inner=np.array(json.loads(row["inner"]));inner0=np.array(json.loads(i["inner"]))
                # Repeated occurrences are retained in event logs; committed direct exposure is checkpoint authority.
                events=[e for e in occurrences[arm][sid,q] if e["update"]<=step] if split=="training384" else []
                history=read_exposure(runs[arm],step);counts=history.get(sid,[{}]*6)[q] if split=="training384" else {}
                clear=float(row["signed_clearance"]) if row["signed_clearance"] else None
                x=dict(split=split,sample_id=sid,query_id=q,arm=arm,step=step,set=group,initial_hard_pass=truth(i),hard_pass=truth(row),failure_types=";".join(taxonomy(row)),hard_reasons=row["hard_reasons"],signed_clearance_m=clear,sampled_margin_gap_m=None if clear is None else clear-CONFIG["required_surface_margin_m"],clearance_delta_m=None if clear is None else clear-float(i["signed_clearance"]),initial_matched=truth(i,"matched"),current_matched=truth(row,"matched"),initial_oracle=oracle[arm,0][sid],oracle=oracle[arm,step][sid],initial_unique_safe=sum(truth(initial[sid,qq],"complete") for qq in range(6))==1,initial_complete=truth(i,"complete"),complete=truth(row,"complete"),head_position=json.dumps(evaluate(c[0],0).tolist()),head_displacement_m=float(np.linalg.norm(evaluate(c[0],0)-evaluate(c0[0],0))),inner=json.dumps(inner.tolist()),inner_displacement_m=json.dumps(np.linalg.norm(inner-inner0,axis=-1).tolist()),terminal_position=json.dumps(evaluate(c[-1],d[-1]).tolist()),terminal_position_displacement_m=float(np.linalg.norm(evaluate(c[-1],d[-1])-evaluate(c0[-1],d0[-1]))),terminal_velocity=json.dumps(evaluate(c[-1],d[-1],1).tolist()),terminal_velocity_displacement_mps=float(np.linalg.norm(evaluate(c[-1],d[-1],1)-evaluate(c0[-1],d0[-1],1))),durations_s=json.dumps(d.tolist()),duration_delta_s=json.dumps((d-d0).tolist()),duration_l1_change_s=float(np.abs(d-d0).sum()),mode_direct_committed=counts.get("direct",0),eligible_committed=counts.get("eligible",0),mode_selected_occurrences=sum(e["mode_selected"] for e in events),geom_sources=json.dumps(dict(collections.Counter(e["geom"] for e in events))),keep_active_training_occurrences=sum(e["keep_active"] for e in events),weighted_keep_raw_norm_max=max([e["weighted_keep_raw_norm"] for e in events] or [0.]),supervision_scope="training occurrences up to checkpoint; Keep norm is local raw gradient, not per-query parameter norm" if split=="training384" else "EVALUATION_ONLY: no direct training supervision; zero is not a measured gradient on this sample",training_events=json.dumps(events))
                transitions.append(x);failure[split+"/"+arm+"/"+str(step)+"/"+(x["failure_types"] or "PASS")]+=1
                if sid=="f00_narrow3d_g5_v1" and q==3:q3.append(x)
                if step==128:
                    exposure.append({k:x[k] for k in ("split","sample_id","query_id","arm","set","initial_hard_pass","hard_pass","initial_matched","mode_direct_committed","eligible_committed","keep_active_training_occurrences","weighted_keep_raw_norm_max","clearance_delta_m")})
            detail=dict(split=split,sample_id=sid,query_id=q,initial_hard_pass=truth(i),A_hard_pass=truth(fa),D_hard_pass=truth(fd),initial_clearance=i["signed_clearance"],A_clearance=fa["signed_clearance"],D_clearance=fd["signed_clearance"],A_reasons=fa["hard_reasons"],D_reasons=fd["hard_reasons"],initial_matched=truth(i,"matched"),D_training_events=occurrences["D"][sid,q] if split=="training384" else "EVALUATION_ONLY")
            if group in sets:sets[group].append(detail)
            if oracle["A",128][sid] and not oracle["D",128][sid]:oracle_cases.append(dict(detail,initial_unique_safe=sum(truth(initial[sid,qq],"complete") for qq in range(6))==1))
        sets["Oracle_lost_tasks"]=sorted({r["sample_id"] for r in oracle_cases if r["split"]==split});allsets[split]=sets
    save_csv(out/"QUERY_TRANSITIONS.csv",transitions);save_csv(out/"ORACLE_LOSS_CASES.csv",oracle_cases);save_csv(out/"EXPOSURE_VS_REPAIR.csv",exposure)
    write(out/"FOCUS_SETS.json",allsets);write(out/"Q3_TRACE.json",q3)
    write(out/"FAILURE_TAXONOMY.json",dict(counts=dict(failure),surface_margin_m=CONFIG["required_surface_margin_m"],body_radius_m=CONFIG["body_radius_m"],Collision="negative sampled signed body-surface clearance in a collision_or_required_margin_violation; magnitude remains in CSV",Margin="nonnegative sampled signed clearance with original margin rejection",Other="unproven certificates and any unclassified original reason; never relabeled as collision",caution="4097-node clearance is a diagnostic approximation, not validator authority or continuous safety proof; original conservative validator hard_pass is preserved"))
    write(out/"GRADIENT_CONFLICT_SUMMARY.json",{arm:summarize_gradients(rr) for arm,rr in records.items()})
    report=["# Safety regression attribution", "", "Verified source artifacts: complete, sealed SHA checks and129-member A/D checkpoint chains PASS. No training or archived weight mutation in this analysis.", "", "## Verified facts", ""]
    for split,sets in allsets.items():
        report += ["### "+split,"", "D-unique regressions%d; D-unique repairs%d; common regressions%d. Oracle lost tasks: %s."%(len(sets["D_unique_regression"]),len(sets["D_unique_repair"]),len(sets["common_regression"]),", ".join(sets["Oracle_lost_tasks"])),"", "| Set | Sample | Query | Initial/A/D signed clearance m | D failure |", "|---|---|---:|---|---|"]
        for name in ("D_unique_regression","D_unique_repair","common_regression"):
            for row in sets[name]:report.append("| %s | %s | %s | %s / %s / %s | %s |"%(name,row["sample_id"],row["query_id"],row["initial_clearance"],row["A_clearance"],row["D_clearance"],row["D_reasons"]))
        report.append("")
    report += ["## Q3: separate first-step and final behavior","", "Identical first-batch A/D both regress Q3. This is not evidence of a PG3-exclusive initial effect. Full frozen checkpoints show A128 PASS and D128 FAIL. Q3_TRACE.json records head/waypoint/endpoint/velocity/duration changes, raw Keep gradients and exact training occurrences. Q3 is initially matched-safe; mode routing on a different dangerous query does not give Q3 direct Mode supervision. Shared model parameters still change its output.", "", "## Associations and unresolved causal hypotheses","", "Full D repairs more initially failed candidates but loses more initially safe candidates and tasks. Per-component Gram/cosines and Adam directions show conflicts; a negative cosine alone cannot identify harmful long-run effects. Historical module norms combine Learnable Queries into query_decoder; this report does not invent a separate historical module measurement.", "", "Evaluation-only development tasks have no direct Mode/Geom/Keep training exposure. Their outputs change through shared parameters trained on Training384. Which of Stop replacement, Mode, Geom or Keep drives final task loss requires S/M/G/K incremental ablation; no attribution from the combined D alone.", "", "Surface clearance values are4097-node diagnostics. Original hard flags, numeric tolerances and safety constraints remain authority. Collision and Margin taxonomy is diagnostic, not a new validator.", "", "QUERY_TRANSITIONS.csv has all3456 (split,sample,query) keys at both arms and five checkpoints. Training event logs distinguish selected supervision, branch source, unique committed exposure and Keep raw norm; absence from a batch is not evidence of zero shared-parameter influence.", "", "Conclusions apply only to Seed42, this static distribution and128 updates. Development192 is not a blind test. Offline Oracle@6 is not selector success. See local replay supplement if available; local counterfactuals are not full-training causal effects."]
    text="\n".join(report)+"\n";(out/"ATTRIBUTION_REPORT.md").write_text(text);(out/"SAFETY_REGRESSION_ATTRIBUTION.md").write_text(text)
    write(out/"RESULT.json",dict(status="COMPLETED",query_checkpoint_rows=len(transitions),oracle_loss_query_rows=len(oracle_cases),sets={k:{n:len(v) for n,v in sets.items()} for k,sets in allsets.items()},training="NOT_RUN",archived_weights_changed=False))
    print(read(out/"RESULT.json"))
_EXPOSURE={}
def read_exposure(run,step):
    key=(str(run),step)
    if key not in _EXPOSURE:
        import torch
        _EXPOSURE[key]=torch.load(run/"commits"/("step%04d"%step)/"checkpoint.pt",map_location="cpu",weights_only=True)["exposure"]["history"]
    return _EXPOSURE[key]
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--base",default="research_outputs/pg3_v21_paired128_v1");p.add_argument("--output",default="research_outputs/pg3_v21_attribution_v1");a=p.parse_args();analyze(a.base,a.output)
