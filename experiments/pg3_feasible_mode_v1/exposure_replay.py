"""Read-only historic eligibility replay, exact slot-constrained coverage upper bounds."""
import copy,csv,json,math,subprocess,collections
from pathlib import Path
import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from query_minco.training.io import read,write,sha,artifact,require
from query_minco.training.run_spec import ROOT,source_files
from query_minco.training.pg1.spec import qualify_bindings
from query_minco.trajectory.parameter_decoder import PhysicalParameterDecoder
from query_minco.training.pg3_feasible_mode_v1.assignment import assign
from query_minco.training.pg3_feasible_mode_v1.exposure_assignment import ExposureRotation
from .fulltrain128 import R as HISTORY
OUT=ROOT/"research_outputs/pg3_feasible_mode_v1/exposure_aware001"

def csv_write(path,rows):
    require(bool(rows),"EXPOSURE_EMPTY_TABLE")
    with path.open("x",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def create_binding():
    old=read(HISTORY/"preparation/CPU0.json")
    for name,s in old["code"]["files"].items():
        require(sha(ROOT/name)==s,"EXPOSURE_LEGACY_SOURCE_DRIFT_"+name)
    for name,s in old["scripts"].items():require(sha(ROOT/name)==s,"EXPOSURE_LEGACY_SCRIPT_DRIFT")
    b=read(artifact(old["bindings"]))
    b["code"]=dict(git_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
                   files=source_files(),allow_uncommitted_sources=True)
    # New exact research-source identity; historical bindings/receipts remain untouched.
    write(OUT/"BINDINGS.json",b)
    ref=dict(path=str(OUT/"BINDINGS.json"),sha256=sha(OUT/"BINDINGS.json"))
    ds,geo,init,q=qualify_bindings(ref,42)
    write(OUT/"QUALIFICATION.json",q)
    return old,ref,ds,geo,init

def verify_seal(path):
    seal=read(path/"RUN_SEAL.json");count=0
    for name,value in seal["files"].items():
        require(sha(path/name)==value,"EXPOSURE_HISTORICAL_SEAL_"+name);count+=1
    return dict(path=str(path),sha256=sha(path/"RUN_SEAL.json"),verified_files=count,status=seal.get("status"))

def upper_bound(events,queries):
    cols=[]
    for e in events:
        for _ in range(e["slots"]):cols.append(e["eligible"])
    if not cols:return 0
    a=np.array([[int(q in col) for col in cols] for q in queries],dtype=np.int8)
    if not a.size:return 0
    return int(np.count_nonzero(maximum_bipartite_matching(csr_matrix(a),perm_type="column")>=0))

def audit():
    old,ref,ds,geo,init=create_binding()
    order=read(artifact(old["order128"]))
    index=read(artifact(old["train_index"]))
    states={s.sample_id:s.state for s in ds.load(list(ds.rows))}
    require(len(states)==384 and sum(len(x["teachers"]) for x in index.values())==672,"EXPOSURE_DATA_COUNTS")
    base={ (x["sample_id"],int(x["query"])):x for x in csv.DictReader((HISTORY/"baseline/training384_step0000.csv").open())}
    results={};all_rows=[];upper_rows=[];seals={}
    for arm in ("B","C"):
        run=HISTORY/("runs/A42_"+arm+"_FullFrozen384_CPU128_run001")
        seals[arm]=verify_seal(run);policy=ExposureRotation()
        counts={sid:[dict(exposures=0,eligible=0,selected=0,matched=0,safe=0,nonzero_mode=0,
                         wait=0,longest=0,reasons=collections.Counter()) for _ in range(6)] for sid in states}
        events=collections.defaultdict(list);teacher_use=collections.Counter();old_rows=0
        for step in range(128):
            folder=run/("commits/step%04d"%(step+1));u=read(folder/"UPDATE.json")
            require(u["ids"]==order["batches"][step] and u["index"]==step,"EXPOSURE_ORDER")
            require(u["lr"]==__import__("experiments.pg3_feasible_mode_v1.fulltrain128",fromlist=["lr_at"]).lr_at(step),"EXPOSURE_LR")
            arrpath=run/("intents/step%04d_ARRAYS.npz"%(step+1))
            started=read(run/("intents/step%04d_STARTED.json"%(step+1)))
            require(sha(arrpath)==started["arrays_sha256"],"EXPOSURE_ARRAY_SHA")
            with np.load(arrpath,allow_pickle=False) as z:
                raw=torch.from_numpy(z["raw"].copy())
                dec=PhysicalParameterDecoder()(raw,torch.stack([states[s] for s in u["ids"]]))
                inner=z["inner"].copy()
            require(np.array_equal(inner,dec.inner_pts.numpy()),"EXPOSURE_RAW_INNER")
            seen={}
            for b,sid in enumerate(u["ids"]):
                a=u["assignments"][b];h=torch.tensor(u["hard_pass"][b]);ts=index[sid]["teachers"]
                matches=dict(queries=[q for q in range(6) if a["matched"][q]],
                             teachers=[a["route_teacher_id"][q] for q in range(6) if a["matched"][q]])
                check=assign(dec.inner_pts[b],dec.tail_pva[b,:,0],matches,h,ts,step,2)
                for key in ("selected","matched","route_teacher_id","occupancy","reasons"):
                    require(check[key]==a[key],"EXPOSURE_ORIGINAL_ASSIGNMENT_REPLAY_"+key)
                pa=policy.choose(sid,dec.inner_pts[b],dec.tail_pva[b,:,0],matches,h,ts,None,None,None,step,2)
                if sid in seen:
                    require(seen[sid]==a,"EXPOSURE_DUPLICATE_HISTORICAL_ASSIGNMENT");continue
                seen[sid]=a;old_rows+=1
                slots=sum(2-sum(int(m)==j for m in matches["teachers"]) for j,t in enumerate(ts) if t["certified"] and t["representable"])
                eligible=[q for q in range(6) if not a["matched"][q] and not bool(h[q])]
                events[sid].append(dict(step=step,eligible=eligible,slots=slots))
                for q,c in enumerate(counts[sid]):
                    c["exposures"]+=1;c["eligible"]+=int(q in eligible);c["selected"]+=int(q in a["selected"])
                    c["matched"]+=int(a["matched"][q]);c["safe"]+=int(h[q])
                    c["reasons"][a["reasons"][q]]+=1
                    if q in eligible:
                        c["wait"]=0 if q in a["selected"] else c["wait"]+1;c["longest"]=max(c["longest"],c["wait"])
                    else:c["wait"]=0
                    c["nonzero_mode"]+=int(q in a["selected"] and u["components"]["mode"]["raw_per_query_norm"][b][q]>0)
                for q in pa["selected"]:teacher_use[(sid,pa["route_teacher_id"][q])]+=1
            policy.commit(step)
        final={(x["sample_id"],int(x["query"])):x for x in csv.DictReader((run/"evaluation/training384_step0128.csv").open())}
        rr=[]
        for sid,cs in counts.items():
            for q,c in enumerate(cs):
                b=base[sid,q];f=final[sid,q];hist=policy.state[sid]
                initial_bad=b["matched"]=="False" and b["hard_pass"]=="False"
                oldinner=np.array(json.loads(b["inner"]));newinner=np.array(json.loads(f["inner"]))
                reason=("SELECTED" if c["selected"] else "STATE_CHANGED_NO_ELIGIBLE_EXPOSURE" if not c["eligible"] else
                        "NO_AVAILABLE_SLOT" if not any(q in e["eligible"] and e["slots"] for e in events[sid]) else
                        "GLOBAL_ROTATION_MISSED_REPLAY_COVERS" if hist["count"][q] else
                        "FINITE_EXPOSURE_CAPACITY_COMPETITION")
                rr.append(dict(arm=arm,sample_id=sid,query=q,M=int(b["M"]),regime=b["regime"],initial_bad=initial_bad,
                    exposures=c["exposures"],eligible=c["eligible"],matched_exposures=c["matched"],safe_exposures=c["safe"],
                    historical_count=c["selected"],nonzero_direct_mode_gradient=c["nonzero_mode"],replay_count=hist["count"][q],
                    historical_max_wait=c["longest"],replay_max_wait=hist["longest"][q],reason=reason,
                    historical_reasons=json.dumps(c["reasons"],sort_keys=True),teacher_certified=all(t["certified"] for t in index[sid]["teachers"]),
                    final_matched=f["matched"],final_hard_pass=f["hard_pass"],inner_shift_m=float(np.linalg.norm(newinner-oldinner)),
                    first_inner_shift_m=float(np.linalg.norm(newinner[0]-oldinner[0])),second_inner_shift_m=float(np.linalg.norm(newinner[1]-oldinner[1]))))
        for sid,ev in events.items():
            qq=[q for q in range(6) if base[sid,q]["matched"]=="False" and base[sid,q]["hard_pass"]=="False"]
            bound=upper_bound(ev,qq);eligible_set=set(q for e in ev if e["slots"] for q in e["eligible"])&set(qq)
            actual=sum(counts[sid][q]["selected"]>0 for q in qq);replay=sum(policy.state[sid]["count"][q]>0 for q in qq)
            require(actual<=bound and replay<=bound,"EXPOSURE_UPPER_BOUND")
            upper_rows.append(dict(arm=arm,sample_id=sid,M=index[sid]["M"],exposures=len(ev),initial_bad=len(qq),
                ever_eligible_with_slot=len(eligible_set),max_unique_bound=bound,historical_unique=actual,replay_unique=replay,
                state_or_no_teacher_exclusion=len(qq)-len(eligible_set),finite_exposure_capacity_shortfall=len(eligible_set)-bound,
                global_rotation_extra_missed=bound-actual,replay_residual=bound-replay,
                per_exposure_extra_slot_counts=json.dumps([e["slots"] for e in ev])))
        all_rows+=rr
        summary={}
        for M in (1,4):
            xs=[x for x in rr if x["M"]==M and x["initial_bad"]];us=[u for u in upper_rows if u["arm"]==arm and u["M"]==M]
            summary[str(M)]=dict(initial_bad=len(xs),historical_covered=sum(x["historical_count"]>0 for x in xs),
                replay_covered=sum(x["replay_count"]>0 for x in xs),upper_bound=sum(x["max_unique_bound"] for x in us),
                state_or_no_teacher_exclusion=sum(x["state_or_no_teacher_exclusion"] for x in us),
                finite_exposure_capacity_shortfall=sum(x["finite_exposure_capacity_shortfall"] for x in us),
                global_rotation_extra_missed=sum(x["global_rotation_extra_missed"] for x in us))
        bins={}
        for label,test in [("1",lambda e:e==1),("2",lambda e:e==2),("3-4",lambda e:3<=e<=4),("5+",lambda e:e>=5)]:
            for M in (1,4):
                xs=[x for x in rr if x["initial_bad"] and x["M"]==M and test(x["exposures"])]
                bins[str(M)+"/"+label]=dict(queries=len(xs),historical=sum(x["historical_count"]>0 for x in xs),replay=sum(x["replay_count"]>0 for x in xs))
        write(OUT/(arm+"_REPLAY_STATE.json"),policy.state_dict())
        results[arm]=dict(strata=summary,exposure_bins=bins,row_exposures=1024,unique_sample_updates=old_rows,
            duplicates_in_same_update=1024-old_rows,initial_bad_queries=sum(x["initial_bad"] for x in rr),
            original_solver_exact_replay=True,
            replay_available_teacher_ids=len(teacher_use),historical_result_not_replaced=True)
    csv_write(OUT/"HISTORICAL_QUERY_EXPOSURE.csv",all_rows)
    csv_write(OUT/"CAPACITY_UPPER_BOUNDS.csv",upper_rows)
    b=results["B"]["strata"];n=sum(s["initial_bad"] for s in b.values());gain=sum(s["replay_covered"]-s["historical_covered"] for s in b.values())
    gate=dict(coverage_gain=gain,minimum_preregistered_gain=math.ceil(.01*n),passed=gain>=math.ceil(.01*n),
              meaning=">=1 percentage point initial bad Query coverage; replay is NOT trained performance")
    write(OUT/"REPLAY_RESULT.json",dict(arms=results,gate=gate,historical_seals=seals,bindings=ref,initialization=init,
          optimizer_steps_executed=0,semantics="OFFLINE_ASSIGNMENT_REPLAY_ON_FROZEN_HISTORICAL_ELIGIBILITY"))
    print("REPLAY_COMPLETE",json.dumps(dict(arms=results,gate=gate)),flush=True)

if __name__=="__main__":audit()
