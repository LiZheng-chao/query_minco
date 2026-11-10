"""Read-only actual-run aggregation; no optimizer or parameter update."""
import csv,json,collections,subprocess,difflib
from pathlib import Path
import numpy as np
import torch
from query_minco.training.io import read,write,sha,artifact,require,digest
from query_minco.training.run_spec import ROOT
from query_minco.training.pg3_feasible_mode_v1.bottleneck import crossing_events
from query_minco.training.pg3_feasible_mode_v1.exposure_assignment import ExposureRotation
from .exposure_replay import OUT,HISTORY,csv_write,verify_seal
from .exposure_experiment import RUN_ID,scripts

def rows(p):return list(csv.DictReader(p.open()))
def yes(x):return x is True or x=="True"
def key(x):return x["sample_id"],int(x["query"])
def qs(xs):
    a=[float(x) for x in xs if x is not None]
    return np.quantile(a,[0,.1,.5,.9,1]).tolist() if a else None
def doc(name,body):
    with (OUT/name).open("x") as f:f.write(body)
def table(headers,rr):
    return "| "+" | ".join(headers)+" |\n|"+"|".join(["---"]*len(headers))+"|\n"+"".join(
        "| "+" | ".join(str(x) for x in r)+" |\n" for r in rr)
def section(c,d,target,scene):
    records=[];deficits=[]
    for b in target["bottlenecks"]:
        if b.get("side") is None:continue
        e=crossing_events(c,d,b["plane_x"])
        if e["status"]!="UNIQUE_FORWARD_CROSSING":
            records.append(dict(status=e["status"],deficit=None));continue
        axis,sgn=b["side"];box=scene["obstacles"][b["box"]];p=e["selected"]["position"]
        boundary=box["max"][axis]+.15 if sgn==1 else box["min"][axis]-.15
        v=max(0.,sgn*(boundary-p[axis]));deficits.append(v)
        records.append(dict(box=b["box"],axis=axis,side=sgn,position=p,boundary=boundary,deficit=v,status=e["status"]))
    return max(deficits) if deficits else None,records

def run():
    rp=OUT/"runs"/RUN_ID;require((rp/"RESULT.json").exists(),"REPORT_TRAIN_NOT_COMPLETE")
    result=read(rp/"RESULT.json");require(result["optimizer_updates"]==128 and result["status"]=="COMPLETED","REPORT_NOT128")
    spec=read(rp/"RUN_SPEC.json");require(scripts()==spec["experiment_sources"],"REPORT_SOURCE")
    control=HISTORY/"runs/A42_B_FullFrozen384_CPU128_run001"
    seals={a:verify_seal(p) for a,p in [("new",rp),("control",control),("C_crosscheck",HISTORY/"runs/A42_C_FullFrozen384_CPU128_run001")]}
    bases={k:{key(x):x for x in rows(HISTORY/"baseline"/(k+"_step0000.csv"))} for k in ("training384","development192")}
    index=read(artifact(spec["train_index"]))
    ev={};metrics=[];trans=[];modes=[];sections=[];quality=[];pair=[];strata=[];deep_rows=[]
    deep={(x["sample_id"],int(x["query"])) for x in read(ROOT/"research_outputs/pg3_feasible_mode_v1/raw_learnability001/DEEP14_INITIAL_GRADIENTS.json")}
    for arm,path in (("Global",control),("Exposure",rp)):
        for step in (0,32,64,96,128):
            for dataset in bases:
                root=HISTORY/"baseline" if step==0 else path/"evaluation"
                rr=rows(root/(dataset+"_step%04d.csv"%step));ev[arm,step,dataset]={key(x):x for x in rr}
                summary=read(root/(dataset+"_step%04d_SUMMARY.json"%step))
                metrics.append(dict(arm=arm,step=step,dataset=dataset,**{k:summary[k] for k in
                    ("physics","goal","complete","oracle","geometric_route_count","mean_qa","mean_qj","mean_time","max_v","max_a","max_j")}))
                groups=collections.defaultdict(list)
                with np.load(root/(dataset+"_step%04d.npz"%step),allow_pickle=False) as ar:
                    ix={str(s):i for i,s in enumerate(ar["ids"])}
                    for x in rr:
                        sid,q=key(x);b=bases[dataset][sid,q];groups[sid].append(x)
                        bad=not yes(b["matched"]) and not yes(b["hard_pass"])
                        trans.append(dict(arm=arm,dataset=dataset,step=step,sample_id=sid,query=q,M=int(x["M"]),regime=x["regime"],
                            initial_matched=b["matched"],current_matched=x["matched"],initial_pass=b["hard_pass"],hard_pass=x["hard_pass"],
                            hard_reasons=x["hard_reasons"],goal=x["goal_pass"],complete=x["complete"],initial_bad=bad,
                            fail_to_pass=not yes(b["hard_pass"]) and yes(x["hard_pass"]),
                            pass_to_fail=yes(b["hard_pass"]) and not yes(x["hard_pass"]),
                            clearance=x["signed_clearance"],clearance_delta=float(x["signed_clearance"])-float(b["signed_clearance"]) if x["signed_clearance"] and b["signed_clearance"] else None,
                            inner_shift_m=float(np.linalg.norm(np.array(json.loads(x["inner"]))-np.array(json.loads(b["inner"])))),
                            bottleneck_error=x["bottleneck_error"],qa=x["qa"],qj=x["qj"],time=x["time"],
                            vmax=x["vmax"],amax=x["amax"],jmax=x["jmax"],goal_progress=x["goal_progress"],mode_potential=x["mode_loss_potential"]))
                        if dataset=="development192" and (sid,q) in deep:deep_rows.append(dict(x,arm=arm))
                        if dataset=="training384" and bad:
                            target=index[sid]["teachers"][int(b["fixed_initial_teacher"])]
                            deficit,detail=section(ar["coeffs"][ix[sid],q],ar["durations"][ix[sid],q],target,index[sid]["scene"])
                            sections.append(dict(arm=arm,step=step,sample_id=sid,query=q,M=int(x["M"]),fixed_teacher=b["fixed_initial_teacher"],
                                section_deficit_m=deficit,events=json.dumps(detail,separators=(",",":")),hard_pass=x["hard_pass"]))
                for sid,xx in groups.items():
                    a=np.array([json.loads(x["inner"]) for x in xx]).reshape(6,-1)
                    spread=[float(np.linalg.norm(a[i]-a[j])) for i in range(6) for j in range(i)]
                    routes={x["route_signature"] for x in xx if x["route_signature"]}
                    modes.append(dict(arm=arm,step=step,dataset=dataset,sample_id=sid,M=int(xx[0]["M"]),
                        geometric_routes=len(routes),oracle=any(yes(x["complete"]) for x in xx),inner_pairwise_mean_m=float(np.mean(spread)),
                        inner_pairwise_min_m=min(spread)))
    for step in (0,32,64,96,128):
        for dataset in bases:
            aa=ev["Global",step,dataset];bb=ev["Exposure",step,dataset]
            for k,x in bb.items():
                a=aa[k];b=bases[dataset][k]
                pair.append(dict(dataset=dataset,step=step,sample_id=k[0],query=k[1],M=int(x["M"]),initial_matched=b["matched"],
                    initial_pass=b["hard_pass"],global_pass=a["hard_pass"],exposure_pass=x["hard_pass"],global_goal=a["goal_pass"],exposure_goal=x["goal_pass"],
                    global_clearance=a["signed_clearance"],exposure_clearance=x["signed_clearance"],
                    clearance_delta=float(x["signed_clearance"])-float(a["signed_clearance"]) if x["signed_clearance"] and a["signed_clearance"] else None,
                    change="FAIL_TO_PASS" if not yes(a["hard_pass"]) and yes(x["hard_pass"]) else
                           "PASS_TO_FAIL" if yes(a["hard_pass"]) and not yes(x["hard_pass"]) else "SAME"))
            for M in (1,4):
                for mt in (True,False):
                    for hp in (True,False):
                        kk=[k for k,b in bases[dataset].items() if int(b["M"])==M and yes(b["matched"])==mt and yes(b["hard_pass"])==hp]
                        strata.append(dict(dataset=dataset,step=step,M=M,initial_matched=mt,initial_pass=hp,queries=len(kk),
                            global_physics=sum(yes(aa[k]["hard_pass"]) for k in kk),exposure_physics=sum(yes(bb[k]["hard_pass"]) for k in kk)))
            for group in ("all","both_safe_goal"):
                kk=[k for k in aa if group=="all" or all(yes(v[k][f]) for v in (aa,bb) for f in ("hard_pass","goal_pass"))]
                if not kk:continue
                for m in ("qa","qj","time","vmax","amax","jmax","goal_progress","mode_loss_potential"):
                    quality.append(dict(dataset=dataset,step=step,group=group,metric=m,queries=len(kk),
                        global_mean=float(np.mean([float(aa[k][m]) for k in kk])),exposure_mean=float(np.mean([float(bb[k][m]) for k in kk])),
                        paired_delta=float(np.mean([float(bb[k][m])-float(aa[k][m]) for k in kk]))))
    for name,rr in [("CHECKPOINT_METRICS",metrics),("QUERY_SAFETY_TRANSITIONS",trans),("PAIRED_QUERY_DIFF",pair),
                    ("INITIAL_STRATA",strata),("QUALITY_COMPARISON",quality),("MODE_COVERAGE",modes),
                    ("SECTION_REPAIR",sections),("DEEP14_DIAGNOSTIC",deep_rows)]:csv_write(OUT/(name+".csv"),rr)
    counts=[];grad=[];cover={};teacher=[]
    for arm,path in (("Global",control),("Exposure",rp)):
        state={sid:[dict(exposures=0,eligible=0,selected=0,nonzero=0,wait=0,longest=0,last=-1,reasons=collections.Counter()) for _ in range(6)] for sid in index}
        parent=None;use=collections.Counter()
        for step in range(129):
            folder=path/("commits/step%04d"%step);c=read(folder/"COMMIT.json")
            require(c["parent_hash"]==parent and c["step"]==step,"REPORT_CHAIN")
            require(all(sha(folder/n)==h for n,h in c["files"].items()),"REPORT_COMMIT_SHA");parent=sha(folder/"COMMIT.json")
            if not step:continue
            u=read(folder/"UPDATE.json")
            grad.append(dict(arm=arm,step=step,loss=u["loss"],lr=u["lr"],ratio=u["physical_imitation_ratio"],gradnorm=u["gradnorm"],
                mode_loss=u["losses"]["mode"],weighted_mode_norm=u["weighted_component_norms"]["mode"],
                imitation_norm=u["components"]["imitation"]["parameter_norm"],imitation_mode_cosine=u["imitation_mode_cosine"],
                adam_negative_mode_cosine=u["adam_delta_vs_negative_mode"],mode_waypoint_raw_norm=float(np.linalg.norm(np.array(
                u["components"]["mode"]["raw_waypoint_components"])))))
            seen=set()
            for b,sid in enumerate(u["ids"]):
                a=u["assignments"][b]
                require(all(not u["hard_pass"][b][q] and not a["matched"][q] for q in a["selected"]),"REPORT_SAFE_PULL")
                require(max(a["occupancy"],default=0)<=2,"REPORT_CAPACITY")
                if sid in seen:continue
                seen.add(sid)
                for q,h in enumerate(state[sid]):
                    eligible=not a["matched"][q] and not u["hard_pass"][b][q];selected=q in a["selected"]
                    h["exposures"]+=1;h["eligible"]+=eligible;h["selected"]+=selected
                    h["nonzero"]+=selected and u["components"]["mode"]["raw_per_query_norm"][b][q]>0
                    h["wait"]=(0 if selected else h["wait"]+1) if eligible else 0;h["longest"]=max(h["longest"],h["wait"])
                    if selected:h["last"]=step-1;use[sid,a["route_teacher_id"][q]]+=1
                    h["reasons"][a["reasons"][q]]+=1
            if arm=="Exposure":
                saved=read(folder/"ASSIGNMENT_STATE.json");p=ExposureRotation();p.load_state_dict(saved)
                for sid,hh in state.items():
                    if sid not in saved["samples"]:
                        require(not hh[0]["exposures"],"REPORT_MISSING_HISTORY");continue
                    sh=saved["samples"][sid];require(sh["exposures"]==hh[0]["exposures"],"REPORT_DOUBLE_COUNT")
                    for q,h in enumerate(hh):
                        require(sh["count"][q]==h["selected"] and sh["eligible"][q]==h["eligible"] and sh["last_selected"][q]==h["last"],"REPORT_HISTORY")
        for sid,hh in state.items():
            for q,h in enumerate(hh):
                b=bases["training384"][sid,q];bad=not yes(b["matched"]) and not yes(b["hard_pass"])
                f=ev[arm,128,"training384"][sid,q]
                counts.append(dict(arm=arm,sample_id=sid,query=q,M=int(b["M"]),initial_bad=bad,exposures=h["exposures"],eligible=h["eligible"],
                    selected=h["selected"],nonzero_direct_mode=h["nonzero"],longest_wait=h["longest"],last_selected=h["last"],
                    final_hard_pass=f["hard_pass"],reasons=json.dumps(h["reasons"],sort_keys=True)))
        cover[arm]={}
        for M in (1,4):
            xx=[x for x in counts if x["arm"]==arm and x["M"]==M and x["initial_bad"]]
            cover[arm][str(M)]=dict(initial_bad=len(xx),covered=sum(x["selected"]>0 for x in xx),never=sum(x["selected"]==0 for x in xx),
                nonzero_mode=sum(x["nonzero_direct_mode"]>0 for x in xx),final_pass=sum(yes(x["final_hard_pass"]) for x in xx),
                count_quantiles=qs([x["selected"] for x in xx]),longest_wait=max((x["longest_wait"] for x in xx),default=0))
        teacher.extend(dict(arm=arm,sample_id=sid,teacher=t,uses=n,M=index[sid]["M"]) for (sid,t),n in use.items())
    csv_write(OUT/"ACTUAL_SUPERVISION_COVERAGE.csv",counts);csv_write(OUT/"TRAINING_GRADIENTS.csv",grad);csv_write(OUT/"TEACHER_USE.csv",teacher)
    initial=torch.load(control/"commits/step0000/checkpoint_step0000.pt",map_location="cpu",weights_only=True)
    final=torch.load(rp/"commits/step0128/checkpoint_step0128.pt",map_location="cpu",weights_only=True)
    require(all(int(s["step"])==128 for s in final["optimizer_state_dict"]["state"].values()),"REPORT_ADAM")
    for n,v in final["state_dict"].items():
        if n.startswith("trajectory_head.") and v.shape[0]==21:require(torch.equal(v[19:21],initial["state_dict"][n][19:21]),"REPORT_SCORE")
    protected=read(OUT/"PROTECTION_BEFORE.json");changed=[n for n,h in protected.items() if sha(ROOT/n)!=h]
    require(not changed,"REPORT_PROTECTION");write(OUT/"PROTECTION_AFTER.json",dict(unchanged=True,tracked_files=len(protected),changed=changed,historical_seals=seals))
    transitions={};ci={};rng=np.random.default_rng(20261030)
    for dataset in bases:
        xx=[x for x in pair if x["step"]==128 and x["dataset"]==dataset]
        transitions[dataset]=dict(fail_to_pass=sum(x["change"]=="FAIL_TO_PASS" for x in xx),pass_to_fail=sum(x["change"]=="PASS_TO_FAIL" for x in xx),clearance_delta_quantiles=qs(x["clearance_delta"] for x in xx))
        groups=collections.defaultdict(list)
        for x in xx:
            b=bases[dataset][x["sample_id"],x["query"]];groups[b["family"],b["kind"]].append(int(yes(x["exposure_pass"]))-int(yes(x["global_pass"])))
        vv=np.array([np.mean(v) for v in groups.values()]);boot=np.mean(rng.choice(vv,size=(4000,len(vv)),replace=True),axis=1)
        ci[dataset]=dict(clusters=len(vv),rate_delta=float(vv.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),caution="single Seed; reused Development")
    write(OUT/"ANALYSIS.json",dict(result=result,coverage=cover,paired_transitions=transitions,map_cluster_ci=ci,
          state_matches_all129_commits=True,score_rows_unchanged=True,optimizer_steps_performed_by_analysis=0))
    replay=read(OUT/"REPLAY_RESULT.json")
    boundtable=table(["Arm","M","initial bad","old","replay","upper","state excluded","finite-budget shortfall","extra global miss"],
        [[a,m,s["initial_bad"],s["historical_covered"],s["replay_covered"],s["upper_bound"],s["state_or_no_teacher_exclusion"],
          s["finite_exposure_capacity_shortfall"],s["global_rotation_extra_missed"]] for a in ("B","C") for m,s in replay["arms"][a]["strata"].items()])
    bins=table(["Arm","M/exposures","initial bad","old covered","replay covered"],
        [[a,k,v["queries"],v["historical"],v["replay"]] for a in ("B","C") for k,v in replay["arms"][a]["exposure_bins"].items()])
    actualtable=table(["Arm","M","initial bad","actual covered","nonzero Mode","final PASS","longest eligible wait"],
        [[a,m,s["initial_bad"],s["covered"],s["nonzero_mode"],s["final_pass"],s["longest_wait"]] for a in cover for m,s in cover[a].items()])
    metrictable=table(["Arm","step","dataset","Physics","Goal","Complete","Oracle","routes"],
        [[s[k] for k in ("arm","step","dataset","physics","goal","complete","oracle","geometric_route_count")] for s in metrics])
    pairtable=table(["dataset","Global FAIL to Exposure PASS","Global PASS to Exposure FAIL","net"],
        [[k,v["fail_to_pass"],v["pass_to_fail"],v["fail_to_pass"]-v["pass_to_fail"]] for k,v in transitions.items()])
    doc("FULLTRAIN_EXPOSURE_ROOT_CAUSE.md","# Exposure root cause\n\n"+boundtable+"\n"+bins+
        "\nAll384 tasks and672 certified Teachers remain bound.1024 batch rows are1014 unique sample/update events:10 duplicate rows do not advance counts twice. "
        "Each original slot and Teacher assignment was exactly reconstructed. HISTORICAL_QUERY_EXPOSURE.csv retains per-Query state changes, reasons, gradients, waypoint shifts and final hard status. "
        "No extra direct Mode is not no gradient: old physical losses and shared parameters remain active.\n")
    doc("SUPERVISION_CAPACITY_UPPER_BOUND.md","# Exact recorded-state coverage bound\n\n"+boundtable+
        "\nFor each sample, left nodes are initial bad Queries; right nodes are extra legal Teacher slots at each unique committed exposure. "
        "An edge exists only when that Query is then dangerous/unmatched. Maximum bipartite matching is the exact hindsight coverage bound for these recorded states. "
        "M1 has one Hungarian occupant plus one extra slot. M4 has four extra slots for at most two unmatched Queries; its slot bottleneck is absent here.\n\n"
        "Ninitial = state/no-valid-slot exclusions + fixed-capacity finite-exposure shortfall + maximum coverage. "
        "Old uncovered additionally includes maximum minus actual covered. Capacity and exposure interact; the finite-budget shortage cannot be uniquely split into independent causal capacity-only and exposure-only counts. "
        "It is unavoidable under both fixed constraints. Increasing either capacity or exposures could change it, and neither was changed. Global-rotation extra misses are separately identifiable.\n")
    doc("EXPOSURE_AWARE_ASSIGNMENT_MATH.md","# Exposure-aware rotation\n\n"
        "Order current detached dangerous-unmatched Queries by (c[s,q], (q-e[s]) mod6), lexicographically. "
        "c counts successful distinct updates giving extra Mode; e counts successful distinct updates containing sample s. "
        "No risk, distance weight or development statistic enters Query priority. Original detached position-distance slot matching is reused after Query selection.\n\n"
        "Original Hungarian routes occupy unchanged capacity2; safe unmatched Queries receive no extra Teacher. "
        "Counts/eligible exposure/last selected/longest wait advance only after successful optimizer update, once per unique sample/update. "
        "Duplicate rows must agree or fail closed. Abort/preflight does not advance state. Fairness needs persistent eligibility and available slots, not an unconditional coverage guarantee.\n")
    doc("EXPOSURE_AWARE_IMPLEMENTATION.md","# Implementation evidence\n\n"
        "New method "+spec["method_id"]+" uses the existing assignment_policy hook and original assign() Teacher solver. "
        "No old source file changed. ASSIGNMENT_STATE.json is SHA-bound in the atomic immutable checkpoint commit alongside .pt, receipt and UPDATE. "
        "Restore rejects missing/tampered state; .pt alone is not a resumable exposure method checkpoint. "
        "Checkpoint namespace also has state companions and BUNDLE records. All129 committed histories were independently reconstructed.\n\n"
        "232 mathematical/regression tests passed. First attempt preserved import-path and legacy exact-three-files inspection failures; "
        "the new method uses its own four-file commit schema, leaving the old inspector unchanged. "
        "All384 initial raw/coefficient arrays and first-batch total losses/gradients matched the control bitwise. "
        "Historical B was reused after exact initializer, weight, order, AdamW, schedule and source verification.\n\n"
        "Base HEAD "+spec["code"]["git_commit"]+". New files are explicit uncommitted research sources with exact SHA snapshots "
        "(allow_uncommitted_sources=true); this is not a clean-commit or GitHub-sync claim. No commit/push was performed.\n")
    doc("EXPOSURE_COVERAGE_COMPARISON.md","# Replay and real supervision\n\n"+boundtable+"\n"+actualtable+
        "\nReplay holds historical network eligibility fixed and is not trained performance. ACTUAL_SUPERVISION_COVERAGE.csv is reconstructed from the new real updates. "
        "Teacher use, all reasons, waiting, nonzero raw Mode and initial-danger strata remain in CSV. "
        "All slots<=2; matched/safe protection held throughout. More coverage does not by itself prove improved hard safety.\n")
    doc("EXPOSURE_AWARE_PILOT128_RESULTS.md","# Actual CPU128 paired results\n\n"+metrictable+"\n"+pairtable+
        "\nNew run "+RUN_ID+" completed128 updates; wall="+str(result["wall_s"])+"s; peak RSS="+str(result["peak_rss_mib"])+"MiB. "
        "Same C1/Seed42/Frozen384/FrozenOrder/Batch8/FP32/AdamW/clip10 and128-endpoint cosine2e-5 to1e-6. Original ratio, resource and checkpoint evaluation guards remained active. "
        "No retries, extensions or endpoint selection. Hard status comes from original continuous Bernstein and polynomial-extrema checks. "
        "Signed-clearance values use4097 nodes/segment and remain numerical diagnostics, not continuous minimum certificates.\n\n"
        "QUERY_SAFETY_TRANSITIONS is relative to initialization; PAIRED_QUERY_DIFF compares arms at identical checkpoints. "
        "SECTION_REPAIR contains actual polynomial-root necessary boundary deficits under a fixed initial Teacher mode; invalid events are retained. "
        "TRAINING_GRADIENTS contains real Imitation/Mode conflict and AdamW delta cosines. Local section, gradient alignment and full hard safety are distinct. "
        "MODE_COVERAGE records conservative AABB-side routes, not homotopy. Oracle is GT evaluation; online Selected@6=NOT_AVAILABLE because Ranking/Selector is untrained.\n")
    cg=sum(v["covered"] for v in cover["Exposure"].values())-sum(v["covered"] for v in cover["Global"].values())
    nt=transitions["training384"]["fail_to_pass"]-transitions["training384"]["pass_to_fail"]
    nd=transitions["development192"]["fail_to_pass"]-transitions["development192"]["pass_to_fail"]
    doc("NEXT_RESEARCH_DECISION.md","# Decision\n\n"
        "Actual initial-danger coverage gain="+str(cg)+" Queries. Final net hard candidates versus Global: train="+str(nt)+", development="+str(nd)+". "
        "This single Seed128 experiment cannot establish stable planning gain. All gains/regressions are retained; no claim of improvement without harming existing candidates is made.\n\n"
        "Fixed capacity combined with sparse exposures dominates the historical shortfall. Fair rotation addresses a smaller avoidable deficit. "
        "If better coverage fails to improve hard safety, study effective geometric displacement per supervised exposure and shared-network/Imitation competition, rather than risk-priority tuning. "
        "No additional experiment, weight change,128 extension or training-data change is performed. Historical14 is diagnostic only.\n")
    doc("FAILURE_AND_LIMITATIONS.md","# Failures and limitations\n\n"
        "- First regression attempt failed on legacy test imports and an old strict-three-files commit inspector; attempt002 passed232 after isolated corrections. Original failure log retained.\n"
        "- The initial report-file write hit Windows command-length limits; a chunked write was used. No training restart or old output change occurred.\n"
        "- All existing tracked files and historical B/C seals verified unchanged. All actual128 updates, optimizer steps, score rows and129 count states verified.\n"
        "- Replay/upper bounds use recorded eligibility, not counterfactual trained states. Capacity and exposure effects are coupled.\n"
        "- One Seed42; reused family02/03 Development192, not blind test. Final endpoint fixed128. No stable superiority or generalization claim.\n"
        "- Exact SHA working snapshot, not a newly committed clean identity; no GitHub sync assumption.\n"
        "- Fine signed clearance is numerical. Continuous hard decisions are unchanged. Teacher identities/route signatures are not certified topology counts.\n"
        "- Per-Query raw Mode gradients are available; full per-Query network Jacobians for all128 updates are NOT_AVAILABLE. Batch Adam cosine is not per-Query causal attribution.\n"
        "- Fixed-reference section absence/ambiguity remains NOT_AVAILABLE, not zero. Local section compliance is not full safety.\n")
    doc("REPRODUCE_COMMANDS.md","# Reproduction\n\n"
        "CWD=/home/lizheng/query_minco_mainline; Python=/home/lizheng/anaconda3/envs/yopo/bin/python; CUDA_VISIBLE_DEVICES=''; "
        "PYTHONDONTWRITEBYTECODE=1; OMP/MKL/OPENBLAS_NUM_THREADS=1; PYTHONPATH=project:project/tests.\n\n"
        "Recorded modules: experiments.pg3_feasible_mode_v1.exposure_replay; exposure_experiment prepare; exposure_experiment train; exposure_report. "
        "All writes are exclusive: do not rerun against exposure_aware001. A fresh binding/output identity and separate authorization is required for another training run.\n")
    names=["query_minco/training/pg3_feasible_mode_v1/exposure_assignment.py","query_minco/training/pg3_feasible_mode_v1/test_exposure_assignment.py",
        "experiments/pg3_feasible_mode_v1/exposure_replay.py","experiments/pg3_feasible_mode_v1/exposure_experiment.py",
        "experiments/pg3_feasible_mode_v1/test_exposure_experiment.py","experiments/pg3_feasible_mode_v1/exposure_report.py"]
    diff="".join("".join(difflib.unified_diff([], (ROOT/n).read_text().splitlines(True),fromfile="/dev/null",tofile=n)) for n in names)
    doc("SOURCE.diff",diff);write(OUT/"CHANGED_FILES_SHA256.json",{n:sha(ROOT/n) for n in names})
    print("FINAL_ANALYSIS",json.dumps(dict(result=result,coverage=cover,transitions=transitions,ci=ci)),flush=True)
if __name__=="__main__":run()
