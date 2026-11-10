"""R3a CPU FP32 zero-update diagnostics; production UNKNOWN remains fail-closed."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from .e1_dataset_binding import load_e1_bound
from .short_train import initialize,INITIAL_SHA
from query_minco.training.physics_only_v1.smoke import parameter_hash
from query_minco.training.physics_only_v1_2.spec import PhysicsV12Spec
from query_minco.training.physics_only_v1_2_gate1.objective import BoundPhysicsV12Objective,feasibility_math
from query_minco.training.physics_only_v1_2.goal_v12 import goal_terms
from query_minco.training.physics_only_v1_2.costs_v12 import smoothness_terms
from query_minco.training.physics_only_v1.decoder import FixedTimeDecoder
from query_minco.trajectory.minco_adapter import MincoAdapter
from query_minco.training.pg1.losses import analytic_samples
from .geometry import world_points
from .hard_validator import validate_result
from query_minco.evaluation.validation import axis_range,evaluate
from .e1_research import similarities
BASE=Path(__file__).resolve().parents[3]
OUT=BASE/"research_outputs/physics_only_v1_2_e1_r3_zero_update_v1"
PREP=OUT/"_preparation/r3a_001"
DATA=BASE/"research_outputs/physics_only_v1_2_e1_r2c_dataset_v1"
PRIMARY=("C_viol","F","G_task_train","C_shape","P","S")
EXTRA=("F_dyn","F_domain","V","Q","J")
PHASE_COEFFICIENTS=dict(C_viol=1.,F=1.,G_task_train=1.,C_shape=.1,P=.025,S=.1)

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    with Path(p).open("x") as f:json.dump(x,f,indent=2,allow_nan=False);f.write("\n")
def guard():
    if (PREP/"STOP.json").exists():raise ValueError("PERSISTENT_STOP")
    e=read(PREP/"ENTRY.json")
    for p,h in e["protected_sha256"].items():
        if sha(p)!=h:raise ValueError("HISTORICAL_SHA_DRIFT "+p)
    for p,h in read(PREP/"SOURCE_FREEZE.json").items():
        if sha(BASE/p)!=h:raise ValueError("SOURCE_FREEZE_DRIFT "+p)
    if time.time()-e["started_unix"]>1800:raise ValueError("TOTAL_TIME_BUDGET")
def phase_spec():return PhysicsV12Spec(phase="A",aggregation="foundation")

def norm_stats(x):
    a=x.detach().double().reshape(-1)
    if not a.numel() or not bool(torch.isfinite(a).all()):raise ValueError("NONFINITE_GRADIENT")
    return dict(norm=float(a.norm()),rms=float(a.square().mean().sqrt()),numel=a.numel(),
        max_abs=float(a.abs().max()),zero_ratio=float((a==0).double().mean()),finite=True,
        accumulation="FP64 after flatten/cast")
def module_groups(named):
    groups={}
    for label,prefix in (("CNN","depth_encoder."),("StateEncoder","state_encoder."),("Transformer","query_decoder.layers."),
                         ("TrajectoryHead","trajectory_head.")):
        groups[label]=[i for i,(n,p) in enumerate(named) if n.startswith(prefix)]
    groups["LearnableQueries"]=[i for i,(n,p) in enumerate(named) if n=="query_decoder.learned_queries"]
    groups["shared"]=groups["CNN"]+groups["StateEncoder"]+groups["Transformer"]
    groups["all_network"]=list(range(len(named)))
    if any(not ids for ids in groups.values()):raise ValueError("MODULE_GROUP_MISSING")
    if set(groups["shared"])&set(groups["TrajectoryHead"]+groups["LearnableQueries"]):raise ValueError("SHARED_GROUP_OVERLAP")
    return groups
def packet(scalars,raw,named,groups):
    vectors={};stats={};raw_values={}
    for cost,value in scalars.items():
        grads=torch.autograd.grad(value,(raw,*[p for _,p in named]),retain_graph=True,allow_unused=True)
        rg=grads[0] if grads[0] is not None else torch.zeros_like(raw)
        if bool(rg[...,15:21].count_nonzero()):raise ValueError("INACTIVE_RAW_GRADIENT")
        gv=[g.detach() if g is not None else torch.zeros_like(p) for g,(_,p) in zip(grads[1:],named)]
        vg={"Raw":rg[...,:15].detach().double().reshape(-1)}
        for label,ids in groups.items():vg[label]=torch.cat([gv[i].reshape(-1) for i in ids]).double()
        vectors[cost]=vg;stats[cost]={g:norm_stats(v) for g,v in vg.items()}
        raw_values[cost]=rg.detach().tolist()
        for g in stats[cost]:
            stats[cost][g]["zero_reason"]=("zero term/upstream/hinge or locally inactive parameter path" if stats[cost][g]["norm"]==0 else None)
    gram={};cos={}
    for group in next(iter(vectors.values())):
        names=list(vectors);gram[group]={};cos[group]={}
        for c in names:
            gram[group][c]={};cos[group][c]={}
            for d in names:
                x=vectors[c][group];y=vectors[d][group]
                dot=float(torch.dot(x,y));gram[group][c][d]=dot/x.numel()
                denom=stats[c][group]["norm"]*stats[d][group]["norm"]
                cos[group][c][d]=None if denom==0 else dot/denom
    return dict(stats=stats,gram_mean_per_parameter=gram,cosine=cos,raw_gradients=raw_values,
        Raw_space="active 0:15 across six queries; full 21D gradients retained; 15:21 verified zero",
        shared_definition="CNN+StateEncoder+Transformer only; excludes TrajectoryHead/LearnableQueries")

def roi_diagnostics(points,g,coefficients,durations):
    world=world_points(points,torch.tensor(g.origin,dtype=points.dtype),torch.tensor(g.rotation,dtype=points.dtype)).detach().double().numpy()
    low=g.roi[0];high=g.roi[1];outside=((world<low)|(world>high)).any(-1)
    rows=[]
    for k in range(6):
        c=coefficients[k];d=durations[k]
        wc=np.einsum("ij,sjk->sik",g.rotation,c);wc[:,:,0]+=g.origin
        ranges=np.stack([axis_range(a,t) for a,t in zip(wc,d)])
        continuous=bool((ranges[:,:,0]<low).any() or (ranges[:,:,1]>high).any())
        ids=np.where(outside[k])[0]
        rows.append(dict(query=k,sampled_outside_count=len(ids),sample_count=len(world[k]),
            first_sample_index=int(ids[0]) if len(ids) else None,first_world_point=world[k,ids[0]].tolist() if len(ids) else None,
            first_B0_point=points[k,ids[0]].detach().tolist() if len(ids) else None,
            sampled_world_min=world[k].min(0).tolist(),sampled_world_max=world[k].max(0).tolist(),
            exact_polynomial_axis_ranges_world=ranges.tolist(),continuous_center_outside_ROI=continuous))
    return dict(roi_world=g.roi.tolist(),B0_origin_world=g.origin.tolist(),R_WB0=g.rotation.tolist(),
        any_sample_outside=bool(outside.any()),any_continuous_center_outside=any(q["continuous_center_outside_ROI"] for q in rows),queries=rows)
def prepare(raw,batch,spec):
    params=FixedTimeDecoder(spec)(raw,batch.state)
    tr=MincoAdapter()(params.head_pva,params.tail_pva,params.inner_pts,params.durations)
    ana=analytic_samples(tr,spec.points_per_segment)
    partial={**feasibility_math(ana,[batch.geometry[0].domain],spec),**smoothness_terms(tr,spec),
        **goal_terms(ana["endpoint"],ana["terminal_velocity"],batch.state[:,6:9],torch.ones(1,dtype=torch.bool),("normal",))}
    coeff=tr.coeffs[0].detach().double().numpy();duration=tr.durations[0].detach().double().numpy()
    roi=roi_diagnostics(ana["p"][0],batch.geometry[0],coeff,duration)
    from types import SimpleNamespace
    hard=validate_result(SimpleNamespace(parameters=params,trajectory=tr),batch)
    continuity=[]
    for c,d in zip(coeff,duration):
        continuity.append([max(float(np.abs(evaluate(c[i],d[i],k)-evaluate(c[i+1],0,k)).max()) for i in range(2)) for k in range(5)])
    record=dict(raw_logits=raw.detach().tolist(),raw_stats=norm_stats(raw),
        waypoints_B0=torch.cat([params.inner_pts,params.tail_pva[:,:,0:1,:]],dim=2).detach().tolist(),
        head_PVA=params.head_pva.detach().tolist(),tail_PVA=params.tail_pva.detach().tolist(),
        coefficients_B0=tr.coeffs.detach().tolist(),durations=tr.durations.detach().tolist(),
        endpoint=ana["endpoint"].detach().tolist(),terminal_velocity=ana["terminal_velocity"].detach().tolist(),
        sampled_p_B0=ana["p"].detach().tolist(),solver_diagnostics={k:v.detach().tolist() for k,v in tr.diagnostics.items()},
        C0_C4_errors=continuity,roi=roi,independent_hard=hard,
        non_collision_diagnostics={k:v.detach().tolist() for k,v in partial.items()},
        partial_terms_scope="diagnostic values only; no Collision/V/Q/J/total is invented when production Objective fails",
        pairwise=similarities(coeff,duration),hard_valid_candidate_count=sum(h["passed"] for h in hard),
        certified_effective_channel_count=0 if not any(h["passed"] for h in hard) else None)
    return record

def actual_gradients(result,raw,model):
    named=list(model.named_parameters());groups=module_groups(named)
    scalars={};upstreams={};metadata={}
    for n in PRIMARY+EXTRA:
        term=result.metrics[n]
        up=torch.autograd.grad(result.loss,term,retain_graph=True,allow_unused=True)[0]
        if up is None:
            scalars[n]=term.mean();metadata[n]="UNWEIGHTED_DIAGNOSTIC_ONLY_NOT_ON_PHASE_A_LOSS_PATH"
            upstreams[n]=None
        else:
            scalars[n]=(up.detach()*term).sum();upstreams[n]=up.detach().tolist()
            metadata[n]="actual Phase A contribution; overlapping diagnostics never summed again"
    scalars["TOTAL"]=result.loss
    full=packet(scalars,raw,named,groups)
    full.update(upstream=upstreams,term_scope=metadata,Phase_A_coefficients=PHASE_COEFFICIENTS,
        primary_names=list(PRIMARY),module_parameters={g:[dict(name=named[i][0],shape=list(named[i][1].shape)) for i in ids] for g,ids in groups.items()})
    # Each query's full actual contribution and primary component paths, including shared attention.
    queries=[]
    for k in range(6):
        parts={n:result.metrics[n][0,k]*float(upstreams[n][0][k]) for n in PRIMARY}
        parts["TOTAL"]=(result.metrics["V"][0,k]+.1*result.metrics["Q"][0,k])/6
        q=packet(parts,raw,named,groups);q["query"]=k;queries.append(q)
    # True backward after diagnostic autograd calls; no cross-sample .grad accumulation.
    raw.retain_grad();result.loss.backward()
    if raw.grad is None or not torch.isfinite(raw.grad).all():raise ValueError("RAW_BACKWARD_INVALID")
    actual=torch.cat([p.grad.reshape(-1) if p.grad is not None else torch.zeros_like(p).reshape(-1) for _,p in named])
    full["actual_total_backward"]=dict(raw=norm_stats(raw.grad),all_network=norm_stats(actual),
        inactive_raw_time_score_zero=not bool(raw.grad[...,15:21].count_nonzero()))
    return full,queries

def scene_equal_summary(records,expected_rows):
    completed=[r for r in records if r["status"]=="PASS"]
    scopes={}
    for scope in ("all","near","far"):
        want=[r for r in expected_rows if scope=="all" or r["sample_id"].endswith("_"+scope)]
        got=[r for r in completed if scope=="all" or r["sample_id"].endswith("_"+scope)]
        scenes=sorted({r["scene_id"] for r in want});counts={s:sum(r["scene_id"]==s for r in want) for s in scenes}
        have={s:sum(r["scene_id"]==s for r in got) for s in scenes}
        complete=counts==have
        averaged={}
        # Partial numeric summaries are explicitly limited to available scenes, never zero-imputed.
        available=[s for s in scenes if have[s]]
        if got:
            first=got[0]["gradients"]["gram_mean_per_parameter"]
            for group in first:
                costs=list(first[group]);G={}
                for c in costs:
                    G[c]={}
                    for d in costs:
                        vals=[]
                        for scene in available:
                            local=[r["gradients"]["gram_mean_per_parameter"][group][c][d] for r in got if r["scene_id"]==scene]
                            vals.append(float(np.mean(local)))
                        G[c][d]=float(np.mean(vals))
                rms={c:float(np.sqrt(max(G[c][c],0))) for c in costs}
                cos={c:{d:None if rms[c]*rms[d]==0 else G[c][d]/(rms[c]*rms[d]) for d in costs} for c in costs}
                averaged[group]=dict(RMS=rms,cosine=cos)
        scopes[scope]=dict(status="COMPLETE" if complete else "BLOCKED_INCOMPLETE_CORPUS",expected_scene_tasks=counts,
            completed_scene_tasks=have,intended_equal_scene_weights={s:1/len(scenes) for s in scenes},
            available_scene_summary=averaged,numeric_scope="formal scene-equal only if COMPLETE; otherwise descriptive completed scenes only, not calibration",
            no_missing_zero_imputation=True)
    return dict(rule="mean per-task gradient dot/n within scene; mean scenes equally; sqrt diagonal and normalized Gram cosine",
        scopes=scopes,Progress_actual_PhaseA_coefficient=.025,weight_selected=None)
def save_svg(record,path):
    # Standalone scientific projection; source data are the saved MINCO samples.
    points=np.asarray(record["sampled_p_B0"])[0];origin=np.asarray(record["roi"]["B0_origin_world"])
    world=points+origin
    roi=np.asarray(record["roi"]["roi_world"]);parts=['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="500">','<rect width="1000" height="500" fill="white"/>']
    colors=["#1565c0","#e65100","#2e7d32","#6a1b9a","#ad1457","#00838f"]
    for panel,(ax,ay,title) in enumerate([(0,1,"world XY / known ROI"),(0,2,"world XZ / known ROI")]):
        low=np.minimum(world.reshape(-1,3).min(0),roi[0]);high=np.maximum(world.reshape(-1,3).max(0),roi[1]);span=np.maximum(high-low,.1)
        def pt(x,y):return (40+panel*500+420*(x-low[ax])/span[ax],450-380*(y-low[ay])/span[ay])
        a,b=pt(roi[0,ax],roi[0,ay]);c,d=pt(roi[1,ax],roi[1,ay])
        parts.append('<text x="%d" y="30" font-size="16">%s</text>'%(40+panel*500,title))
        parts.append('<rect x="%g" y="%g" width="%g" height="%g" fill="#eef5ff" stroke="black"/>'%(a,d,c-a,b-d))
        for k,p in enumerate(world):
            coords=" ".join("%g,%g"%pt(q[ax],q[ay]) for q in p)
            parts.append('<polyline fill="none" stroke="%s" stroke-width="1.5" points="%s"/>'%(colors[k],coords))
    parts.append('</svg>')
    with Path(path).open("x") as f:f.write("".join(parts))
def finalize(records,expected,status,error=None):
    run=OUT/"runs/zero_update_001"
    write(run/"SCENE_EQUAL_GRADIENT_SUMMARY.json",scene_equal_summary(records,expected))
    losses=[];gradient_rows=[]
    for record in records:
        metrics=record.get("metrics",record["non_collision_diagnostics"])
        for k in range(6):
            row=dict(sample_id=record["sample_id"],scene_id=record["scene_id"],query=k,status=record["status"],loss=record.get("loss"),
                Hard_pass=record["independent_hard"][k]["passed"],sampled_outside_ROI=record["roi"]["queries"][k]["sampled_outside_count"],
                Hard_reasons=";".join(record["independent_hard"][k]["reasons"]),max_v=record["independent_hard"][k]["dynamic_polynomial_maxima"][0],
                max_a=record["independent_hard"][k]["dynamic_polynomial_maxima"][1],max_j=record["independent_hard"][k]["dynamic_polynomial_maxima"][2])
            for n in ("C_viol","C_shape","F_dyn","F_domain","G_task_train","P","S","V","Q","J","min_sdf_m"):
                val=metrics.get(n);row[n]=val[0][k] if val is not None else None
            losses.append(row)
        if record["status"]=="PASS":
            for label,packet_ in [("sample",record["gradients"])]+[("query_"+str(k),q) for k,q in enumerate(record["per_query_gradients"])]:
                for cost,groups in packet_["stats"].items():
                    for group,stat in groups.items():gradient_rows.append(dict(sample_id=record["sample_id"],scene_id=record["scene_id"],scope=label,cost=cost,module=group,**stat))
    for name,rows in [("PER_QUERY_LOSS.csv",losses),("GRADIENT_MODULE_STATS.csv",gradient_rows)]:
        if rows:
            keys=list(dict.fromkeys(k for row in rows for k in row))
            with (run/name).open("x",newline="") as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    completed={r["sample_id"]:r["status"] for r in records}
    write(run/"RESULT.json",dict(R3a=status,R3b="NO-GO" if status!="PASS" else "GO_FOR_SEPARATE_APPROVAL",
        successful=sum(r["status"]=="PASS" for r in records),failed=sum(r["status"]!="PASS" for r in records),expected=13,
        task_status=[dict(sample_id=r["sample_id"],status=completed.get(r["sample_id"],"NOT_RUN_AFTER_STOP")) for r in expected],
        optimizer_constructed=False,optimizer_updates=0,checkpoint_loaded=False,Dev_forward_count=0,CUDA_calls=0,error=error,weight_selected=None))
def preflight():
    guard();rows=read(PREP/"TRAIN13.json");reports=[]
    for row in rows:
        b=load_e1_bound(row["source_manifest_path"],row["source_manifest_sha256"]);b.validate()
        reports.append(dict(sample_id=row["sample_id"],scene_id=row["scene_id"],state=b.state.tolist(),roi=b.geometry[0].roi.tolist(),
            manifest_sha256=row["source_manifest_sha256"],source_qualified=True))
    model=initialize()
    if parameter_hash(model)!=INITIAL_SHA:raise ValueError("INITIAL_MODEL_IDENTITY")
    write(PREP/"PREFLIGHT_PASS.json",dict(status="PASS",source_verified_Train13=reports,initial_model_sha256=INITIAL_SHA))
def run_zero():
    from unittest.mock import patch
    guard();expected=read(PREP/"TRAIN13.json");run=OUT/"runs/zero_update_001";run.mkdir(parents=True);records=[]
    with patch.object(torch.optim.Optimizer,"__init__",side_effect=RuntimeError("OPTIMIZER_FORBIDDEN")),patch.object(torch,"load",side_effect=RuntimeError("CHECKPOINT_LOAD_FORBIDDEN")):
        model=initialize();before=parameter_hash(model);spec=phase_spec()
        if before!=INITIAL_SHA:raise ValueError("INITIAL_MODEL_IDENTITY")
        write(run/"MODEL_IDENTITY.json",dict(seed=20261104,before=before,expected=INITIAL_SHA,checkpoint=None,model_mode="eval+autograd",spec=__import__("dataclasses").asdict(spec)))
        try:
            for row in expected:
                guard();model.zero_grad(set_to_none=True)
                if any(p.grad is not None for p in model.parameters()):raise ValueError("GRADIENT_ACCUMULATION")
                b=load_e1_bound(row["source_manifest_path"],row["source_manifest_sha256"]);raw=model(b.depth,b.state)
                if not torch.isfinite(raw).all():raise ValueError("NONFINITE_RAW")
                record=prepare(raw,b,spec);record.update(sample_id=row["sample_id"],scene_id=row["scene_id"],split="train",source_manifest_sha256=row["source_manifest_sha256"])
                save_svg(record,OUT/"visualization/r3a_001"/(row["sample_id"]+"_ROI.svg"))
                objective=BoundPhysicsV12Objective(spec)
                if record["roi"]["any_sample_outside"] or record["roi"]["any_continuous_center_outside"]:
                    actual_exception=None
                    if record["roi"]["any_sample_outside"]:
                        try:objective(raw,b)
                        except ValueError as ex:
                            actual_exception=str(ex)
                            if "UNKNOWN" not in actual_exception:raise
                        else:raise ValueError("UNKNOWN_NOT_REJECTED")
                    record.update(status="STOP_ROI_UNKNOWN",production_exception=actual_exception,loss=None,gradients=None,per_query_gradients=None,
                        reason="sampled or exact polynomial center outside unchanged known ROI; no selective Query replay/clamp/geometry extension")
                    records.append(record);write(run/(row["sample_id"]+"_FAILURE.json"),record)
                    raise ValueError("KNOWN_ROI_UNKNOWN "+row["sample_id"])
                result=objective(raw,b)
                if not torch.isfinite(result.loss):raise ValueError("NONFINITE_OBJECTIVE")
                grad,perq=actual_gradients(result,raw,model)
                record.update(status="PASS",loss=float(result.loss),metrics={k:v.detach().tolist() for k,v in result.metrics.items()},gradients=grad,per_query_gradients=perq,
                    phase_A_per_query_weighted_total=(result.metrics["V"]+.1*result.metrics["Q"]).detach().tolist(),
                    model_sha256_after=parameter_hash(model))
                if parameter_hash(model)!=before:raise ValueError("MODEL_UPDATED")
                records.append(record);write(run/(row["sample_id"]+"_RESULT.json"),record)
                print(json.dumps(dict(sample_id=row["sample_id"],status="PASS",loss=record["loss"])),flush=True)
        except BaseException:
            import traceback
            error=traceback.format_exc()
            write(run/"MODEL_AFTER.json",dict(after=parameter_hash(model),before=before,unchanged=parameter_hash(model)==before))
            finalize(records,expected,"STOP",error)
            raise
        write(run/"MODEL_AFTER.json",dict(after=parameter_hash(model),before=before,unchanged=parameter_hash(model)==before))
        finalize(records,expected,"PASS")
def main():
    p=argparse.ArgumentParser();p.add_argument("--worker",choices=["preflight","zero"],required=True);a=p.parse_args()
    {"preflight":preflight,"zero":run_zero}[a.worker]()
if __name__=="__main__":main()
