"""E1 read-only evidence, exact polynomial similarities, and zero-update gradients."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/"research_outputs/physics_only_v1_2_e1_research_v1"
E0=ROOT/"research_outputs/physics_only_v1_2_first_short_train_v1"
RUN=E0/"runs/phase_a_seed20261104_train001"
PRIMARY=("C_viol","F","G_task_train","C_shape","P","S")


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p,data):
    with Path(p).open("x") as f:json.dump(data,f,indent=2,allow_nan=False);f.write("\n")


def protect():
    import os
    review=os.environ.get("QUERY_MINCO_E1_REVIEW_ENTRY","entry_001")
    if review not in ("entry_001","r1_001"):raise ValueError("UNKNOWN_REVIEW_ENTRY")
    preparation=OUT/"_preparation"/review
    entry=json.loads((preparation/"ENTRY.json").read_text())
    for p,h in entry["protected_sha256"].items():
        if sha(p)!=h:raise ValueError("HISTORICAL_SHA_DRIFT "+p)
    frozen=preparation/"SOURCE_FREEZE.json"
    if review=="r1_001" and not frozen.is_file():raise ValueError("SOURCE_FREEZE_REQUIRED")
    if frozen.exists():
        for p,h in json.loads(frozen.read_text()).items():
            if sha(ROOT/p)!=h:raise ValueError("E1_SOURCE_DRIFT "+p)


def cosine(a,b):
    a=np.asarray(a,np.float64).ravel();b=np.asarray(b,np.float64).ravel()
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError("GRADIENT_VECTOR")
    den=np.linalg.norm(a)*np.linalg.norm(b)
    return None if den==0 else float(np.dot(a,b)/den)


def trajectory_points(c,dt,n=2049):
    from query_minco.evaluation.validation import evaluate
    c=np.asarray(c,float);dt=np.asarray(dt,float)
    grid=np.linspace(0,float(dt.sum()),n)
    ids=np.minimum(np.searchsorted(np.cumsum(dt),grid,side="right"),2)
    start=np.r_[0,np.cumsum(dt)[:-1]]
    points=np.empty((n,3))
    for i in range(3):
        mask=ids==i
        points[mask]=evaluate(c[i],grid[mask]-start[i])
    return points


def arc_resample(p,n=501):
    p=np.asarray(p,float)
    lengths=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
    if lengths[-1]==0:return np.repeat(p[:1],n,axis=0)
    keep=np.r_[True,np.diff(lengths)>0]
    target=np.linspace(0,lengths[-1],n)
    return np.stack([np.interp(target,lengths[keep],p[keep,j]) for j in range(3)],axis=1)


def similarities(coefficients,durations):
    """Exact degree-10 squared-distance integral by 6-node Gauss; independent maxima."""
    from query_minco.evaluation.validation import evaluate,norm_max
    c=np.asarray(coefficients,float);dt=np.asarray(durations,float)
    if c.shape!=(6,3,3,6) or dt.shape!=(6,3) or not np.isfinite(c).all() or not (dt>0).all():
        raise ValueError("TRAJECTORY_SHAPE")
    if not np.allclose(dt,dt[:1],atol=0,rtol=0):raise ValueError("COMMON_TIME_GRID_REQUIRED")
    nodes,weights=np.polynomial.legendre.leggauss(6)
    paths=[trajectory_points(p,t) for p,t in zip(c,dt)]
    arcs=[arc_resample(p) for p in paths]
    endpoint=np.stack([evaluate(p[-1],t[-1]) for p,t in zip(c,dt)])
    er=np.zeros((6,6));rms=er.copy();maximum=er.copy();arc=er.copy()
    for i in range(6):
        for j in range(i):
            squared=0.;peaks=[]
            for segment,T in enumerate(dt[i]):
                diff=c[i,segment]-c[j,segment]
                val=evaluate(diff,(nodes+1)*T/2)
                squared+=T/2*float(np.sum(weights*np.square(val).sum(-1)))
                peaks.append(norm_max(diff,T,0))
            er[i,j]=er[j,i]=float(np.linalg.norm(endpoint[i]-endpoint[j]))
            rms[i,j]=rms[j,i]=float(np.sqrt(squared/dt[i].sum()))
            maximum[i,j]=maximum[j,i]=max(peaks)
            arc[i,j]=arc[j,i]=float(np.sqrt(np.square(arcs[i]-arcs[j]).sum(-1).mean()))
    ix=np.triu_indices(6,1)
    return dict(endpoint_distance_m=er.tolist(),physical_time_exact_rms_m=rms.tolist(),
        physical_time_polynomial_max_offset_m=maximum.tolist(),arc_aligned_sampled_rms_m=arc.tolist(),
        endpoint_pair_mean_m=float(er[ix].mean()),endpoint_pair_max_m=float(er[ix].max()),
        trajectory_pair_rms_mean_m=float(rms[ix].mean()),trajectory_pair_rms_max_m=float(rms[ix].max()),
        trajectory_pair_max_offset_m=float(maximum[ix].max()),
        arc_pair_rms_mean_m=float(arc[ix].mean()),
        full_trajectory_within_5cm=bool(maximum[ix].max()<.05),
        similarity_threshold_scope="5cm is a declared descriptive similarity scale, never a safety threshold",
        homotopy_class="NOT_IDENTIFIED")


def crossing_signature(c,dt,planes=(.5,1.,1.5,2.,2.5,3.)):
    """Descriptive gate crossing fingerprint only, NOT a certified portal/channel."""
    from query_minco.evaluation.validation import evaluate,roots
    c=np.asarray(c,float);dt=np.asarray(dt,float);records=[]
    for plane in planes:
        crossings=[];time=0.
        for poly,T in zip(c,dt):
            x=poly[0].copy();x[0]-=plane
            for t in sorted(roots(x,T)):
                p=evaluate(poly,t);v=evaluate(poly,t,1)
                crossings.append(dict(t=float(time+t),y=float(p[1]),z=float(p[2]),direction=int(np.sign(v[0]))))
            time+=T
        records.append(dict(plane_x_m=plane,crossings=crossings))
    return dict(gates=records,channel_label="UNKNOWN_NO_INDEPENDENT_PORTAL_REGISTRY",
        topology_certified=False,feasible_channel_count=None)


def scene_split_guard(rows):
    """Identity de-dup includes counts, spatial semantics and source scene, never seed alone."""
    seen={}
    for row in rows:
        if row["split"] not in ("train","dev"):raise ValueError("SPLIT")
        for kind in ("scene_id","occupancy_semantic_sha256","canonical_geometry_sha256"):
            value=row.get(kind)
            if not value:raise ValueError("MISSING_GEOMETRY_ID")
            key=(kind,value)
            if key in seen and seen[key]!=row["split"]:raise ValueError("CROSS_SPLIT_GEOMETRY")
            seen[key]=row["split"]
    return True


def choose_progress_weight(group_rms,grid=(.5,1.,2.,4.)):
    ratios=[]
    for name,row in group_rms.items():
        p=float(row["P"]);target=max(float(row["S"]),float(row["C_shape"]))
        if not np.isfinite([p,target]).all() or p<=0 or target<=0:raise ValueError("CALIBRATION_UNDEFINED")
        ratios.append(target/p)
    target=.25*float(np.exp(np.median(np.log(ratios))))
    candidate=min(grid,key=lambda x:(abs(np.log2(x/target)),x))
    return dict(baseline_w_p=.25,uncapped_target=target,candidate_w_p=candidate,
        group_target_to_P_ratio=dict(zip(group_rms,ratios)),rule="closest log2 grid value; Train random initialization only",
        tuned_or_optimal=False,scope="provisional one-Train-sample candidate, not multi-scene calibration")


def saved_rows():
    return [json.loads(line) for line in (RUN/"STEPS.jsonl").read_text().splitlines()]


def diagnostic(run):
    protect();rows=saved_rows()
    from query_minco.evaluation.validation import evaluate
    from query_minco.evaluation.config import CONFIG
    all_steps=[];query_rows=[];pair_rows=[];raw_rows=[]
    for step,row in enumerate(rows):
        c=np.array(row["coefficients_B0"][0]);dt=np.array(row["durations"][0])
        similar=similarities(c,dt)
        coefficients=dict(C_viol=1.,F=1.,G_task_train=1.,C_shape=.1,P=.025,S=.1)
        costs={n:float(np.mean(row["per_query"][n][0])) for n in PRIMARY}
        contribution={n:coefficients[n]*v for n,v in costs.items()}
        if not np.isclose(sum(contribution.values()),row["loss"],atol=2e-5,rtol=1e-5):raise ValueError("LOSS_DECOMPOSITION")
        vectors={n:np.array(row["gradients"]["effective"][n]["raw_values"])[...,:15].reshape(-1) for n in PRIMARY}
        total=np.array(row["gradients"]["total"]["raw_values"])[...,:15].reshape(-1)
        np.testing.assert_allclose(sum(vectors.values()),total,atol=2e-5,rtol=2e-4)
        conflict={}
        for i,name in enumerate(PRIMARY):
            for other in PRIMARY[i+1:]:
                co=cosine(vectors[name],vectors[other])
                conflict[name+"__"+other]=co
                raw_rows.append([step,name,other,co])
        local=None
        if step<32:
            delta=np.array(rows[step+1]["raw"])[...,:15].reshape(-1)-np.array(row["raw"])[...,:15].reshape(-1)
            observed=float(np.mean(rows[step+1]["per_query"]["P"][0])-np.mean(row["per_query"]["P"][0]))
            linear={n:float(g@delta) for n,g in vectors.items()}
            local=dict(observed_delta_P=observed,predicted_delta_P=linear["P"]/.025,
                weighted_component_grad_dot_actual_delta_raw=linear,
                cos_P_vs_other=cosine(vectors["P"],sum(v for n,v in vectors.items() if n!="P")),
                actual_parameter_direction="NOT_AVAILABLE: only delta norm/RMS logged; cannot infer AdamW vector from gradient norms",
                causal_limit="raw first-order association; nonlinear shared-network AdamW update is not separate per-loss interventions")
        signatures=[]
        for k,h in enumerate(row["independent_hard"]):
            signatures.append(crossing_signature(c[k],dt[k]))
            residuals=[max(float(np.abs(evaluate(c[k,i],dt[k,i],order)-evaluate(c[k,i+1],0.,order)).max()) for i in (0,1)) for order in range(5)]
            if not all(e<=t for e,t in zip(residuals,CONFIG["continuity_atol_orders_0_to_4"])):raise ValueError("CONTINUITY")
            points=[evaluate(poly,T).tolist() for poly,T in zip(c[k],dt[k])]
            query_rows.append([step,k,*h["dynamic_polynomial_maxima"],row["per_query"]["F_speed"][0][k],
                max(0,h["dynamic_polynomial_maxima"][0]-3.),h["passed"],h["goal"]["passed"],
                all(p["passed"] for p in h["certificates"]),h["reasons"],points,residuals])
        all_steps.append(dict(step=step,loss=row["loss"],coefficients=coefficients,
            mean_cost=costs,actual_loss_contributions=contribution,raw_gradient_cosines=conflict,
            raw_local_update=local,similarity=similar,
            crossing_fingerprints=signatures,certified_hard_trajectories=row["hard_all_pass"],
            certified_distinct_channel_count="UNKNOWN",shared_per_step_vector="NOT_AVAILABLE except recomputed checkpoint milestones"))
        for i in range(6):
            for j in range(i+1,6):
                pair_rows.append([step,i,j,*[similar[n][i][j] for n in (
                    "endpoint_distance_m","physical_time_exact_rms_m","physical_time_polynomial_max_offset_m","arc_aligned_sampled_rms_m")]])
    write(run/"DIAGNOSTIC.json",dict(status="READ_ONLY_DIAGNOSTIC_PASS",steps=all_steps,optimizer_updates=0,
      source_steps_sha256=sha(RUN/"STEPS.jsonl"),new_scene_data="NOT_GENERATED",valid_channel_registry="NOT_AVAILABLE"))
    for name,headers,records in (
      ("RAW_GRADIENT_COSINES.csv",["step","cost_a","cost_b","cosine_null_means_undefined"],raw_rows),
      ("TRAJECTORY_PAIR_METRICS.csv",["step","query_i","query_j","endpoint_m","physical_time_rms_m","max_offset_m","arc_rms_m"],pair_rows),
      ("SPEED_HARD_AND_CONTINUITY.csv",["step","query","max_v","max_a","max_j","F_speed","speed_excess","HardAll","HardGoal","HardCollision","reasons","waypoints","C0_to_C4"],query_rows)):
        with (run/name).open("x",newline="") as f:
            w=csv.writer(f);w.writerow(headers);w.writerows(records)
    protect();print(json.dumps(dict(status="DIAGNOSTIC_PASS",logged_steps=33,optimizer_updates=0)),flush=True)


def tensor_stats(value):
    x=value.detach().double().reshape(-1)
    if not x.numel() or not bool(__import__("torch").isfinite(x).all()):raise ValueError("NONFINITE_GRADIENT")
    return dict(norm=float(x.norm()),rms=float(x.square().mean().sqrt()),zero_ratio=float((x==0).double().mean()))


def checkpoint_gradients(run):
    import torch
    from unittest.mock import patch
    from query_minco.training.physics_only_v1_2_gate1.short_train import initialize,parameter_hash,MANIFEST,MANIFEST_SHA
    from query_minco.training.physics_only_v1_2_gate1.binding import load_bound
    from query_minco.training.physics_only_v1_2_gate1.objective import BoundPhysicsV12Objective
    from query_minco.training.physics_only_v1_2.spec import PhysicsV12Spec
    protect();batch=load_bound(MANIFEST,MANIFEST_SHA);records=[];rows=saved_rows()
    with patch.object(torch.optim.Optimizer,"__init__",side_effect=RuntimeError("E1_OPTIMIZER_FORBIDDEN")):
        for step in (0,8,16,32):
            path=RUN/"checkpoints"/("step_"+str(step).zfill(3)+".pt")
            proof=json.loads(path.with_suffix(".json").read_text())
            if sha(path)!=proof["file_sha256"]:raise ValueError("CHECKPOINT_SHA")
            data=torch.load(path,map_location="cpu",weights_only=True)
            model=initialize();model.load_state_dict(data["model"],strict=True)
            before=parameter_hash(model)
            if before!=proof["model_sha256"]:raise ValueError("MODEL_SHA")
            raw=model(batch.depth,batch.state)
            torch.testing.assert_close(raw,torch.tensor(rows[step]["raw"]),atol=0,rtol=0)
            result=BoundPhysicsV12Objective(PhysicsV12Spec(phase="A",aggregation="foundation"))(raw,batch)
            torch.testing.assert_close(result.loss,torch.tensor(rows[step]["loss"]),atol=0,rtol=0)
            named=[(n,p) for n,p in model.named_parameters() if not n.startswith("trajectory_head.") and n!="query_decoder.learned_queries"]
            parameters=[p for _,p in named]
            indices={name:[i for i,(n,_) in enumerate(named) if n.startswith(prefix)] for name,prefix in (
                ("CNN","depth_encoder."),("Transformer","query_decoder.layers."),("State","state_encoder."))}
            components={};vectors={}
            for name in (*PRIMARY,"TOTAL"):
                if name=="TOTAL":value=result.loss;up=None
                else:
                    term=result.metrics[name]
                    up=torch.autograd.grad(result.loss,term,retain_graph=True)[0].detach()
                    value=(up*term).sum()
                grads=torch.autograd.grad(value,(raw,*parameters),retain_graph=True,allow_unused=True)
                rawg=grads[0] if grads[0] is not None else torch.zeros_like(raw)
                if rawg[...,15:21].count_nonzero():raise ValueError("TIME_SCORE_GRADIENT")
                gs=[g if g is not None else torch.zeros_like(p) for g,p in zip(grads[1:],parameters)]
                flat=torch.cat([g.reshape(-1) for g in gs])
                groups={group:torch.cat([gs[i].reshape(-1) for i in ix]) for group,ix in indices.items()}
                groups.update(raw=rawg[...,:15],shared=flat)
                vectors[name]={group:v.detach().numpy().reshape(-1) for group,v in groups.items()}
                components[name]=dict(upstream_per_query=None if up is None else up.tolist(),
                    groups={group:tensor_stats(v) for group,v in groups.items()})
                original=rows[step]["gradients"]["total"] if name=="TOTAL" else rows[step]["gradients"]["effective"][name]
                np.testing.assert_allclose(rawg.detach().numpy(),original["raw_values"],atol=0,rtol=0)
                np.testing.assert_allclose(tensor_stats(flat)["norm"],original["shared"]["norm"],atol=1e-5,rtol=2e-6)
            cos={}
            for group in vectors["P"]:
                pairs={}
                for i,name in enumerate(PRIMARY):
                    for other in PRIMARY[i+1:]:
                        pairs[name+"__"+other]=cosine(vectors[name][group],vectors[other][group])
                cos[group]=pairs
            # At saved checkpoints actual cumulative parameter displacement is available.
            delta_projections=None
            if step:
                prev=0 if step==8 else 8 if step==16 else 16
                prev_path=RUN/"checkpoints"/("step_"+str(prev).zfill(3)+".pt")
                old=torch.load(prev_path,map_location="cpu",weights_only=True)["model"]
                delta=torch.cat([(data["model"][n]-old[n]).reshape(-1) for n,_ in named]).numpy()
                delta_projections=dict(from_step=prev,to_step=step,
                    note="gradient evaluated at END of interval, dot cumulative displacement; not per-update causal decomposition",
                    end_gradient_dot_cumulative_delta={n:float(vectors[n]["shared"]@delta) for n in PRIMARY})
            records.append(dict(step=step,checkpoint_sha256=sha(path),model_sha256_before=before,
                model_sha256_after=parameter_hash(model),components=components,gradient_cosines=cos,
                interval_shared_direction=delta_projections,optimizer_constructed=False,updates=0))
            if parameter_hash(model)!=before:raise ValueError("MODEL_UPDATE_IN_E1")
            print("ZERO_UPDATE_CHECKPOINT",step,flush=True)
        random=records[0]
        group_rms={group:{n:random["components"][n]["groups"][group]["rms"] for n in ("P","S","C_shape")}
                   for group in ("raw","CNN","Transformer")}
        selection=choose_progress_weight(group_rms)
        selection.update(train_source_manifest_sha256=MANIFEST_SHA,model_seed=20261104,
            model_sha256=random["model_sha256_before"],selected_from_checkpoint=0,
            forbidden_for_selection="steps8/16/32 outcomes and all Dev",
            full_new_Train_dataset_calibration="NOT_RUN_NEW_DATA_NOT_AUTHORIZED")
        write(run/"CHECKPOINT_GRADIENTS.json",dict(status="ZERO_UPDATE_RECOMPUTATION_PASS",
            records=records,calibration=selection,optimizer_updates=0,
            availability="module/full shared angles at 0/8/16/32 only; no per-step replay or optimizer state restoration"))
    protect();batch.validate();print(json.dumps(dict(status="CHECKPOINT_GRADIENTS_PASS",
        recomputed_models=4,optimizer_updates=0,provisional_w_p=selection["candidate_w_p"])),flush=True)


def main():
    import os
    parser=argparse.ArgumentParser();parser.add_argument("--stage",choices=("diagnostic","checkpoint"),required=True)
    parser.add_argument("--run",required=True);args=parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":raise ValueError("CPU_ONLY_ENV_REQUIRED")
    import torch
    torch.set_num_threads(1)
    if args.stage=="diagnostic":diagnostic(Path(args.run))
    else:checkpoint_gradients(Path(args.run))

if __name__=="__main__":main()
