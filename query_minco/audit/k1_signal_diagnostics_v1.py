"""Read-only K1 signal audit. TRAIN gradients only; DEV is strictly forward-only."""
from pathlib import Path
import json,time
import numpy as np
import torch
from query_minco.audit.nonlocal_signal_v1 import SymmetricNonlocalCost,physical_lateral_shift,lateral_tangent
from query_minco.evaluation.avoidance_families_v1 import load_family_batch
from query_minco.evaluation.mirror_holdout_v1 import sha,write
from query_minco.evaluation.crossing_v1 import crossing_metric
from query_minco.evaluation.validation import norm_max,evaluate
from query_minco.training.physics_only_v1_2_gate1.k1_interface import make_planner,make_objective
from query_minco.training.physics_only_v1_2_gate1.hard_candidates_v3 import validate_result_candidates
from query_minco.training.k1_physics_trainer_v1.trainer import state_sha,metric_terms
COMPONENT_ID="k1_train_only_nonlocal_direction_network_jacobian_audit_v1"
GROUPS=("depth_encoder","state_encoder","query_decoder","trajectory_head")
def model_at(root,step):
    folder=Path(root)/"research_outputs/g2b_k1_pilot_candidate_20261110_v1/A_reasonable_seed20261104"
    path=folder/("step%03d.pt"%step);side=json.loads(Path(str(path)+".json").read_text())
    if sha(path)!=side["file_sha256"]:raise ValueError("CHECKPOINT_SHA")
    blob=torch.load(path,map_location="cpu",weights_only=True)
    model=make_planner().eval();model.load_state_dict(blob["model"],strict=True)
    if state_sha(model)!=side["model_sha256"] or blob["cursor"]!=step:raise ValueError("MODEL_CURSOR_SHA")
    return model,dict(step=step,path=str(path),file_sha256=sha(path),model_sha256=state_sha(model))
def dataset(plan):
    path=Path(plan["data_join_path"])
    if sha(path)!=plan["data_join_sha256"]:raise ValueError("DATA_JOIN_SHA")
    d=json.loads(path.read_text())
    for p,h in d["source_sha256"].items():
        if sha(p)!=h:raise ValueError("DATA_SOURCE_SHA")
    if sha(d["index_path"])!=d["index_sha256"] or sha(d["registry_path"])!=d["registry_sha256"]:raise ValueError("DATA_INDEX_REGISTRY")
    return d["samples"]
def parameters(model):
    named=list(model.named_parameters())
    return named,[p for _,p in named]
def dot(xs,ys):return sum(float((a.detach().double()*b.detach().double()).sum()) for a,b in zip(xs,ys))
def norm(xs):return dot(xs,xs)**.5
def group_stats(named,grads):
    result={}
    for group in GROUPS:
        values=[g for (n,_),g in zip(named,grads) if n.startswith(group+".")]
        result[group]=dict(norm=norm(values),all_finite=all(bool(torch.isfinite(v).all()) for v in values))
    return result
def projection(named,qgrads,lossgrads):
    d=dot(qgrads,lossgrads);qn=norm(qgrads);ln=norm(lossgrads)
    return dict(predicted_dy_per_infinitesimal_sgd_step=-d,
        normalized_direction_cosine=-d/(qn*ln) if qn*ln>0 else None,
        per_unit_loss_gradient_step=-d/ln if ln>0 else None,
        group_contributions={g:-dot([a for (n,_),a in zip(named,qgrads) if n.startswith(g+".")],
                                      [b for (n,_),b in zip(named,lossgrads) if n.startswith(g+".")]) for g in GROUPS})
def report_trajectory(result,batch,qualification):
    tr=result.trajectory
    if tr.coeffs.shape!=(1,1,3,3,6) or not tr.solve_valid.all():raise ValueError("K1_SOLVE")
    if not torch.equal(tr.durations,torch.full_like(tr.durations,2/3)):raise ValueError("FIXED_TIME")
    if not torch.equal(result.parameters.head_pva[0,0,1:],batch.state[0,:6].reshape(2,3)):raise ValueError("INITIAL_PVA")
    hard=validate_result_candidates(result,batch)[0]
    if "INDEPENDENT_MINCO" in hard["reasons"]:raise ValueError("MINCO_REFERENCE_FAILED")
    c=tr.coeffs[0,0].detach().double().numpy();dt=tr.durations[0,0].detach().numpy()
    crossing=[crossing_metric(c,dt,s["planes_B0"],s["gap_B0"],qualification["side"]) for s in qualification["sections"]]
    return dict(hard=hard,required_crossings=crossing,
        hard_task_and_safe_crossing=hard["passed"] and all(x["passed"] for x in crossing),
        terms=metric_terms(result),loss=float(result.loss.detach()),coefficients=c.tolist(),durations=dt.tolist())
def crossing_functional(result,report,qualification):
    # Implicit differentiation of y at x=gate, with a frozen simple nominal root.
    last=report["required_crossings"][-1];hits=last["planes"][1]["forward_hits"]
    if len(hits)!=1:raise ValueError("UNIQUE_FORWARD_CROSSING_REQUIRED")
    hit=hits[0];piece=hit["piece"];dt=result.trajectory.durations[0,0].detach()
    local=hit["time_s"]-float(dt[:piece].sum())
    c=result.trajectory.coeffs[0,0,piece]
    powers=c.new_tensor([local**i for i in range(6)])
    deriv=c.new_tensor([0.]+[i*local**(i-1) for i in range(1,6)])
    p=(c*powers).sum(-1);v=(c*deriv).sum(-1)
    if float(v[0].detach())<=1e-7:raise ValueError("SIMPLE_CROSSING_ROOT")
    xgate=qualification["sections"][-1]["planes_B0"][1]
    return p[1]-(v[1]/v[0]).detach()*(p[0]-xgate)
def pair_differences(records):
    result=[]
    keys=sorted(set((x["step"],x["family_id"]) for x in records))
    for step,family in keys:
        xs=[x for x in records if x["step"]==step and x["family_id"]==family]
        if len(xs)!=2:continue
        a,b=sorted(xs,key=lambda x:x["side"]);ca=np.asarray(a["nominal"]["coefficients"]);cb=np.asarray(b["nominal"]["coefficients"])
        delta=max(norm_max(c-d,t,0) for c,d,t in zip(ca,cb,a["nominal"]["durations"]))
        result.append(dict(step=step,family_id=family,raw15_max_abs=float(np.abs(np.asarray(a["raw21"])[...,:15]-np.asarray(b["raw21"])[...,:15]).max()),
                           continuous_path_difference_m=delta,safe_results=[a["nominal"]["hard_task_and_safe_crossing"],b["nominal"]["hard_task_and_safe_crossing"]]))
    return result
def run_train(root,plan_path,output):
    root=Path(root);out=Path(output);out.mkdir();plan=json.loads(Path(plan_path).read_text())
    rows=[r for r in dataset(plan) if r["split"]=="train"]
    if len(rows)!=4:raise ValueError("TRAIN4")
    records=[];arrays={};shared=[];identities=[]
    for step in plan["checkpoints"]:
        model,identity=model_at(root,step);identities.append(identity);named,params=parameters(model)
        before=state_sha(model);object_ids=[id(p) for p in params];cache=[]
        for row in rows:
            batch=load_family_batch(row["manifest"],row["manifest_sha256"],"train")
            q=json.loads((Path(batch.manifest["folder"])/"SCENE_QUALIFICATION.json").read_text())
            original=make_objective();nonlocal_cost=SymmetricNonlocalCost(plan["scales_m"],objective=original)
            started=time.perf_counter();raw=model(batch.depth,batch.state);forward_s=time.perf_counter()-started
            if raw.shape!=(1,1,21):raise ValueError("RAW_SHAPE")
            started=time.perf_counter();a=original(raw,batch);time_a=time.perf_counter()-started
            nominal=report_trajectory(a,batch,q)
            ga=torch.autograd.grad(a.loss,(raw,*params),retain_graph=True)
            started=time.perf_counter();b=nonlocal_cost(raw,batch);time_b=time.perf_counter()-started
            gb=torch.autograd.grad(b.loss,(raw,*params),retain_graph=True)
            if any(not torch.isfinite(g).all() for g in (*ga,*gb)):raise ValueError("NONFINITE_GRADIENT")
            if ga[0][...,15:].count_nonzero() or gb[0][...,15:].count_nonzero():raise ValueError("RAW_DEAD_GRADIENT")
            module_a=group_stats(named,ga[1:]);module_b=group_stats(named,gb[1:])
            if not all(v["all_finite"] and v["norm"]>0 for v in list(module_a.values())+list(module_b.values())):raise ValueError("MODULE_GRADIENT")
            layer=model.trajectory_head.mlp[-1]
            for grads in [ga[1:],gb[1:]]:
                for p in [layer.weight,layer.bias]:
                    ix=next(i for i,t in enumerate(params) if t is p)
                    if grads[ix][15:].count_nonzero():raise ValueError("DEAD_HEAD_ROW_GRADIENT")
            term_grads={name:torch.autograd.grad(a.metrics[name].mean(),raw,retain_graph=True)[0][0,0,:15].detach().tolist() for name in ["C_viol","F","G_task_train","C_shape","P","S"]}
            functional=crossing_functional(a,nominal,q);qgrads=torch.autograd.grad(functional,params,retain_graph=True)
            pa=projection(named,qgrads,ga[1:]);pb=projection(named,qgrads,gb[1:])
            direction=lateral_tangent(raw.detach());da=float((ga[0]*direction).sum());db=float((gb[0]*direction).sum())
            candidate_rows=[]
            for item in b.candidates:
                value=item["result"];rr=report_trajectory(value,batch,q)
                rg=torch.autograd.grad(value.loss,raw,retain_graph=True)[0]
                candidate_rows.append(dict(delta_m=item["delta_m"],sign=item["sign"],raw21=item["raw"].detach().tolist(),
                    unperturbed_raw_lateral_derivative=float((rg*direction).sum()),**rr))
                key="step%s_%s_delta%s_sign%s"%(step,row["scene_id"],item["delta_m"],item["sign"])
                arrays[key+"_coeff"]=value.trajectory.coeffs.detach().numpy()
            scales=[]
            for delta in plan["scales_m"]:
                minus=next(x for x in candidate_rows if x["delta_m"]==delta and x["sign"]==-1);plus=next(x for x in candidate_rows if x["delta_m"]==delta and x["sign"]==1)
                scales.append(dict(delta_m=delta,symmetric_mean=(minus["loss"]+plus["loss"])/2,
                    symmetric_mean_autograd_lateral=(minus["unperturbed_raw_lateral_derivative"]+plus["unperturbed_raw_lateral_derivative"])/2,
                    cost_secant=(plus["loss"]-minus["loss"])/(2*delta),
                    note="Cost secant is NOT the derivative of the symmetric mean."))
            fd=[]
            for eps in plan["nominal_shift_fd_steps_m"]:
                values=[]
                with torch.no_grad():
                    for sign in [-1,1]:
                        shifted=physical_lateral_shift(raw.detach(),sign*eps)
                        aa=original(shifted,batch).loss;bb=nonlocal_cost(shifted,batch).loss
                        values.append((float(aa),float(bb)))
                fd.append(dict(epsilon_m=eps,A=(values[1][0]-values[0][0])/(2*eps),B=(values[1][1]-values[0][1])/(2*eps),
                    A_discrepancy=abs((values[1][0]-values[0][0])/(2*eps)-da),B_discrepancy=abs((values[1][1]-values[0][1])/(2*eps)-db)))
            row_result=dict(step=step,scene_id=row["scene_id"],family_id=row["family_id"],split="train",side=q["side"],raw21=raw.detach().tolist(),
                nominal=nominal,original_term_raw15_gradients=term_grads,
                A=dict(loss=float(a.loss.detach()),raw15_gradient=ga[0][0,0,:15].tolist(),lateral_derivative=da,module_gradients=module_a,parameter_projection=pa),
                B=dict(loss=float(b.loss.detach()),raw15_gradient=gb[0][0,0,:15].tolist(),lateral_derivative=db,module_gradients=module_b,parameter_projection=pb),
                candidates=candidate_rows,scales=scales,finite_differences=fd,
                any_perturbed_safe=any(x["hard_task_and_safe_crossing"] for x in candidate_rows),
                nominal_unsafe_despite_safe_perturbation=not nominal["hard_task_and_safe_crossing"] and any(x["hard_task_and_safe_crossing"] for x in candidate_rows),
                timing=dict(network_forward_s=forward_s,A_physics_graph_s=time_a,B_8_physics_graph_s=time_b),
                raw15_21_grad_nonzero=0,parameters_grad_buffers_unchanged=all(p.grad is None for p in params))
            if not row_result["parameters_grad_buffers_unchanged"]:raise ValueError("PARAMETER_GRAD_BUFFER_CHANGED")
            records.append(row_result)
            print('TRAIN_CASE_COMPLETE',step,row['scene_id'],'A_dir',da,'B_dir',db,flush=True)
            arrays["step%s_%s_A_param_grad"%(step,row["scene_id"])]=torch.cat([g.detach().flatten() for g in ga[1:]]).numpy()
            arrays["step%s_%s_B_param_grad"%(step,row["scene_id"])]=torch.cat([g.detach().flatten() for g in gb[1:]]).numpy()
            arrays["step%s_%s_crossing_y_param_jac"%(step,row["scene_id"])]=torch.cat([g.detach().flatten() for g in qgrads]).numpy()
            arrays["step%s_%s_nominal_coeff"%(step,row["scene_id"])]=a.trajectory.coeffs.detach().numpy()
            cache.append(dict(row=row_result,A=[g.detach() for g in ga[1:]],B=[g.detach() for g in gb[1:]],q=[g.detach() for g in qgrads]))
            if state_sha(model)!=before or [id(p) for p in params]!=object_ids:raise ValueError("MODEL_OR_PARAMETER_OBJECT_CHANGED")
        mean_a=[sum(item["A"][i] for item in cache)/len(cache) for i in range(len(params))]
        mean_b=[sum(item["B"][i] for item in cache)/len(cache) for i in range(len(params))]
        for item in cache:
            shared.append(dict(step=step,scene_id=item["row"]["scene_id"],side=item["row"]["side"],
                               A=projection(named,item["q"],mean_a),B=projection(named,item["q"],mean_b)))
        arrays["step%s_shared_A_grad"%step]=torch.cat([g.flatten() for g in mean_a]).numpy()
        arrays["step%s_shared_B_grad"%step]=torch.cat([g.flatten() for g in mean_b]).numpy()
        if state_sha(model)!=before or any(p.grad is not None for p in params):raise ValueError("MODEL_CHANGED")
    write(out/"TRAIN_SIGNALS.json",dict(component_id=COMPONENT_ID,records=records,shared_train_mean_projections=shared,
        checkpoint_identity=identities,pair_differences=pair_differences(records),network_updates=0,optimizer_constructed=False,
        nominal_B_trajectory_equals_A=True,dev_read_or_used_for_choice=False))
    np.savez_compressed(out/"TRAIN_SIGNAL_ARRAYS.npz",**arrays)
    offsets=[];position=0
    for n,p in named:offsets.append(dict(name=n,shape=list(p.shape),start=position,end=position+p.numel()));position+=p.numel()
    write(out/"PARAMETER_FLATTENING.json",dict(parameters=offsets,count=position))
    return dict(train_cases=len(records),perturbed_cases=sum(len(x["candidates"]) for x in records),network_updates=0)
def seal_train_decision(plan_path,train_folder,output):
    plan=json.loads(Path(plan_path).read_text());folder=Path(train_folder);r=json.loads((folder/"TRAIN_SIGNALS.json").read_text())
    threshold=plan["thresholds"]["direction_abs"]
    summary={}
    for step in plan["checkpoints"]:
        rows=[x for x in r["records"] if x["step"]==step]
        shared=[x for x in r["shared_train_mean_projections"] if x["step"]==step]
        summary[step]={}
        for name in ["A","B"]:
            summary[step][name]=dict(
                raw_correct=sum(-x[name]["lateral_derivative"]*x["side"]>threshold for x in rows),
                individual_parameter_correct=sum(x[name]["parameter_projection"]["predicted_dy_per_infinitesimal_sgd_step"]*x["side"]>threshold for x in rows),
                fd_stable_correct=sum(sum(-v[name]*x["side"]>threshold for v in x["finite_differences"])>=2 for x in rows),
                shared_parameter_correct=sum(x[name]["predicted_dy_per_infinitesimal_sgd_step"]*x["side"]>threshold for x in shared))
    initial=summary[plan["primary_checkpoint"]];stress=summary[plan["stress_checkpoint"]]
    primary_ok=all(initial["B"][key]==4 for key in ["raw_correct","individual_parameter_correct","fd_stable_correct"]) and initial["B"]["raw_correct"]>initial["A"]["raw_correct"]
    all_strong=primary_ok and all(stress["B"][key]==4 for key in ["raw_correct","individual_parameter_correct","fd_stable_correct"]) and initial["B"]["shared_parameter_correct"]==4 and stress["B"]["shared_parameter_correct"]==4
    status="SUPPORTED" if all_strong else "INCONCLUSIVE" if primary_ok else "INSUFFICIENT"
    result=dict(nonlocal_directional_signal=status,original_l0_directional_signal="PASS" if initial["A"]["raw_correct"]==4 and initial["A"]["individual_parameter_correct"]==4 else "INSUFFICIENT",
        summary=summary,rule=plan["decision_rule"],thresholds=plan["thresholds"],TRAIN_SIGNALS_sha256=sha(folder/"TRAIN_SIGNALS.json"),train_arrays_sha256=sha(folder/"TRAIN_SIGNAL_ARRAYS.npz"),
        preregistration_sha256=sha(plan_path),dev_used_for_decision=False,delta_tuned=False,network_updates=0)
    write(output,result);return result
def run_dev(root,plan_path,train_seal,output):
    root=Path(root);plan=json.loads(Path(plan_path).read_text());seal=json.loads(Path(train_seal).read_text())
    if seal["preregistration_sha256"]!=sha(plan_path) or seal["dev_used_for_decision"]:raise ValueError("SEALED_TRAIN_ONLY_DECISION")
    if plan["dev_policy"]!="FORWARD_ONLY_GRADIENTS_NOT_RUN_RESEARCHER_CONFIRMED":raise ValueError("DEV_POLICY")
    rows=[x for x in dataset(plan) if x["split"]=="dev"]
    out=Path(output);out.mkdir();records=[];arrays={}
    for step in plan["checkpoints"]:
        model,identity=model_at(root,step);before=state_sha(model)
        with torch.no_grad():
            for row in rows:
                batch=load_family_batch(row["manifest"],row["manifest_sha256"],"dev");q=json.loads((Path(batch.manifest["folder"])/"SCENE_QUALIFICATION.json").read_text())
                begin=time.perf_counter();raw=model(batch.depth,batch.state);forward_s=time.perf_counter()-begin
                result=make_objective()(raw,batch);nominal=report_trajectory(result,batch,q)
                records.append(dict(step=step,scene_id=row["scene_id"],family_id=row["family_id"],split="dev",side=q["side"],raw21=raw.tolist(),nominal=nominal,
                    Raw15_gradient="NOT_RUN_DEV_FORWARD_ONLY",module_gradients="NOT_RUN_DEV_FORWARD_ONLY",
                    nonlocal_cost_or_offsets="NOT_RUN_TRAIN_ONLY",network_forward_s=forward_s))
                arrays["step%s_%s_raw"%(step,row["scene_id"])]=raw.numpy()
                arrays["step%s_%s_coeff"%(step,row["scene_id"])]=result.trajectory.coeffs.numpy()
        if state_sha(model)!=before or any(p.grad is not None for p in model.parameters()):raise ValueError("DEV_CHANGED_MODEL")
    write(out/"DEV_FORWARD.json",dict(records=records,pair_differences=pair_differences(records),train_decision_sha256=sha(train_seal),
        dev_backward_calls=0,parameter_updates=0,delta_selection=False))
    np.savez(out/"DEV_FORWARD_ARRAYS.npz",**arrays)
    return dict(dev_forward_cases=len(records),gradients="NOT_RUN",updates=0)
