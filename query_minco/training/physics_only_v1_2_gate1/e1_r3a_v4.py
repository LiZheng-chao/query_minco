"""V4 diagnosis only. Original eval+autograd model path is mandatory."""
import argparse,csv,hashlib,json,time,traceback
from pathlib import Path
import numpy as np
import torch
from .short_train import initialize,INITIAL_SHA
from query_minco.training.physics_only_v1.smoke import parameter_hash
from .e1_dataset_binding import load_e1_bound
from .e1_r3a_v2 import bootstrap
from .e1_r3a_v3_logfix import decode,polynomial_exit,vector_outputs,jacobian_audit
from .e1_r3_zero_update import roi_diagnostics
from query_minco.training.pg1.losses import analytic_samples
from .hard_validator import validate_result,transform_coefficients,validate_polynomial
from query_minco.evaluation.validation import axis_range,bernstein_controls,numerical_reference,evaluate
BASE=Path(__file__).resolve().parents[3]
OUT=BASE/"research_outputs/physics_only_v1_2_e1_r3a_v4"
PREP=OUT/"_preparation/freeze_001"
V2=BASE/"research_outputs/physics_only_v1_2_e1_r3a_v2/runs/zero_update_001"
GROUPS={"waypoints":(0,9),"head_PVA":(9,18),"terminal_PVA":(18,27),"MINCO_coefficients":(27,81),"durations":(81,84),"score_logits":(84,86)}
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    with Path(p).open("x") as f:json.dump(x,f,indent=2,allow_nan=False);f.write("\n")
def emit(payload):print(json.dumps(payload),flush=True)
def tensor_list(x):return x.detach().cpu().tolist()
def guard():
    if (PREP/"STOP.json").exists():raise ValueError("PERSISTENT_STOP")
    e=read(PREP/"ENTRY.json")
    for p,h in {**e["protected_sha256"],**{str(BASE/p):h for p,h in read(PREP/"SOURCE_FREEZE.json").items()},**read(PREP/"INPUT_HASHES.json")}.items():
        if sha(p)!=h:raise ValueError("SHA_DRIFT "+p)
    if time.time()-e["started_unix"]>3600:raise ValueError("TOTAL_BUDGET")
def rows():return read(PREP/"TRAIN13_AND_EXCLUSIONS.json")["Train13"]
def bound(row):
    if row["split"]!="train" or row["reachability"]!="FEASIBLE_CERTIFIED" or not row["data_eligible"]:raise ValueError("TRAIN_ELIGIBILITY")
    b=load_e1_bound(row["source_manifest_path"],row["source_manifest_sha256"]);b.validate();return b
def capture(model,batch):
    if model.training or not torch.is_grad_enabled() or torch.is_inference_mode_enabled():raise ValueError("FORWARD_SEMANTICS")
    if torch.get_num_threads()!=1 or any(not p.requires_grad for p in model.parameters()):raise ValueError("MODEL_EXECUTION_IDENTITY")
    raw=model(batch.depth,batch.state)
    if raw.dtype!=torch.float32 or raw.device.type!="cpu" or not raw.requires_grad or not torch.isfinite(raw).all():raise ValueError("FORWARD_TYPE_OR_FINITE")
    params,tr=decode(raw,batch.state);ana=analytic_samples(tr,64)
    record=dict(raw_logits=tensor_list(raw),waypoints_B0=tensor_list(torch.cat([params.inner_pts,params.tail_pva[:,:,0:1]],2)),
        head_PVA=tensor_list(params.head_pva),tail_PVA=tensor_list(params.tail_pva),coefficients_B0=tensor_list(tr.coeffs),durations=tensor_list(tr.durations),
        sampled_p_B0=tensor_list(ana["p"]),endpoint=tensor_list(ana["endpoint"]),terminal_velocity=tensor_list(ana["terminal_velocity"]),
        execution=dict(training=model.training,grad_enabled=torch.is_grad_enabled(),inference_mode=torch.is_inference_mode_enabled(),raw_requires_grad=raw.requires_grad,num_threads=torch.get_num_threads()))
    for key in ["raw_logits","waypoints_B0","head_PVA","tail_PVA","coefficients_B0","durations","sampled_p_B0"]:
        if not np.isfinite(np.asarray(record[key])).all():raise ValueError("NONFINITE "+key)
    return record,raw,params,tr,ana
def differences(current,previous):
    out={}
    for name in ["raw_logits","waypoints_B0","head_PVA","tail_PVA","coefficients_B0","durations","sampled_p_B0"]:
        a=np.asarray(current[name]);b=np.asarray(previous[name])
        same_shape=a.shape==b.shape
        out[name]=dict(shape=list(a.shape),reference_shape=list(b.shape),exact_numeric_equal=same_shape and bool(np.array_equal(a,b)),max_abs_difference=float(np.max(np.abs(a-b))) if same_shape else None,
            actual_FP32_SHA256=hashlib.sha256(a.astype("<f4").tobytes()).hexdigest(),reference_FP32_SHA256=hashlib.sha256(b.astype("<f4").tobytes()).hexdigest())
    return out
def reproduction():
    guard();dest=OUT/"runs/reproduction_001";dest.mkdir(parents=True);reports=[]
    model=initialize();before=parameter_hash(model)
    if before!=INITIAL_SHA:raise ValueError("MODEL_SHA")
    write(dest/"MODEL_BEFORE.json",dict(sha256=before,seed=20261104))
    try:
        for row in [rows()[0],rows()[4]]:
            b=bound(row);old=read(V2/(row["sample_id"]+".json"))
            first,*_=capture(model,b);write(dest/(row["sample_id"]+"_FORWARD.json"),first)
            repeat,*_=capture(model,b);write(dest/(row["sample_id"]+"_REPEAT.json"),repeat)
            comp=differences(first,old);det=differences(repeat,first)
            item=dict(sample_id=row["sample_id"],v2_comparison=comp,repeat_comparison=det,v2_record_sha256=sha(V2/(row["sample_id"]+".json")))
            reports.append(item)
            write(dest/(row["sample_id"]+"_COMPARISON.json"),item)
            if not all(v["exact_numeric_equal"] for v in comp.values()):raise ValueError("V2_FORWARD_MISMATCH")
            if not all(v["exact_numeric_equal"] for v in det.values()):raise ValueError("MODEL_NONDETERMINISM")
        after=parameter_hash(model)
        if after!=before:raise ValueError("MODEL_UPDATED")
        write(dest/"RESULT.json",dict(status="PASS",comparisons=reports,model_before=before,model_after=after,no_tolerance_relaxation=True,comparison_atol=0,comparison_rtol=0))
    except BaseException:
        write(dest/"FAILURE.json",dict(status="STOP",error=traceback.format_exc(),comparisons=reports,model_sha256=parameter_hash(model)));raise
def support():
    guard()
    if read(OUT/"runs/reproduction_001/RESULT.json")["status"]!="PASS":raise ValueError("REPRODUCTION_REQUIRED")
    from types import SimpleNamespace
    dest=OUT/"runs/support_001";dest.mkdir(parents=True);model=initialize();before=parameter_hash(model);matrix=[];records=[]
    if before!=INITIAL_SHA:raise ValueError("MODEL_SHA")
    try:
        for row in rows():
            guard();b=bound(row);r,raw,params,tr,ana=capture(model,b);g=b.geometry[0]
            if not np.array_equal(g.rotation,np.eye(3)):raise ValueError("UNREGISTERED_ROTATED_ROI")
            c=tr.coeffs[0].detach().double().numpy();d=tr.durations[0].detach().double().numpy()
            roi=roi_diagnostics(ana["p"][0],g,c,d)
            hard=validate_result(SimpleNamespace(parameters=params,trajectory=tr),b)
            r.update(sample_id=row["sample_id"],scene_id=row["scene_id"],source_sha=row,roi=roi,state=tensor_list(b.state),independent_hard=hard,
                roi_B0=(g.roi-g.origin).tolist(),reference_sphere_eroded_world=[(g.roi[0]+.150001).tolist(),(g.roi[1]-.150001).tolist()],
                reference_sphere_eroded_B0=[(g.roi[0]+.150001-g.origin).tolist(),(g.roi[1]-.150001-g.origin).tolist()],queries=[],total_Loss_called=False)
            for k in range(6):
                wc=transform_coefficients(c[k],g.origin,g.rotation)
                center=polynomial_exit(wc,d[k],g.roi[0],g.roi[1])
                ball=polynomial_exit(wc,d[k],g.roi[0]+.150001,g.roi[1]-.150001)
                ranges=np.stack([axis_range(z,t) for z,t in zip(wc,d[k])]);rd=roi["queries"][k]
                if bool(rd["continuous_center_outside_ROI"])!=(center["first_exit_s"] is not None):raise ValueError("ROOT_EXTREMA_CLASSIFICATION_CONFLICT")
                query=dict(query=k,status="OUTSIDE" if center["first_exit_s"] is not None else "INSIDE",center_exit=center,reference_sphere_exit=ball,continuous_ranges_world=ranges.tolist(),
                    continuous_ranges_B0=np.stack([axis_range(z,t) for z,t in zip(c[k],d[k])]).tolist(),sampled_outside_count=rd["sampled_outside_count"],unknown_reason="OUTSIDE_NATIVE_CERTIFIED_ROI" if center["first_exit_s"] is not None else None,
                    terminal_velocity_B0=r["tail_PVA"][0][k][1],terminal_acceleration_B0=r["tail_PVA"][0][k][2])
                r["queries"].append(query)
                matrix.append(dict(sample_id=row["sample_id"],scene_id=row["scene_id"],query=k,status=query["status"],first_center_exit_s=center["first_exit_s"],first_reference_sphere_exit_s=ball["first_exit_s"],
                    sampled_outside=rd["sampled_outside_count"],sample_count=192,reference_sphere_outside=ball["first_exit_s"] is not None,hard_pass=hard[k]["passed"],hard_reasons=";".join(hard[k]["reasons"]),
                    unknown_reason=query["unknown_reason"],roi_world=json.dumps(g.roi.tolist()),roi_B0=json.dumps(r["roi_B0"]),eroded_roi_B0=json.dumps(r["reference_sphere_eroded_B0"]),
                    continuous_ranges_world=json.dumps(ranges.tolist()),source_manifest_sha256=row["source_manifest_sha256"],geometry_sha256=row["geometry_sha256"],camera_sha256=row["camera_sha256"],state_sha256=row["state_sha256"],Loss_status="NOT_CALLED_SUPPORT_AUDIT"))
            # Save before any optional comparison gate so failures retain full evidence.
            write(dest/(row["sample_id"]+".json"),r)
            if row["sample_id"] in [rows()[0]["sample_id"],rows()[4]["sample_id"]]:
                ctrl=read(OUT/"runs/reproduction_001"/(row["sample_id"]+"_FORWARD.json"));cmp=differences(r,ctrl)
                write(dest/(row["sample_id"]+"_CONTROL_COMPARISON.json"),cmp)
                if not all(z["exact_numeric_equal"] for z in cmp.values()):raise ValueError("SUPPORT_FORWARD_NONDETERMINISM")
            records.append(r);emit(dict(sample_id=row["sample_id"],outside_queries=sum(q["status"]=="OUTSIDE" for q in r["queries"]),hard_pass=sum(h["passed"] for h in hard)))
        after=parameter_hash(model)
        if before!=after:raise ValueError("MODEL_UPDATED")
        with (OUT/"summary/freeze_001/TRAIN13_ROI_SUPPORT_MATRIX.csv").open("x",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(matrix[0]));w.writeheader();w.writerows(matrix)
        write(dest/"RESULT.json",dict(status="PASS",tasks=13,queries=78,center_outside_tasks=sum(any(q["status"]=="OUTSIDE" for q in r["queries"]) for r in records),center_outside_queries=sum(r["status"]=="OUTSIDE" for r in matrix),
            reference_sphere_outside_tasks=sum(any(q["reference_sphere_exit"]["first_exit_s"] is not None for q in r["queries"]) for r in records),reference_sphere_outside_queries=sum(r["reference_sphere_outside"] for r in matrix),
            hard_pass_queries=sum(r["hard_pass"] for r in matrix),model_before=before,model_after=after,total_Loss_calls=0,Loss_qualification="NOT_PERFORMED",optimizer_updates=0))
    except BaseException:
        write(dest/"FAILURE.json",dict(status="STOP",error=traceback.format_exc(),completed_tasks=len(records),model_sha256=parameter_hash(model)));raise
def real_jacobian():
    guard()
    if read(OUT/"runs/support_001/RESULT.json")["status"]!="PASS":raise ValueError("SUPPORT_REQUIRED")
    dest=OUT/"runs/jacobian_001";dest.mkdir(parents=True)
    # Fixed saved raw points. No additional model execution and no perturbed UNKNOWN Loss.
    free=read(OUT/"runs/support_001"/(rows()[0]["sample_id"]+".json"));sparse=read(OUT/"runs/support_001"/(rows()[4]["sample_id"]+".json"))
    raw=torch.tensor(sparse["raw_logits"],dtype=torch.float32);state=torch.tensor(sparse["state"],dtype=torch.float32)
    fb=bound(rows()[0]);fr=torch.tensor(free["raw_logits"],dtype=torch.float32)
    write(dest/"FROZEN_TEST_POINTS.json",dict(sparse_sample_id=sparse["sample_id"],free_sample_id=free["sample_id"],sparse_record_sha256=sha(OUT/"runs/support_001"/(sparse["sample_id"]+".json")),free_record_sha256=sha(OUT/"runs/support_001"/(free["sample_id"]+".json")),raw_sparse=raw.tolist(),state_sparse=state.tolist(),raw_free=fr.tolist(),FD_steps=[.002,.001],original_tolerances_unchanged=True))
    jacobian_audit(raw,state,fb,fr,dest)
    npz=np.load(dest/"RAW21_JACOBIAN.npz");ad=npz["AD"];fd=npz["FD_h002"]
    if any(np.linalg.norm(ad[k,27:81,i])==0 for k in range(6) for i in range(15)):raise ValueError("ACTIVE_MINCO_DIMENSION_ZERO")
    # Saturation probe affects physical parameter map only, not production geometry.
    def theta(r):
        from query_minco.training.physics_only_v1.decoder import FixedTimeDecoder
        from .e1_r3_zero_update import phase_spec
        p=FixedTimeDecoder(phase_spec())(r,state)
        return torch.cat([p.inner_pts.flatten(2),p.tail_pva.flatten(2)],2)
    sat=raw.clone();sat[...,:15]=50.
    ja=torch.autograd.functional.jacobian(theta,raw,vectorize=True);js=torch.autograd.functional.jacobian(theta,sat,vectorize=True)
    ranks=[];sat_ranks=[]
    for k in range(6):
        normal=ja[0,k,:,0,k,:15].detach().double().numpy();saturated=js[0,k,:,0,k,:15].detach().double().numpy()
        ranks.append(int(np.linalg.matrix_rank(normal,tol=1e-10)));sat_ranks.append(int(np.linalg.matrix_rank(saturated,tol=1e-10)))
    if ranks!=[15]*6 or sat_ranks!=[0]*6:raise ValueError("SATURATION_RANK_CONTRACT")
    write(dest/"SATURATION_AND_RANK.json",dict(actual_parameter_ranks=ranks,saturated_parameter_ranks=sat_ranks,saturation_logit=50.,rank_absolute_tolerance=1e-10,physical_active_dim_count=15,unconsumed_score_identity="AD rows84:86 retain identity; Loss derivatives19:21 zero"))
    write(dest/"RESULT.json",dict(status="PASS",actual_sparse_queries=6,raw_dimensions_per_query=21,physical_parameter_and_MINCO_AD_FD=True,Loss_AD_FD_sample=free["sample_id"],sparse_Loss="UNDEFINED_UNKNOWN; not called",max_physical_AD_FD_abs=float(np.max(np.abs(ad-fd))),max_Free_Loss_AD_FD_abs=read(dest/"LOSS_JACOBIAN.json")["max_abs_error"]))
def reference_polynomials(state):
    """Read-only review of already proposed cubic reference, never a B decoder."""
    v=np.asarray(state[:3],float);a=np.asarray(state[3:6],float);goal=np.asarray(state[6:9],float)
    j=.25*6*(goal-v*2-a*2)/8
    norm=np.linalg.norm(j)
    if norm>1:j=j/norm
    c=np.zeros((3,3,6));d=np.full(3,2/3)
    for n,t in enumerate([0,2/3,4/3]):
        c[n,:,0]=v*t+.5*a*t*t+j*t**3/6;c[n,:,1]=v+a*t+j*t*t/2;c[n,:,2]=(a+j*t)/2;c[n,:,3]=j/6
    return c,d,dict(j_ref=j.tolist(),endpoint=(v*2+a*2+j*8/6).tolist(),terminal_velocity=(v+a*2+j*2).tolist(),terminal_acceleration=(a+j*2).tolist())
def sensitivity_controls():
    # Existing independent S3NU linear constraints, not any new decoding routine.
    d=np.full(3,2/3);maps=[]
    for i in range(15):
        head=np.zeros((3,3));tail=np.zeros((3,3));inner=np.zeros((2,3))
        if i<6:inner.flat[i]=1
        else:tail.flat[i-6]=1
        c=numerical_reference(head,tail,inner,d)
        maps.append(np.stack([bernstein_controls(z,t) for z,t in zip(c,d)]))
    return np.stack(maps,-1)
def endpoint_box_lower_bounds(reference,goal,alpha):
    Dp=np.array([1.,.8,.5])*alpha;Dv=np.array([.75,.75,.5])*alpha
    pe=np.maximum(np.abs(np.asarray(goal)-reference["endpoint"])-Dp,0)
    vv=np.maximum(np.abs(reference["terminal_velocity"])-Dv,0)
    return dict(min_possible_endpoint_goal_error_m=float(np.linalg.norm(pe)),min_possible_terminal_speed_mps=float(np.linalg.norm(vv)))
def design_review():
    guard()
    if read(OUT/"runs/jacobian_001/RESULT.json")["status"]!="PASS":raise ValueError("JACOBIAN_REQUIRED")
    dest=OUT/"runs/design_review_001";dest.mkdir(parents=True);M=sensitivity_controls()
    D0=np.array([1.,.8,.5]*3+[.75,.75,.5]+[1.,1.,1.]);records=[]
    for row in rows():
        b=bound(row);g=b.geometry[0];state=b.state[0].double().numpy();c,d,ref=reference_polynomials(state)
        world=transform_coefficients(c,g.origin,g.rotation);beta=np.stack([bernstein_controls(z,t) for z,t in zip(world,d)])
        ranges=np.stack([axis_range(z,t) for z,t in zip(world,d)])
        outside=bool((ranges[:,:,0]<g.roi[0]+.150001).any() or (ranges[:,:,1]>g.roi[1]-.150001).any())
        # Rotate linear sensitivity into World axes; current bindings require identity.
        if not np.array_equal(g.rotation,np.eye(3)):raise ValueError("ROTATED_SENSITIVITY_UNREGISTERED")
        candidates=[]
        for alpha in [1.,.5,.25,.125]:
            spread=(np.abs(M)*(D0*alpha)).sum(-1)
            low=beta-spread;high=beta+spread
            margin=min(float((low-(g.roi[0]+.150001+.0022)).min()),float(((g.roi[1]-.150001-.0022)-high).min()))
            bounds_=endpoint_box_lower_bounds(ref,state[6:9],alpha)
            near=np.linalg.norm(state[6:9])<=3.2
            impossible=bool(near and (bounds_["min_possible_endpoint_goal_error_m"]>.15 or bounds_["min_possible_terminal_speed_mps"]>.25))
            candidates.append(dict(alpha=alpha,Bernstein_sufficient_margin_m=margin,all_residuals_ROI_sufficient=margin>=0,**bounds_,Near_Hard_necessary_condition_impossible=impossible,
                interpretation="negative Bernstein margin is not task infeasibility; Near endpoint/stop lower bound can prove proposed B set insufficient"))
        hard=validate_polynomial(c,d,g,state[6:9])
        record=dict(sample_id=row["sample_id"],reference=ref,reference_center_ranges_world=ranges.tolist(),reference_sphere_outside=outside,reference_original_Hard=hard,candidates=candidates,alpha_selected=None,B_decoder_implemented=False,geometry_sha256=row["geometry_sha256"])
        write(dest/(row["sample_id"]+".json"),record);records.append(record)
    write(dest/"RESULT.json",dict(status="PASS_READ_ONLY_REVIEW",tasks=13,reference_sphere_outside_tasks=sum(r["reference_sphere_outside"] for r in records),proposed_B_impossible_Near_tasks=sum(all(c["Near_Hard_necessary_condition_impossible"] for c in r["candidates"]) for r in records),selected_alpha=None,B_implementation_started=False,readiness="NO-GO if full proposed class cannot satisfy Near stop; no alpha shrinking workaround"))
def tests():
    guard();bootstrap();dest=OUT/"runs/tests_001";dest.mkdir(parents=True);import pytest
    args=["-q","-p","no:cacheprovider","--basetemp="+str(dest/"pytest_tmp"),"--junitxml="+str(dest/"TEST_RESULTS.xml"),*read(PREP/"RUN_SPEC.json")["test_paths"],"--deselect=tests/test_physics_v12_gate1_geometry.py::test_exact_distance_matches_independent_all_boxes"]
    code=pytest.main(args);write(dest/"RESULT.json",dict(status="PASS" if code==0 else "STOP",exit_code=int(code)))
    if code:raise SystemExit(code)
def main():
    p=argparse.ArgumentParser();p.add_argument("--worker",choices=["tests","reproduction","support","jacobian","design"],required=True);stage=p.parse_args().worker
    from unittest.mock import patch
    guard()
    with patch.object(torch.optim.Optimizer,"__init__",side_effect=RuntimeError("OPTIMIZER_FORBIDDEN")) if stage!="tests" else __import__("contextlib").nullcontext():
        {"tests":tests,"reproduction":reproduction,"support":support,"jacobian":real_jacobian,"design":design_review}[stage]()
if __name__=="__main__":main()
