"""Original decoder support audit; no residual decoder or training implementation."""
import argparse,csv,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from .e1_r3a_v2 import bootstrap
from .e1_r3_zero_update import phase_spec,roi_diagnostics
from .e1_dataset_binding import load_e1_bound
from .short_train import initialize,INITIAL_SHA
from query_minco.training.physics_only_v1.smoke import parameter_hash
from query_minco.training.physics_only_v1.decoder import FixedTimeDecoder
from query_minco.trajectory.parameter_decoder import PhysicalParameterDecoder
from query_minco.trajectory.minco_adapter import MincoAdapter
from query_minco.training.pg1.losses import analytic_samples
from .geometry import world_points
from .hard_validator import validate_result,transform_coefficients
from query_minco.evaluation.validation import evaluate,axis_range
BASE=Path(__file__).resolve().parents[3]
OUT=BASE/"research_outputs/physics_only_v1_2_e1_r3a_v3"
PREP=OUT/"_preparation/freeze_001"
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
        if sha(BASE/p)!=h:raise ValueError("SOURCE_SHA_DRIFT "+p)
    for p,h in read(PREP/"INPUT_HASHES.json").items():
        if sha(p)!=h:raise ValueError("INPUT_SHA_DRIFT "+p)
    if time.time()-e["started_unix"]>3600:raise ValueError("TOTAL_BUDGET")
def polynomial_exit(coefficients,durations,low,high):
    """Partition real polynomial boundary roots; classify every open interval."""
    intervals=[];offset=0.;roots=[]
    for segment,(c,T) in enumerate(zip(coefficients,durations)):
        ts=[0.,float(T)]
        for axis in range(3):
            for bound in (low[axis],high[axis]):
                a=c[axis].copy();a[0]-=bound
                for z in np.polynomial.polynomial.polyroots(a):
                    if abs(z.imag)<=1e-7 and -1e-9<=z.real<=T+1e-9:ts.append(float(np.clip(z.real,0,T)))
        ts=sorted(ts);knots=[]
        for t in ts:
            if not knots or t-knots[-1]>1e-9:knots.append(t)
        for left,right in zip(knots[:-1],knots[1:]):
            mid=evaluate(c,(left+right)/2)
            if ((mid<low)|(mid>high)).any():intervals.append(dict(segment=segment,local_start=left,local_end=right,start_s=offset+left,end_s=offset+right))
        roots.append(knots);offset+=float(T)
    return dict(first_exit_s=intervals[0]["start_s"] if intervals else None,outside_intervals=intervals,piece_boundary_roots=roots,total_time_s=offset,semantics="infimum of continuous-time open outside intervals; not sampled index time")
def decode(raw,state):
    p=FixedTimeDecoder(phase_spec())(raw,state)
    tr=MincoAdapter()(p.head_pva,p.tail_pva,p.inner_pts,p.durations)
    return p,tr
def vector_outputs(raw,state):
    p,tr=decode(raw,state)
    return torch.cat([torch.cat([p.inner_pts,p.tail_pva[:,:,0:1]],2).flatten(2),p.head_pva.flatten(2),p.tail_pva.flatten(2),tr.coeffs.flatten(2),p.durations,p.score[...,None],p.feas_logit[...,None]],2)
def jacobian_audit(raw,state,free_batch,free_raw,dest):
    raw=raw.detach().requires_grad_(True)
    f=lambda x:vector_outputs(x,state)
    J=torch.autograd.functional.jacobian(f,raw,vectorize=True)
    # Shape [1,6,86,1,6,21]; only same-query blocks may be nonzero.
    ad=np.stack([J[0,k,:,0,k,:].detach().double().numpy() for k in range(6)])
    groups={"waypoints":(0,9),"head_PVA":(9,18),"terminal_PVA":(18,27),"MINCO_coefficients":(27,81),"durations":(81,84),"score_logits":(84,86)}
    fd=np.zeros_like(ad);fd2=np.zeros_like(ad);h=.002
    for k in range(6):
        for dim in range(21):
            for epsilon,store in ((h,fd),(h/2,fd2)):
                plus=raw.detach().clone();minus=raw.detach().clone();plus[0,k,dim]+=epsilon;minus[0,k,dim]-=epsilon
                with torch.no_grad():v=(f(plus)-f(minus))/(2*epsilon)
                store[k,:,dim]=v[0,k].double().numpy()
    np.savez(dest/"RAW21_JACOBIAN.npz",AD=ad,FD_h002=fd,FD_h001=fd2)
    rows=[]
    for k in range(6):
        for dim in range(21):
            row=dict(query=k,raw_dim=dim)
            for name,(a,b) in groups.items():
                aa=ad[k,a:b,dim];ff=fd[k,a:b,dim]
                row[name+"_AD_norm"]=float(np.linalg.norm(aa));row[name+"_FD_norm"]=float(np.linalg.norm(ff))
                row[name+"_max_abs_error"]=float(np.max(np.abs(aa-ff)))
                if not np.allclose(aa,ff,atol=.003,rtol=.003):raise ValueError("NEW_FP32_JACOBIAN_MISMATCH "+str((k,dim,name,row)))
            row["FD_halving_max_abs_change"]=float(np.max(np.abs(fd[k,:,dim]-fd2[k,:,dim])))
            rows.append(row)
    legacy=torch.autograd.functional.jacobian(lambda r:PhysicalParameterDecoder()(r,state).durations,raw,vectorize=True).detach()
    write(dest/"LEGACY_TIME_JACOBIAN.json",dict(legacy_per_dimension_norm=[float(legacy[...,i].double().norm()) for i in range(21)],fixed_time_per_dimension_norm=[float(np.linalg.norm(ad[:,81:84,i])) for i in range(21)]))
    # Loss Jacobian uses certified Free sample ONLY, no fake loss outside sparse ROI.
    from .objective import BoundPhysicsV12Objective
    q=free_raw.detach().requires_grad_(True);obj=BoundPhysicsV12Objective(phase_spec());loss=obj(q,free_batch).loss
    g=torch.autograd.grad(loss,q)[0].detach().double().numpy();fl=np.zeros((6,21));fl2=np.zeros_like(fl)
    for k in range(6):
        for dim in range(21):
            for epsilon,store in ((h,fl),(h/2,fl2)):
                x=q.detach().clone();y=q.detach().clone();x[0,k,dim]+=epsilon;y[0,k,dim]-=epsilon
                with torch.no_grad():store[k,dim]=float((obj(x,free_batch).loss-obj(y,free_batch).loss)/(2*epsilon))
    if not np.allclose(g[0],fl,atol=.005,rtol=.005):raise ValueError("NEW_FP32_LOSS_FD_MISMATCH")
    if np.count_nonzero(ad[:,:84,15:]) or np.count_nonzero(fd[:,:84,15:]) or np.count_nonzero(g[...,15:]):raise ValueError("INACTIVE_PHYSICAL_PATH_NONZERO")
    write(dest/"LOSS_JACOBIAN.json",dict(sample_id="train_free_roi_1101_p0_near",loss=float(loss),AD=g.tolist(),FD_h002=fl.tolist(),FD_h001=fl2.tolist(),max_abs_error=float(np.max(np.abs(g[0]-fl))),sparse_loss="UNDEFINED_UNKNOWN; no Jacobian or finite difference fabricated",new_FP32_diagnostic_tolerance=dict(physical_atol=.003,physical_rtol=.003,loss_atol=.005,loss_rtol=.005),original_gradcheck_tolerances_unchanged=True))
    with (dest/"RAW21_JACOBIAN.csv").open("x",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def audit():
    guard();bootstrap();dest=OUT/"runs/support_audit_001";dest.mkdir(parents=True)
    from unittest.mock import patch
    from types import SimpleNamespace
    records=[];matrix=[];rows=read(PREP/"TRAIN13_AND_EXCLUSIONS.json")["Train13"]
    with patch.object(torch.optim.Optimizer,"__init__",side_effect=RuntimeError("OPTIMIZER_FORBIDDEN")),patch.object(torch,"load",side_effect=RuntimeError("CHECKPOINT_FORBIDDEN")),patch.object(torch,"save",side_effect=RuntimeError("CHECKPOINT_SAVE_FORBIDDEN")):
        model=initialize();before=parameter_hash(model)
        if before!=INITIAL_SHA:raise ValueError("INITIAL_MODEL_SHA")
        write(dest/"MODEL_BEFORE.json",dict(sha256=before,seed=20261104,checkpoint=None))
        try:
            for row in rows:
                guard();b=load_e1_bound(row["source_manifest_path"],row["source_manifest_sha256"]);b.validate()
                with torch.no_grad():
                    raw=model(b.depth,b.state);p,tr=decode(raw,b.state);ana=analytic_samples(tr,64)
                if not torch.isfinite(raw).all():raise ValueError("NONFINITE_RAW")
                c=tr.coeffs[0].double().numpy();d=tr.durations[0].double().numpy();g=b.geometry[0]
                roi=roi_diagnostics(ana["p"][0],g,c,d)
                hard=validate_result(SimpleNamespace(parameters=p,trajectory=tr),b)
                # Unknowns remain unqueried. This partition is geometry audit, not masked Loss.
                points=ana["p"][0];world=world_points(points,torch.tensor(g.origin,dtype=points.dtype),torch.tensor(g.rotation,dtype=points.dtype)).double().numpy()
                known=((world>=g.roi[0])&(world<=g.roi[1])).all(-1)
                distances=np.full(known.shape,np.nan)
                if known.any():
                    with torch.no_grad():distances[known]=g.sdf(points[known]).double().numpy()
                rec=dict(sample_id=row["sample_id"],scene_id=row["scene_id"],source_sha=row,raw21=raw.tolist(),waypoints_B0=torch.cat([p.inner_pts,p.tail_pva[:,:,0:1]],2).tolist(),head_PVA=p.head_pva.tolist(),tail_PVA=p.tail_pva.tolist(),coefficients_B0=tr.coeffs.tolist(),durations=tr.durations.tolist(),sampled_p_B0=ana["p"].tolist(),roi=roi,independent_hard=hard,queries=[],physical_objective_executed=False,Loss_backward_qualification="NOT_ATTEMPTED_SUPPORT_AUDIT_ONLY")
                for k in range(6):
                    wc=transform_coefficients(c[k],g.origin,g.rotation)
                    exit_=polynomial_exit(wc,d[k],g.roi[0],g.roi[1])
                    sphere=polynomial_exit(wc,d[k],g.roi[0]+.150001,g.roi[1]-.150001)
                    kk=known[k];ds=distances[k]
                    counts=dict(UNKNOWN=int((~kk).sum()),KNOWN_NONPOSITIVE_SDF=int((kk&(ds<=0)).sum()),KNOWN_POSITIVE_UNDER_015=int((kk&(ds>0)&(ds<.15)).sum()),KNOWN_CLEARANCE_GE_015=int((kk&(ds>=.15)).sum()))
                    # Boundary-limited SDF is distinguished from actual voxel/floor penetration.
                    native_occupied=0
                    for point in world[k,kk]:
                        q=np.abs(point-g.centers)-g.halves
                        sd=np.linalg.norm(np.maximum(q,0),axis=1)+np.minimum(q.max(1),0)
                        native_occupied+=int(sd.min()<=0 or point[2]<=g.floor)
                    rec["queries"].append(dict(query=k,center_exit=exit_,reference_sphere_exit=sphere,sampled_classes=counts,native_occupied_or_floor_centers=native_occupied,known_free_center_count=int(kk.sum())-native_occupied))
                    local=(g.roi-g.origin)@g.rotation
                    matrix.append(dict(sample_id=row["sample_id"],scene_id=row["scene_id"],query=k,roi_B0_low=json.dumps(local[0].tolist()),roi_B0_high=json.dumps(local[1].tolist()),raw_min=float(raw[0,k].min()),raw_max=float(raw[0,k].max()),waypoint_x_min=float(np.asarray(rec["waypoints_B0"])[0,k,:,0].min()),waypoint_x_max=float(np.asarray(rec["waypoints_B0"])[0,k,:,0].max()),continuous_center_outside=exit_["first_exit_s"] is not None,first_center_exit_s=exit_["first_exit_s"],first_reference_sphere_exit_s=sphere["first_exit_s"],sampled_unknown=counts["UNKNOWN"],native_occupied_centers=native_occupied,known_free_centers=int(kk.sum())-native_occupied,Hard_pass=hard[k]["passed"],representation_box_contained_in_roi=bool((np.array([0,-4,-2])>=local[0]).all() and (np.array([10,4,2])<=local[1]).all()),loss_qualification="NOT_EVALUATED",**counts))
                if row["sample_id"]=="train_sparse_1102_p0_near":
                    old=read(BASE/"research_outputs/physics_only_v1_2_e1_r3a_v2/runs/zero_update_001/train_sparse_1102_p0_near.json")
                    diffs={name:float(np.max(np.abs(np.asarray(rec[newname])-np.asarray(old[oldname])))) for name,newname,oldname in [("raw","raw21","raw_logits"),("waypoints","waypoints_B0","waypoints_B0"),("coefficients","coefficients_B0","coefficients_B0"),("sampled_p","sampled_p_B0","sampled_p_B0")]}
                    if any(diffs.values()):raise ValueError("V2_EXACT_REPRODUCTION_MISMATCH")
                    try:g.sdf(points)
                    except ValueError as ex:
                        if str(ex)!="UNKNOWN_OUTSIDE_CERTIFIED_ROI":raise
                        rejection=str(ex)
                    else:raise ValueError("UNKNOWN_NOT_REJECTED")
                    write(dest/"V2_EXACT_REPRODUCTION.json",dict(status="PASS",max_abs_differences=diffs,production_sdf_exception=rejection,old_stop_sha256=sha(BASE/"research_outputs/physics_only_v1_2_e1_r3a_v2/_preparation/freeze_001/STOP.json")))
                    sparse_raw=raw.clone();sparse_state=b.state.clone()
                if row==rows[0]:free_raw=raw.clone();free_batch=b
                write(dest/(row["sample_id"]+".json"),rec);records.append(rec)
                print(json.dumps(dict(sample_id=row["sample_id"],center_outside_queries=sum(q["center_exit"]["first_exit_s"] is not None for q in rec["queries"]),sampled_unknown=sum(q["sampled_classes"]["UNKNOWN"] for q in rec["queries"])),flush=True))
            jacobian_audit(sparse_raw,sparse_state,free_batch,free_raw,dest)
            after=parameter_hash(model)
            if after!=before:raise ValueError("MODEL_UPDATED")
            write(dest/"MODEL_AFTER.json",dict(before=before,after=after,unchanged=True))
            with (OUT/"summary/freeze_001/TRAIN13_ROI_SUPPORT_MATRIX.csv").open("x",newline="") as f:
                w=csv.DictWriter(f,fieldnames=list(matrix[0]));w.writeheader();w.writerows(matrix)
            write(dest/"RESULT.json",dict(research_diagnostic="PASS",support_audit_tasks=len(records),queries=len(matrix),center_outside_tasks=sum(any(q["center_exit"]["first_exit_s"] is not None for q in r["queries"]) for r in records),center_outside_queries=sum(x["continuous_center_outside"] for x in matrix),all13_physical_Loss_forward_backward_PASS=False,optimizer_updates=0,CUDA_calls=0,checkpoint_loaded=False,R3b="NO-GO"))
        except BaseException:
            import traceback
            write(dest/"FAILURE.json",dict(status="STOP",error=traceback.format_exc(),completed_tasks=len(records),model_sha256=parameter_hash(model)))
            raise
def tests():
    guard();bootstrap();dest=OUT/"runs/tests_001";dest.mkdir(parents=True)
    import pytest
    spec=read(PREP/"RUN_SPEC.json")
    args=["-q","-p","no:cacheprovider","--basetemp="+str(dest/"pytest_tmp"),"--junitxml="+str(dest/"TEST_RESULTS.xml"),*spec["test_paths"],"--deselect=tests/test_physics_v12_gate1_geometry.py::test_exact_distance_matches_independent_all_boxes"]
    code=pytest.main(args);write(dest/"RESULT.json",dict(status="PASS" if code==0 else "STOP",exit_code=int(code)))
    if code:raise SystemExit(code)
def main():
    p=argparse.ArgumentParser();p.add_argument("--worker",choices=["tests","audit"],required=True);a=p.parse_args();{"tests":tests,"audit":audit}[a.worker]()
if __name__=="__main__":main()
