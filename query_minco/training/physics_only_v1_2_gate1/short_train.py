"""Read-only initialization audit and separately authorized single-sample trainer.

No training is reachable without an exact researcher approval and frozen RunSpec.
No model, decoder, MINCO, loss, or hard threshold is reimplemented here.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import torch
import numpy as np
from query_minco.models import QueryMincoPlanner
from query_minco.training.physics_only_v1.smoke import parameter_hash
from query_minco.training.physics_only_v1_2.spec import PhysicsV12Spec
from query_minco.training.physics_only_v1_2_gate1.binding import load_bound
from query_minco.training.physics_only_v1_2_gate1.objective import BoundPhysicsV12Objective
from query_minco.training.physics_only_v1_2_gate1.hard_validator import validate_result
from query_minco.training.physics_only_v1_2_gate1.preflight import stats
from query_minco.training.pg1.losses import analytic_samples
from query_minco.evaluation.validation import evaluate,norm_max

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/"research_outputs/physics_only_v1_2_first_short_train_v1"
OLD=ROOT/"research_outputs/physics_only_v1_2_gate1_forest_v1"
INITIAL_SHA="73bac72015dcaf768d92414c90a6162cc298431843b527bd9c202481b002b1d1"
MANIFEST=OLD/"runs/bind_posecert_001/BOUND_MANIFEST.json"
MANIFEST_SHA="fd8237790ec5bb2d7b33fce6f8b66a4398c321b0d4ee6a01a2284f449b68445f"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,data):
    with Path(path).open("x") as f:json.dump(data,f,indent=2,allow_nan=False);f.write("\n")


def verify_files(hashes):
    for p,h in hashes.items():
        if sha(ROOT/p if not Path(p).is_absolute() else p)!=h:
            raise ValueError("SHA_DRIFT "+p)


def initialize():
    torch.set_num_threads(1);torch.manual_seed(20261104)
    return QueryMincoPlanner().cpu().eval()


def decode_rows(result):
    p=result.parameters;tr=result.trajectory
    rows=[]
    for k in range(6):
        c=tr.coeffs[0,k].detach().double().numpy()
        dt=tr.durations[0,k].detach().double().numpy()
        points=np.vstack([np.zeros(3),p.inner_pts[0,k].detach().numpy(),
                          p.tail_pva[0,k,0].detach().numpy()])
        segments=[]
        for i,T in enumerate(dt):
            # Average speed from numerical arc length, independently of extrema.
            times=np.linspace(0,T,8193);v=evaluate(c[i],times,1)
            speed=np.linalg.norm(v,axis=-1)
            segments.append(dict(segment=i,duration_s=float(T),
                displacement_m=(points[i+1]-points[i]).tolist(),
                displacement_over_time_mps=float(np.linalg.norm(points[i+1]-points[i])/T),
                mean_velocity_mps=((points[i+1]-points[i])/T).tolist(),
                mean_speed_arc_mps=float(np.trapz(speed,times)/T),
                peak_speed_mps=norm_max(c[i],T,1),
                peak_acceleration_mps2=norm_max(c[i],T,2),peak_jerk_mps3=norm_max(c[i],T,3)))
        rows.append(dict(query=k,waypoints_B0_m=points[1:].tolist(),
            tail_velocity_mps=p.tail_pva[0,k,1].detach().tolist(),
            tail_acceleration_mps2=p.tail_pva[0,k,2].detach().tolist(),segments=segments,
            goal_task=float(result.metrics["G_task_train"][0,k]),progress=float(result.metrics["P"][0,k])))
    return rows


def audit_initialization(run):
    from unittest.mock import patch
    entry=json.loads((OUT/"_preparation/plan_001/ENTRY.json").read_text())
    verify_files(entry["protected_sha256"])
    batch=load_bound(MANIFEST,MANIFEST_SHA)
    config=PhysicsV12Spec(phase="A",aggregation="foundation")
    with patch.object(torch.optim.Optimizer,"__init__",side_effect=RuntimeError("NO_OPTIMIZER_IN_AUDIT")):
        model=initialize();before=parameter_hash(model)
        raw=model(batch.depth,batch.state)
        objective=BoundPhysicsV12Objective(config)
        result=objective(raw,batch)
        reference=torch.tensor(json.loads((OLD/"runs/sdf_c_phase_a_001/FORWARD.json").read_text())["raw"])
        torch.testing.assert_close(raw,reference,atol=0,rtol=0)
        zero=objective(torch.zeros_like(raw),batch)
        if parameter_hash(model)!=before or before!=INITIAL_SHA:raise ValueError("INITIAL_MODEL_SHA")
        record=dict(status="AUDIT_ONLY_ZERO_UPDATES",model_sha256_before=before,
            model_sha256_after=parameter_hash(model),raw=raw.detach().tolist(),raw_stats=stats(raw),
            rows=decode_rows(result),zero_logits_reference_rows=decode_rows(zero),
            per_query={k:v.detach().tolist() for k,v in result.metrics.items()},
            independent_hard=validate_result(result,batch),loss=float(result.loss),
            head_state=batch.state.tolist(),manifest_sha256=MANIFEST_SHA,
            actual_input_shapes=[list(batch.depth.shape),list(batch.state.shape)],
            raw_matches_gate1_exactly=True,optimizer_constructed=False,optimizer_updates=0)
    verify_files(entry["protected_sha256"]);write(run/"INITIALIZATION.json",record)
    print(json.dumps(dict(status=record["status"],loss=record["loss"],model_sha256=before)),flush=True)


def check_authorization(spec_path,approval_path):
    """Reject before constructing a model, optimizer, run, or checkpoint."""
    spec_path=Path(spec_path)
    if approval_path is None:raise PermissionError("RESEARCHER_APPROVAL_REQUIRED")
    approval=json.loads(Path(approval_path).read_text());spec=json.loads(spec_path.read_text())
    if (approval.get("decision")!="APPROVE_EXACT_RUNSPEC" or
        approval.get("runspec_sha256")!=sha(spec_path) or
        approval.get("authority")!="researcher_explicit_message" or
        not approval.get("message_reference")):raise PermissionError("APPROVAL_NOT_BOUND")
    if spec["schema"]!="physics_v12_first_short_train_runspec_v1":raise ValueError("RUNSPEC_SCHEMA")
    if spec["max_updates"]!=32 or spec["evaluation_steps"]!=[0,8,16,32]:raise ValueError("UPDATE_BUDGET")
    if spec["optimizer"]!=dict(name="AdamW",lr=1e-4,betas=[.9,.999],eps=1e-8,
                              weight_decay=0.,amsgrad=False,foreach=False):raise ValueError("OPTIMIZER_SPEC")
    if spec["gradient_clip_norm"]!=10. or spec["seed"]!=20261104:raise ValueError("TRAIN_SPEC")
    if spec["physics_spec"]!=__import__("dataclasses").asdict(PhysicsV12Spec(phase="A",aggregation="foundation")):
        raise ValueError("FROZEN_PHYSICS_SPEC")
    if spec["initial_model_sha256"]!=INITIAL_SHA or spec["manifest_sha256"]!=MANIFEST_SHA:
        raise ValueError("INITIAL_DATA_IDENTITY")
    for repo,key in ((ROOT,"query_head"),(Path("/home/lizheng/YOPO"),"yopo_head")):
        head=subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()
        if head!=spec[key]:raise ValueError("HEAD_DRIFT")
        if subprocess.check_output(["git","-C",str(repo),"diff","HEAD","--name-only"],text=True):
            raise ValueError("TRACKED_DRIFT")
    verify_files(spec["source_sha256"]);verify_files(spec["input_sha256"])
    if sha(spec["history_manifest_path"])!=spec["history_manifest_sha256"]:raise ValueError("HISTORY_LIST_DRIFT")
    verify_files(json.loads(Path(spec["history_manifest_path"]).read_text())["protected_sha256"])
    status=json.loads((OLD/"audit/sdf_zero_update_001/FINAL_STATUS.json").read_text())
    if status["gate1_B"]!="GATE1_B_PASS" or status["gate1_C"]!="GATE1_C_PASS":raise ValueError("GATE1_NOT_READY")
    destination=Path(spec["run_directory"]).resolve()
    expected=(OUT/"runs/phase_a_seed20261104_train001").resolve()
    if destination!=expected or destination.exists():raise ValueError("RUN_EXISTS_OR_PATH")
    return spec


def gradients(result,raw,model):
    named=[(n,p) for n,p in model.named_parameters() if not n.startswith("trajectory_head.")
           and n!="query_decoder.learned_queries"]
    shared=[p for _,p in named]
    def grad(value):
        gs=torch.autograd.grad(value,(raw,*shared),retain_graph=True,allow_unused=True)
        rg=gs[0] if gs[0] is not None else torch.zeros_like(raw)
        sg=torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g,p in zip(gs[1:],shared)])
        if rg[...,15:21].count_nonzero():raise ValueError("TIME_SCORE_GRADIENT")
        return dict(raw=stats(rg[...,:15]),raw_values=rg.detach().tolist(),shared=stats(sg),
                    per_query_raw=[stats(rg[0,k,:15]) for k in range(6)])
    # Both diagnostic components and actual upstream-weighted contributions.
    names=("C_viol","C_shape","G_N","G_F","G_task_train","P","S","F_dyn","F_domain","F","V","Q","J")
    report={n:grad(result.metrics[n].mean()) for n in names}
    effective={}
    for n in ("C_viol","F","G_task_train","C_shape","P","S"):
        term=result.metrics[n];up=torch.autograd.grad(result.loss,term,retain_graph=True)[0].detach()
        effective[n]=dict(upstream=up.tolist(),**grad((up*term).sum()))
    return dict(components=report,effective=effective,total=grad(result.loss))


def evaluation_record(result,batch):
    hard=validate_result(result,batch)
    return dict(per_query={k:v.detach().tolist() for k,v in result.metrics.items()},
                independent_hard=hard,hard_all_pass=sum(x["passed"] for x in hard),
                solver_residuals={k:v.detach().tolist() for k,v in result.trajectory.diagnostics.items()},
                coefficients_B0=result.trajectory.coeffs.detach().tolist(),
                durations=result.trajectory.durations.detach().tolist(),
                note="sampled min_sdf is not a certified global minimum; failed Hard certificate has no invented positive clearance")


def execute_training(spec_path,approval_path):
    # This function is NOT executed or optimizer-tested in the planning task.
    spec=check_authorization(spec_path,approval_path)
    batch=load_bound(MANIFEST,spec["manifest_sha256"])
    model=initialize()
    if parameter_hash(model)!=spec["initial_model_sha256"]:raise ValueError("INIT_MODEL")
    run=Path(spec["run_directory"]);run.mkdir()
    write(run/"APPROVAL_BINDING.json",dict(approval_sha256=sha(approval_path),runspec_sha256=sha(spec_path)))
    write(run/"RUN_SPEC.json",spec)
    opt=spec["optimizer"]
    optimizer=torch.optim.AdamW(model.parameters(),lr=opt["lr"],betas=tuple(opt["betas"]),
        eps=opt["eps"],weight_decay=opt["weight_decay"],amsgrad=opt["amsgrad"],foreach=opt["foreach"])
    objective=BoundPhysicsV12Objective(PhysicsV12Spec(**spec["physics_spec"]))
    checkpoint_dir=run/"checkpoints";checkpoint_dir.mkdir()
    baseline=None;last=None;updates=0
    def save_checkpoint(step):
        path=checkpoint_dir/("step_"+str(step).zfill(3)+".pt")
        with path.open("xb") as f:torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
            step=step,runspec_sha256=sha(spec_path),manifest_sha256=MANIFEST_SHA),f)
        write(checkpoint_dir/("step_"+str(step).zfill(3)+".json"),
            dict(path=str(path),file_sha256=sha(path),model_sha256=parameter_hash(model),step=step,resume_allowed=False))
    try:
        with (run/"STEPS.jsonl").open("x") as log:
            for step in range(spec["max_updates"]+1):
                verify_files(spec["source_sha256"]);batch.validate()
                optimizer.zero_grad(set_to_none=True)
                raw=model(batch.depth,batch.state);result=objective(raw,batch)
                value=float(result.loss);baseline=value if baseline is None else baseline
                if value>2*baseline and last is not None and last>2*baseline:
                    raise ValueError("LOSS_DIVERGENCE_TWO_CONSECUTIVE")
                last=value
                grad_report=gradients(result,raw,model)
                record=dict(completed_updates=updates,model_sha256=parameter_hash(model),loss=value,
                    gradients=grad_report,raw=raw.detach().tolist(),**evaluation_record(result,batch))
                if step in spec["evaluation_steps"]:save_checkpoint(step)
                if step==spec["max_updates"]:
                    record["update_executed"]=False
                    log.write(json.dumps(record,allow_nan=False)+"\n");log.flush();break
                result.loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),spec["gradient_clip_norm"],error_if_nonfinite=True)
                before=[p.detach().clone() for p in model.parameters()]
                optimizer.step();updates+=1
                delta=torch.cat([(p.detach()-b).reshape(-1) for p,b in zip(model.parameters(),before)])
                if not bool(torch.isfinite(delta).all()):raise ValueError("NONFINITE_UPDATE")
                record.update(update_executed=True,gradient_norm_before_clip=float(norm),parameter_delta=stats(delta),
                              model_sha256_after_update=parameter_hash(model),completed_updates_after=updates)
                log.write(json.dumps(record,allow_nan=False)+"\n");log.flush()
        write(run/"TRAIN_RESULT.json",dict(status="COMPLETED_ENGINEERING_CHECK",updates=updates,
            model_sha256=parameter_hash(model),generalization_established=False))
    except BaseException as e:
        write(run/"TRAIN_FAILURE.json",dict(error=repr(e),completed_updates=updates,automatic_resume=False))
        raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",required=True,choices=("audit","train"))
    parser.add_argument("--output-run");parser.add_argument("--spec");parser.add_argument("--approval")
    args=parser.parse_args()
    if args.stage=="audit":audit_initialization(Path(args.output_run))
    else:execute_training(args.spec,args.approval)

if __name__=="__main__":main()
