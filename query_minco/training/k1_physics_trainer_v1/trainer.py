"""Independent shared-network K1 physics lifecycle, exact authorization and checkpoints."""
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from query_minco.training.physics_only_v1_2_gate1.k1_interface import make_objective
from query_minco.training.physics_only_v1_2_gate1.e1_dataset_binding import load_e1_bound
from query_minco.training.physics_only_v1_2_gate1.hard_validator import validate_polynomial
from query_minco.evaluation.validation import numerical_reference, evaluate as poly_evaluate
from .initialization import build
from .optimizer import DeadRowGuard, digest

TRAINER_ID = "k1_shared_physics_L0_p0c_trainer_v1"
ROOT = Path(__file__).resolve().parents[3]
TERM_NAMES = ("C_viol", "F", "G_task_train", "C_shape", "P", "S")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def state_sha(model):
    h = hashlib.sha256()
    for name, p in model.state_dict().items():
        h.update(name.encode()); h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def gradient_sha(model):
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        h.update(name.encode())
        h.update(b"NONE" if p.grad is None else p.grad.detach().numpy().tobytes())
    return h.hexdigest()


def write(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)


def verify_spec(path):
    path = Path(path)
    spec = json.loads(path.read_text())
    if spec["schema"] != "k1_physics_pilot_runspec_v1" or spec["trainer_id"] != TRAINER_ID:
        raise ValueError("RUNSPEC_SCHEMA")
    if spec["batch_size"] != 1 or spec["device"] != "cpu" or spec["dtype"] != "float32":
        raise ValueError("CPU_FP32_B1_REQUIRED")
    if spec["max_updates"] != 128 or spec["evaluation_steps"] != [0, 8, 32, 64, 128]:
        raise ValueError("PILOT_BUDGET_FROZEN")
    if spec["optimizer"] != dict(lr=1e-4, betas=[.9,.999], eps=1e-8, weight_decay=0.,
                                 amsgrad=False, foreach=False):
        raise ValueError("OPTIMIZER_SPEC")
    if spec["clip_norm"] != 10. or spec["engineering"]["max_steps"] != 2:
        raise ValueError("GRADIENT_OR_ENGINEERING_BUDGET")
    for name, expected in spec["source_sha256"].items():
        if sha(name) != expected:
            raise ValueError("SOURCE_SHA_DRIFT " + name)
    for row in spec["train"] + spec["dev"]:
        if sha(row["manifest"]) != row["manifest_sha256"]:
            raise ValueError("DATA_MANIFEST_DRIFT")
        if row["split"] != ("train" if row in spec["train"] else "dev"):
            raise ValueError("SPLIT_IDENTITY")
    if len(spec["train"]) != 13 or len(spec["dev"]) != 8:
        raise ValueError("TRAIN13_DEV8_REQUIRED")
    if spec["train_order"] != [x["sample_id"] for x in spec["train"]]:
        raise ValueError("FROZEN_ORDER")
    if spec["order_sha256"] != canonical(spec["train_order"]):
        raise ValueError("ORDER_SHA")
    return spec


def authorize(spec_path, approval_path, seed, initialization, output):
    spec = verify_spec(spec_path)
    if approval_path is None:
        raise PermissionError("EXACT_RESEARCHER_AUTHORIZATION_REQUIRED")
    approval = json.loads(Path(approval_path).read_text())
    run_id = initialization + "_seed" + str(seed)
    if (approval.get("decision") != "APPROVE_EXACT_RUNSPEC" or
        approval.get("authority") != "researcher_explicit_message" or
        approval.get("runspec_sha256") != sha(spec_path) or
        not approval.get("message_reference") or
        run_id not in approval.get("approved_run_ids", [])):
        raise PermissionError("AUTHORIZATION_NOT_BOUND_TO_RUNSPEC")
    expected = spec["formal_run_directories"].get(run_id)
    if expected is None or Path(output).resolve() != Path(expected).resolve():
        raise PermissionError("EXACT_OUTPUT_RUN_REQUIRED")
    if seed not in spec["seeds"] or initialization not in spec["initializations"]:
        raise ValueError("SEED_OR_INITIALIZATION")
    return spec


def claim_run(path):
    path = Path(path).resolve()
    base = (ROOT / "research_outputs").resolve()
    if base not in path.parents:
        raise ValueError("OUTPUT_OUTSIDE_RESEARCH_ROOT")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir()
    return path


def model_for(spec, mode, seed):
    random.seed(seed);np.random.seed(seed)
    model = build(mode, seed)
    record = spec["initial_states"][mode][str(seed)]
    if sha(record["path"]) != record["sha256"]:
        raise ValueError("INITIAL_STATE_FILE_DRIFT")
    frozen = np.load(record["path"])
    state = {n:torch.from_numpy(frozen[n].copy()) for n in frozen.files}
    model.load_state_dict(state, strict=True)
    if state_sha(model) != record["model_sha256"]:
        raise ValueError("INITIAL_MODEL_IDENTITY")
    return model


def load(row, required_split):
    if row["split"] != required_split:
        raise PermissionError("TRAIN_DEV_BOUNDARY")
    return load_e1_bound(row["manifest"], row["manifest_sha256"])


def result_numeric(out, batch):
    p, tr = out.parameters, out.trajectory
    if tr.coeffs.shape != (1,1,3,3,6) or not tr.solve_valid.all():
        raise ValueError("K1_MINCO_SHAPE_OR_SOLVE")
    if not torch.equal(tr.durations, torch.full_like(tr.durations, 2/3)):
        raise ValueError("FIXED_TIME_REQUIRED")
    head = torch.stack([torch.zeros(3), batch.state[0,:3], batch.state[0,3:6]])
    if not torch.equal(p.head_pva[0,0], head):
        raise ValueError("HEAD_PVA")
    c = tr.coeffs[0,0].detach().double().numpy()
    dt = tr.durations[0,0].detach().numpy()
    ref = numerical_reference(p.head_pva[0,0].detach().numpy(),p.tail_pva[0,0].detach().numpy(),
                              p.inner_pts[0,0].detach().numpy(),dt)
    errors = [max(float(np.abs(poly_evaluate(a,np.linspace(0,t,65),d)-
                               poly_evaluate(b,np.linspace(0,t,65),d)).max())
                    for a,b,t in zip(c,ref,dt)) for d in (0,1,2)]
    if any(e>tol for e,tol in zip(errors,[.002,.01,.05])):
        raise ValueError("INDEPENDENT_MINCO")
    hard = validate_polynomial(c,dt,batch.geometry[0],batch.state[0,6:9].numpy())
    if any(x.startswith("INVALID") or x=="NONPOSITIVE_DURATION" for x in hard["reasons"]):
        raise ValueError("INVALID_HARD_INPUT")
    return errors, hard


def group_stats(model, before=None):
    report = {};offset = 0
    for name, mod in [("depth",model.depth_encoder),("state",model.state_encoder),
                      ("decoder",model.query_decoder),("head",model.trajectory_head)]:
        params=list(mod.parameters())
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in params):
            raise ValueError("MISSING_OR_NONFINITE_GRADIENT")
        norm=float(torch.cat([p.grad.flatten() for p in params]).norm())
        if not norm>0:
            raise ValueError("ZERO_MODULE_GRADIENT")
        row=dict(gradient_norm=norm)
        if before is not None:
            row["parameter_update_norm"]=float(torch.cat([
                (p.detach()-b).flatten() for p,b in zip(params,before[offset:offset+len(params)])]).norm())
        offset+=len(params);report[name]=row
    return report


def metric_terms(out):
    return {name:out.metrics[name].detach().tolist() for name in TERM_NAMES}


def evaluate(model, objective, rows):
    before=state_sha(model);before_grad=gradient_sha(model);was_training=model.training
    model.eval();records=[]
    with torch.no_grad():
        for row in rows:
            batch=load(row,row["split"])
            raw=model(batch.depth,batch.state);out=objective(raw,batch)
            if not torch.isfinite(out.loss):
                raise ValueError("NONFINITE_EVALUATION")
            errors,hard=result_numeric(out,batch)
            goal=hard["goal"]
            quality=None
            if hard["passed"]:
                quality=dict(S=float(out.metrics["S"].mean()),P=float(out.metrics["P"].mean()),
                             C_shape=float(out.metrics["C_shape"].mean()))
            records.append(dict(sample_id=row["sample_id"],split=row["split"],loss=float(out.loss),
                terms=metric_terms(out),hard=hard,task_passed=goal["passed"],
                hard_and_task=hard["passed"] and goal["passed"],quality_if_feasible=quality,
                independent_minco_errors=errors))
    model.train(was_training)
    if state_sha(model)!=before or gradient_sha(model)!=before_grad:
        raise ValueError("EVALUATION_CHANGED_MODEL_OR_GRADIENTS")
    def summary(split):
        xs=[x for x in records if x["split"]==split]
        feasible=[x["quality_if_feasible"] for x in xs if x["quality_if_feasible"] is not None]
        return dict(samples=len(xs),mean_loss=sum(x["loss"] for x in xs)/len(xs),
                    hard_passed=sum(x["hard"]["passed"] for x in xs),
                    hard_and_task=sum(x["hard_and_task"] for x in xs),
                    feasible_quality=feasible if feasible else None)
    return dict(train=summary("train"),dev=summary("dev"),records=records,
                model_sha_before=before,model_sha_after=state_sha(model),dev_backward_calls=0)


def rng_state():
    n=np.random.get_state()
    return dict(torch=torch.get_rng_state(),python=random.getstate(),
                numpy=[n[0],n[1].tolist(),n[2],n[3],n[4]])


def save_checkpoint(path, model, optimizer, guard, metadata, cursor):
    guard.assert_frozen()
    if any(x["nonzero"] for x in guard.state_report(optimizer).values()):
        raise ValueError("DIRTY_STATE_AT_CHECKPOINT")
    blob=dict(schema="k1_physics_checkpoint_v1",model=model.state_dict(),
              optimizer=optimizer.state_dict(),dead_reference=guard.reference,
              rng=rng_state(),cursor=cursor,metadata=metadata)
    with Path(path).open("xb") as f:torch.save(blob,f)
    write(str(path)+".json",dict(file_sha256=sha(path),model_sha256=state_sha(model),
                                 cursor=cursor,metadata=metadata))
    return blob


def load_checkpoint(path, expected_sha, model, optimizer, guard, metadata):
    if sha(path)!=expected_sha:
        raise ValueError("CHECKPOINT_FILE_SHA")
    blob=torch.load(path,map_location="cpu",weights_only=True)
    if blob["schema"]!="k1_physics_checkpoint_v1" or blob["metadata"]!=metadata:
        raise ValueError("CHECKPOINT_RUN_IDENTITY")
    if len(blob["dead_reference"])!=2:
        raise ValueError("CHECKPOINT_DEAD_REFERENCE")
    model.load_state_dict(blob["model"],strict=True)
    optimizer.load_state_dict(blob["optimizer"])
    guard.load_reference(blob["dead_reference"])
    state_before_sanitize=guard.sanitize(optimizer)
    torch.set_rng_state(blob["rng"]["torch"]);random.setstate(blob["rng"]["python"])
    n=blob["rng"]["numpy"]
    np.random.set_state((n[0],np.asarray(n[1],dtype=np.uint32),n[2],n[3],n[4]))
    return blob["cursor"],state_before_sanitize


def optimizer_for(model, options):
    return torch.optim.AdamW(model.parameters(),lr=options["lr"],betas=tuple(options["betas"]),
        eps=options["eps"],weight_decay=options["weight_decay"],
        amsgrad=options["amsgrad"],foreach=options["foreach"])


def optimizer_report(optimizer):
    report=[]
    for i,p in enumerate(optimizer.param_groups[0]["params"]):
        s=optimizer.state.get(p,{})
        report.append(dict(parameter=i,shape=list(p.shape),
            tensors={k:dict(sha256=digest([v]),norm=float(v.norm()),shape=list(v.shape))
                     for k,v in s.items() if isinstance(v,torch.Tensor)}))
    return report


def update(model, optimizer, guard, objective, row, ledger, budget, spec_path):
    spec=verify_spec(spec_path)
    if row["sample_id"] not in spec["train_order"] or row["split"]!="train":
        raise PermissionError("DEV_MUST_NOT_UPDATE")
    batch=load(row,"train")
    optimizer.zero_grad(set_to_none=True)
    raw=model(batch.depth,batch.state);raw.retain_grad();out=objective(raw,batch)
    if not torch.isfinite(out.loss):
        raise ValueError("NONFINITE_TRAIN_LOSS")
    numeric,hard=result_numeric(out,batch)
    out.loss.backward()
    if not torch.isfinite(raw.grad).all() or raw.grad[...,15:].count_nonzero():
        raise ValueError("RAW_GRADIENT_CONTRACT")
    gradients=group_stats(model)
    norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),spec["clip_norm"],error_if_nonfinite=True))
    before=[p.detach().clone() for p in model.parameters()]
    before_hash=state_sha(model);live_before=model.trajectory_head.mlp[-1].weight.detach()[:15].clone()
    isolation=guard.step(optimizer,ledger,budget)
    updates=group_stats(model,before)
    live_delta=float((model.trajectory_head.mlp[-1].weight.detach()[:15]-live_before).norm())
    if not live_delta>0 or any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("NONFINITE_OR_NO_LIVE_UPDATE")
    return dict(sample_id=row["sample_id"],split="train",loss_before=float(out.loss),
        terms=metric_terms(out),raw0_15_gradient=raw.grad[0,0,:15].tolist(),
        raw15_21_nonzero=0,module_gradients_before_clip=gradients,gradient_norm_before_clip=norm,
        module_updates=updates,head_live_update_norm=live_delta,dead_row_isolation=isolation,
        model_sha_before=before_hash,model_sha_after=state_sha(model),
        independent_minco_errors=numeric,hard_before_update=hard,
        optimizer_state_after=optimizer_report(optimizer))


def shared_parameter_identity(model):
    return canonical([id(p) for p in model.parameters()])


def preflight(spec_path, output, seed, mode):
    spec=verify_spec(spec_path)
    run=claim_run(output)
    from .initialization import audit
    audit(spec,run/"INITIALIZATION_AUDIT.json")
    model=model_for(spec,mode,seed).train();objective=make_objective()
    before=state_sha(model);pid=shared_parameter_identity(model);rows=[]
    for row in spec["train"]:
        batch=load(row,"train");model.zero_grad(set_to_none=True)
        raw=model(batch.depth,batch.state);raw.retain_grad();out=objective(raw,batch);out.loss.backward()
        if not torch.isfinite(raw.grad).all() or raw.grad[...,15:].count_nonzero():
            raise ValueError("PREFLIGHT_GRADIENT")
        errors,hard=result_numeric(out,batch)
        rows.append(dict(sample_id=row["sample_id"],groups=group_stats(model),
                         parameter_identity=shared_parameter_identity(model),numeric=errors))
        if state_sha(model)!=before or shared_parameter_identity(model)!=pid:
            raise ValueError("NOT_ONE_SHARED_NETWORK")
    evaluation=evaluate(model,objective,spec["train"]+spec["dev"])
    write(run/"RESULT.json",dict(status="PASS",train_backward_tasks=rows,evaluation=evaluation,
          model_sha_before=before,model_sha_after=state_sha(model),engineering_optimizer_steps=0))
    return run


def smoke(spec_path, output):
    spec=verify_spec(spec_path)
    root=Path(spec["engineering_owner_root"]).resolve()
    if Path(output).resolve().parent!=root:
        raise ValueError("ENGINEERING_OUTPUT_ROOT")
    run=claim_run(output)
    ledger=root/"ENGINEERING_STEP_LEDGER.json"
    if json.loads(ledger.read_text())["attempted"]!=0:
        raise PermissionError("ENGINEERING_BUDGET_ALREADY_USED")
    seed=spec["seeds"][0];model=model_for(spec,"A_random",seed).train()
    objective=make_objective();options=spec["engineering"]["optimizer"]
    optimizer=optimizer_for(model,options);guard=DeadRowGuard(model)
    metadata=dict(trainer_id=TRAINER_ID,runspec_sha256=sha(spec_path),engineering_only=True,
                  optimizer=options,seed=seed,initialization="A_random",deployable=False)
    pid=shared_parameter_identity(model)
    save_checkpoint(run/"step000.pt",model,optimizer,guard,metadata,0)
    evaluation0=evaluate(model,objective,spec["train"]+spec["dev"])
    write(run/"evaluation000.json",evaluation0)
    reports=[]
    ids=spec["engineering"]["sample_order"]
    for index,sid in enumerate(ids):
        row=next(x for x in spec["train"] if x["sample_id"]==sid)
        if index==1:
            # Restore a separate, deliberately contaminated engineering checkpoint.
            dirty=torch.load(run/"step001.pt",map_location="cpu",weights_only=True)
            for p in guard.parameters:
                ix=next(i for i,x in enumerate(optimizer.param_groups[0]["params"]) if x is p)
                key=dirty["optimizer"]["param_groups"][0]["params"][ix]
                for name,value in [("exp_avg",.4),("exp_avg_sq",.7),("max_exp_avg_sq",.9)]:
                    dirty["optimizer"]["state"][key][name][15:]=value
            path=run/"ENGINEERING_DIRTY_STATE_PROBE.pt"
            with path.open("xb") as f:torch.save(dirty,f)
            expected_model=state_sha(model)
            cursor,dirty_report=load_checkpoint(path,sha(path),model,optimizer,guard,metadata)
            if cursor!=1 or state_sha(model)!=expected_model or shared_parameter_identity(model)!=pid:
                raise ValueError("CHECKPOINT_RESUME_DRIFT")
            if not any(x["nonzero"] for x in dirty_report.values()):
                raise ValueError("DIRTY_CHECKPOINT_PROBE_MISSING")
            # Active state must roundtrip exactly; sanitized dead state equals clean saved checkpoint.
            clean=torch.load(run/"step001.pt",map_location="cpu",weights_only=True)["optimizer"]
            restored=optimizer.state_dict()
            for key,values in clean["state"].items():
                for name,value in values.items():
                    if isinstance(value,torch.Tensor) and not torch.equal(value,restored["state"][key][name]):
                        raise ValueError("ACTIVE_OPTIMIZER_STATE_RESUME_DRIFT")
            write(run/"CHECKPOINT_RESUME.json",dict(status="PASS",cursor=cursor,
                dirty_state_before_sanitize=dirty_report,active_and_sanitized_state_exact=True,
                same_shared_parameter_objects=True,model_sha256=expected_model,probe_sha256=sha(path)))
            # Also exercise the step guard against nonzero in-memory historical moments.
            for p in guard.parameters:
                for name,value in [("exp_avg",.2),("exp_avg_sq",.5),("max_exp_avg_sq",.8)]:
                    optimizer.state[p][name][15:]=value
        report=update(model,optimizer,guard,objective,row,ledger,2,spec_path)
        report.update(step=index+1,classification="ENGINEERING_OPTIMIZER_STEPS",
                      shared_parameter_identity=shared_parameter_identity(model))
        if report["shared_parameter_identity"]!=pid:
            raise ValueError("NETWORK_REINITIALIZED_PER_SAMPLE")
        reports.append(report)
        write(run/("step"+str(index+1).zfill(3)+".json"),report)
        save_checkpoint(run/("step"+str(index+1).zfill(3)+".pt"),model,optimizer,guard,metadata,index+1)
        write(run/("evaluation"+str(index+1).zfill(3)+".json"),
              evaluate(model,objective,spec["train"]+spec["dev"]))
    write(run/"RESULT.json",dict(status="PASS",ENGINEERING_OPTIMIZER_STEPS=2,
          FORMAL_RESEARCH_TRAINING_STEPS=0,shared_parameter_identity=pid,
          samples=ids,final_model_sha256=state_sha(model),dead_sha256=guard.sha(),
          optimizer=options,checkpoint_resume="PASS",deployable=False))
    return run


def train(spec_path, approval, output, seed, mode):
    # Authorization and frozen identity are verified BEFORE creating any Run or optimizer.
    spec=authorize(spec_path,approval,seed,mode,output)
    run=claim_run(output);model=model_for(spec,mode,seed).train();objective=make_objective()
    optimizer=optimizer_for(model,spec["optimizer"]);guard=DeadRowGuard(model)
    ledger=run/"FORMAL_STEP_LEDGER.json";write(ledger,dict(attempted=0,completed=0))
    metadata=dict(trainer_id=TRAINER_ID,runspec_sha256=sha(spec_path),engineering_only=False,
                  optimizer=spec["optimizer"],seed=seed,initialization=mode,
                  approval_sha256=sha(approval))
    write(run/"AUTHORIZATION.json",metadata)
    pid=shared_parameter_identity(model)
    save_checkpoint(run/"step000.pt",model,optimizer,guard,metadata,0)
    write(run/"evaluation000.json",evaluate(model,objective,spec["train"]+spec["dev"]))
    try:
        with (run/"STEPS.jsonl").open("x") as stream:
            for i in range(spec["max_updates"]):
                row=spec["train"][i%len(spec["train"])]
                record=update(model,optimizer,guard,objective,row,ledger,spec["max_updates"],spec_path)
                if shared_parameter_identity(model)!=pid:
                    raise ValueError("SHARED_NETWORK_IDENTITY")
                record.update(step=i+1,classification="FORMAL_RESEARCH_TRAINING")
                stream.write(json.dumps(record,allow_nan=False)+"\n");stream.flush()
                if i+1 in spec["evaluation_steps"]:
                    save_checkpoint(run/("step"+str(i+1).zfill(3)+".pt"),model,optimizer,guard,metadata,i+1)
                    write(run/("evaluation"+str(i+1).zfill(3)+".json"),
                          evaluate(model,objective,spec["train"]+spec["dev"]))
        write(run/"RESULT.json",dict(status="COMPLETED_PILOT_EXECUTION",
              FORMAL_RESEARCH_TRAINING_STEPS=spec["max_updates"],planning_success_not_inferred=True))
    except BaseException as e:
        write(run/"STOP.json",dict(error=repr(e),ledger=json.loads(ledger.read_text()),automatic_resume=False))
        raise
    return run
