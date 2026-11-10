"""P0c staged launcher: development attempts and immutable frozen acceptance are distinct."""
import argparse,hashlib,importlib.util,json,os,subprocess,sys,time,traceback
from pathlib import Path
BASE=Path(__file__).resolve().parents[2]
OUT=BASE/"research_outputs/physics_only_v1_2_p0c_world_boundary_v1"
ENTRY=OUT/"_preparation/entry_p0c_001"
FREEZE=OUT/"_preparation/freeze_p0c_001"
PYTHON="/home/lizheng/anaconda3/envs/yopo/bin/python"
CODE=["query_minco/training/physics_only_v1_2_gate1/geometry.py","query_minco/training/physics_only_v1_2_gate1/objective.py","query_minco/training/physics_only_v1_2_gate1/roi_safe_training.py","tests/test_physics_v12_p0c_world_boundary.py","experiments/physics_only_v1_2_p0c_world_boundary_v1/PROTOCOL.md","experiments/physics_only_v1_2_p0c_world_boundary_v1/run_audit.py"]
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    with Path(p).open("x") as f:json.dump(x,f,indent=2,allow_nan=False);f.write("\n")
def guard(phase):
    e=read(ENTRY/"ENTRY.json");allowed={str(BASE/p) for p in e["allowed_existing"]}
    for p,h in e["protected_sha256"].items():
        if p not in allowed and sha(p)!=h:raise RuntimeError("HISTORY_DRIFT "+p)
    for p,h in read(ENTRY/"INPUT_HASHES.json").items():
        if p not in allowed and sha(p)!=h:raise RuntimeError("INPUT_DRIFT "+p)
    for label,repo in (("query",BASE),("yopo",Path("/home/lizheng/YOPO")),("history",Path("/home/lizheng/reach_planner"))):
        for key,cmd in (("head",["rev-parse","HEAD"]),("branch",["branch","--show-current"]),("diff",["diff","--binary"]),("staged",["diff","--cached","--binary"])):
            if subprocess.check_output(["git","-C",str(repo),*cmd],text=True)!=e["git_before"][label][key]:raise RuntimeError("GIT_DRIFT")
    if time.time()-e["started_unix"]>3600:raise RuntimeError("TOTAL_BUDGET")
    if phase=="formal":
        for p,h in read(FREEZE/"INPUT_HASHES.json").items():
            if sha(p)!=h:raise RuntimeError("FROZEN_INPUT_DRIFT "+p)
        if (FREEZE/"STOP.json").exists():raise RuntimeError("PERSISTENT_STOP")
        for p,h in read(FREEZE/"SOURCE_FREEZE.json").items():
            if sha(BASE/p)!=h:raise RuntimeError("FROZEN_SOURCE_DRIFT "+p)
def test_worker(stage,dest):
    from query_minco.training.physics_only_v1_2_gate1.e1_r3a_v2 import bootstrap
    write(dest/"DEPENDENCIES.json",bootstrap());import pytest
    if stage=="numeric":
        paths=["tests/test_physics_v12_p0c_world_boundary.py"];extra=["-k","not integration"]
    else:
        paths=read(BASE/"research_outputs/physics_only_v1_2_e1_r3a_v4/_preparation/freeze_001/RUN_SPEC.json")["test_paths"]
        paths=paths+["tests/test_physics_v12_p0c_world_boundary.py"]
        extra=["--deselect=tests/test_physics_v12_gate1_geometry.py::test_exact_distance_matches_independent_all_boxes"]
    args=["-q","-p","no:cacheprovider","--basetemp="+str(dest/"pytest_tmp"),"--junitxml="+str(dest/"TEST_RESULTS.xml"),*paths,*extra]
    write(dest/"PYTEST_COMMAND.json",dict(args=args))
    code=int(pytest.main(args));write(dest/"RESULT.json",dict(status="PASS" if code==0 else "FAIL",pytest_exit=code))
    if code:raise SystemExit(code)
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--phase",choices=["dev","formal"],required=True);ap.add_argument("--stage",choices=["numeric","tests","zero"],required=True);ap.add_argument("--run-id",required=True);ap.add_argument("--worker",action="store_true");a=ap.parse_args()
    if "/" in a.run_id or ".." in a.run_id:raise ValueError("RUN_ID")
    dest=OUT/("dev_preflight" if a.phase=="dev" else "runs")/a.run_id
    sys.path.insert(0,str(BASE))
    if a.worker:
        guard(a.phase)
        try:
            if a.stage=="zero":zero_worker(dest,a.phase)
            else:test_worker(a.stage,dest)
        except BaseException:
            write(dest/"WORKER_FAILURE.json",dict(status="FAIL" if a.phase=="dev" else "STOP",error=traceback.format_exc()));raise
        return
    dest.mkdir()
    write(dest/"ENTRY_COMMAND.json",dict(argv=[sys.executable]+sys.argv,cwd=str(Path.cwd()),phase=a.phase))
    write(dest/"SOURCE_SNAPSHOT.json",{p:dict(sha256=sha(BASE/p),content=(BASE/p).read_text()) for p in CODE if (BASE/p).exists()})
    try:
        guard(a.phase)
        if sys.executable!=PYTHON:raise RuntimeError("PYTHON_IDENTITY")
        if a.stage=="zero":
            prerequisite=OUT/("runs/acceptance_tests_001/RESULT.json" if a.phase=="formal" else "dev_preflight/integration_dev_001/RESULT.json")
            if read(prerequisite)["status"]!="PASS":raise RuntimeError("TESTS_REQUIRED")
        s=importlib.util.spec_from_file_location("_accepted_full_tree_launcher",BASE/"experiments/physics_only_v1_2_e1_r3a_v2/run_zero_update.py")
        m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
        (dest/"tmp").mkdir()
        env=dict(os.environ,PYTHONPATH=str(BASE),PYTHONDONTWRITEBYTECODE="1",CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1",PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",TMPDIR=str(dest/"tmp"))
        env.pop("PYTEST_ADDOPTS",None)
        cmd=[PYTHON,"-B",str(Path(__file__).resolve()),"--phase",a.phase,"--stage",a.stage,"--run-id",a.run_id,"--worker"]
        stop=dest/"DEV_STOP.json" if a.phase=="dev" else FREEZE/"STOP.json"
        m.launch(cmd,dest/"driver",env,stop,OUT,min(1200 if a.stage=="zero" else 600,3600-(time.time()-read(ENTRY/"ENTRY.json")["started_unix"])))
        guard(a.phase);write(dest/"ENTRY_EXIT.json",dict(status="PASS",exit_code=0))
    except BaseException:
        error=traceback.format_exc()
        with (dest/"ENTRY_stderr.log").open("x") as f:f.write(error)
        write(dest/"ENTRY_EXIT.json",dict(status="FAIL" if a.phase=="dev" else "STOP",exit_code=1,error=error))
        if a.phase=="formal" and not (FREEZE/"STOP.json").exists():write(FREEZE/"STOP.json",dict(error=error))
        raise
def zero_worker(dest,phase):
    import numpy as np
    import torch
    from unittest.mock import patch
    from query_minco.training.physics_only_v1_2_gate1.short_train import initialize,INITIAL_SHA
    from query_minco.training.physics_only_v1.smoke import parameter_hash
    from query_minco.training.physics_only_v1_2_gate1.e1_dataset_binding import load_e1_bound
    from query_minco.training.physics_only_v1_2_gate1.roi_safe_training import RoiSafeTrainingObjective
    from query_minco.training.physics_only_v1_2_gate1.e1_r3_zero_update import phase_spec,packet,module_groups,norm_stats
    from query_minco.training.physics_only_v1_2_gate1.hard_validator import validate_result
    from query_minco.training.physics_only_v1_2_gate1.geometry import world_points
    from query_minco.training.pg1.losses import analytic_samples
    import csv
    def val(x):return x.detach().tolist()
    rows=read(ENTRY/"TRAIN13_AND_EXCLUSIONS.json")
    bindings=read(OUT/"_preparation/bindings_p0c_001/BINDING_INDEX.json")
    bmap={r["sample_id"]:r for r in bindings}
    # Verify ALL original task identities while accounting only for the two explicitly approved source versions.
    inv=[];verified={}
    for row in rows["all28"]:
        assert sha(row["source_manifest_path"])==row["source_manifest_sha256"]
        if row.get("hard_evidence_path"):assert sha(row["hard_evidence_path"])==row["hard_evidence_sha256"]
        original=read(row["source_manifest_path"]);derived=read(bmap[row["sample_id"]]["p0c_manifest_path"])
        for p,h in original["files_sha256"].items():
            if p not in verified:verified[p]=sha(p)
            if verified[p]!=h:
                changes=derived["p0c_source_rebinding"]["authorized_changes"]
                if p not in changes or changes[p]["original_sha256"]!=h or changes[p]["p0c_sha256"]!=verified[p]:raise ValueError("UNAUTHORIZED_SOURCE_DRIFT "+p)
        inv.append(dict(sample_id=row["sample_id"],scene_id=row["scene_id"],split=row["split"],source=row["source"],reachability=row["reachability"],eligible=row["data_eligible"],source_manifest_sha256=row["source_manifest_sha256"],p0c_manifest_sha256=bmap[row["sample_id"]]["p0c_manifest_sha256"],geometry_sha256=row.get("geometry_sha256"),camera_sha256=row.get("camera_sha256"),state_sha256=row.get("state_sha256"),static_hashes="PASS",action="eligible after P0c" if row["data_eligible"] else "ISOLATED; no recertification"))
    with (dest/"ALL28_TASK_INVENTORY.csv").open("x",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(inv[0]));w.writeheader();w.writerows(inv)
    write(dest/"ORIGINAL_INPUT_REHASH.json",verified)
    expected=rows["Train13"] if phase=="formal" else [rows["Train13"][0],rows["Train13"][4]]
    source=read(OUT/"audit/entry_p0c_001/ORIGINAL_SOURCE.json")
    scope={"__name__":"_p0c_frozen_objective_oracle","__package__":"query_minco.training.physics_only_v1_2_gate1"}
    exec(compile(source["query_minco/training/physics_only_v1_2_gate1/objective.py"],"<frozen_original_objective>","exec"),scope)
    LegacyObjective=scope["BoundPhysicsV12Objective"]
    records=[];current=None;model=None;before=None
    def export(status,error=None):
        matrix=[];have={r["sample_id"]:r for r in records}
        for row in expected:
            rec=have.get(row["sample_id"])
            for k in range(6):
                line=dict(sample_id=row["sample_id"],scene_id=row["scene_id"],query=k,status="PASS" if rec else "NOT_RUN_AFTER_STOP",
                    source_manifest_sha256=row["source_manifest_sha256"],p0c_manifest_sha256=bmap[row["sample_id"]]["p0c_manifest_sha256"])
                if rec:
                    metrics=rec["metrics"];h=rec["hard"][k];gr=rec["per_query_gradients"][k]
                    line.update(known_count=rec["known_counts"][k],UNKNOWN_count=rec["outside_counts"][k],
                        inside_margin_insufficient=rec["margin_counts"][k],task_total_loss=rec["loss"],Hard_pass=h["passed"],Hard_reasons=";".join(h["reasons"]),
                        max_v=h["dynamic_polynomial_maxima"][0],max_a=h["dynamic_polynomial_maxima"][1],max_j=h["dynamic_polynomial_maxima"][2],
                        Raw_gradient_norm=gr["stats"]["TOTAL"]["Raw"]["norm"],network_gradient_norm=gr["stats"]["TOTAL"]["all_network"]["norm"],
                        gradient_finite=True,gradient_nonzero=True,legacy_loss_status=rec["legacy_status"],training_distance_label="training_proxy outside; strict known inside")
                    for name in ["C_viol","C_shape","F","F_dyn","F_domain","G_task_train","G_N","G_F","P","S","V","Q","J","pi"]:line[name]=metrics[name][0][k]
                    line["training_min_distance_m"]=metrics["min_sdf_m"][0][k]
                matrix.append(line)
        fields=list(dict.fromkeys(k for r in matrix for k in r))
        with (dest/"TRAIN13_ZERO_UPDATE.csv").open("x",newline="") as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(matrix)
        write(dest/"RESULT.json",dict(status=status,phase=phase,tasks_completed=len(records),expected_tasks=len(expected),candidate_records=len(matrix),
            Hard_pass=sum(h["passed"] for r in records for h in r["hard"]),inside_comparison_tasks=sum(r["legacy_status"]=="EXACT_VALUES_RAW_AND_NETWORK_GRADIENTS" for r in records),
            optimizer_updates=0,model_forwards=len(records),model_before=before,model_after=parameter_hash(model) if model is not None else None,error=error))
    with patch.object(torch.optim.Optimizer,"__init__",side_effect=RuntimeError("OPTIMIZER_FORBIDDEN")),patch.object(torch,"load",side_effect=RuntimeError("CHECKPOINT_FORBIDDEN")):
        model=initialize();before=parameter_hash(model)
        if before!=INITIAL_SHA:raise ValueError("MODEL_INIT_SHA")
        write(dest/"MODEL_BEFORE.json",dict(sha256=before,seed=20261104,mode="eval+autograd",FP32=True,threads=1))
        named=list(model.named_parameters());groups=module_groups(named);objective=RoiSafeTrainingObjective(phase_spec())
        try:
            for row in expected:
                current=row["sample_id"];guard(phase);model.zero_grad(set_to_none=True)
                br=bmap[current];batch=load_e1_bound(br["p0c_manifest_path"],br["p0c_manifest_sha256"])
                if model.training or not torch.is_grad_enabled() or torch.is_inference_mode_enabled():raise ValueError("FORWARD_SEMANTICS")
                raw=model(batch.depth,batch.state)
                result=objective(raw,batch);ana=analytic_samples(result.trajectory,64)
                geo=result.training_geometry_records[0];outside=geo["outside_mask"]
                hard=validate_result(result,batch)
                reference=read(BASE/"research_outputs/physics_only_v1_2_e1_r3a_v4/runs/support_001"/(current+".json"))
                if not torch.equal(raw.detach(),torch.tensor(reference["raw_logits"],dtype=torch.float32)):raise ValueError("FROZEN_RAW_REPRODUCTION")
                if hard!=reference["independent_hard"]:raise ValueError("HARD_CHANGED_FOR_IDENTICAL_TRAJECTORY")
                rec=dict(sample_id=current,source_identity=row,p0c_binding_identity=br,raw=val(raw),state=val(batch.state),
                    parameters=dict(head=val(result.parameters.head_pva),tail=val(result.parameters.tail_pva),inner=val(result.parameters.inner_pts),durations=val(result.parameters.durations)),
                    coefficients=val(result.trajectory.coeffs),sampled_p_B0=val(ana["p"]),sampled_v=val(ana["v"]),sampled_a=val(ana["a"]),sampled_jerk=val(ana["jerk"]),
                    loss=float(result.loss),metrics={k:val(v) for k,v in result.metrics.items()},hard=hard,Hard_matches_v4=True,
                    known_counts=(~outside).sum(-1).tolist(),outside_counts=outside.sum(-1).tolist(),margin_counts=geo["inside_insufficient_margin"].sum(-1).tolist(),
                    known_mask=val(geo["known_mask"]),outside_mask=val(outside),training_distance=val(geo["training_distance"]),
                    max_outside_distance_m=float(geo["outside_distance_m"].max()),proxy_cast_max_abs_m=geo["proxy_cast_max_abs_m"],
                    FP32_vs_FP64_membership_disagreement=geo["FP32_vs_recomputed_FP64_mask_disagreement"],world_recompute_difference_m=geo["max_world_recompute_difference_m"])
                write(dest/(current+"_PRE_BACKWARD.json"),rec)
                m=result.metrics;coeff=dict(C_viol=1.,F=1.,G_task_train=1.,C_shape=.1,P=.025,S=.1)
                if not outside.any():
                    old=LegacyObjective(phase_spec())(raw,batch);comparisons={}
                    if not torch.equal(old.loss,result.loss):raise ValueError("KNOWN_TOTAL_DRIFT")
                    for key in old.metrics:
                        a=old.metrics[key];b=m[key]
                        if not torch.equal(a,b):raise ValueError("KNOWN_TERM_DRIFT "+key)
                        if a.requires_grad:
                            ga=torch.autograd.grad(a.sum(),raw,retain_graph=True,allow_unused=True)[0]
                            gb=torch.autograd.grad(b.sum(),raw,retain_graph=True,allow_unused=True)[0]
                            if not ((ga is None and gb is None) or (ga is not None and gb is not None and torch.equal(ga,gb))):raise ValueError("KNOWN_RAW_TERM_DRIFT "+key)
                        comparisons[key]=dict(value_exact=True,Raw_gradient_exact=True if a.requires_grad else "NOT_DIFFERENTIABLE_CONSTANT")
                    for key in [*coeff,"TOTAL"]:
                        a=old.loss if key=="TOTAL" else old.metrics[key].mean()*coeff[key]
                        b=result.loss if key=="TOTAL" else m[key].mean()*coeff[key]
                        ga=torch.autograd.grad(a,tuple(p for _,p in named),retain_graph=True,allow_unused=True)
                        gb=torch.autograd.grad(b,tuple(p for _,p in named),retain_graph=True,allow_unused=True)
                        if any(not ((x is None and y is None) or (x is not None and y is not None and torch.equal(x,y))) for x,y in zip(ga,gb)):raise ValueError("KNOWN_NETWORK_DRIFT "+key)
                    rec["known_comparison"]=dict(all_terms=comparisons,network_primary_and_total_exact=True,atol=0,rtol=0)
                    rec["legacy_status"]="EXACT_VALUES_RAW_AND_NETWORK_GRADIENTS"
                else:
                    try:LegacyObjective(phase_spec())(raw,batch)
                    except ValueError as ex:
                        if str(ex)!="UNKNOWN_OUTSIDE_CERTIFIED_ROI":raise
                        rec["legacy_status"]="UNDEFINED_UNKNOWN_OUTSIDE_CERTIFIED_ROI"
                    else:raise ValueError("OLD_UNKNOWN_NOT_REJECTED")
                    point_grad=torch.autograd.grad(m["C_viol"].mean()+.1*m["C_shape"].mean(),geo["points_B0"],retain_graph=True)[0]
                    g=batch.geometry[0];p=geo["points_B0"]
                    w=world_points(p,torch.tensor(g.origin,dtype=p.dtype),torch.tensor(g.rotation,dtype=p.dtype)).double()
                    q=torch.maximum(torch.minimum(w,torch.tensor(g.roi[1])),torch.tensor(g.roi[0]))
                    normal=(w-q)[outside]/(w-q).norm(dim=-1)[outside].unsqueeze(-1)
                    world_grad=point_grad.double()@torch.tensor(g.rotation).T
                    radial=(world_grad[outside]*normal).sum(-1)
                    if not torch.isfinite(radial).all() or not (radial>0).all():raise ValueError("COLLISION_OUTSIDE_GRADIENT_DIRECTION")
                    rec["outside_collision_gradient"]=dict(count=radial.numel(),min_outward_cost_derivative=float(radial.min()),max_outward_cost_derivative=float(radial.max()),gradient_descent_inward=True,
                        scope="Collision contribution only; not guaranteed for total Loss")
                scalars={k:m[k].mean()*v for k,v in coeff.items()};scalars["TOTAL"]=result.loss
                rec["full_gradients"]=packet(scalars,raw,named,groups)
                perq=[]
                for k in range(6):
                    gr=packet({"TOTAL":(m["V"][0,k]+.1*m["Q"][0,k])/6},raw,named,groups)
                    rr=torch.tensor(gr["raw_gradients"]["TOTAL"])[0,k]
                    if any(float(rr[a:b].norm())==0 for a,b in [(0,9),(9,12),(12,15)]) or gr["stats"]["TOTAL"]["all_network"]["norm"]==0:raise ValueError("QUERY_GRADIENT_INACTIVE")
                    perq.append(gr)
                rec["per_query_gradients"]=perq
                raw.retain_grad();result.loss.backward()
                if raw.grad is None or not torch.isfinite(raw.grad).all() or raw.grad[...,15:21].count_nonzero():raise ValueError("RAW_BACKWARD_INVALID")
                network=torch.cat([p.grad.reshape(-1) if p.grad is not None else torch.zeros_like(p).reshape(-1) for _,p in named])
                rec["actual_backward"]=dict(Raw=norm_stats(raw.grad),network=norm_stats(network))
                if rec["actual_backward"]["network"]["norm"]==0:raise ValueError("NETWORK_ZERO")
                if parameter_hash(model)!=before:raise ValueError("MODEL_UPDATED")
                rec["status"]="PASS";write(dest/(current+"_RESULT.json"),rec);records.append(rec)
                print(json.dumps(dict(sample_id=current,status="PASS",loss=rec["loss"],UNKNOWN=sum(rec["outside_counts"]),Hard_pass=sum(h["passed"] for h in hard))),flush=True)
            write(dest/"MODEL_AFTER.json",dict(before=before,after=parameter_hash(model),unchanged=before==parameter_hash(model)))
            export("PASS")
        except BaseException:
            write(dest/"MODEL_AFTER.json",dict(before=before,after=parameter_hash(model),unchanged=before==parameter_hash(model)))
            write(dest/"CURRENT_TASK_FAILURE.json",dict(sample_id=current,error=traceback.format_exc()))
            export("STOP",traceback.format_exc());raise

if __name__=="__main__":main()
