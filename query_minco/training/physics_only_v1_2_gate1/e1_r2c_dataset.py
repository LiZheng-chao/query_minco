"""E1-R2d immutable-source continuation. No generators, training or new search."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np

BASE=Path(__file__).resolve().parents[3]
OUT=BASE/"research_outputs/physics_only_v1_2_e1_r2c_dataset_v1"
PREP=OUT/"_preparation/continuation_001"
PYTHON="/home/lizheng/anaconda3/envs/yopo/bin/python"
PROPOSAL_SHA="7d04d5cb91ff39d0e8b1723f716b964a249a7b33aaf450fafe5127334dbb9bb1"
AUDIT=BASE/"research_outputs/physics_only_v1_2_e1_r2c_audit_v1"
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    with Path(p).open("x") as f:json.dump(x,f,indent=2,allow_nan=False);f.write("\n")
def proposal():
    p=PREP/"AUTHORIZED_PROPOSAL.json"
    if sha(p)!=PROPOSAL_SHA:raise ValueError("PROPOSAL_SHA")
    return read(p)
def verify_hashes(mapping):
    for p,h in mapping.items():
        if sha(p)!=h:raise ValueError("SOURCE_SHA_DRIFT "+str(p))
def guard():
    if (PREP/"STOP.json").exists():raise ValueError("PERSISTENT_STOP")
    e=read(PREP/"ENTRY.json")
    verify_hashes(e["protected_sha256"])
    verify_hashes({str(BASE/p):h for p,h in read(PREP/"SOURCE_FREEZE.json").items()})
    if time.time()-e["started_unix"]>=3600:raise ValueError("TOTAL_TIME_BUDGET")
def validate_depth_request(pose_id,p,attempts,kind="depth"):
    if kind!="depth":raise ValueError("COUNTS_ONLY_FORBIDDEN")
    pending={q["pose_id"] for q in p["pending_pose_plan"]}
    if pose_id in p["forbidden_rerender_pose_ids"]:raise ValueError("COMPLETED_POSE_RERENDER_FORBIDDEN")
    if pose_id not in pending:raise ValueError("POSE_NOT_IN_FROZEN_PLAN")
    if any(q["pose_id"]==pose_id for q in attempts):raise ValueError("DUPLICATE_DEPTH_ATTEMPT")
    if len(attempts)>=11 or p["historical_ledger"]["depth"]+len(attempts)>=14:raise ValueError("LIFETIME_DEPTH_BUDGET")
    if p["historical_ledger"]["actual_CUDA_processes"]+len(attempts)>=21:raise ValueError("LIFETIME_CUDA_BUDGET")
    return True
def reserve_depth(pose_id,p,ledger,argv):
    ledger=Path(ledger);old=[json.loads(x) for x in ledger.read_text().splitlines()] if ledger.exists() else []
    validate_depth_request(pose_id,p,old)
    row=dict(kind="depth",pose_id=pose_id,new_attempt=len(old)+1,lifetime_CUDA_index=11+len(old),lifetime_depth_index=4+len(old),
        previous_ledger_sha256=p["historical_ledger"]["sha256"],argv=[str(x) for x in argv],unix=time.time(),
        failure_consumes_attempt=True)
    with ledger.open("a") as f:f.write(json.dumps(row)+"\n");f.flush();os.fsync(f.fileno())
    return row
def child_env(tmp,gpu=False):
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1",PYTHONPATH=str(BASE),OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1",CUDA_VISIBLE_DEVICES="0" if gpu else "",
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",TMPDIR=str(tmp),LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:/opt/ros/noetic/lib")
    env.pop("PYTEST_ADDOPTS",None);return env
def prereqs():
    guard();p=proposal()
    verify_hashes({str(BASE/q):h for q,h in p["approved_new_contract_source_sha256"].items()})
    for key,hkey in [("old_data_inventory_path","old_data_inventory_sha256"),("old_dataset_identity_path","old_dataset_identity_sha256")]:
        if sha(p[key])!=p[hkey]:raise ValueError("INPUT_INVENTORY_DRIFT")
    verify_hashes(read(p["old_data_inventory_path"]))
    ledger=p["historical_ledger"]
    if sha(ledger["path"])!=ledger["sha256"]:raise ValueError("HISTORICAL_LEDGER_DRIFT")
    rows=[json.loads(x) for x in Path(ledger["path"]).read_text().splitlines()]
    if len(rows)!=10 or sum(q["kind"]=="counts" for q in rows)!=7 or sum(q["kind"]=="depth" for q in rows)!=3:raise ValueError("HISTORICAL_CUDA_COUNT")
    binary=p["source_native_binary"]
    if sha(binary["path"])!=binary["sha256"]:raise ValueError("NATIVE_BINARY_DRIFT")
    verify_hashes(binary["original_build_receipt"]["native_objects"])
    idx=p["reachability_evidence_index"]
    if sha(idx["path"])!=idx["sha256"]:raise ValueError("REACHABILITY_INDEX_DRIFT")
    verify_hashes({str(BASE/q):h for q,h in read(AUDIT/"audit/review_001/DELIVERY_SHA256.json").items()})
    return p
def preflight():
    import importlib,sys
    sys.path.append("/home/lizheng/anaconda3/lib/python3.11/site-packages")
    p=prereqs()
    modules={n:importlib.import_module(n) for n in ("numpy","scipy","torch","pytest")}
    from .e1_resource_supervisor import monitor_preflight
    from .e1_dataset_qualification import gpu_sample
    result=dict(status="PASS",dependencies={n:dict(file=m.__file__,version=m.__version__) for n,m in modules.items()},
        CPU_monitor=monitor_preflight(),GPU_samples=[gpu_sample() for _ in range(3)],native_binary_sha256=p["source_native_binary"]["sha256"],
        lifetime_counts=7,lifetime_depth=3,new_counts_allowed=0,pending_depth_poses=11)
    write(PREP/"PREFLIGHT_PASS.json",result)
def make_state(case):
    return dict(v0_world_mps=case["v0_B0_mps"],a0_world_mps2=case["a0_B0_mps2"],g_B0_m=case["g_B0_m"],
        gravity_world_unit=case["gravity_B0"],snapshot_time_s=0.,origin="explicit offline initial condition, not measured odometry")
def bind_task(scene,pose,case,sub,sample_id,p):
    scene=Path(scene);pose=Path(pose);sub=Path(sub);sub.mkdir(parents=True)
    statepath=sub/"STATE.json";write(statepath,make_state(case))
    info=read(scene/"scene.json")
    binary=p["source_native_binary"]["path"];old=Path(p["old_data_root"])
    files=[scene/n for n in ("cloud_xyz.f32","occupancy_counts.i32","grid.json","scene.json")]
    files += [pose/n for n in ("renderer.json","depth_z.f32","POSE_CERTIFICATE.json","CAMERA_BINDING.json","depth_input_2x128x160.f32")]
    files += [statepath,Path(binary),old/"runs/qualification_001/ALL_SCENE_IDENTITIES.json",PREP/"AUTHORIZED_PROPOSAL.json",
              PREP/"SOURCE_FREEZE.json",PREP/"CONFIG.json",AUDIT/"_preparation/review_001/PROTOCOL.json"]
    files += [BASE/q for q in read(PREP/"SOURCE_FREEZE.json")]
    manifest=dict(schema="e1_r2_bound_sample_v1",sample_id=sample_id,scene_id=info["scene_id"],scene_identity=info["identity"],
        split=info["split"],frame="B0-FLU",teacher_loaded=False,scene_dir=str(scene),pose_dir=str(pose),binary_path=binary,
        scene_registry_path=str(old/"runs/qualification_001/ALL_SCENE_IDENTITIES.json"),
        files_sha256={str(q.resolve()):sha(q) for q in files},
        geometry_authority="immutable original native occupancy; outside known base/top UNKNOWN; no no-hit free inference")
    path=sub/"SAMPLE_MANIFEST.json";write(path,manifest)
    return path
def verify_reachability(evidence,batch,audit_geometry):
    from .e1_reachability_v2 import evaluate_witness,verify_exclusion,FEASIBLE,INFEASIBLE,UNKNOWN
    g=batch.geometry[0]
    for name in ("origin","rotation","roi","boxes"):
        if not np.array_equal(getattr(g,name),getattr(audit_geometry,name)):raise ValueError("GEOMETRY_FRAME_REBIND_MISMATCH "+name)
    if g.floor!=audit_geometry.floor or not np.array_equal(g.domain.numpy(),audit_geometry.domain.numpy()):
        raise ValueError("FROZEN_PHYSICAL_DOMAIN_MISMATCH")
    state=batch.state[0].numpy()
    if not np.array_equal(np.asarray(evidence["task_state"],np.float32),state):raise ValueError("FROZEN_STATE_MISMATCH")
    label=evidence["reachability"];proof=evidence["proof"];checked=[]
    if label==FEASIBLE:
        if not evidence["witnesses"] or proof is not None:raise ValueError("FEASIBLE_WITHOUT_VALID_WITNESS")
        for index,w in evidence["witnesses"].items():
            result=evaluate_witness(w["coeffs_B0"],w["durations_s"],state,g)
            original=evidence["attempts"][int(index)]
            if not result["qualified_witness"] or result["hard"]!=original["hard"]:raise ValueError("HARD_CERTIFICATE_REPRODUCTION_MISMATCH")
            checked.append(dict(index=index,Hard=result["hard"],initial_PVA_error=result["initial_PVA_error"],C0_C4_errors=result["C0_C4_errors"]))
    elif label==INFEASIBLE:
        if evidence["witnesses"] or not verify_exclusion(proof,state[6:9],audit_geometry):raise ValueError("INVALID_INFEASIBILITY_PROOF")
    elif label==UNKNOWN:
        if evidence["witnesses"] or proof is not None or any(a["qualified_witness"] for a in evidence["attempts"]):raise ValueError("UNKNOWN_EVIDENCE_CONTRADICTION")
    else:raise ValueError("INVALID_TASK_STATE")
    return dict(reachability=label,checked_Hard_certificates=checked,independent_infeasible_proof=proof,
        unknown_not_infeasible=label==UNKNOWN,portal_coverage=evidence["portal_coverage"],new_candidate_search=False,
        offline_audit_only=True,Teacher_label=False)
def render_pose(plan,data_scene,new_pose,p,run):
    from .pose_preflight import certify_position
    from .e1_dataset_qualification import scene_data,camera_qualification,gpu_sample
    from .e1_resource_supervisor import supervise
    scene=Path(data_scene);new_pose=Path(new_pose);new_pose.mkdir(parents=True)
    cloud,counts,meta=scene_data(scene);cert=dict(plan["certificate"])
    proof=certify_position(cert["position_world"],meta,counts)
    if not proof["passed"] or abs(proof["clearance_lower_bound_m"]-cert["clearance_lower_bound_m"])>1e-9:raise ValueError("INITIAL_CLEARANCE_DRIFT")
    if sha(scene/"cloud_xyz.f32")!=plan["raw_cloud_sha256"] or sha(scene/"occupancy_counts.i32")!=plan["counts_sha256"]:raise ValueError("POSE_SOURCE_DRIFT")
    cert.update(source_cloud_sha256=plan["raw_cloud_sha256"],source_occupancy_sha256=plan["counts_sha256"],
        scene_id=plan["scene_id"],split=plan["split"])
    write(new_pose/"POSE_CERTIFICATE.json",cert)
    render=OUT/"runs"/("render_"+plan["pose_id"]+"_001");render.mkdir();raw=render/"raw";raw.mkdir();tmp=render/"tmp";tmp.mkdir()
    argv=[p["source_native_binary"]["path"],"depth",scene/"cloud_xyz.f32",new_pose/"POSE_CERTIFICATE.json",raw]
    guard();attempt=reserve_depth(plan["pose_id"],p,PREP/"GPU_CALLS.jsonl",argv)
    write(render/"DEPTH_ATTEMPT.json",attempt)
    supervise(argv,render,"render",cwd=BASE,env=child_env(tmp,True),stop_path=PREP/"STOP.json",budget_root=OUT,
        timeout=120,rss_limit=4294967296,disk_limit=2147483648,guard=guard,gpu_monitor=gpu_sample)
    if sha(raw/"occupancy_counts.i32")!=plan["counts_sha256"]:raise ValueError("RENDER_OCCUPANCY_DRIFT")
    rm=read(raw/"renderer.json")
    for key in ("grid_shape","grid_origin","resolution_m","occupy_threshold","point_count"):
        if rm[key]!=meta[key]:raise ValueError("RENDER_GRID_DRIFT")
    if not np.array_equal(np.asarray(rm["position_world"],np.float32),np.asarray(cert["position_world"],np.float32)):raise ValueError("RENDER_POSE_DRIFT")
    depth=np.fromfile(raw/"depth_z.f32","<f4").reshape(90,160);x,camera=camera_qualification(depth,counts,rm)
    for n in ("renderer.json","depth_z.f32"):
        with (new_pose/n).open("xb") as f:f.write((raw/n).read_bytes())
    write(new_pose/"CAMERA_BINDING.json",camera)
    with (new_pose/"depth_input_2x128x160.f32").open("xb") as f:f.write(x.tobytes())
    write(render/"QUALIFICATION.json",dict(status="SOURCE_RENDER_PASS",source_counts_sha256=plan["counts_sha256"],
        camera=camera,pose=cert,raw_sha256={q.name:sha(q) for q in raw.iterdir() if q.is_file()}))
    return new_pose
def unknown_and_frame(batch):
    from .e1_dataset_qualification import unknown_and_frame as check
    return check(batch)
def final_snapshot(error=None):
    p=proposal();old=Path(p["old_data_root"]);cfg=read(old/"_preparation/qualification_001/CONFIG.json")
    plan=[sid for s in cfg["scenes"] for sid in s["sample_ids"]];run=OUT/"runs/qualification_001"
    samples=[]
    for sid in plan:
        path=run/(sid+"_QUALIFICATION.json")
        if path.exists():samples.append(read(path))
        else:
            split=next(s["split"] for s in cfg["scenes"] if sid in s["sample_ids"])
            samples.append(dict(sample_id=sid,split=split,source="SOURCE_PENDING_BINDING",reachability="NOT_RUN",data_eligible=False))
    stats={}
    for split in ("train","dev"):
        subset=[s for s in samples if s["split"]==split]
        stats[split]=dict(total=len(subset),SOURCE_QUALIFIED=sum(s["source"]=="SOURCE_QUALIFIED" for s in subset),
            FEASIBLE=sum(s["reachability"]=="FEASIBLE_CERTIFIED" for s in subset),
            UNKNOWN=sum(s["reachability"]=="REACHABILITY_UNKNOWN" for s in subset),
            INFEASIBLE=sum(s["reachability"]=="INFEASIBLE_CERTIFIED" for s in subset),
            NOT_RUN=sum(s["reachability"]=="NOT_RUN" for s in subset))
    attempts=[json.loads(x) for x in (PREP/"GPU_CALLS.jsonl").read_text().splitlines()] if (PREP/"GPU_CALLS.jsonl").exists() else []
    status="STOP" if error else "PARTIAL" if any(v["UNKNOWN"] or v["INFEASIBLE"] or v["NOT_RUN"] for v in stats.values()) else "PASS"
    (OUT/"data").mkdir(exist_ok=True)
    split_record=dict(scene_level=True,scenes=read(old/"runs/qualification_001/ALL_SCENE_IDENTITIES.json"),old_source1001="Train-only, excluded",
        original_split_sha256=sha(old/"data/SCENE_SPLIT.json"),scene_dedup_unchanged=True)
    write(OUT/"data/SCENE_SPLIT.json",split_record)
    order=dict(fixed_complete_order=plan,source_qualified_order=[s["sample_id"] for s in samples if s["source"]=="SOURCE_QUALIFIED"],
        eligible_train_order=[s["sample_id"] for s in samples if s["split"]=="train" and s["data_eligible"]],
        eligible_dev_order=[s["sample_id"] for s in samples if s["split"]=="dev" and s["data_eligible"]])
    write(OUT/"data/SAMPLE_ORDER.json",order)
    with (OUT/"data/QUARANTINE.jsonl").open("x") as f:
        for s in samples:
            if not s["data_eligible"]:f.write(json.dumps(s)+"\n")
    write(OUT/"data/DATASET_MANIFEST.json",dict(schema="e1_r2d_qualified_dataset_v1",status=status,samples=samples,split_statistics=stats,
        proposal_sha256=PROPOSAL_SHA,source_freeze_sha256=sha(PREP/"SOURCE_FREEZE.json"),new_counts=0,new_depth_attempts=len(attempts),
        lifetime_counts=7,lifetime_depth_attempts=3+len(attempts),lifetime_CUDA_attempts=10+len(attempts),
        scene_split_sha256=sha(OUT/"data/SCENE_SPLIT.json"),sample_order_sha256=sha(OUT/"data/SAMPLE_ORDER.json"),
        quarantine_sha256=sha(OUT/"data/QUARANTINE.jsonl"),global_failure=error,training_executed=False,calibration_executed=False,
        sparse_dev_certified_feasible=sum(s.get("scene_id")=="dev_sparse_2102" and s["data_eligible"] for s in samples),
        task_goal_feasibility_is_not_portal_coverage=True))
def qualify():
    import torch
    from .e1_dataset_binding import load_e1_bound
    from .e1_dataset_qualification import scene_data
    from .e1_r2c_audit import make_audit_geometry
    torch.set_num_threads(1)
    p=prereqs()
    if read(PREP/"TESTS_PASS.json")["status"]!="PASS" or read(PREP/"PREFLIGHT_PASS.json")["status"]!="PASS":raise ValueError("STAGES_REQUIRED")
    old=Path(p["old_data_root"]);cfg=read(old/"_preparation/qualification_001/CONFIG.json");run=OUT/"runs/qualification_001";run.mkdir(parents=True)
    evidence_index={q["sample_id"]:q for q in read(p["reachability_evidence_index"]["path"])}
    pending={q["pose_id"]:q for q in p["pending_pose_plan"]}
    allposes={q["pose_id"]:q for q in read(AUDIT/"runs/audit_001/POSE_PLAN.json")}
    for scene in cfg["scenes"]:
        guard();sid=scene["scene_id"];source_scene=old/"data"/scene["split"]/sid;dest=OUT/"data"/scene["split"]/sid;dest.mkdir(parents=True)
        write(dest/"SCENE_REF.json",dict(source_scene=str(source_scene),source_identity=read(source_scene/"scene.json")["identity"],
            files_sha256={str(source_scene/n):sha(source_scene/n) for n in ("cloud_xyz.f32","occupancy_counts.i32","grid.json","scene.json")}))
        cloud,counts,meta=scene_data(source_scene)
        for pi in (0,1):
            pose_id=sid+"_p"+str(pi);pose_dest=dest/("p"+str(pi))
            if pose_id in pending:pose=render_pose(pending[pose_id],source_scene,pose_dest,p,run)
            else:
                if pose_id not in p["forbidden_rerender_pose_ids"]:raise ValueError("UNLISTED_REUSE_POSE")
                pose=source_scene/("p"+str(pi));pose_dest.mkdir(parents=True)
                write(pose_dest/"SOURCE_REF.json",dict(readonly_existing_pose=str(pose),files_sha256={str(q):sha(q) for q in pose.iterdir() if q.is_file()}))
            cert=read(pose/"POSE_CERTIFICATE.json");auditg=make_audit_geometry(source_scene,counts,meta,cert)
            for case in cfg["state_cases"]:
                sample_id=pose_id+"_"+case["case"];existing=source_scene/("p"+str(pi))/case["case"]/"SAMPLE_MANIFEST.json"
                path=existing if existing.exists() else bind_task(source_scene,pose,case,pose_dest/case["case"],sample_id,p)
                batch=load_e1_bound(path,sha(path));checks=unknown_and_frame(batch)
                ep=evidence_index[sample_id]["file"];evidence=read(ep)
                proof=verify_reachability(evidence,batch,auditg)
                proof.update(sample_id=sample_id,source_evidence_path=ep,source_evidence_sha256=sha(ep),bound_manifest_sha256=sha(path))
                proofpath=run/(sample_id+"_HARD_EVIDENCE.json");write(proofpath,proof)
                row=dict(sample_id=sample_id,scene_id=sid,split=scene["split"],source="SOURCE_QUALIFIED",
                    reachability=proof["reachability"],data_eligible=proof["reachability"]=="FEASIBLE_CERTIFIED",
                    source_manifest_path=str(path),source_manifest_sha256=sha(path),reused_source_manifest=existing.exists(),
                    hard_evidence_path=str(proofpath),hard_evidence_sha256=sha(proofpath),portal_coverage=proof["portal_coverage"],
                    geometry_sha256=read(source_scene/"scene.json")["identity"]["spatial_occupancy_sha256"],
                    camera_sha256=sha(pose/"CAMERA_BINDING.json"),state_sha256=sha(path.parent/"STATE.json"),
                    checks=checks,depth_pose_id=pose_id,depth_reused_readonly=pose_id not in pending)
                write(run/(sample_id+"_QUALIFICATION.json"),row)
                print(json.dumps(dict(sample_id=sample_id,source=row["source"],reachability=row["reachability"])),flush=True)
    guard();final_snapshot()
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--worker",required=True,choices=("preflight","qualify"));args=parser.parse_args()
    try:
        {"preflight":preflight,"qualify":qualify}[args.worker]()
    except BaseException:
        import traceback
        error=traceback.format_exc()
        if args.worker=="qualify":
            try:final_snapshot(error)
            except BaseException:print("SNAPSHOT_ERROR",traceback.format_exc(),flush=True)
        raise
if __name__=="__main__":main()
