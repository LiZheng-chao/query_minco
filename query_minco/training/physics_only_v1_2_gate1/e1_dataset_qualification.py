"""E1-R2 source-qualified data experiment. No optimizer, policy, or loss changes."""
import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

BASE=Path(__file__).resolve().parents[3]
_REVIEW=os.environ.get("QUERY_MINCO_E1_DATA_REVIEW","qualification_001")
if _REVIEW not in ("qualification_001","r2a_001","r2a_002","r2b_001"):raise ValueError("UNAUTHORIZED_REVIEW")
OUT=BASE/({"r2b_001":"research_outputs/physics_only_v1_2_e1_dataset_v2","r2a_001":"research_outputs/physics_only_v1_2_e1_r2a_supervisor_v1","r2a_002":"research_outputs/physics_only_v1_2_e1_r2a_supervisor_v2"}.get(_REVIEW,"research_outputs/physics_only_v1_2_e1_dataset_v1"))
PREP=OUT/("_preparation/validation_001" if _REVIEW in ("r2a_001","r2a_002") else "_preparation/qualification_001")
PROPOSAL_SHA=("6ba1f74c2ca5834b6805f91af99c1567e63cbe0b953e856d262558a0afde15f9" if _REVIEW=="r2b_001" else "6d32ff77a9d142149a57f62699c074817ec007b8fdac4cf3d14820c9b13b71fa")
PYTHON="/home/lizheng/anaconda3/envs/yopo/bin/python"
YOPO=Path("/home/lizheng/YOPO")
GPU_SMI="/usr/lib/wsl/lib/nvidia-smi"
MEMORY=1610612736
DOMAIN=[[0,-4,-2],[10,4,2]]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    with Path(p).open("x") as f:json.dump(x,f,indent=2,allow_nan=False);f.write("\n")
def proposal():
    p=PREP/"AUTHORIZED_PROPOSAL.json"
    if sha(p)!=PROPOSAL_SHA:raise ValueError("PROPOSAL_SHA")
    value=read(p)
    if _REVIEW=="r2b_001":
        value["future_code_allowlist"]=list(read(PREP/"SOURCE_FREEZE.json"))
        value["source_inputs_sha256"]=read(PREP/"ENTRY.json")["source_inputs_sha256"]
    return value
def protect():
    if (PREP/"STOP.json").exists():raise ValueError("PERSISTENT_STOP "+str(PREP/"STOP.json"))
    e=read(PREP/"ENTRY.json")
    for p,h in e["protected_sha256"].items():
        if sha(p)!=h:raise ValueError("HISTORICAL_SHA_DRIFT "+p)
    for p,h in e["source_inputs_sha256"].items():
        if sha(p)!=h:raise ValueError("SOURCE_SHA_DRIFT "+p)
    if (PREP/"SOURCE_FREEZE.json").exists():
        for p,h in read(PREP/"SOURCE_FREEZE.json").items():
            if sha(BASE/p)!=h:raise ValueError("SOURCE_FREEZE_DRIFT "+p)
    if _REVIEW=="r2b_001":
        approved=read(PREP/"AUTHORIZED_PROPOSAL.json")
        if sha(approved["accepted_binary"]["path"])!=approved["accepted_binary"]["sha256"]:raise ValueError("BINARY_SHA_DRIFT")
        for q,h in approved["immutable_source_sha256"].items():
            if sha(BASE/q)!=h:raise ValueError("IMMUTABLE_SOURCE_DRIFT "+q)
        for repo,key in ((BASE,"query"),(YOPO,"yopo")):
            head=subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True)
            if head!=e["git_before"][key]["head"]:raise ValueError("HEAD_DRIFT")
    if time.time()-e["started_unix"]>=3600:raise ValueError("TOTAL_WALL_BUDGET")
    return e
def gpu_sample():
    cmd=[GPU_SMI,"--query-gpu=index,memory.total,memory.used","--format=csv,noheader,nounits"]
    q=subprocess.run(cmd,capture_output=True,text=True,timeout=3)
    if q.returncode:raise ValueError("GPU_MONITOR_EXIT "+q.stderr)
    rows=[x.strip().split(",") for x in q.stdout.strip().splitlines()]
    if len(rows)!=1:raise ValueError("ONE_GPU_MONITOR_REQUIRED")
    index,total,used=[int(x.strip()) for x in rows[0]]
    if index!=0 or total*1048576<MEMORY or used*1048576>MEMORY:raise ValueError("GPU_MEMORY_BUDGET")
    return dict(unix=time.time(),total_bytes=total*1048576,used_bytes=used*1048576,scope="whole device conservative bound; sampled, native allocations also checked")
def reconstruct(cloud,counts,m):
    h=np.float32(m["resolution_m"]);o=np.asarray(m["grid_origin"],np.float32)
    if h!=np.float32(.1) or m["occupy_threshold"]!=0:raise ValueError("GRID_PROFILE")
    if cloud.ndim!=2 or cloud.shape[1]!=3 or not len(cloud) or not np.isfinite(cloud).all():raise ValueError("CLOUD")
    if not np.array_equal(cloud.min(0),o):raise ValueError("ORIGIN")
    shape=np.ceil((cloud.max(0)-cloud.min(0)+np.float32(.002))/h).astype(int)
    if tuple(shape)!=counts.shape or list(shape)!=m["grid_shape"]:raise ValueError("SHAPE")
    i=np.floor((cloud+np.float32(.001)-o)/h).astype(np.int64)
    if (i<0).any() or (i>=shape).any():raise ValueError("POINT_INDEX")
    flat=(i[:,0]*shape[1]+i[:,1])*shape[2]+i[:,2]
    expected=np.bincount(flat,minlength=int(np.prod(shape))).reshape(tuple(shape))
    if not np.array_equal(expected,counts):raise ValueError("NATIVE_COUNTS_RECONSTRUCTION")
    if counts.sum()!=len(cloud) or len(cloud)!=m["point_count"]:raise ValueError("POINT_COUNT")
    return True
def geometry_identity(cloud,counts,m):
    occupied=np.argwhere(counts>0)
    # Translation independent occupied lattice identity; repetition does not alter it.
    canonical=hashlib.sha256(np.asarray(occupied,"<i4").tobytes()).hexdigest()
    semantics=dict(shape=m["grid_shape"],origin=m["grid_origin"],resolution=m["resolution_m"],
                   threshold=0,xy="symmetricIndex:2*n-2",z="z<=0 solid; above native top UNKNOWN",
                   counts_sha256=hashlib.sha256(counts.tobytes()).hexdigest())
    return dict(scene_source_sha256=hashlib.sha256(cloud.tobytes()).hexdigest(),
                spatial_occupancy_sha256=digest(semantics),canonical_occupied_geometry_sha256=canonical)
def check_split(scenes):
    for i,a in enumerate(scenes):
        for b in scenes[:i]:
            for key in ("scene_source_sha256","spatial_occupancy_sha256","canonical_occupied_geometry_sha256"):
                if a[key]==b[key]:raise ValueError("DUPLICATE_SCENE "+key+" "+a["scene_id"]+" "+b["scene_id"])
    return True
def choose_poses(m,counts):
    from .pose_preflight import grid_boxes,certify_position
    boxes,_=grid_boxes(m,counts)
    o=np.asarray(m["grid_origin"],np.float32).astype(float);h=float(np.float32(.1))
    center=(o[1]+o[1]+counts.shape[1]*h)/2
    poses=[]
    for offset in (-.2,.2):
        target=center+offset
        ys=sorted(range(2,counts.shape[1]-2),key=lambda j:(abs(o[1]+(j+.5)*h-target),j))
        visited=0;selected=None
        for i in range(2,counts.shape[0]-2):
            for j in ys:
                visited+=1
                if visited>4096:break
                pos=np.array([o[0]+(i+.5)*h,o[1]+(j+.5)*h,1.8],np.float32)
                proof=certify_position(pos,m,counts,boxes)
                if proof["passed"]:
                    selected=proof;break
            if selected or visited>=4096:break
        if selected is None:raise ValueError("NO_CERTIFIED_POSE")
        selected.update(schema="e1_r2_pose_v1",candidates_visited=visited,reference_radius_m=.12,margin_m=.03)
        poses.append(selected)
    if poses[0]["position_world"]==poses[1]["position_world"]:raise ValueError("POSE_DUPLICATE")
    return poses
def camera_qualification(depth,counts,m):
    from .camera import camera_input,replay_depth
    if [m["camera"][k] for k in ("fx","fy","cx","cy")]!=[80,80,80,45]:raise ValueError("CAMERA_K")
    x,camera=camera_input(depth,m);replay=replay_depth(counts,m)
    err=np.abs(replay.astype(float)-depth)
    if (err>2e-5).any():raise ValueError("DEPTH_REPLAY_MISMATCH max="+str(float(err.max())))
    camera.update(independent_replay_max_error_m=float(err.max()),replay_tolerance_m=2e-5)
    return x,camera
def scene_data(path):
    path=Path(path);m=read(path/"grid.json")
    counts=np.fromfile(path/"occupancy_counts.i32","<i4").reshape(m["grid_shape"])
    cloud=np.fromfile(path/"cloud_xyz.f32","<f4").reshape(-1,3)
    reconstruct(cloud,counts,m);return cloud,counts,m
def polynomial_witness(state,geometry,side=None,gate_x=None):
    from .hard_validator import validate_polynomial
    from query_minco.evaluation.validation import numerical_reference,evaluate
    v=np.asarray(state[:3],float);a=np.asarray(state[3:6],float);g=np.asarray(state[6:9],float)
    near=np.linalg.norm(g)<=3
    end=g.copy() if near else np.array([3.6,0,0.])
    durations=np.full(3,2/3.)
    head=np.stack([np.zeros(3),v,a])
    tail=np.stack([end,np.zeros(3) if near else np.array([1.2,0,0]),np.zeros(3)])
    # Deterministic audit witness, not network target or teacher supervision.
    inner=np.stack([end/3,end*2/3])
    if side is not None:
        inner[:,1]=side
    coeff=numerical_reference(head,tail,inner,durations)
    hard=validate_polynomial(coeff,durations,geometry,g)
    boundary=max(float(np.abs(evaluate(coeff[0],0,k)-head[k]).max()) for k in range(3))
    boundary=max(boundary,max(float(np.abs(evaluate(coeff[-1],durations[-1],k)-tail[k]).max()) for k in range(3)))
    hard["boundary_PVA_max_abs"]=boundary
    if boundary>1e-7:raise ValueError("WITNESS_BOUNDARY")
    if gate_x is not None:
        world_end=geometry.origin+end
        hard["gate_crossed"]=bool(geometry.origin[0]<gate_x<world_end[0])
        hard["gate_x_world_m"]=gate_x
    return dict(status="PASS" if hard["passed"] else "UNKNOWN",hard=hard,
                coefficients_B0=coeff.tolist(),durations=durations.tolist(),used_as_training_label=False)

def command(argv,folder,name,gpu=False):
    from .e1_resource_supervisor import supervise
    if gpu:
        require_data_gate()
        approved=proposal()["accepted_binary"]
        args=[str(x) for x in argv]
        if args[0]!=approved["path"] or args[1] not in ("counts","depth"):raise ValueError("GPU_COMMAND_NOT_AUTHORIZED")
        ledger=PREP/"GPU_CALLS.jsonl"
        previous=[json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
        kind=args[1]
        if kind=="depth":
            if sum(x["kind"]=="counts" for x in previous)!=7 or read(OUT/"runs/qualification_001/COUNTS_GATE_PASS.json")["status"]!="PASS":
                raise ValueError("COUNTS_GATE_REQUIRED_BEFORE_DEPTH")
        if sum(x["kind"]==kind for x in previous)>=(7 if kind=="counts" else 14) or len(previous)>=21:raise ValueError("GPU_CALL_BUDGET")
        record=dict(index=len(previous)+1,kind=kind,name=name,argv=args,unix=time.time(),semantics="attempt recorded before monitoring/Popen; no refund/retry")
        with ledger.open("a") as f:f.write(json.dumps(record)+"\n");f.flush();os.fsync(f.fileno())
    folder=Path(folder)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",
             NUMEXPR_NUM_THREADS="1",PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
             PYTHONPATH=str(BASE),CUDA_VISIBLE_DEVICES="0" if gpu else "",TMPDIR=str(folder/"tmp"),
             LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:/opt/ros/noetic/lib")
    env.pop("PYTEST_ADDOPTS",None)
    (folder/"tmp").mkdir(parents=True,exist_ok=True)
    return supervise(argv,folder,name,cwd=BASE,env=env,stop_path=PREP/"STOP.json",budget_root=OUT,guard=protect,timeout=120 if gpu else 300,gpu_monitor=gpu_sample if gpu else None)
def freeze():
    protect();p=proposal()
    for file in p["future_code_allowlist"]:
        if not (BASE/file).is_file():raise ValueError("MISSING_NEW_SOURCE "+file)
    write(PREP/"SOURCE_FREEZE.json",{q:sha(BASE/q) for q in p["future_code_allowlist"]})
    write(PREP/"CONFIG.json",dict(proposal_sha256=PROPOSAL_SHA,scenes=p["scenes"],layout_precommit=p["layout_precommit"],
           poses=p["pose_rule"],camera=p["camera"],states=p["state_cases"],center_domain_B0_m=DOMAIN,
           source1001="permanent Train only; excluded from new sample order; read-only dedup input",
           source_qualification_separate_from_task_reachability=True))
    samples=[gpu_sample() for _ in range(3)]
    write(PREP/"GPU_MONITOR_PREFLIGHT.json",dict(status="PASS",samples=samples,
          enforcement="whole-device sampling plus native cudaMemGetInfo and exact explicit allocation precheck; not a claim of continuous peak measurement",
          peak_budget_bytes=MEMORY))
def build():
    protect();run=OUT/"runs/build_001";run.mkdir(parents=True);art=run/"artifacts";art.mkdir()
    native=BASE/"research_outputs/physics_only_v1_2_gate1_forest_v1/runs/build_002/artifacts"
    oldobjects=sorted(native.glob("*.o"))
    needed=[q for q in oldobjects if q.name!="renderer_host.host.o"]
    write(run/"READONLY_NATIVE_OBJECTS.json",{str(q):sha(q) for q in needed})
    inc=["-I"+str(YOPO/"Simulator/src/include"),"-I"+str(YOPO/"Simulator/src/src"),"-I/usr/include/opencv4","-I/usr/include/pcl-1.10","-I/usr/include/eigen3","-I/opt/ros/noetic/include"]
    src=BASE/"query_minco/training/physics_only_v1_2_gate1"
    plans=[
        ["/usr/bin/g++-9","-std=c++17","-O2"]+inc+["-c",src/"e1_renderer_host.cpp","-o",art/"e1_renderer_host.o"],
        ["/usr/local/cuda-11.8/bin/nvcc","-std=c++17","-O2","-arch=sm_75","-ccbin","/usr/bin/g++-9"]+inc+["-c",src/"e1_renderer_bridge.cu","-o",art/"e1_renderer_bridge.o"],
        ["/usr/bin/g++-9",art/"e1_renderer_host.o",art/"e1_renderer_bridge.o"]+needed+["-o",art/"e1_renderer_bridge",
         "-L/usr/local/cuda-11.8/lib64","-Wl,-rpath,/usr/local/cuda-11.8/lib64","-L/opt/ros/noetic/lib","-Wl,-rpath,/opt/ros/noetic/lib",
         "-lopencv_core","-lopencv_imgproc","-lopencv_calib3d","-lpcl_common","-lpcl_io","-lyaml-cpp","-lrosconsole","-lrostime","-lcpp_common","-lroscpp","-lcurand","-lcudart"]]
    for i,cmd in enumerate(plans):command(cmd,run,"build_"+str(i))
    command(["ldd",art/"e1_renderer_bridge"],run,"ldd")
    for q,h in read(run/"READONLY_NATIVE_OBJECTS.json").items():
        if sha(q)!=h:raise ValueError("NATIVE_OBJECT_DRIFT")
    write(run/"BINARY.json",dict(path=str(art/"e1_renderer_bridge"),sha256=sha(art/"e1_renderer_bridge"),
          source_sha256=read(PREP/"SOURCE_FREEZE.json"),native_objects=read(run/"READONLY_NATIVE_OBJECTS.json")))
def tests():
    run=OUT/"runs/tests_001";run.mkdir(parents=True)
    args=["-q","-p","no:cacheprovider",
          "--deselect=tests/test_physics_v12_gate1_geometry.py::test_exact_distance_matches_independent_all_boxes","--basetemp="+str(run/"pytest_tmp"),"--junitxml="+str(run/"TEST_RESULTS.xml"),
          "tests/test_physics_v12_e1_dataset.py","tests/test_physics_v12_e1_supervisor.py","tests/test_physics_v12_gate1_geometry.py",
          "tests/test_physics_v12_gate1_objective.py","tests/test_physics_v12_gate1_pose.py","tests/test_physics_v12_e1_research.py",
          "tests/test_physics_v12_gate1_sdf_nonsmooth.py","query_minco/training/physics_only_v1_2/tests",
          "query_minco/training/physics_only_v1/tests"]
    code="import sys;sys.path.append('/home/lizheng/anaconda3/lib/python3.11/site-packages');import pytest;raise SystemExit(pytest.main("+repr(args)+"))"
    command([PYTHON,"-B","-c",code],run,"pytest")
def unknown_and_frame(batch):
    import torch
    g=batch.geometry[0]
    from .geometry import world_points
    p=torch.tensor([[.2,.1,.1]],dtype=torch.float64)
    actual=world_points(p,torch.tensor(g.origin),torch.tensor(g.rotation))
    np.testing.assert_allclose((actual.numpy()-g.origin)@g.rotation,p.numpy(),atol=1e-12,rtol=0)
    outside=torch.tensor((g.roi[1]+1-g.origin)@g.rotation,dtype=torch.float64)
    try:g.sdf(outside)
    except ValueError as e:
        if "UNKNOWN" not in str(e):raise
    else:raise ValueError("UNKNOWN_NOT_REJECTED")
    return dict(world_B0_roundtrip="PASS",UNKNOWN_outside_ROI_rejected=True,network_inputs=["depth+mask","B0-FLU 12D state"])
def require_data_gate():
    if _REVIEW!="r2b_001" or os.environ.get("QUERY_MINCO_E1_DATA_STAGE")!="qualify":
        raise ValueError("DATA_GENERATION_REQUIRES_SEPARATE_R2B_AUTHORIZATION")
    protect()
    for stage in ("PREFLIGHT","TESTS"):
        if read(PREP/(stage+"_PASS.json"))["status"]!="PASS":raise ValueError("STAGE_GATE_"+stage)
def qualify():
    require_data_gate()
    from .e1_dataset_binding import load_e1_bound
    protect();p=proposal();run=OUT/"runs/qualification_001";run.mkdir(parents=True)
    binary=read(OUT/"runs/build_001/BINARY.json");exe=Path(binary["path"])
    if sha(exe)!=binary["sha256"]:raise ValueError("BINARY_SHA")
    all_scenes=[];samples=[];order=[];renders=0;counts_calls=0;task_unknown=[]
    old=BASE/"research_outputs/physics_only_v1_2_gate1_forest_v1/data/train_sparse_1001_001"
    oldmeta=read(old/"renderer.json");oldc=np.fromfile(old/"occupancy_counts.i32","<i4").reshape(oldmeta["grid_shape"]);oldcloud=np.fromfile(old/"cloud_xyz.f32","<f4").reshape(-1,3)
    oldidentity=geometry_identity(oldcloud,oldc,oldmeta)
    for spec in p["scenes"]:
        protect();sid=spec["scene_id"];data=OUT/"data"/spec["split"]/sid;data.mkdir(parents=True)
        specpath=run/(sid+"_SCENE_SPEC.json")
        frozen=dict(spec);frozen["layout"]=p["layout_precommit"].get(sid,{})
        write(specpath,frozen)
        export=run/(sid+"_counts_export");export.mkdir()
        counts_calls+=1
        if counts_calls>7:raise ValueError("COUNTS_CALL_BUDGET")
        command([exe,"counts",YOPO,specpath,export],run,sid+"_counts",gpu=True)
        for n in ("cloud_xyz.f32","occupancy_counts.i32","grid.json"):
            with (data/n).open("xb") as f:f.write((export/n).read_bytes())
        cloud,counts,meta=scene_data(data);identity=geometry_identity(cloud,counts,meta)
        record=dict(scene_id=sid,split=spec["split"],seed=spec["seed"],source_type=spec["source_type"],identity=identity,
                    raw_point_count=len(cloud),grid_shape=meta["grid_shape"],canonical_note="shared tree asset is not asset-level OOD",source_spec_sha256=sha(specpath))
        flat=dict(scene_id=sid,split=spec["split"],**identity)
        write(run/(sid+"_COUNTS_NUMERICAL.json"),dict(scene_id=sid,identity=identity,independent_reconstruction_exact=True,
              counts_sha256=sha(data/"occupancy_counts.i32"),source_sha256=sha(data/"cloud_xyz.f32"),grid=meta))
        check_split(all_scenes+[flat])
        if any(identity[k]==oldidentity[k] for k in identity):raise ValueError("SOURCE1001_DUPLICATE")
        all_scenes.append(flat);write(data/"scene.json",record)
    # All seven actual native occupancies must precede any pose selection/render.
    write(run/"ALL_SCENE_IDENTITIES.json",all_scenes)
    write(run/"COUNTS_GATE_PASS.json",dict(status="PASS",scenes=len(all_scenes),source1001_cross_split="not reused",attempted_counts=counts_calls))
    for spec in p["scenes"]:
        sid=spec["scene_id"];data=OUT/"data"/spec["split"]/sid
        specpath=run/(sid+"_SCENE_SPEC.json")
        cloud,counts,meta=scene_data(data);identity=read(data/"scene.json")["identity"]
        poses=choose_poses(meta,counts)
        for pi,cert in enumerate(poses):
            pose=data/("p"+str(pi));pose.mkdir()
            cert.update(source_cloud_sha256=sha(data/"cloud_xyz.f32"),source_occupancy_sha256=sha(data/"occupancy_counts.i32"),
                        scene_id=sid,split=spec["split"],source_scene_identity=identity)
            certpath=pose/"POSE_CERTIFICATE.json";write(certpath,cert)
            # Recompute proof and hashes immediately before depth; never render a guessed pose.
            from .pose_preflight import certify_position
            if not certify_position(cert["position_world"],meta,counts)["passed"]:raise ValueError("POSE_PRE_RENDER")
            if sha(data/"cloud_xyz.f32")!=cert["source_cloud_sha256"] or sha(data/"occupancy_counts.i32")!=cert["source_occupancy_sha256"]:raise ValueError("POSE_SOURCE_DRIFT")
            render=OUT/"runs"/("render_"+sid+"_p"+str(pi)+"_001");render.mkdir()
            raw=render/"raw";raw.mkdir()
            renders+=1
            if renders>14:raise ValueError("RENDER_BUDGET")
            command([exe,"depth",data/"cloud_xyz.f32",certpath,raw],render,"render",gpu=True)
            if sha(raw/"occupancy_counts.i32")!=cert["source_occupancy_sha256"]:raise ValueError("DEPTH_OCCUPANCY_DRIFT")
            for n in ("renderer.json","depth_z.f32"):
                with (pose/n).open("xb") as f:f.write((raw/n).read_bytes())
            depth=np.fromfile(pose/"depth_z.f32","<f4").reshape(90,160);rm=read(pose/"renderer.json")
            x,cam=camera_qualification(depth,counts,rm);write(pose/"CAMERA_BINDING.json",cam)
            with (pose/"depth_input_2x128x160.f32").open("xb") as f:f.write(x.tobytes())
            for statecase in p["state_cases"]:
                case=statecase["case"];sub=pose/case;sub.mkdir();sample_id=sid+"_p"+str(pi)+"_"+case
                state=dict(v0_world_mps=statecase["v0_B0_mps"],a0_world_mps2=statecase["a0_B0_mps2"],g_B0_m=statecase["g_B0_m"],
                           gravity_world_unit=statecase["gravity_B0"],snapshot_time_s=0.,origin="explicit offline initial condition, not measured odometry")
                write(sub/"STATE.json",state)
                boundfiles=[data/n for n in ("scene.json","cloud_xyz.f32","occupancy_counts.i32","grid.json")]
                boundfiles += list(pose.glob("*.*"))+[sub/"STATE.json",exe,PREP/"CONFIG.json",PREP/"AUTHORIZED_PROPOSAL.json",specpath,OUT/"runs/build_001/BINARY.json",run/"ALL_SCENE_IDENTITIES.json"]
                boundfiles += [BASE/q for q in p["future_code_allowlist"]]+[Path(q) for q in p["source_inputs_sha256"]]
                manifest=dict(schema="e1_r2_bound_sample_v1",sample_id=sample_id,scene_id=sid,scene_identity=identity,split=spec["split"],frame="B0-FLU",
                              teacher_loaded=False,scene_dir=str(data),pose_dir=str(pose),binary_path=str(exe),scene_registry_path=str(run/"ALL_SCENE_IDENTITIES.json"),
                              files_sha256={str(q.resolve()):sha(q) for q in boundfiles if q.is_file()},
                              geometry_authority="complete generated native occupancy only; no sensor-free inference; outside native ROI/top UNKNOWN")
                path=sub/"SAMPLE_MANIFEST.json";write(path,manifest)
                batch=load_e1_bound(path,sha(path));checks=unknown_and_frame(batch)
                witness=polynomial_witness(batch.state[0].numpy(),batch.geometry[0])
                tasks=dict(nominal=witness,portal_coverage="NOT_APPLICABLE")
                if "two_portals" in sid:
                    gate=p["layout_precommit"][sid]["gate_x_m"]
                    block=p["layout_precommit"][sid]["block_box_m"]
                    sides=[block[0][1]-.25-batch.geometry[0].origin[1],block[1][1]+.25-batch.geometry[0].origin[1]]
                    tasks["portals"]=[polynomial_witness(batch.state[0].numpy(),batch.geometry[0],side,gate) for side in sides]
                    tasks["portal_coverage"]="PASS" if all(w["status"]=="PASS" and w["hard"]["gate_crossed"] for w in tasks["portals"]) else "UNKNOWN"
                    tasks["separator"]=portal_separator(batch.geometry[0],meta,counts,block,gate)
                    if not tasks["separator"]["passed"]:tasks["portal_coverage"]="UNKNOWN"
                if "single_slit" in sid:
                    layout=p["layout_precommit"][sid];tasks["actual_gap"]=slit_gap(meta,counts,layout)
                    if not tasks["actual_gap"]["center_corridor_positive"]:raise ValueError("SLIT_GEOMETRY_REJECTED")
                taskstatus=witness["status"]
                if "two_portals" in sid and case=="far":taskstatus=tasks["portal_coverage"]
                row=dict(sample_id=sample_id,scene_id=sid,split=spec["split"],manifest_path=str(path),manifest_sha256=sha(path),
                         source_qualified=True,task_reachability=taskstatus,clearance_lower_bound_m=cert["clearance_lower_bound_m"],checks=checks,
                         task_evidence_path=str(run/(sample_id+"_TASK.json")))
                write(run/(sample_id+"_TASK.json"),tasks)
                if taskstatus!="PASS":task_unknown.append(sample_id)
                samples.append(row);order.append(sample_id)
                write(run/(sample_id+"_QUALIFICATION.json"),row)
                if taskstatus!="PASS":raise ValueError("REQUIRED_TASK_REACHABILITY_UNKNOWN "+sample_id)
            # Preserve qualified source contracts even when reachability is UNKNOWN.
        write(run/(sid+"_QUALIFICATION.json"),dict(scene_id=sid,source_qualified=True,counts_calls=counts_calls,depth_calls=renders))
    write(OUT/"data/SCENE_SPLIT.json",dict(scene_level=True,scenes=all_scenes,existing_source1001="Train-only excluded from new order",dedup="PASS"))
    write(OUT/"data/SAMPLE_ORDER.json",order)
    with (OUT/"data/REJECTIONS.jsonl").open("x") as f:
        for sid in task_unknown:f.write(json.dumps(dict(sample_id=sid,reason="TASK_REACHABILITY_UNKNOWN; not qualified for calibration"))+"\n")
    write(OUT/"data/DATASET_MANIFEST.json",dict(status="PASS" if not task_unknown else "PARTIAL",samples=samples,
          qualified_source_samples=len(samples),task_qualified_samples=sum(x["task_reachability"]=="PASS" for x in samples),
          source_train=sum(x["split"]=="train" for x in samples),source_dev=sum(x["split"]=="dev" for x in samples),
          counts_only_calls=counts_calls,depth_calls=renders,train_only_calibration_eligible=not task_unknown,
          optimizer_updates=0,unknown_tasks=task_unknown))
    return
def slit_gap(meta,counts,layout):
    o=np.asarray(meta["grid_origin"],np.float32).astype(float);h=float(np.float32(.1))
    slab=np.asarray(layout["slab_box_m"],float);yz=1.8
    xlo=max(0,int(np.floor((slab[0,0]-o[0])/h)))
    xhi=min(counts.shape[0],int(np.ceil((slab[1,0]-o[0])/h)))
    zlo=max(1,int(np.floor((yz-.150011-o[2])/h)))
    zhi=min(counts.shape[2],int(np.ceil((yz+.150011-o[2])/h)))
    occupied=(counts[xlo:xhi,:,zlo:zhi]>0).any(axis=(0,2))
    center=sum(layout["remove_y_interval_m"])/2
    j=int(np.floor((center-o[1])/h))
    if occupied[j]:return dict(center_corridor_positive=False,actual_closed_voxel_gap_m=0.)
    low=high=j
    while low>0 and not occupied[low-1]:low-=1
    while high+1<len(occupied) and not occupied[high+1]:high+=1
    gap=(high-low+1)*h-2e-5
    return dict(actual_closed_voxel_gap_m=float(gap),center_corridor_width_m=float(gap-.3),
                center_corridor_positive=bool(gap>.300002),y_bounds_world_m=[float(o[1]+low*h+1e-5),float(o[1]+(high+1)*h-1e-5)],
                slab_x_indices=[xlo,xhi],body_z_indices=[zlo,zhi],
                dynamic_reachability="requires independent polynomial Hard; width alone not sufficient")
def portal_separator(g,meta,counts,block,gate):
    # Actual occupied closed voxels must form a vertical connected separator
    # across the collision-free center altitude; no assumed nominal full height.
    o=np.asarray(meta["grid_origin"],np.float32).astype(float);h=float(np.float32(.1))
    ix=int(np.floor((gate-o[0])/h))
    domain=g.domain.numpy();top=min(g.roi[1,2]-.150001,g.origin[2]+domain[1,2])
    bottom=max(g.floor+.150001,g.origin[2]+domain[0,2])
    y0=int(np.floor((block[0][1]-o[1])/h));y1=int(np.ceil((block[1][1]-o[1])/h))
    # Native occupied support, not rounded nominal boundary indices, defines the separator.
    y0=max(0,y0-1);y1=min(counts.shape[1],y1+1)
    occupied=np.argwhere(counts[ix,y0:y1,1:]>0)
    if not len(occupied):return dict(passed=False,reason="NO_NATIVE_SEPARATOR")
    ylow=y0+int(occupied[:,0].min());yhigh=y0+int(occupied[:,0].max())+1
    z=sorted(set(int(k)+1 for k in occupied[:,1]))
    continuous=all(b==a+1 for a,b in zip(z,z[1:]))
    low=o[2]+z[0]*h-1e-5;high=o[2]+(z[-1]+1)*h+1e-5
    vertical=bool(continuous and low-.15<=bottom and high+.15>=top)
    # The discrete column must also be occupied across every intended y/z cell.
    solid=bool((counts[ix,ylow:yhigh,z[0]:z[-1]+1]>0).all())
    return dict(passed=vertical and solid,gate_x_world_m=gate,native_vertical_range_m=[float(low),float(high)],
                certified_center_altitude_m=[float(bottom),float(top)],solid_slice=solid,vertical_no_over_under=vertical)
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--worker",choices=("build","tests","qualify"),required=True);args=parser.parse_args()
    {"build":build,"tests":tests,"qualify":qualify}[args.worker]()
if __name__=="__main__":main()
