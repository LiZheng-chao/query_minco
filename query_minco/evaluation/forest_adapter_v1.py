"""Read-only YOPO forest source bridge and strict snapshot loader. No planner implementation."""
from pathlib import Path
import json,os,itertools
import numpy as np
import torch
from scipy.spatial import cKDTree
from query_minco.audit.run_provenance_v1 import run_logged
from query_minco.evaluation.mirror_holdout_v1 import sha,write
from query_minco.evaluation.validation import numerical_reference
from query_minco.training.physics_only_v1_2_gate1.pose_preflight import certify_position,grid_boxes
from query_minco.training.physics_only_v1_2_gate1.e1_dataset_qualification import reconstruct,camera_qualification
from query_minco.training.physics_only_v1_2_gate1.geometry import TypedBoundOccupancy,_SEAL
from query_minco.training.physics_only_v1_2_gate1.binding import BoundBatch
from query_minco.training.physics_only_v1_2_gate1.hard_validator import validate_polynomial
COMPONENT_ID="strict_yopo_forest_voxel_equivalent_adapter_v1"
def verify_files(files):
    for p,h in files.items():
        if sha(p)!=h:raise ValueError("FOREST_SOURCE_SHA "+p)
def compact_source(path,out):
    out=Path(out);cloud=np.fromfile(path,"<f4").reshape(-1,3)
    if not len(cloud) or not np.isfinite(cloud).all():raise ValueError("FOREST_SOURCE_POINTS")
    origin=cloud.min(0);maximum=cloud.max(0);h=np.float32(.1)
    shape=np.ceil((maximum-origin+np.float32(.002))/h).astype(int)
    if int(np.prod(shape))>30000000:raise ValueError("NATIVE_GRID_BUDGET")
    ids=np.floor((cloud+np.float32(.001)-origin)/h).astype(np.int64)
    flat=np.ravel_multi_index(ids.T,shape)
    counts=np.bincount(flat,minlength=int(np.prod(shape))).reshape(tuple(shape)).astype("<i4")
    unique,first=np.unique(flat,return_index=True)
    extrema=np.r_[np.argmin(cloud,axis=0),np.argmax(cloud,axis=0)]
    chosen=np.unique(np.r_[first,extrema]);compact=cloud[chosen].copy()
    cids=np.floor((compact+np.float32(.001)-origin)/h).astype(np.int64)
    ccounts=np.bincount(np.ravel_multi_index(cids.T,shape),minlength=int(np.prod(shape))).reshape(tuple(shape)).astype("<i4")
    if not np.array_equal(counts>0,ccounts>0) or not np.array_equal(compact.min(0),origin) or not np.array_equal(compact.max(0),maximum):raise ValueError("OCCUPANCY_COMPACTION_DRIFT")
    if len(compact)>3000000:raise ValueError("RENDERER_POINT_BUDGET")
    meta=dict(grid_origin=origin.tolist(),grid_shape=shape.tolist(),resolution_m=float(h),occupy_threshold=0,point_count=len(compact))
    reconstruct(compact,ccounts,meta)
    compact.tofile(out/"compact_cloud.f32");ccounts.tofile(out/"COMPACT_COUNTS.i32");counts.tofile(out/"FULL_SOURCE_COUNTS.i32")
    write(out/"COMPACT_GRID.json",meta)
    proof=dict(source_points=len(cloud),compact_points=len(compact),occupied_voxels=int((counts>0).sum()),grid_cells=int(np.prod(shape)),
        occupied_fraction=float((counts>0).mean()),min_max_preserved=True,positive_occupancy_exact=True,
        multiplicities_preserved=False,threshold=0,mask_sha256=__import__("hashlib").sha256(np.asarray(counts>0,dtype=np.uint8).tobytes()).hexdigest(),
        source_cloud_sha256=sha(path),compact_cloud_sha256=sha(out/"compact_cloud.f32"))
    write(out/"COMPACTION_PROOF.json",proof)
    return ccounts,meta,proof
def geometry(counts,meta,pose,identity,verify):
    origin=np.asarray(meta["grid_origin"],np.float32).astype(float);h=float(np.float32(meta["resolution_m"]))
    roi=np.stack([origin+1e-5,origin+np.asarray(counts.shape)*h-1e-5])
    return TypedBoundOccupancy(seal=_SEAL,counts=counts,meta=meta,pose=pose,roi=roi,domain=[[0,-4,-2],[10,4,2]],identity=identity,revalidate=verify)
def choose_poses(counts,meta,plan):
    boxes,_=grid_boxes(meta,counts);centers=boxes.mean(1);tree=cKDTree(centers);circ=float(np.linalg.norm((boxes[:,1]-boxes[:,0])/2,axis=1).max())
    origin=np.asarray(meta["grid_origin"],np.float32).astype(float);h=float(np.float32(.1));upper=origin+np.array(counts.shape)*h
    selected=[];used=[]
    for frac in plan["pose_anchor_fractions"]:
        anchor=origin+(upper-origin)*np.array([frac[0],frac[1],.5]);anchor[2]=plan["pose_height_m"]
        base=np.rint((anchor[:2]-origin[:2])/h-.5).astype(int)
        offsets=sorted(itertools.product(range(-6,7),range(-6,7)),key=lambda p:(p[0]*p[0]+p[1]*p[1],p))
        result=None;reasons=[]
        for dx,dy in offsets[:128]:
            ij=base+np.array([dx,dy]);p=np.array([origin[0]+(ij[0]+.5)*h,origin[1]+(ij[1]+.5)*h,plan["pose_height_m"]],dtype=np.float32)
            if any(np.array_equal(p,x) for x in used):continue
            bound=float(np.min(np.r_[p-origin,upper-p]))-1e-5
            nearest_lower=float(tree.query(p)[0])-circ
            if min(bound,nearest_lower)<plan["initial_clearance_sampling_m"]:continue
            cert=certify_position(p,meta,counts,boxes)
            if cert["passed"] and cert["clearance_lower_bound_m"]>=plan["initial_clearance_sampling_m"]:
                cert["schema"]="e1_r2_pose_v1";cert["anchor_world_m"]=anchor.tolist();cert["nearest_center_lower_bound_m"]=nearest_lower
                result=cert;used.append(p);break
        selected.append(result)
    return selected
def witness_bank(state,plan):
    s=np.asarray(state,dtype=float);goal=s[6:9];end=goal*min(1.,2.4/np.linalg.norm(goal))
    u=np.array([1/3,2/3]);base=(10*u**3-15*u**4+6*u**5)[:,None]*end
    head=np.stack([np.zeros(3),s[:3],s[3:6]]);tail=np.stack([end,np.zeros(3),np.zeros(3)]);dt=np.full(3,2/3)
    for y in plan["witness_bank"]["inner_y_m"]:
        for z in plan["witness_bank"]["inner_z_m"]:
            inner=base.copy();inner[:,1]+=y;inner[:,2]+=z
            yield dict(inner_y_m=y,inner_z_m=z,coefficients=numerical_reference(head,tail,inner,dt),durations=dt)
def prepare(root,plan_path,out):
    root=Path(root);out=Path(out);out.mkdir();plan=json.loads(Path(plan_path).read_text())
    source=json.loads((Path(plan_path).parent/"YOPO_SOURCE_IDENTITY.json").read_text());verify_files(source)
    build=json.loads((Path(plan_path).parent/"build/BUILD_IDENTITY.json").read_text())
    if sha(build["binary"])!=build["binary_sha256"]:raise ValueError("GENERATOR_BINARY_SHA")
    binary=root/"research_outputs/physics_only_v1_2_e1_r2a_supervisor_v2/runs/build_001/artifacts/e1_renderer_bridge"
    if sha(binary)!="7cb7ead45a858da5f56fc0d9ebc3b9c35a6274ede690f54c58564fcae5b0069b":raise ValueError("RENDER_BINARY_SHA")
    env=dict(os.environ,CUDA_VISIBLE_DEVICES="0",LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:/opt/ros/noetic/lib",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
    maps=[];snapshots=[];excluded=[]
    for spec in plan["maps"]:
        folder=out/spec["map_id"];folder.mkdir();write(folder/"MAP_SPEC.json",spec);raw=folder/"raw_source";raw.mkdir()
        p=run_logged([build["binary"],str(root.parent/"YOPO"),str(folder/"MAP_SPEC.json"),str(raw)],folder/"generator_process",cwd=root,env=env,runspec=plan_path,timeout_s=180)
        if p["exit_code"]!=0:
            excluded.append(dict(map_id=spec["map_id"],planned_snapshots=4,reason="GENERATOR_EXIT",exit_code=p["exit_code"]));continue
        counts,meta,proof=compact_source(raw/"source_cloud.f32",folder)
        native_paths=[raw/"source_cloud.f32",folder/"compact_cloud.f32",folder/"COMPACT_COUNTS.i32",folder/"FULL_SOURCE_COUNTS.i32",folder/"COMPACT_GRID.json",folder/"COMPACTION_PROOF.json",folder/"MAP_SPEC.json",Path(build["binary"]),binary,Path(plan_path)]
        map_manifest=dict(schema="forest_map_source_v1",map=spec,files_sha256={str(p.resolve()):sha(p) for p in native_paths},yopo_sources_sha256=source,
            original_map_type=5,tree_count=(int(spec["map_width_m"]/spec["tree_dist_m"]))**2,scope="12m local native forest window, not original60m benchmark")
        write(folder/"MAP_MANIFEST.json",map_manifest)
        maps.append(dict(map_id=spec["map_id"],seed=spec["seed"],density=spec["density"],map_manifest=str((folder/"MAP_MANIFEST.json").resolve()),map_manifest_sha256=sha(folder/"MAP_MANIFEST.json"),**proof))
        poses=choose_poses(counts,meta,plan)
        for pose_index,cert in enumerate(poses):
            if cert is None:
                excluded.append(dict(map_id=spec["map_id"],pose_index=pose_index,planned_snapshots=2,reason="NO_CLEARANCE_POSE_AMONG_REGISTERED128"));continue
            pd=folder/("pose%d"%pose_index);pd.mkdir();write(pd/"POSE_CERTIFICATE.json",cert);render=pd/"render";render.mkdir()
            p=run_logged([str(binary),"depth",str(folder/"compact_cloud.f32"),str(pd/"POSE_CERTIFICATE.json"),str(render)],pd/"renderer_process",cwd=root,env=env,runspec=plan_path,timeout_s=120)
            if p["exit_code"]!=0:
                excluded.append(dict(map_id=spec["map_id"],pose_index=pose_index,planned_snapshots=2,reason="RENDER_EXIT",exit_code=p["exit_code"]));continue
            actual=json.loads((render/"renderer.json").read_text());got=np.fromfile(render/"occupancy_counts.i32","<i4").reshape(actual["grid_shape"])
            if not np.array_equal(got,counts):raise ValueError("NATIVE_COMPACT_COUNTS_DRIFT")
            depth=np.fromfile(render/"depth_z.f32","<f4").reshape(90,160);processed,camera=camera_qualification(depth,counts,actual)
            processed.tofile(pd/"depth_input_2x128x160.f32");write(pd/"CAMERA_BINDING.json",camera)
            for task in plan["tasks_per_pose"]:
                sd=pd/task;sd.mkdir();motion=plan["states"][pose_index];state=np.r_[motion["v"],motion["a"],plan["goals_B0"][task],[0,0,-1]].astype("<f4")
                state.tofile(sd/"state12.f32")
                identity=dict(map_id=spec["map_id"],seed=spec["seed"],split="forest_test",component=COMPONENT_ID)
                g=geometry(counts,meta,cert,identity,lambda:None);bank=[]
                for w in witness_bank(state,plan):
                    hard=validate_polynomial(w["coefficients"],w["durations"],g,state[6:9])
                    bank.append(dict(inner_y_m=w["inner_y_m"],inner_z_m=w["inner_z_m"],hard=hard,coefficients=w["coefficients"].tolist(),durations=w["durations"].tolist(),used_as_training_label=False))
                feasible=any(w["hard"]["passed"] for w in bank)
                classification="CERTIFIED_FEASIBLE" if feasible else "FEASIBILITY_UNDETERMINED"
                stats=dict(classification=classification,proof="pre-registered finite witness bank, not optimizer or policy selection",witnesses=bank,
                    feasible_directions=[dict(y=w["inner_y_m"],z=w["inner_z_m"]) for w in bank if w["hard"]["passed"]],
                    certified_infeasible=False,nearest_initial_obstacle_m=cert["native_closed_voxel_distance_m"],
                    initial_clearance_lower_bound_m=cert["clearance_lower_bound_m"],depth_valid_fraction=float(processed[1,19:109].mean()),
                    no_return_fraction=float((depth>=20).mean()),representation=dict(position_bounds=[[0,-4,-2],[10,4,2]],goal_in_position_support=True,
                    far_means_progress_not_arrival=task=="far_progress"),motion_source="explicit offline initial condition, not measured odometry")
                write(sd/"PROBLEM_QUALIFICATION.json",stats)
                operational=[folder/"MAP_MANIFEST.json",folder/"compact_cloud.f32",folder/"COMPACT_COUNTS.i32",folder/"COMPACT_GRID.json",folder/"COMPACTION_PROOF.json",pd/"POSE_CERTIFICATE.json",pd/"CAMERA_BINDING.json",pd/"depth_input_2x128x160.f32",render/"renderer.json",render/"depth_z.f32",render/"occupancy_counts.i32",sd/"state12.f32",sd/"PROBLEM_QUALIFICATION.json",Path(__file__)]
                m=dict(schema="strict_forest_snapshot_v1",component_id=COMPONENT_ID,snapshot_id=spec["map_id"]+"_p%d_"%pose_index+task,map_id=spec["map_id"],seed=spec["seed"],density=spec["density"],pose_index=pose_index,task=task,
                    classification=classification,split="forest_test",folder=str(sd.resolve()),map_folder=str(folder.resolve()),pose_folder=str(pd.resolve()),
                    map_manifest_sha256=sha(folder/"MAP_MANIFEST.json"),files_sha256={str(p.resolve()):sha(p) for p in operational},frame="B0-FLU",teacher_loaded=False)
                write(sd/"SNAPSHOT_MANIFEST.json",m)
                snapshots.append(dict(snapshot_id=m["snapshot_id"],map_id=spec["map_id"],seed=spec["seed"],density=spec["density"],task=task,classification=classification,
                    manifest=str((sd/"SNAPSHOT_MANIFEST.json").resolve()),manifest_sha256=sha(sd/"SNAPSHOT_MANIFEST.json")))
                print("FOREST_SNAPSHOT_QUALIFIED",m["snapshot_id"],classification,flush=True)
        verify_files(source)
    write(out/"FOREST_INDEX.json",snapshots);write(out/"MAP_INDEX.json",maps)
    result=dict(input_qualified_snapshots=len(snapshots),maps=len(maps),planned_snapshots=24,excluded=excluded,
        independently_feasible=sum(x["classification"]=="CERTIFIED_FEASIBLE" for x in snapshots),
        infeasible=0,undetermined=sum(x["classification"]=="FEASIBILITY_UNDETERMINED" for x in snapshots),
        source_profile="independent map-seed TEST-only; shared tree asset",network_evaluated=False,network_updates=0)
    write(out/"FOREST_QUALIFICATION.json",result);return result
class ForestAdapter:
    component_id=COMPONENT_ID
    @staticmethod
    def load_snapshot(path,expected_sha):
        path=Path(path)
        if sha(path)!=expected_sha:raise ValueError("FOREST_MANIFEST_SHA")
        raw=path.read_bytes();m=json.loads(raw)
        if m["schema"]!="strict_forest_snapshot_v1" or m["component_id"]!=COMPONENT_ID or m["frame"]!="B0-FLU" or m["teacher_loaded"] is not False:raise ValueError("FOREST_SCHEMA")
        folder=Path(m["map_folder"]);pd=Path(m["pose_folder"]);sd=Path(m["folder"]);map_path=folder/"MAP_MANIFEST.json"
        if sha(map_path)!=m["map_manifest_sha256"]:raise ValueError("MAP_MANIFEST_SHA")
        source=json.loads(map_path.read_text());verify_files(source["files_sha256"]);verify_files(source["yopo_sources_sha256"])
        def verify():
            if path.read_bytes()!=raw:raise ValueError("SNAPSHOT_MANIFEST_DRIFT")
            verify_files(m["files_sha256"])
        verify();pose=json.loads((pd/"POSE_CERTIFICATE.json").read_text());meta=json.loads((folder/"COMPACT_GRID.json").read_text())
        if pose["R_WB0"]!=np.eye(3).tolist():raise ValueError("FOREST_ROTATION_PROFILE_UNSUPPORTED")
        counts=np.fromfile(folder/"COMPACT_COUNTS.i32","<i4").reshape(meta["grid_shape"])
        g=geometry(counts,meta,pose,dict(map_id=m["map_id"],seed=m["seed"],split="forest_test",component=COMPONENT_ID),verify)
        depth=torch.from_numpy(np.fromfile(pd/"depth_input_2x128x160.f32","<f4").reshape(1,2,128,160).copy())
        state=torch.from_numpy(np.fromfile(sd/"state12.f32","<f4").reshape(1,12).copy())
        if not torch.isfinite(state).all() or not torch.isclose(state[0,9:12].norm(),torch.tensor(1.)):raise ValueError("FOREST_STATE")
        b=BoundBatch(depth,state,(g,),m,verify);b._depth=depth.numpy().tobytes();b._state=state.numpy().tobytes();b.validate()
        return b
