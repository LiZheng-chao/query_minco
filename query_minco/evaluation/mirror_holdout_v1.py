"""Pre-registered offline mirror world generation, qualification and strict loading."""
import hashlib,json,os
from pathlib import Path
import numpy as np
import torch
from query_minco.training.physics_only_v1_2_gate1.geometry import TypedBoundOccupancy,_SEAL
from query_minco.training.physics_only_v1_2_gate1.pose_preflight import certify_position
from query_minco.training.physics_only_v1_2_gate1.e1_dataset_qualification import reconstruct,geometry_identity,camera_qualification
from query_minco.training.physics_only_v1_2_gate1.hard_validator import validate_polynomial
from query_minco.evaluation.validation import numerical_reference,norm_max
from query_minco.evaluation.crossing_v1 import crossing_metric
from query_minco.audit.run_provenance_v1 import run_logged
COMPONENT_ID="registered_native_renderer_mirror_holdout_v1"
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    with Path(path).open("x") as f:json.dump(value,f,indent=2,allow_nan=False)
def cloud_for(pair,side):
    # Exact occupied-cell reflection, not an image operation.
    x=np.arange(141,dtype=np.float32);y=np.arange(121,dtype=np.float32)
    xx,yy=np.meshgrid(x,y,indexing="ij")
    floor=np.stack([-2+xx.ravel()*.1,-6+yy.ravel()*.1,np.zeros(xx.size)],1).astype("<f4")
    anchors=np.array([[x,y,6.] for x in [-2.,12.] for y in [-6.,6.]],dtype="<f4")
    gap=range(*pair["positive_gap_y_indices"])
    if side<0:gap={120-j for j in gap}
    indices=np.array([(i,j,k) for i in range(*pair["slab_x_indices"]) for j in range(121) if j not in gap for k in range(1,61)])
    barrier=(np.array([-2.,-6.,0.],dtype=np.float32)+(indices.astype(np.float32)+.5)*np.float32(.1)).astype("<f4")
    cloud=np.concatenate([floor,anchors,barrier])
    origin=cloud.min(0);shape=np.ceil((cloud.max(0)-origin+np.float32(.002))/np.float32(.1)).astype(int)
    idx=np.floor((cloud+np.float32(.001)-origin)/np.float32(.1)).astype(int)
    counts=np.bincount(np.ravel_multi_index(idx.T,shape),minlength=int(np.prod(shape))).reshape(tuple(shape)).astype("<i4")
    meta=dict(grid_origin=origin.tolist(),grid_shape=shape.tolist(),resolution_m=float(np.float32(.1)),occupy_threshold=0,point_count=len(cloud))
    reconstruct(cloud,counts,meta)
    return cloud,counts,meta
def geometry(counts,meta,pose):
    origin=np.asarray(meta["grid_origin"],np.float32).astype(float);h=float(np.float32(.1))
    roi=np.stack([origin+1e-5,origin+np.asarray(counts.shape)*h-1e-5])
    return TypedBoundOccupancy(seal=_SEAL,counts=counts,meta=meta,pose=pose,roi=roi,domain=[[0,-4,-2],[10,4,2]],
        identity=dict(component=COMPONENT_ID,split="holdout",not_a_training_label=True),revalidate=lambda:None)
def gates(pair,side,meta,pose):
    o=np.asarray(meta["grid_origin"],np.float32).astype(float);h=float(np.float32(.1))
    a,b=pair["slab_x_indices"];lo,hi=pair["positive_gap_y_indices"]
    if side<0:lo,hi=121-hi,121-lo
    planes=[o[0]+q*h-pose["position_world"][0] for q in [a,(a+b)/2,b]]
    gap=[o[1]+q*h-pose["position_world"][1] for q in [lo,hi]]
    return planes,gap
def witness(plan,pair,side):
    state=np.asarray(plan["state_B0"]);head=np.stack([np.zeros(3),state[:3],state[3:6]])
    tail=np.stack([state[6:9],np.zeros(3),np.zeros(3)])
    u=np.array([1/3,2/3]);inner=np.zeros((2,3));inner[:,0]=2.4*(10*u**3-15*u**4+6*u**5);inner[:,1]=side*pair["witness_inner_y"]
    dt=np.full(3,2/3)
    return numerical_reference(head,tail,inner,dt),dt
def qualification(plan_path,output,binary,baseline_registry):
    plan=json.loads(Path(plan_path).read_text());out=Path(output);out.mkdir()
    registry=json.loads(Path(baseline_registry).read_text());prior={x[k] for x in registry for k in ["spatial_occupancy_sha256","canonical_occupied_geometry_sha256"]}
    results=[];pair_rows=[]
    for pair in plan["pairs"]:
        native_counts=[];pair_pass=[]
        for side in [1,-1]:
            name=pair["pair_id"]+("_left" if side>0 else "_right");folder=out/name;folder.mkdir()
            cloud,counts,meta=cloud_for(pair,side);native_counts.append(counts)
            cloud.tofile(folder/"cloud_xyz.f32")
            pose=certify_position(plan["pose_world"],meta,counts)
            pose["schema"]="e1_r2_pose_v1"
            write(folder/"POSE_CERTIFICATE.json",pose);write(folder/"EXPECTED_GRID.json",meta)
            g=geometry(counts,meta,pose);planes,gap=gates(pair,side,meta,pose)
            coeff,dt=witness(plan,pair,side);hard=validate_polynomial(coeff,dt,g,np.array(plan["state_B0"])[6:9])
            crossing=crossing_metric(coeff,dt,planes,gap,side)
            opposite=coeff.copy();opposite[:,1]*=-1
            opposite_hard=validate_polynomial(opposite,dt,g,np.array(plan["state_B0"])[6:9])
            initial=dict(scene_id=name,side=side,pose_passed=pose["passed"],witness_hard=hard,witness_crossing=crossing,
                         wrong_sign_witness_hard=opposite_hard,witness_coefficients=coeff.tolist(),durations=dt.tolist(),
                         planes_B0=planes,gap_B0=gap,used_for_model_input_or_training=False)
            # Real native rendering only after predetermined witness checks; no model read anywhere here.
            if not (pose["passed"] and hard["passed"] and crossing["passed"] and not opposite_hard["passed"]):
                initial.update(status="NOT_QUALIFIED",reason="PRE_RENDER_POSE_WITNESS_OR_DIRECTION")
                write(folder/"QUALIFICATION_FAILURE.json",initial);results.append(initial);pair_pass.append(False);continue
            render=folder/"render";render.mkdir()
            env=dict(os.environ,CUDA_VISIBLE_DEVICES="0",LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:/opt/ros/noetic/lib")
            process=run_logged([str(binary),"depth",str(folder/"cloud_xyz.f32"),str(folder/"POSE_CERTIFICATE.json"),str(render)],folder/"renderer_process",cwd=Path.cwd(),env=env,runspec=plan_path,timeout_s=120)
            if process["exit_code"]!=0:
                initial.update(status="NOT_QUALIFIED",reason="NATIVE_RENDERER_EXIT",renderer_exit=process["exit_code"])
                write(folder/"QUALIFICATION_FAILURE.json",initial);results.append(initial);pair_pass.append(False);continue
            actual=json.loads((render/"renderer.json").read_text())
            got=np.fromfile(render/"occupancy_counts.i32","<i4").reshape(actual["grid_shape"])
            if not np.array_equal(got,counts):raise ValueError("NATIVE_COUNTS_MISMATCH")
            depth=np.fromfile(render/"depth_z.f32","<f4").reshape(90,160)
            x,camera=camera_qualification(depth,counts,actual)
            x.tofile(folder/"depth_input_2x128x160.f32");write(folder/"CAMERA_BINDING.json",camera)
            # Independent source occupancy identity and observability at horizontal center-row rays.
            identity=geometry_identity(cloud,counts,meta)
            independent=identity["spatial_occupancy_sha256"] not in prior and identity["canonical_occupied_geometry_sha256"] not in prior
            ratio=-(np.arange(160)-80)/80
            ys=ratio*planes[1]
            gap_pixels=(ys>gap[0]+.15)&(ys<gap[1]-.15)
            wall_pixels=(ys<gap[0]-.15)|(ys>gap[1]+.15)
            visible_gap=int((gap_pixels&(depth[45]>planes[-1]+.1)).sum())
            visible_wall=int((wall_pixels&(depth[45]>=planes[0]-.1)&(depth[45]<=planes[-1]+.1)).sum())
            visible=visible_gap>=3 and visible_wall>=3
            initial.update(status="QUALIFIED" if visible and independent else "NOT_QUALIFIED",
                independent_geometry=independent,identity=identity,native_counts_exact=True,
                observable_gap_pixels_center_row=visible_gap,observable_wall_pixels_center_row=visible_wall,
                depth_replay_max_error_m=camera["independent_replay_max_error_m"],renderer_exit=0)
            write(folder/"SCENE_QUALIFICATION.json",initial)
            if initial["status"]=="QUALIFIED":
                state=np.array(plan["state_B0"],dtype="<f4");state.tofile(folder/"state12.f32")
                bound=[folder/"cloud_xyz.f32",folder/"EXPECTED_GRID.json",folder/"POSE_CERTIFICATE.json",folder/"CAMERA_BINDING.json",folder/"depth_input_2x128x160.f32",folder/"state12.f32",folder/"SCENE_QUALIFICATION.json",render/"renderer.json",render/"occupancy_counts.i32",render/"depth_z.f32",Path(binary),Path(plan_path)]
                manifest=dict(component_id=COMPONENT_ID,scene_id=name,side=side,pair=pair,pose=pose,files_sha256={str(p.resolve()):sha(p) for p in bound},folder=str(folder.resolve()),preregistration_sha256=sha(plan_path))
                write(folder/"SCENE_MANIFEST.json",manifest)
                initial["manifest"]=str((folder/"SCENE_MANIFEST.json").resolve());initial["manifest_sha256"]=sha(folder/"SCENE_MANIFEST.json")
            results.append(initial);pair_pass.append(initial["status"]=="QUALIFIED")
        mirrored=np.array_equal(native_counts[0][:,::-1,:],native_counts[1])
        qualified=all(pair_pass) and mirrored
        pair_rows.append(dict(pair_id=pair["pair_id"],exact_counts_reflection=mirrored,qualified=qualified))
    final=dict(status="QUALIFIED" if all(x["qualified"] for x in pair_rows) and len(pair_rows)>=2 else "NOT_QUALIFIED",pairs=pair_rows,scenes=results,model_evaluated=False,new_network_updates=0)
    write(out/"QUALIFICATION_RESULT.json",final)
    return final
def load_scene(path,expected_sha):
    path=Path(path)
    if sha(path)!=expected_sha:raise ValueError("SCENE_MANIFEST_DRIFT")
    raw=path.read_bytes();m=json.loads(raw);folder=Path(m["folder"])
    def verify():
        if path.read_bytes()!=raw:raise ValueError("MANIFEST_MEMORY_DRIFT")
        for name,h in m["files_sha256"].items():
            if sha(name)!=h:raise ValueError("SCENE_SOURCE_DRIFT "+name)
    verify()
    q=json.loads((folder/"SCENE_QUALIFICATION.json").read_text())
    if q["status"]!="QUALIFIED":raise ValueError("UNQUALIFIED_SCENE")
    meta=json.loads((folder/"EXPECTED_GRID.json").read_text());counts=np.fromfile(folder/"render/occupancy_counts.i32","<i4").reshape(meta["grid_shape"])
    g=geometry(counts,meta,m["pose"]);g._revalidate=verify
    depth=torch.from_numpy(np.fromfile(folder/"depth_input_2x128x160.f32","<f4").reshape(1,2,128,160).copy())
    state=torch.from_numpy(np.fromfile(folder/"state12.f32","<f4").reshape(1,12).copy())
    return depth,state,g,m,q
