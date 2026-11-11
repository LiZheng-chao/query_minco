"""New native-rendered avoidance families; shared MINCO/Hard/camera are imported, not copied."""
from pathlib import Path
import json,os
import numpy as np
import torch
from query_minco.evaluation.mirror_holdout_v1 import write,sha
from query_minco.evaluation.crossing_v1 import crossing_metric
from query_minco.audit.run_provenance_v1 import run_logged
from query_minco.training.physics_only_v1_2_gate1.geometry import TypedBoundOccupancy,_SEAL
from query_minco.training.physics_only_v1_2_gate1.pose_preflight import certify_position
from query_minco.training.physics_only_v1_2_gate1.e1_dataset_qualification import reconstruct,geometry_identity,camera_qualification
from query_minco.training.physics_only_v1_2_gate1.hard_validator import validate_polynomial
from query_minco.training.physics_only_v1_2_gate1.binding import BoundBatch
from query_minco.evaluation.validation import numerical_reference
COMPONENT_ID="native_three_family_avoidance_scene_qualification_v1"
def obstacle_mask(spec,side):
    mask=np.zeros((141,121,61),dtype=bool);name=spec["family_id"]
    def wall(x,gap):
        mask[x[0]:x[1],:,1:]=True;mask[x[0]:x[1],gap[0]:gap[1],1:]=False
    if name=="staggered_double_aperture":
        for s in spec["transverse_walls"]:wall(s["x"],s["gap_y"])
        sections=spec["transverse_walls"]
    elif name=="blind_branch_splitter":
        s=spec["spine"];mask[s["x"][0]:s["x"][1],s["y"][0]:s["y"][1],1:]=True
        s=spec["exit_baffle"];wall(s["x"],s["gap_y"]);sections=[s]
    elif name=="column_row_narrow_decoys":
        x=spec["column_x"]
        for j in range(121):
            if j%spec["period_y_cells"]<spec["filled_per_period"]:mask[x[0]:x[1],j,1:]=True
        gap=spec["wide_gap_y"];mask[x[0]:x[1],gap[0]:gap[1],1:]=False
        sections=[dict(x=x,gap_y=gap)]
    else:raise ValueError("UNKNOWN_GEOMETRY_FAMILY")
    if side<0:mask=mask[:,::-1,:].copy()
    return mask,sections
def cloud_and_counts(mask):
    i,j=np.meshgrid(np.arange(141,dtype=np.float32),np.arange(121,dtype=np.float32),indexing="ij")
    floor=np.stack([-2+i.ravel()*.1,-6+j.ravel()*.1,np.zeros(i.size)],1).astype("<f4")
    anchors=np.array([[x,y,6] for x in [-2,12] for y in [-6,6]],dtype="<f4")
    cells=np.argwhere(mask).astype(np.float32)
    obstacles=(np.array([-2,-6,0],dtype=np.float32)+(cells+.5)*np.float32(.1)).astype("<f4")
    cloud=np.concatenate([floor,anchors,obstacles]);o=cloud.min(0);h=np.float32(.1)
    shape=np.ceil((cloud.max(0)-o+np.float32(.002))/h).astype(int)
    ids=np.floor((cloud+np.float32(.001)-o)/h).astype(int)
    counts=np.bincount(np.ravel_multi_index(ids.T,shape),minlength=int(np.prod(shape))).reshape(tuple(shape)).astype("<i4")
    meta=dict(grid_origin=o.tolist(),grid_shape=shape.tolist(),resolution_m=float(h),occupy_threshold=0,point_count=len(cloud))
    reconstruct(cloud,counts,meta)
    return cloud,counts,meta
def make_geometry(counts,meta,pose,identity,verify):
    o=np.asarray(meta["grid_origin"],np.float32).astype(float);h=float(np.float32(.1))
    roi=np.stack([o+1e-5,o+np.asarray(counts.shape)*h-1e-5])
    return TypedBoundOccupancy(seal=_SEAL,counts=counts,meta=meta,pose=pose,roi=roi,domain=[[0,-4,-2],[10,4,2]],identity=identity,revalidate=verify)
def section_contract(section,side,meta,pose):
    o=np.asarray(meta["grid_origin"],np.float32).astype(float);h=float(np.float32(.1));a,b=section["x"];lo,hi=section["gap_y"]
    if side<0:lo,hi=121-hi,121-lo
    p=np.asarray(pose["position_world"])
    return [o[0]+x*h-p[0] for x in [a,(a+b)/2,b]],[o[1]+y*h-p[1] for y in [lo,hi]]
def fixed_witness(plan,spec,side):
    state=np.asarray(plan["data"]["state_B0"]);u=np.array([1/3,2/3])
    inner=np.zeros((2,3));inner[:,0]=2.4*(10*u**3-15*u**4+6*u**5);inner[:,1]=side*spec["witness_inner_y"]
    head=np.stack([np.zeros(3),state[:3],state[3:6]]);tail=np.stack([state[6:9],np.zeros(3),np.zeros(3)])
    d=np.full(3,2/3);return numerical_reference(head,tail,inner,d),d
def qualify_dataset(plan_path,output,binary,old_identities):
    plan=json.loads(Path(plan_path).read_text());out=Path(output);out.mkdir()
    rows=[];families=[];all_ids=set(old_identities)
    for spec in plan["data"]["families"]:
        sides=[];counts_pair=[]
        for side in [1,-1]:
            scene=spec["family_id"]+("_left" if side>0 else "_right");folder=out/scene;folder.mkdir()
            mask,sections=obstacle_mask(spec,side);cloud,counts,meta=cloud_and_counts(mask);counts_pair.append(counts)
            cloud.tofile(folder/"cloud_xyz.f32");write(folder/"EXPECTED_GRID.json",meta);write(folder/"FAMILY_SPEC.json",dict(spec=spec,side=side))
            pose=certify_position(plan["data"]["pose_world"],meta,counts);pose["schema"]="e1_r2_pose_v1";write(folder/"POSE_CERTIFICATE.json",pose)
            identity=geometry_identity(cloud,counts,meta)
            unique=all(identity[k] not in all_ids for k in identity)
            all_ids.update(identity.values())
            g=make_geometry(counts,meta,pose,dict(scene_id=scene,split=spec["split"],family_id=spec["family_id"],component=COMPONENT_ID),lambda:None)
            c,d=fixed_witness(plan,spec,side);hard=validate_polynomial(c,d,g,np.array(plan["data"]["state_B0"])[6:9])
            contracts=[section_contract(s,side,meta,pose) for s in sections]
            crossings=[crossing_metric(c,d,p,gap,side) for p,gap in contracts]
            wrong=c.copy();wrong[:,1]*=-1;wrong_hard=validate_polynomial(wrong,d,g,np.array(plan["data"]["state_B0"])[6:9])
            rec=dict(scene_id=scene,family_id=spec["family_id"],split=spec["split"],side=side,identity=identity,pose_passed=pose["passed"],
                witness_hard=hard,witness_crossings=crossings,wrong_side_hard=wrong_hard,witness_coefficients=c.tolist(),durations=d.tolist(),
                sections=[dict(planes_B0=p,gap_B0=gap) for p,gap in contracts],independent_identity=unique,witness_used_as_training_label=False)
            if not(pose["passed"] and unique and hard["passed"] and all(x["passed"] for x in crossings) and not wrong_hard["passed"]):
                rec.update(status="NOT_QUALIFIED",reason="PRE_RENDER_WITNESS_DIRECTION_OR_IDENTITY");write(folder/"QUALIFICATION_FAILURE.json",rec);rows.append(rec);sides.append(False);continue
            render=folder/"render";render.mkdir()
            env=dict(os.environ,CUDA_VISIBLE_DEVICES="0",LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:/opt/ros/noetic/lib")
            process=run_logged([str(binary),"depth",str(folder/"cloud_xyz.f32"),str(folder/"POSE_CERTIFICATE.json"),str(render)],folder/"renderer_process",cwd=Path.cwd(),env=env,runspec=plan_path,timeout_s=120)
            if process["exit_code"]!=0:
                rec.update(status="NOT_QUALIFIED",reason="NATIVE_RENDER_EXIT",renderer_exit=process["exit_code"]);write(folder/"QUALIFICATION_FAILURE.json",rec);rows.append(rec);sides.append(False);continue
            actual=json.loads((render/"renderer.json").read_text());got=np.fromfile(render/"occupancy_counts.i32","<i4").reshape(actual["grid_shape"])
            if not np.array_equal(got,counts):raise ValueError("NATIVE_COUNTS")
            depth=np.fromfile(render/"depth_z.f32","<f4").reshape(90,160);x,camera=camera_qualification(depth,counts,actual)
            x.tofile(folder/"depth_input_2x128x160.f32");write(folder/"CAMERA_BINDING.json",camera)
            np.asarray(plan["data"]["state_B0"],dtype="<f4").tofile(folder/"state12.f32")
            observed=[]
            for planes,gap in contracts:
                ys=-(np.arange(160)-80)/80*planes[1]
                aperture=(ys>gap[0]+.15)&(ys<gap[1]-.15)
                barrier=(ys<gap[0]-.15)|(ys>gap[1]+.15)
                observed.append(dict(gap_pixels=int((aperture&(depth[45]>planes[-1]+.1)).sum()),
                    wall_pixels=int((barrier&(depth[45]>=planes[0]-.1)&(depth[45]<=planes[-1]+.1)).sum())))
            visible=all(v["gap_pixels"]>=3 and v["wall_pixels"]>=3 for v in observed)
            rec.update(status="QUALIFIED" if visible else "NOT_QUALIFIED",observability=observed,native_counts_exact=True,
                       depth_replay_max_error_m=camera["independent_replay_max_error_m"],renderer_exit=0)
            write(folder/"SCENE_QUALIFICATION.json",rec);rows.append(rec);sides.append(visible)
        mirrored=np.array_equal(counts_pair[0][:,::-1,:],counts_pair[1])
        families.append(dict(family_id=spec["family_id"],split=spec["split"],pair_qualified=all(sides) and mirrored,exact_occupancy_reflection=mirrored))
    registry=[dict(scene_id=x["scene_id"],family_id=x["family_id"],split=x["split"],identity=x["identity"],status=x["status"]) for x in rows]
    write(out/"FAMILY_SPLIT_REGISTRY.json",registry)
    index=[]
    for row in rows:
        if row["status"]!="QUALIFIED":continue
        folder=out/row["scene_id"]
        names=["cloud_xyz.f32","EXPECTED_GRID.json","FAMILY_SPEC.json","POSE_CERTIFICATE.json","CAMERA_BINDING.json","depth_input_2x128x160.f32","state12.f32","SCENE_QUALIFICATION.json","render/renderer.json","render/occupancy_counts.i32","render/depth_z.f32"]
        paths=[folder/n for n in names]+[out/"FAMILY_SPLIT_REGISTRY.json",Path(binary),Path(plan_path),Path(__file__)]
        m=dict(schema="qualified_avoidance_family_scene_v1",component_id=COMPONENT_ID,scene_id=row["scene_id"],family_id=row["family_id"],split=row["split"],side=row["side"],
               identity=row["identity"],folder=str(folder.resolve()),registry_path=str((out/"FAMILY_SPLIT_REGISTRY.json").resolve()),files_sha256={str(p.resolve()):sha(p) for p in paths},
               qualification_only=True,teacher_loaded=False)
        write(folder/"SCENE_MANIFEST.json",m);index.append(dict(scene_id=row["scene_id"],family_id=row["family_id"],split=row["split"],manifest=str((folder/"SCENE_MANIFEST.json").resolve()),manifest_sha256=sha(folder/"SCENE_MANIFEST.json")))
    train={x["family_id"] for x in families if x["split"]=="train"};dev={x["family_id"] for x in families if x["split"]=="dev"}
    qualified=all(x["pair_qualified"] for x in families) and len(train)>=2 and len(dev)>=1 and not train&dev
    write(out/"DATASET_INDEX.json",index)
    result=dict(status="QUALIFIED" if qualified else "NOT_QUALIFIED",families=families,scenes=rows,train_scenes=sum(x["split"]=="train" for x in index),
        dev_scenes=sum(x["split"]=="dev" for x in index),family_leakage=bool(train&dev),network_inference_on_new_data=False,network_updates=0)
    write(out/"DATASET_QUALIFICATION.json",result);return result
def load_family_batch(path,expected_sha,required_split):
    path=Path(path)
    if sha(path)!=expected_sha:raise ValueError("FAMILY_MANIFEST_SHA")
    raw=path.read_bytes();m=json.loads(raw);folder=Path(m["folder"])
    if m["schema"]!="qualified_avoidance_family_scene_v1" or m["component_id"]!=COMPONENT_ID or m["split"]!=required_split or m["teacher_loaded"] is not False:
        raise ValueError("FAMILY_SPLIT_SCHEMA")
    def verify():
        if path.read_bytes()!=raw:raise ValueError("MANIFEST_DRIFT")
        for name,h in m["files_sha256"].items():
            if sha(name)!=h:raise ValueError("FAMILY_SOURCE_DRIFT "+name)
    verify();registry=json.loads(Path(m["registry_path"]).read_text())
    same=[x for x in registry if x["scene_id"]==m["scene_id"] and x["identity"]==m["identity"] and x["family_id"]==m["family_id"] and x["split"]==required_split and x["status"]=="QUALIFIED"]
    if len(same)!=1:raise ValueError("REGISTRY_BINDING")
    if {x["family_id"] for x in registry if x["split"]=="train"}&{x["family_id"] for x in registry if x["split"]=="dev"}:raise ValueError("FAMILY_LEAKAGE")
    q=json.loads((folder/"SCENE_QUALIFICATION.json").read_text())
    if q["status"]!="QUALIFIED":raise ValueError("UNQUALIFIED")
    meta=json.loads((folder/"EXPECTED_GRID.json").read_text());pose=json.loads((folder/"POSE_CERTIFICATE.json").read_text())
    counts=np.fromfile(folder/"render/occupancy_counts.i32","<i4").reshape(meta["grid_shape"])
    g=make_geometry(counts,meta,pose,dict(scene_id=m["scene_id"],family_id=m["family_id"],split=required_split,component=COMPONENT_ID),verify)
    depth=torch.from_numpy(np.fromfile(folder/"depth_input_2x128x160.f32","<f4").reshape(1,2,128,160).copy())
    state=torch.from_numpy(np.fromfile(folder/"state12.f32","<f4").reshape(1,12).copy())
    b=BoundBatch(depth,state,(g,),m,verify);b._depth=depth.numpy().tobytes();b._state=state.numpy().tobytes();b.validate()
    return b
