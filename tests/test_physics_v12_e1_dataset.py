"""E1-R2 negative contracts and independent witnesses; no GPU or optimizer."""
import ast
import copy
from pathlib import Path
import numpy as np
import pytest
from query_minco.training.physics_only_v1_2_gate1.e1_dataset_qualification import (
 reconstruct,geometry_identity,check_split,camera_qualification,choose_poses,polynomial_witness,write,sha)
from query_minco.training.physics_only_v1_2_gate1.camera import camera_input,replay_depth
from query_minco.training.physics_only_v1_2_gate1.pose_preflight import certify_position

def fixture():
 cloud=np.array([[-2,-6,0],[12,6,6],[-2,6,6],[12,-6,6],[5.05,3.05,1.85]],dtype=np.float32)
 origin=cloud.min(0);shape=np.ceil((cloud.max(0)-origin+np.float32(.002))/np.float32(.1)).astype(int)
 idx=np.floor((cloud+np.float32(.001)-origin)/np.float32(.1)).astype(int)
 counts=np.zeros(shape,np.int32)
 for a in idx:counts[tuple(a)]+=1
 m=dict(grid_origin=origin.tolist(),grid_shape=shape.tolist(),point_count=len(cloud),resolution_m=float(np.float32(.1)),occupy_threshold=0,
        camera=dict(width=160,height=90,fx=80,fy=80,cx=80,cy=45,max_depth_m=20,normalized=False),
        body_quaternion_wxyz=[1,0,0,0],camera_pitch_deg=0,position_world=[0,0,1.8])
 return cloud,counts,m

def test_counts_independent_reconstruction():
 cloud,c,m=fixture();assert reconstruct(cloud,c,m)
 c=c.copy();c[1,1,1]+=1
 with pytest.raises(ValueError,match="COUNTS"):reconstruct(cloud,c,m)

@pytest.mark.parametrize("kind",["origin","shape","threshold"])
def test_grid_drift_rejected(kind):
 cloud,c,m=fixture();m=copy.deepcopy(m)
 if kind=="origin":m["grid_origin"][0]+=.1
 if kind=="shape":m["grid_shape"][0]+=1
 if kind=="threshold":m["occupy_threshold"]=1
 with pytest.raises(ValueError):reconstruct(cloud,c,m)

def test_scene_repetition_and_translation_not_split_independence():
 cloud,c,m=fixture();a=geometry_identity(cloud,c,m)
 moved=cloud+np.array([10,20,0],np.float32);meta=copy.deepcopy(m);meta["grid_origin"]=(np.array(m["grid_origin"])+[10,20,0]).tolist()
 b=geometry_identity(moved,c,meta)
 assert a["canonical_occupied_geometry_sha256"]==b["canonical_occupied_geometry_sha256"]
 with pytest.raises(ValueError,match="DUPLICATE_SCENE"):check_split([dict(a,scene_id="a"),dict(b,scene_id="b")])
 doubled=np.concatenate([cloud,cloud]);c2=c*2;m2=dict(m,point_count=len(doubled))
 assert geometry_identity(doubled,c2,m2)["canonical_occupied_geometry_sha256"]==a["canonical_occupied_geometry_sha256"]

def test_source_pose_native_bound_and_occupied_negative():
 cloud,c,m=fixture();poses=choose_poses(m,c)
 assert all(p["passed"] and p["clearance_lower_bound_m"]>=.150001 for p in poses)
 assert poses[0]["position_world"]!=poses[1]["position_world"]
 assert not certify_position([-3,0,1.8],m,c)["passed"]
 assert not certify_position([5.05,3.05,1.85],m,c)["passed"]
 assert not certify_position([0,0,6.5],m,c)["passed"]

def test_camera_replay_padding_no_hit_and_bad_intrinsics():
 cloud,c,m=fixture();depth=replay_depth(c,m);x,b=camera_qualification(depth,c,m)
 assert x.shape==(2,128,160);assert not x[:,:19].any() and not x[:,109:].any()
 assert b["processed_K"][1][2]==64
 d=np.full((90,160),20,np.float32);v,_=camera_input(d,m);assert not v.any()
 bad=copy.deepcopy(m);bad["camera"]["fx"]=81
 with pytest.raises(ValueError,match="CAMERA_K"):camera_qualification(depth,c,bad)
 depth[0,0]+=.01
 with pytest.raises(ValueError,match="DEPTH_REPLAY"):camera_qualification(depth,c,m)

def test_exclusive_write_no_overwrite(tmp_path):
 p=tmp_path/"receipt.json";write(p,{"a":1});old=sha(p)
 with pytest.raises(FileExistsError):write(p,{"a":2})
 assert sha(p)==old

def test_new_factory_rejects_wrong_manifest_sha(tmp_path):
 from query_minco.training.physics_only_v1_2_gate1.e1_dataset_binding import load_e1_bound
 p=tmp_path/"manifest.json";write(p,{"schema":"synthetic"})
 with pytest.raises(ValueError,match="MANIFEST_SHA"):load_e1_bound(p,"0"*64)

def test_no_optimizer_or_policy_in_new_source():
 root=Path(__file__).resolve().parents[1]
 for name in ("e1_dataset_binding.py","e1_dataset_qualification.py"):
  text=(root/"query_minco/training/physics_only_v1_2_gate1"/name).read_text()
  tree=ast.parse(text)
  for n in ast.walk(tree):
   if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute):
    assert n.func.attr not in ("step","backward")
  assert "SyntheticGeometry" not in text
