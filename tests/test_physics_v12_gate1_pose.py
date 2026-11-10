import json
from pathlib import Path
import numpy as np
from query_minco.training.physics_only_v1_2_gate1.pose_preflight import load_source,certify_position

ROOT=Path("research_outputs/physics_only_v1_2_gate1_forest_v1")

def test_historical_mirrored_collision_and_new_clearance():
    meta,counts,_=load_source(ROOT)
    old=certify_position(meta["position_world"],meta,counts)
    assert not old["passed"] and old["start_count"]==28 and old["mirrored_voxel"]==[20,12,18]
    cert=json.loads((ROOT/"runs/pose_preflight_001/POSE_CERTIFICATE.json").read_text())
    proof=certify_position(cert["position_world"],meta,counts)
    assert proof["passed"] and proof["clearance_lower_bound_m"]>=.15
    assert proof["native_closed_voxel_distance_m"]==proof["independently_clipped_voxel_distance_m"]
    assert abs(proof["clearance_lower_bound_m"]-cert["clearance_lower_bound_m"])<1e-12

def test_free_center_is_insufficient_at_ground_and_base_boundary():
    meta,counts,_=load_source(ROOT)
    p=json.loads((ROOT/"runs/pose_preflight_001/POSE_CERTIFICATE.json").read_text())["position_world"]
    p[2]=float(np.float32(meta["grid_origin"][2]))+.2
    assert not certify_position(p,meta,counts)["passed"]
    p[2]=1.8;p[0]=meta["grid_origin"][0]+.01
    assert not certify_position(p,meta,counts)["passed"]
