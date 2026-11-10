"""Continuation quota, immutable binding and existing proof revalidation negatives."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
from query_minco.training.physics_only_v1_2_gate1 import e1_r2c_dataset as d
from query_minco.training.physics_only_v1_2_gate1.e1_dataset_binding import load_e1_bound
from query_minco.training.physics_only_v1_2_gate1.e1_r2c_audit import make_audit_geometry
from query_minco.training.physics_only_v1_2_gate1.e1_dataset_qualification import scene_data
OLD=d.BASE/"research_outputs/physics_only_v1_2_e1_dataset_v2"

def p():return d.proposal()

@pytest.mark.parametrize("pose",["train_free_roi_1101_p0","train_free_roi_1101_p1","train_sparse_1102_p0"])
def test_completed_poses_must_not_rerender(pose):
    with pytest.raises(ValueError,match="RERENDER"):d.validate_depth_request(pose,p(),[])

def test_counts_and_new_layout_forbidden():
    with pytest.raises(ValueError,match="COUNTS_ONLY"):d.validate_depth_request("train_sparse_1102_p1",p(),[],"counts")
    with pytest.raises(ValueError,match="FROZEN_PLAN"):d.validate_depth_request("new_seed_pose",p(),[])

def test_attempt_ledger_persists_even_if_GPU_not_started(tmp_path):
    proposal=p();pose=proposal["pending_pose_plan"][0]["pose_id"];ledger=tmp_path/"GPU_CALLS.jsonl"
    row=d.reserve_depth(pose,proposal,ledger,["not-executed-test-only"])
    assert row["lifetime_CUDA_index"]==11 and row["lifetime_depth_index"]==4
    before=ledger.read_bytes()
    with pytest.raises(ValueError,match="DUPLICATE"):d.reserve_depth(pose,proposal,ledger,["no retry"])
    assert ledger.read_bytes()==before

def test_lifetime_limit_cannot_reset():
    proposal=p();attempts=[dict(pose_id=x["pose_id"]) for x in proposal["pending_pose_plan"]]
    with pytest.raises(ValueError):d.validate_depth_request(proposal["pending_pose_plan"][0]["pose_id"],proposal,attempts)
    proposal=copy.deepcopy(proposal);proposal["historical_ledger"]["depth"]=14
    with pytest.raises(ValueError,match="LIFETIME_DEPTH"):d.validate_depth_request("train_sparse_1102_p1",proposal,[])

def test_reuse_depth_to_bind_missing_far_without_GPU(tmp_path):
    proposal=p();scene=OLD/"data/train/train_sparse_1102";pose=scene/"p0"
    cfg=d.read(OLD/"_preparation/qualification_001/CONFIG.json");case=next(x for x in cfg["state_cases"] if x["case"]=="far")
    path=d.bind_task(scene,pose,case,tmp_path/"far","train_sparse_1102_p0_far",proposal)
    b=load_e1_bound(path,d.sha(path))
    assert b.state.shape==(1,12) and b.depth.shape==(1,2,128,160)
    np.testing.assert_array_equal(b.state[0,6:9].numpy(),[6,0,0])
    assert b.manifest["pose_dir"]==str(pose) and not b.manifest["teacher_loaded"]

def test_old_near_feasible_evidence_reverified_without_new_search():
    scene=OLD/"data/train/train_sparse_1102";mp=scene/"p0/near/SAMPLE_MANIFEST.json"
    b=load_e1_bound(mp,d.sha(mp));cloud,c,m=scene_data(scene)
    g=make_audit_geometry(scene,c,m,d.read(scene/"p0/POSE_CERTIFICATE.json"))
    ep=d.AUDIT/"runs/audit_001/train_sparse_1102_p0_near_REACHABILITY.json"
    proof=d.verify_reachability(d.read(ep),b,g)
    assert proof["reachability"]=="FEASIBLE_CERTIFIED"
    assert len(proof["checked_Hard_certificates"])==1 and not proof["new_candidate_search"]

@pytest.mark.parametrize("fault",["state","frame","forged_Hard"])
def test_evidence_binding_drift_is_global_error(fault):
    scene=OLD/"data/train/train_sparse_1102";mp=scene/"p0/near/SAMPLE_MANIFEST.json"
    b=load_e1_bound(mp,d.sha(mp));cloud,c,m=scene_data(scene)
    g=make_audit_geometry(scene,c,m,d.read(scene/"p0/POSE_CERTIFICATE.json"))
    evidence=copy.deepcopy(d.read(d.AUDIT/"runs/audit_001/train_sparse_1102_p0_near_REACHABILITY.json"))
    if fault=="state":evidence["task_state"][6]+=.1
    if fault=="frame":g.origin[0]+=.01
    if fault=="forged_Hard":evidence["attempts"][18]["hard"]["passed"]=False
    with pytest.raises(ValueError):d.verify_reachability(evidence,b,g)

def test_immutable_hash_validation(tmp_path):
    q=tmp_path/"input";q.write_bytes(b"source");bound={str(q):d.sha(q)}
    d.verify_hashes(bound);q.write_bytes(b"drift")
    with pytest.raises(ValueError,match="SHA_DRIFT"):d.verify_hashes(bound)
