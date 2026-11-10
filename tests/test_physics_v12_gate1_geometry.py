import copy,hashlib,json
from pathlib import Path
import numpy as np
import pytest
import torch
from query_minco.training.physics_only_v1_2_gate1.binding import load_bound,sha
from query_minco.training.physics_only_v1_2_gate1.geometry import world_points,TypedBoundOccupancy
from query_minco.training.physics_only_v1_2_gate1.qualification import qualify_export,digest
from query_minco.training.physics_only_v1_2_gate1.hard_validator import transform_coefficients,piece_certificate

ROOT=Path("research_outputs/physics_only_v1_2_gate1_forest_v1")
MANIFEST=ROOT/"runs/bind_posecert_001/BOUND_MANIFEST.json"

@pytest.fixture(scope="module")
def bound():
    return load_bound(MANIFEST,sha(MANIFEST))

def test_actual_camera_state_and_qualified_geometry(bound):
    bound.validate()
    assert type(bound.geometry[0]) is TypedBoundOccupancy
    assert bound.depth.shape==(1,2,128,160) and bound.state.shape==(1,12)
    assert not bound.depth[:,:,0:19].any() and not bound.depth[:,:,109:].any()
    mask=bound.depth[:,1]
    assert set(mask.unique().tolist())=={0.,1.}
    assert torch.count_nonzero(bound.depth[:,0][mask==0])==0
    assert torch.isfinite(bound.state).all()
    torch.testing.assert_close(bound.state[0],torch.tensor([.8,0,0,.1,0,0,6,0,0,0,0,-1]))
    q=json.loads((ROOT/"runs/bind_posecert_001/EXPORT_QUALIFICATION.json").read_text())
    assert q["camera"]["processed_K"]==[[80,0,80],[0,80,64],[0,0,1]]
    assert q["independent_depth_replay_mismatch_pixels"]==0
    assert not bound.manifest["external_certificate_authority"]

def test_rotation_translation_not_false_B0_aabb():
    R=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
    o=np.array([2.,3.,4.]);p=np.array([[1.,2.,3.]])
    np.testing.assert_allclose(world_points(p,o,R),[[0,4,7]])
    c=np.zeros((3,3,6));c[:,0,1]=1
    wc=transform_coefficients(c,o,R)
    np.testing.assert_allclose(wc[:,:,0],np.tile(o,(3,1)))
    np.testing.assert_allclose(wc[:,:,1],np.tile([0,1,0],(3,1)))

def test_exact_distance_matches_independent_all_boxes(bound):
    g=bound.geometry[0];p=torch.tensor([[[0.,0.,0.],[.1,.02,.03],[.2,-.02,.1]]],dtype=torch.float64,requires_grad=True)
    d=g.sdf(p)
    w=p.detach().numpy()@g.rotation.T+g.origin
    q=np.abs(w[...,None,:]-g.boxes.mean(1))-(g.boxes[:,1]-g.boxes[:,0])/2
    boxes=np.linalg.norm(q.clip(min=0),axis=-1)+q.max(-1).clip(max=0)
    roi=np.minimum(w-g.roi[0],g.roi[1]-w).min(-1)
    expected=np.minimum(np.minimum(boxes.min(-1),w[...,2]-g.floor),roi)
    np.testing.assert_allclose(d.detach().numpy(),expected,atol=1e-10)
    assert torch.autograd.gradcheck(lambda x:g.sdf(x),(p,),atol=1e-5)

def test_unknown_and_top_free_are_rejected(bound):
    g=bound.geometry[0]
    for w in (g.roi[1]+[1,0,0],g.origin+[0,0,100]):
        p=torch.tensor(w-g.origin,dtype=torch.float64)
        with pytest.raises(ValueError,match="UNKNOWN"):g.sdf(p)
    c=np.zeros((3,6));c[:,0]=g.roi[1]+[1,0,0]
    assert not piece_certificate(c,2/3,g)["passed"]

@pytest.mark.parametrize("kind",["counts","depth","manifest"])
def test_byte_drift_fails_closed(bound,monkeypatch,kind):
    target=MANIFEST.resolve() if kind=="manifest" else Path(bound.manifest["raw_dir"])/("occupancy_counts.i32" if kind=="counts" else "depth_z.f32")
    original=Path.read_bytes
    def changed(p):
        b=original(p)
        return bytes([b[0]^1])+b[1:] if p.resolve()==target.resolve() else b
    monkeypatch.setattr(Path,"read_bytes",changed)
    with pytest.raises(ValueError,match="DRIFT"):bound.validate()

@pytest.mark.parametrize("fault",["frame","split","pose","state","units"])
def test_manifest_semantic_mismatch_rejected(tmp_path,fault):
    m=json.loads(MANIFEST.read_text())
    if fault=="frame":m["frame"]="world"
    if fault=="split":m["split"]="dev";m["split_registry"][m["geometry_scene_id"]]="dev"
    if fault=="pose":m["pose"]["position_world"][0]+=.1
    if fault=="state":m["state"]["v0_world_mps"][0]=100
    if fault=="units":m["units"]["position"]="cm"
    m.pop("contract_sha256");m["contract_sha256"]=digest(m)
    p=tmp_path/"bad.json";p.write_text(json.dumps(m))
    with pytest.raises(ValueError):load_bound(p,sha(p))

def test_wrong_camera_K_rejected_even_with_fresh_byte_hash(bound,monkeypatch):
    m=bound.manifest;d=Path(m["raw_dir"]);hashes=json.loads(Path(m["export_hashes_path"]).read_text())
    target=d/"renderer.json";meta=json.loads(target.read_text());meta["camera"]["fx"]=81
    modified=json.dumps(meta).encode();hashes["renderer.json"]=hashlib.sha256(modified).hexdigest()
    original=Path.read_bytes
    monkeypatch.setattr(Path,"read_bytes",lambda p:modified if p==target else original(p))
    with pytest.raises(ValueError,match="CAMERA_K"):
        qualify_export(d,hashes,json.loads(Path(m["binary_binding_path"]).read_text()),m["pose"])

def test_unsealed_constructor_and_corrupt_memory_rejected(bound):
    with pytest.raises(ValueError):TypedBoundOccupancy(seal=None,counts=None,meta=None,pose=None,roi=None,domain=None,identity=None,revalidate=None)
    before=bound.state.clone()
    try:
        bound.state[0,0]+=1
        with pytest.raises(ValueError,match="MEMORY_DRIFT"):bound.validate()
    finally:bound.state.copy_(before)
