import copy
import hashlib
import json
import numpy as np
import pytest
import torch

from query_minco.training.physics_only_v1.dataset import (
    PhysicsOnlyDataset, CAMERA, FRAME, optical_points_to_b0)


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False))
    return binding(path)


def binding(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def fixture_files(root, *, empty=False, qualification="SYNTHETIC_TEST_ONLY"):
    state = np.zeros(12, np.float32)
    state[6] = 2.4
    state[11] = -1
    depth = np.zeros((2, 128, 160), np.float32)
    depth[0] = .1
    depth[1] = 1
    sample = root / "sample.npz"
    np.savez(sample, sample_id=np.array("s0"), depth_input=depth, state=state)
    geom = dict(schema="physics_native_aabb_v1", sample_id="s0", split="train",
                representation="native_aabb", frame=FRAME, units="m_s",
                domain_semantics="center_bounds_not_occupied_solids",
                boxes=[] if empty else [[[.9, -.3, -.3], [1.2, .3, .3]]],
                domain=[[0, -4, -2], [10, 4, 2]], empty_certified=empty)
    gb = write_json(root / "geometry.json", geom)
    cert = dict(schema="physics_binding_certificate_v1", sample_id="s0", split="train",
                sample=binding(sample), geometry=gb, qualification=qualification,
                frame=FRAME, units="m_s", camera=copy.deepcopy(CAMERA), state=state.tolist(),
                empty_certified=empty)
    cb = write_json(root / "certificate.json", cert)
    manifest = dict(schema="physics_only_manifest_v1", qualification=qualification,
                    selection_provenance="independent", rows=[
                        dict(sample_id="s0", split="train", sample=binding(sample), geometry=gb, certificate=cb)])
    mb = write_json(root / "manifest.json", manifest)
    return mb, manifest, geom, cert


def dataset(root, binding_):
    return PhysicsOnlyDataset(binding_, roots=[root], qualification="SYNTHETIC_TEST_ONLY")


def repin(root, manifest, geom, cert):
    gb = write_json(root / "geometry.json", geom)
    cert["geometry"] = gb
    manifest["rows"][0]["geometry"] = gb
    manifest["rows"][0]["certificate"] = write_json(root / "certificate.json", cert)
    return write_json(root / "manifest.json", manifest)


@pytest.mark.parametrize("empty", [False, True])
def test_independent_teacher_absent_and_revalidation(tmp_path, empty):
    mb, _, _, _ = fixture_files(tmp_path, empty=empty)
    ds = dataset(tmp_path, mb)
    assert ds.sample_ids == ("s0",)
    x = ds.load(["s0"])[0]
    assert x.state.shape == (12,) and x.depth_input.shape == (2, 128, 160)
    assert x.empty_certified == empty and not hasattr(x, "teacher_coeffs")
    # Mutating returned tensors cannot bypass the next disk authentication.
    x.state.fill_(99)
    assert ds.load(["s0"])[0].state[0] == 0
    with (tmp_path / "sample.npz").open("ab") as f:
        f.write(b"tampered")
    with pytest.raises(ValueError, match="SHA256"):
        ds.load(["s0"])


@pytest.mark.parametrize("target", ["manifest.json", "geometry.json", "certificate.json"])
def test_rehash_every_artifact(tmp_path, target):
    mb, _, _, _ = fixture_files(tmp_path)
    ds = dataset(tmp_path, mb)
    with (tmp_path / target).open("ab") as f:
        f.write(b" ")
    with pytest.raises(ValueError, match="SHA256"):
        ds.load(["s0"])


@pytest.mark.parametrize("change", ["split", "id", "frame", "units", "empty", "forest", "camera", "state", "cert_id", "cert_split", "extents", "domain"])
def test_even_repinned_inconsistent_semantics_rejected(tmp_path, change):
    _, m, g, c = fixture_files(tmp_path)
    if change == "split": m["rows"][0]["split"] = "test"
    if change == "id": g["sample_id"] = "other"
    if change == "frame": g["frame"] = "world"
    if change == "units": g["units"] = "cm"
    if change == "empty": g["empty_certified"] = c["empty_certified"] = True
    if change == "forest": g["representation"] = "yopo_forest"
    if change == "camera": c["camera"]["optical_to_B0_rotation"] = np.eye(3).tolist()
    if change == "state": c["state"][0] = 1
    if change == "cert_id": c["sample_id"] = "other"
    if change == "cert_split": c["split"] = "validation"
    if change == "extents": g["boxes"][0][1] = g["boxes"][0][0]
    if change == "domain": g["domain"][1] = g["domain"][0]
    mb = repin(tmp_path, m, g, c)
    with pytest.raises(ValueError):
        dataset(tmp_path, mb)


def test_pin_root_empty_and_qualification(tmp_path):
    mb, m, g, c = fixture_files(tmp_path, empty=True)
    with pytest.raises(ValueError, match="QUALIFICATION"):
        PhysicsOnlyDataset(mb, roots=[tmp_path])
    bad = dict(mb, sha256="0"*64)
    with pytest.raises(ValueError, match="SHA256"):
        dataset(tmp_path, bad)
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(ValueError, match="PATH_ESCAPE"):
        PhysicsOnlyDataset(mb, roots=[other])
    g["empty_certified"] = c["empty_certified"] = False
    with pytest.raises(ValueError, match="EMPTY"):
        dataset(tmp_path, repin(tmp_path, m, g, c))


def test_train_metadata_supported_without_teacher_status(tmp_path):
    mb, _, _, _ = fixture_files(tmp_path, qualification="TRAIN")
    # Synthetic content carrying TRAIN metadata checks schema mechanics only.
    ds = PhysicsOnlyDataset(mb, roots=[tmp_path])
    assert ds.load(["s0"])[0].qualification == "TRAIN"
    with pytest.raises(ValueError, match="UNKNOWN"):
        ds.load(["missing"])


@pytest.mark.parametrize("fault", ["teacher", "mask", "dtype", "nan", "gravity", "sample_id"])
def test_bad_sample_arrays(tmp_path, fault):
    _, m, g, c = fixture_files(tmp_path)
    p = tmp_path / "sample.npz"
    with np.load(p, allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    if fault == "teacher": a["teacher_coeffs"] = np.zeros(1)
    if fault == "mask": a["depth_input"][1, 0, 0] = .5
    if fault == "dtype": a["state"] = a["state"].astype(np.float64)
    if fault == "nan": a["state"][0] = np.nan
    if fault == "gravity": a["state"][11] = 0; c["state"][11] = 0
    if fault == "sample_id": a["sample_id"] = np.array("other")
    np.savez(p, **a)
    c["sample"] = m["rows"][0]["sample"] = binding(p)
    with pytest.raises(ValueError):
        dataset(tmp_path, repin(tmp_path, m, g, c))


def test_cross_sample_certificate_binding(tmp_path):
    _, m, g, c = fixture_files(tmp_path)
    c["sample"] = dict(c["sample"], sha256="0"*64)
    with pytest.raises(ValueError, match="BINDING"):
        dataset(tmp_path, repin(tmp_path, m, g, c))


def test_depth_camera_basis_and_single_obstacle():
    # 2m optical forward (normalized .1 at 20m) becomes +2m B0 forward.
    points = torch.tensor([[0., 0., .1 * CAMERA["depth_scale_m"]], [1., 0., 0.], [0., 1., 0.]])
    torch.testing.assert_close(optical_points_to_b0(points),
                               torch.tensor([[2., 0., 0.], [0., -1., 0.], [0., 0., -1.]]))
    from query_minco.training.losses.set_physical_risk import box_sdf
    boxes = torch.tensor([[[1.9, -.1, -.1], [2.1, .1, .1]]])
    assert box_sdf(optical_points_to_b0(points[:1]), boxes).item() < 0
