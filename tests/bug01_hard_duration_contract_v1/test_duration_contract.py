"""BUG-01 regression; legacy source is a hash-checked, read-only audit snapshot."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import numpy as np
import pytest
import torch
from query_minco.training.physics_only_v1_2_gate1 import hard_validator as current
from query_minco.training.physics_only_v1_2_gate1.geometry import TypedBoundOccupancy, _SEAL
from query_minco.evaluation.validation import numerical_reference

ROOT = Path(__file__).resolve().parents[2]
AUDIT = Path(os.environ.get("BUG01_AUDIT_ROOT", ROOT / "research_outputs/bug01_hard_duration_contract_20261110_01"))
OLD_SHA = "f55159ffc879b3b1d0062638bcde42690a40719cfcd3c23b34a9f8ec11cbeacf"
TARGET = ROOT / "query_minco/training/physics_only_v1_2_gate1/hard_validator.py"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def legacy():
    path = AUDIT / "snapshots/hard_validator_before.py"
    assert sha(path) == OLD_SHA
    spec = importlib.util.spec_from_file_location("_bug01_legacy_hard", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def geometry():
    counts = np.zeros((60, 60, 60), np.int32)
    counts[50, 50, 40] = 1
    return TypedBoundOccupancy(
        seal=_SEAL, counts=counts,
        meta=dict(grid_origin=[-2, -2, 0], resolution_m=.1, occupy_threshold=0),
        pose=dict(position_world=[0, 0, 1], R_WB0=np.eye(3).tolist()),
        roi=[[-1.9, -1.9, .2], [3.8, 3.8, 5.8]],
        domain=[[-1, -1, -.5], [3, 1, 2]],
        identity={"fixture": "SYNTHETIC_TEST_ONLY BUG01"}, revalidate=lambda: None)


def stationary():
    c = np.zeros((3, 3, 6))
    c[:, :, 0] = [.4, .4, .4]
    return c, np.array([.4, .4, .4])


@pytest.mark.parametrize("segment", range(3))
@pytest.mark.parametrize("value,reason", [
    (-2/3, "NONPOSITIVE_DURATION"), (0., "NONPOSITIVE_DURATION"),
    (-0., "NONPOSITIVE_DURATION"), (float("nan"), "INVALID_DURATION_NONFINITE"),
    (float("inf"), "INVALID_DURATION_NONFINITE"),
    (-float("inf"), "INVALID_DURATION_NONFINITE")])
def test_each_illegal_segment_rejected(geometry, segment, value, reason):
    c, goal = stationary()
    dt = np.full(3, 2/3); dt[segment] = value
    result = current.validate_polynomial(c, dt, geometry, goal)
    assert result == dict(passed=False, reasons=[reason])


@pytest.mark.parametrize("duration", [
    2/3, [], [2/3], [2/3, 2/3], [2/3]*4,
    [[2/3]*3], [[2/3], [2/3], [2/3]], np.ones((1, 1, 3)),
    torch.ones((3, 1))])
def test_shape_and_segment_count_rejected(geometry, duration):
    c, goal = stationary()
    assert current.validate_polynomial(c, duration, geometry, goal) == dict(
        passed=False, reasons=["INVALID_DURATION_SHAPE"])


@pytest.mark.parametrize("duration", [None, ["bad", 1, 1], [[1], [1, 2], [3]],
                                     torch.ones(3, requires_grad=True)])
def test_unconvertible_time_controlled_rejection(geometry, duration):
    c, goal = stationary()
    result = current.validate_polynomial(c, duration, geometry, goal)
    assert result["passed"] is False
    assert result["reasons"][0] in ("INVALID_DURATION_TYPE", "INVALID_DURATION_SHAPE")


@pytest.mark.parametrize("segments", [0, 1, 2, 4])
def test_unsupported_coefficient_segment_count_rejected(geometry, segments):
    result = current.validate_polynomial(np.zeros((segments, 3, 6)),
                                         np.ones(segments), geometry, [1, 0, 0])
    assert result == dict(passed=False, reasons=["INVALID_POLYNOMIAL"])


@pytest.mark.parametrize("duration", [
    [2/3]*3, [.5, .75, .75], [1e-8, 1., 1.], torch.ones(3),
    np.ones(3, dtype=np.float32)])
def test_legal_stationary_result_exactly_unchanged(geometry, duration):
    c, goal = stationary()
    before = legacy().validate_polynomial(c, duration, geometry, goal)
    after = current.validate_polynomial(c, duration, geometry, goal)
    assert before["passed"]
    assert before == after


@pytest.mark.parametrize("duration", [[2/3]*3, [.4, .6, 1.], [.5, .75, .75]])
def test_known_legal_moving_minco_result_exactly_unchanged(geometry, duration):
    dt = np.asarray(duration); total = dt.sum()
    head = np.zeros((3, 3)); tail = np.zeros((3, 3)); tail[0, 0] = 1.
    u = np.cumsum(dt)[:2] / total
    inner = np.zeros((2, 3)); inner[:, 0] = 10*u**3-15*u**4+6*u**5
    c = numerical_reference(head, tail, inner, dt)
    before = legacy().validate_polynomial(c, dt, geometry, [1, 0, 0])
    after = current.validate_polynomial(c, dt, geometry, [1, 0, 0])
    assert before["passed"], before
    assert before == after


@pytest.mark.parametrize("case", ["goal_fail", "domain_fail", "dynamic_fail", "unknown"])
def test_positive_time_failure_results_exactly_unchanged(geometry, case):
    c, goal = stationary(); dt = np.full(3, 2/3)
    if case == "goal_fail":
        goal = np.array([2., 0., 0.])
    elif case == "domain_fail":
        c[:, 0, 0] = -1.2
    elif case == "dynamic_fail":
        c[:, 0, 1] = 4.
    else:
        c[:, 0, 0] = 4.
    before = legacy().validate_polynomial(c, dt, geometry, goal)
    assert not before["passed"]
    assert current.validate_polynomial(c, dt, geometry, goal) == before


def test_version_and_all_physical_math_unchanged():
    assert current.COMPONENT_ID == "physics_v12_independent_hard_positive_duration_v2"
    assert current.LEGACY_SOURCE_SHA256 == OLD_SHA
    before = ast.parse((AUDIT / "snapshots/hard_validator_before.py").read_text())
    after = ast.parse(TARGET.read_text())
    old_funcs = {n.name: n for n in before.body if isinstance(n, ast.FunctionDef)}
    new_funcs = {n.name: n for n in after.body if isinstance(n, ast.FunctionDef)}
    assert old_funcs.keys() == new_funcs.keys()
    for name, node in old_funcs.items():
        if name != "validate_polynomial":
            assert ast.dump(node) == ast.dump(new_funcs[name])
    def physics_body(node):
        for i, stmt in enumerate(node.body):
            if isinstance(stmt, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "world" for t in stmt.targets):
                return [ast.dump(x) for x in node.body[i:]]
        raise AssertionError("physics body not found")
    assert physics_body(old_funcs["validate_polynomial"]) == physics_body(new_funcs["validate_polynomial"])


def prepare_rebound():
    """New manifests retain full loader checks; change only approved source identity."""
    index = json.loads((ROOT / "research_outputs/physics_only_v1_2_p0c_world_boundary_v1/_preparation/bindings_p0c_001/BINDING_INDEX.json").read_text())
    selected = [r for r in index if r["data_eligible"]]
    mapping = {}
    evidence = []
    for row in selected:
        original = Path(row["p0c_manifest_path"])
        assert sha(original) == row["p0c_manifest_sha256"]
        manifest = json.loads(original.read_text())
        assert manifest["files_sha256"][str(TARGET)] == OLD_SHA
        for name, expected in manifest["files_sha256"].items():
            assert sha(name) == (sha(TARGET) if name == str(TARGET) else expected)
        destination = AUDIT / "rebound" / row["sample_id"]
        destination.mkdir(parents=True)
        state = destination / "STATE.json"
        state.write_bytes((original.parent / "STATE.json").read_bytes())
        manifest["files_sha256"][str(TARGET)] = sha(TARGET)
        manifest["files_sha256"][str(state)] = sha(state)
        new = destination / "SAMPLE_MANIFEST.json"
        new.write_text(json.dumps(manifest, indent=2))
        mapping[str(original)] = (str(new), sha(new), row["p0c_manifest_sha256"])
        evidence.append(dict(sample_id=row["sample_id"],original=str(original),
            original_sha=sha(original),rebound=str(new),rebound_sha=sha(new),
            authorized_source_change={str(TARGET):dict(before=OLD_SHA,after=sha(TARGET))},
            copied_state_sha=sha(state),geometry_unchanged=True))
    (AUDIT / "REBINDING.json").write_text(json.dumps(evidence, indent=2))
    return mapping


@pytest.mark.parametrize("sample_id", [
    "train_free_roi_1101_p0_near", "train_free_roi_1101_p0_far",
    "dev_single_slit_2104_p0_near", "dev_two_portals_2103_p0_far"])
def test_frozen_real_legal_witnesses_exactly_unchanged(sample_id):
    from query_minco.training.physics_only_v1_2_gate1.e1_dataset_binding import load_e1_bound
    index = json.loads((ROOT / "research_outputs/physics_only_v1_2_p0c_world_boundary_v1/_preparation/bindings_p0c_001/BINDING_INDEX.json").read_text())
    row = next(r for r in index if r["sample_id"] == sample_id)
    manifest = AUDIT / "rebound" / sample_id / "SAMPLE_MANIFEST.json"
    batch = load_e1_bound(manifest, sha(manifest))
    evidence = json.loads(Path(row["hard_evidence_path"]).read_text())
    original = Path(evidence["source_evidence_path"])
    assert sha(original) == evidence["source_evidence_sha256"]
    frozen = json.loads(original.read_text())
    assert frozen["witnesses"]
    for witness in frozen["witnesses"].values():
        c = np.asarray(witness["coeffs_B0"]); dt = np.asarray(witness["durations_s"])
        goal = batch.state[0, 6:9].numpy()
        before = legacy().validate_polynomial(c, dt, batch.geometry[0], goal)
        after = current.validate_polynomial(c, dt, batch.geometry[0], goal)
        assert before["passed"], before
        assert before == after
