"""Optimizer isolation and lifecycle contracts. These tests never execute optimizer.step."""
import hashlib
import json
import os
from pathlib import Path
import pytest
import torch
from unittest.mock import patch
from query_minco.training.k1_physics_trainer_v1.optimizer import DeadRowGuard
from query_minco.training.k1_physics_trainer_v1.initialization import build
from query_minco.training.k1_physics_trainer_v1 import trainer

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(os.environ.get("G2_AUDIT_ROOT",ROOT/"research_outputs/g2a_k1_trainer_qualification_20261110_v1"))
SPEC=OUT/"K1_TRAINING_RUNSPEC_DRAFT.json"
torch.set_num_threads(1)


@pytest.mark.parametrize("amsgrad",[False,True])
@pytest.mark.parametrize("weight_decay",[0.,.1])
def test_sanitize_dirty_moments_without_step(amsgrad,weight_decay):
    model=build("A_random",20261104)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=weight_decay,amsgrad=amsgrad)
    guard=DeadRowGuard(model);before=guard.sha()
    for p in guard.parameters:
        p.grad=torch.ones_like(p)
        optimizer.state[p]={"step":torch.tensor(7.),"exp_avg":torch.full_like(p,.2),
                            "exp_avg_sq":torch.full_like(p,.5)}
        if amsgrad:optimizer.state[p]["max_exp_avg_sq"]=torch.full_like(p,.8)
    report=guard.sanitize(optimizer)
    assert all(x["nonzero"]>0 for x in report.values())
    assert all(x["nonzero"]==0 for x in guard.state_report(optimizer).values())
    assert guard.sha()==before
    for p in guard.parameters:
        assert p.grad[15:].count_nonzero()==0 and torch.equal(p.grad[:15],torch.ones_like(p.grad[:15]))
        assert torch.equal(optimizer.state[p]["exp_avg"][:15],torch.full_like(p[:15],.2))
        assert optimizer.state[p]["step"].item()==7.


def test_step_budget_rejects_before_backend(tmp_path):
    model=build("A_random",20261104);guard=DeadRowGuard(model)
    optimizer=torch.optim.AdamW(model.parameters())
    ledger=tmp_path/"ledger.json";ledger.write_text(json.dumps(dict(attempted=2,completed=2)))
    with patch.object(optimizer,"step",side_effect=AssertionError("BACKEND_MUST_NOT_RUN")):
        with pytest.raises(PermissionError,match="BUDGET_EXHAUSTED"):
            guard.step(optimizer,ledger,2)


def test_malformed_state_fails_closed():
    model=build("A_random",20261104);guard=DeadRowGuard(model)
    optimizer=torch.optim.AdamW(model.parameters())
    optimizer.state[guard.parameters[0]]={"exp_avg":torch.zeros(1)}
    with pytest.raises(ValueError,match="STATE_SHAPE"):
        guard.sanitize(optimizer)


def test_exact_authorization_required_no_output_created(tmp_path):
    output=tmp_path/"must_not_exist"
    with pytest.raises(PermissionError,match="AUTHORIZATION_REQUIRED"):
        trainer.train(SPEC,None,output,20261104,"A_reasonable")
    assert not output.exists()


@pytest.mark.parametrize("case",["wrong_sha","empty_message","wrong_authority","wrong_run"])
def test_bad_approval_cannot_enable_training(tmp_path,case):
    spec=trainer.verify_spec(SPEC)
    mode="A_reasonable";seed=20261104;rid=mode+"_seed"+str(seed)
    output=spec["formal_run_directories"][rid]
    a=dict(decision="APPROVE_EXACT_RUNSPEC",authority="researcher_explicit_message",
           runspec_sha256=trainer.sha(SPEC),message_reference="TEST_ONLY_NOT_AUTHORIZATION",
           approved_run_ids=[rid])
    if case=="wrong_sha":a["runspec_sha256"]="0"*64
    if case=="empty_message":a["message_reference"]=""
    if case=="wrong_authority":a["authority"]="test"
    if case=="wrong_run":a["approved_run_ids"]=[]
    p=tmp_path/"approval.json";p.write_text(json.dumps(a))
    with pytest.raises(PermissionError):
        trainer.authorize(SPEC,p,seed,mode,output)
    assert not Path(output).exists()


def test_dev_cannot_enter_update_before_optimizer(tmp_path):
    spec=trainer.verify_spec(SPEC)
    with pytest.raises(PermissionError,match="DEV_MUST_NOT_UPDATE"):
        trainer.update(None,None,None,None,spec["dev"][0],tmp_path/"unused.json",2,SPEC)


def test_new_initialization_retains_state_dict_and_architecture():
    random=build("A_random",20261104);reasonable=build("A_reasonable",20261104)
    assert sum(p.numel() for p in random.parameters())==469821
    assert random.state_dict().keys()==reasonable.state_dict().keys()
    reasonable.load_state_dict(random.state_dict(),strict=True)
    assert random.config==reasonable.config


def test_checkpoint_bad_hash_rejected_before_mutation(tmp_path):
    p=tmp_path/"bad.pt";p.write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError,match="CHECKPOINT_FILE_SHA"):
        trainer.load_checkpoint(p,"0"*64,None,None,None,{})


def test_artifact_preflight_shared_model_and_dev_no_updates():
    result=json.loads((OUT/"preflight_001/RESULT.json").read_text())
    assert result["status"]=="PASS"
    rows=result["train_backward_tasks"]
    assert len(rows)==13 and len({r["parameter_identity"] for r in rows})==1
    assert result["model_sha_before"]==result["model_sha_after"]
    assert result["evaluation"]["dev_backward_calls"]==0
    assert result["evaluation"]["model_sha_before"]==result["evaluation"]["model_sha_after"]


def test_artifact_exactly_two_engineering_steps_and_real_live_updates():
    result=json.loads((OUT/"smoke_001/RESULT.json").read_text())
    ledger=json.loads((OUT/"ENGINEERING_STEP_LEDGER.json").read_text())
    backend=json.loads((OUT/"BACKEND_STEPS.json").read_text())
    assert result["ENGINEERING_OPTIMIZER_STEPS"]==2 and result["FORMAL_RESEARCH_TRAINING_STEPS"]==0
    assert ledger==dict(attempted=2,completed=2)
    assert backend["AdamW_step_calls"]==2
    steps=[json.loads((OUT/("smoke_001/step"+str(i).zfill(3)+".json")).read_text()) for i in (1,2)]
    assert steps[0]["model_sha_after"]==steps[1]["model_sha_before"]
    assert steps[0]["sample_id"]!=steps[1]["sample_id"]
    assert steps[0]["shared_parameter_identity"]==steps[1]["shared_parameter_identity"]
    for s in steps:
        assert s["head_live_update_norm"]>0
        assert all(v["parameter_update_norm"]>0 for v in s["module_updates"].values())
        d=s["dead_row_isolation"]
        assert d["dead_sha_before"]==d["dead_sha_after"]
        assert d["backend_dead_delta_before_restore"]>0
        assert all(v["nonzero"]==0 for v in d["state_after"].values())
    assert any(v["nonzero"]>0 for v in steps[1]["dead_row_isolation"]["state_before_sanitization"].values())


def test_artifact_checkpoint_resume_and_full_state():
    r=json.loads((OUT/"smoke_001/CHECKPOINT_RESUME.json").read_text())
    assert r["status"]=="PASS" and r["cursor"]==1
    assert r["active_and_sanitized_state_exact"] and r["same_shared_parameter_objects"]
    assert any(x["nonzero"] for x in r["dirty_state_before_sanitize"].values())
    blob=torch.load(OUT/"smoke_001/step002.pt",map_location="cpu",weights_only=True)
    assert blob["cursor"]==2 and blob["metadata"]["engineering_only"]
    assert blob["metadata"]["deployable"] is False
    assert "rng" in blob and "optimizer" in blob and "model" in blob


def test_artifact_initialization_not_tuned_and_all_cases_reported():
    d=json.loads((OUT/"preflight_001/INITIALIZATION_AUDIT.json").read_text())
    assert d["tuning_rounds"]==1 and d["no_general_state_conditioned_reference_claim"]
    for mode in ["A_random","A_reasonable"]:
        cases=d["records"][mode]["cases"]
        assert len(cases)==22 and sum(c["case"]["in_scope"] for c in cases)==18
    # Scope outcome is reported honestly; no test forces this candidate to pass.
    assert d["status"] in ["QUALIFIED","NOT_QUALIFIED"]
