"""Planning tests never create an optimizer or take any update."""
import json
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch
import pytest
import torch
from query_minco.training.physics_only_v1_2_gate1.short_train import check_authorization,execute_training,sha
from query_minco.training.physics_only_v1_2.spec import PhysicsV12Spec
from query_minco.training.physics_only_v1.decoder import FixedTimeDecoder


def test_no_approval_rejects_before_model_optimizer_or_directory(tmp_path):
    with patch("query_minco.training.physics_only_v1_2_gate1.short_train.initialize") as model,patch.object(torch.optim,"AdamW") as optimizer:
        with pytest.raises(PermissionError,match="APPROVAL_REQUIRED"):
            execute_training(tmp_path/"absent_spec.json",None)
        model.assert_not_called();optimizer.assert_not_called()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("fault",["decision","sha","authority","reference"])
def test_approval_must_bind_exact_researcher_message(tmp_path,fault):
    spec=tmp_path/"spec.json";spec.write_text("{}")
    approval=dict(decision="APPROVE_EXACT_RUNSPEC",runspec_sha256=sha(spec),
                  authority="researcher_explicit_message",message_reference="test only, never execute")
    if fault=="decision":approval["decision"]="pending"
    if fault=="sha":approval["runspec_sha256"]="0"*64
    if fault=="authority":approval["authority"]="automatic"
    if fault=="reference":approval["message_reference"]=""
    path=tmp_path/"approval.json";path.write_text(json.dumps(approval))
    with pytest.raises(PermissionError,match="APPROVAL_NOT_BOUND"):check_authorization(spec,path)


@pytest.mark.parametrize("fault",["updates","schedule","optimizer","clip","physics"])
def test_bad_budget_optimizer_or_math_rejected_before_runtime(tmp_path,fault):
    s=dict(schema="physics_v12_first_short_train_runspec_v1",max_updates=32,
        evaluation_steps=[0,8,16,32],seed=20261104,gradient_clip_norm=10.,
        optimizer=dict(name="AdamW",lr=1e-4,betas=[.9,.999],eps=1e-8,weight_decay=0.,amsgrad=False,foreach=False),
        physics_spec=asdict(PhysicsV12Spec(phase="A",aggregation="foundation")))
    if fault=="updates":s["max_updates"]=33
    if fault=="schedule":s["evaluation_steps"]=[0,32]
    if fault=="optimizer":s["optimizer"]["lr"]=1e-3
    if fault=="clip":s["gradient_clip_norm"]=100
    if fault=="physics":s["physics_spec"]["w_p"]=0.
    p=tmp_path/"spec.json";p.write_text(json.dumps(s))
    approval=tmp_path/"approval.json"
    approval.write_text(json.dumps(dict(decision="APPROVE_EXACT_RUNSPEC",runspec_sha256=sha(p),
       authority="researcher_explicit_message",message_reference="negative unit fixture, no execution")))
    with pytest.raises(ValueError):check_authorization(p,approval)


def test_zero_logits_are_absolute_coincident_waypoints_not_displacements():
    spec=PhysicsV12Spec(phase="A",aggregation="foundation")
    raw=torch.zeros(1,6,21,requires_grad=True)
    state=torch.tensor([[.8,0,0,.1,0,0,6,0,0,0,0,-1.]])
    params=FixedTimeDecoder(spec)(raw,state)
    points=torch.cat([params.inner_pts,params.tail_pva[...,:1,:]],dim=-2)
    torch.testing.assert_close(points,torch.tensor([5.,0,0]).expand(1,6,3,3))
    torch.testing.assert_close(params.durations,torch.full((1,6,3),2/3))
    grads=torch.autograd.grad(points.sum(),raw)[0]
    assert not grads[...,15:21].any()
    assert torch.all(grads[...,:9].reshape(1,6,3,3)[...,0]==5.)
