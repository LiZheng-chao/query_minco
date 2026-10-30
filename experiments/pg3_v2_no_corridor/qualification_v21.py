"""V2.1 qualification only. This module has no 128-step training command."""
import argparse,copy,io,json,os,subprocess,sys,time,unittest
from pathlib import Path
import torch
from query_minco.training.io import artifact,digest,read,require,sha,write
from query_minco.training.run_spec import ROOT,source_files,verify_code
from query_minco.training.pg1.spec import qualify_bindings
from query_minco.training.initialization import configure,head_rows,rows_unchanged
from query_minco.inference.identity import state_sha
from query_minco.training.pg2_guided_v2.objective import forward as pg2_forward
from query_minco.training.pg2_guided_v1.geometry import TeacherGuidance,tensor_digest
from query_minco.training.pg3_v2_no_corridor.objective import ObjectiveV21,KEYS_V21,METHOD_ID_V21,COMPOSE_ID_V21
from query_minco.training.pg3_v2_no_corridor.revision import section_events_v21,save_v21_checkpoint,optimizer_step_v21
from query_minco.training.pg3_v2_no_corridor.audit import audit_v21,calibrate_v21,make_keep_probe_v21,step_guard_v21
from query_minco.training.pg1_v2_r12.diagnostics import flat,cosine
from experiments.pg3_feasible_mode_v1.common import model_from
CONFIG=Path(__file__).with_name("pilot.json")
DEFAULT_OLD=Path("/home/lizheng/query_minco_mainline/research_outputs/pg3_feasible_mode_v1/fulltrain128_001/preparation/CPU0.json")
BASE="651152c1b3214101b54536c6904f4f38f13ceb57"

def ref(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))

def snapshot():
    tracked=subprocess.check_output(["git","ls-files"],cwd=str(ROOT),text=True).splitlines()
    result={p:sha(ROOT/p) for p in tracked if (ROOT/p).is_file()}
    for external in read(CONFIG)["external_readonly_code"]:
        target=artifact(dict(path=external["path"],sha256=external["sha256"]))
        linked=ROOT/external["worktree_link"]
        require(linked.is_file() and linked.resolve()==target and sha(linked)==external["sha256"],"V21_EXTERNAL_CODE_BINDING")
        result["@external/"+external["worktree_link"]]=external["sha256"]
    return result

def validate_config(cfg):
    require(cfg["method_id"]==METHOD_ID_V21 and cfg["compose_id"]==COMPOSE_ID_V21,"V21_METHOD_CONFIG")
    require(cfg["cpu0_limit"]==10 and cfg["calibration_batches"]==8 and cfg["seed"]==42 and cfg["threads"]==1 and cfg["batch_size"]==8,"V21_FIXED_PROTOCOL")
    require(set(cfg["budgets"])==set(KEYS_V21) and all(v>0 for v in cfg["budgets"].values()),"V21_POSITIVE_BUDGETS")
    require(cfg["section"]==dict(d_safe_m=.15,delta_extra_m=0.,sigma_m=.15,vx_min_mps=1e-4,join_epsilon_s=1e-6)
       and cfg["stop"]==dict(braking_acceleration_mps2=4.,tau_m=.05,d_ref_m=.15),"V21_FIXED_MATH")
    require(cfg["stop_mode"]=="replace_full_stop" and not any(cfg[k] for k in ("convex_corridor","teacher_corridor","ranking","control","projection")),"V21_DISABLED_FEATURES")
    return cfg


def shadow_step(initial,geo,ds,old,weights,config,records,folder):
    """One diagnostic AdamW step on an isolated clone, selected solely in Training8."""
    first=next((i for i,r in enumerate(records) if any(g["geom_source"]!="NONE" and g["query"] in r["selected"][g["batch"]] for g in r["geometry"])),None)
    if first is None:return dict(status="NO_REPAIR_SIGNAL",reason="NO_ACTIVE_DANGEROUS_GEOMETRY",optimizer_steps=0)
    ids,_=geo.orders[42].batch(first);samples=ds.load(ids)
    original_sha=state_sha(initial);clone=copy.deepcopy(initial).train();score=head_rows(clone)
    obj=ObjectiveV21(initial,original_sha,weights);before_history=digest(obj.assignment.state_dict())
    z=obj(clone,samples,geo,old["old_weights"],TeacherGuidance(),0)
    report,vectors,total=audit_v21(clone,z,strata_parameters=False)
    preclip_guard=step_guard_v21(report,config)
    before=torch.cat([p.detach().flatten().clone() for p in clone.parameters()])
    opt=torch.optim.AdamW(clone.parameters(),lr=config["shadow_step"]["lr"],betas=tuple(config["shadow_step"]["betas"]),eps=1e-8,weight_decay=0.)
    offset=0
    for p in clone.parameters():p.grad=total[offset:offset+p.numel()].reshape_as(p).clone();offset+=p.numel()
    require(bool(torch.isfinite(total).all()),"V21_SHADOW_FINITE")
    norm=float(torch.nn.utils.clip_grad_norm_(clone.parameters(),10.))
    receipt=optimizer_step_v21(clone,opt,obj.assignment,0,score)
    after=pg2_forward(clone,samples,geo,"query_minco_pg2_guided_v2_safe_v1",TeacherGuidance(),old["old_weights"])
    gaps=[]
    for row in z["geometry"]:
        if row["geom_source"]!="SECTION":continue
        b,q,m=row["batch"],row["query"],row["teacher"]
        _,info=section_events_v21(after["output"].trajectory.coeffs[b,q],after["output"].parameters.durations[b,q],z["banks"][b]["teachers"][m],z["banks"][b]["scene"])
        later={x["event"]:x for x in info["events"]}
        for event in row["section"]["events"]:
            if event["event"] in later:
                new=later[event["event"]]
                gaps.append(dict(sample_id=samples[b].sample_id,query=q,box=event["box"],
                    before_deficit_m=event["deficit_m"],after_deficit_m=new["deficit_m"],improvement_m=event["deficit_m"]-new["deficit_m"]))
    initial_params=z["output"].parameters;new_params=after["output"].parameters
    safe=z["labels"]["passed"]
    drift=dict(initial_safe_count=int(safe.sum()),pass_to_fail=int((safe&~after["labels"]["passed"]).sum()),
        waypoint_max_m=float((new_params.inner_pts-initial_params.inner_pts).norm(dim=-1)[safe].max()) if bool(safe.any()) else None,
        endpoint_max_m=float((new_params.tail_pva[:,:,0]-initial_params.tail_pva[:,:,0]).norm(dim=-1)[safe].max()) if bool(safe.any()) else None,
        velocity_max_mps=float((new_params.tail_pva[:,:,1]-initial_params.tail_pva[:,:,1]).norm(dim=-1)[safe].max()) if bool(safe.any()) else None,
        duration_l1_max_s=float((new_params.durations-initial_params.durations).abs().sum(-1)[safe].max()) if bool(safe.any()) else None)
    delta=torch.cat([p.detach().flatten() for p in clone.parameters()])-before
    improved=any(x["improvement_m"]>config["shadow_step"]["minimum_gap_improvement_m"] for x in gaps)
    checkpoint=folder/"SHADOW_PROBE_ONLY.pt"
    spec=dict(method_id=METHOD_ID_V21,code_sha256=digest(snapshot()),config_sha256=sha(CONFIG),seed=42,order=old["order128"],purpose="PROBE_ONLY")
    save_v21_checkpoint(checkpoint,clone,opt,obj.assignment,spec)
    require(state_sha(initial)==original_sha,"V21_SHADOW_SOURCE_MUTATION")
    return dict(status="PASS" if improved else "NO_REPAIR_SIGNAL",meaning="PROBE_ONLY_NOT_HARD_SAFETY_BENEFIT",batch_index=first,ids=ids,
        optimizer_steps=1,formal_optimizer_steps=0,initial_model_unchanged=True,initial_sha256=original_sha,
        clone_sha256=state_sha(clone),source_history_sha256=before_history,exposure_receipt=receipt,
        geometry_gaps=gaps,initial_safe_drift=drift,preclip_grad_norm=norm,preclip_guard=preclip_guard,ratios_before=report["ratios"],
        delta_cosine_negative={k:cosine(delta.double(),v.double().neg()) for k,v in dict(total=total,imitation=vectors["pg2/imitation"],geom=vectors["geom"],stop=vectors["stop"]).items()},checkpoint=ref(checkpoint))


def prepare(output,expected_commit,historical_cpu0):
    require(os.environ.get("CUDA_VISIBLE_DEVICES")=="","V21_CPU_ONLY")
    configure(dict(seed=42,threads=1,device="cpu"));cfg=validate_config(read(CONFIG))
    require(not subprocess.check_output(["git","status","--porcelain","--untracked-files=all"],cwd=str(ROOT),text=True).strip(),"V21_STRICT_CLEAN_WORKTREE_REQUIRED")
    head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=str(ROOT),text=True).strip();require(head==expected_commit,"V21_EXPECTED_HEAD")
    subprocess.run(["git","merge-base","--is-ancestor",BASE,head],cwd=str(ROOT),check=True)
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    old=read(historical_cpu0);binding=read(artifact(old["bindings"]))
    historical_reference=binding["historical_reference"]
    binding["historical_reference"]=ref(ROOT/"query_minco/methods/reference.json")
    require(historical_reference["sha256"]==binding["historical_reference"]["sha256"],"V21_REFERENCE_TRANSPORT_SHA")
    code=dict(git_commit=head,files=source_files(),allow_uncommitted_sources=False);binding["code"]=code
    verify_code(code);write(output/"BINDINGS.json",binding)
    ds,geo,init,data_qualification=qualify_bindings(ref(output/"BINDINGS.json"),42)
    write(output/"DATA_QUALIFICATION.json",data_qualification)
    model=model_from(init).train();before=state_sha(model);require(before==init["weights_sha256"],"V21_C1_IDENTITY")
    original_rng=torch.get_rng_state().clone();initial_history=None
    weights={k:1. for k in KEYS_V21};obj=ObjectiveV21(model,before,weights)
    initial_history=digest(obj.assignment.state_dict())
    probe,probe_meta=make_keep_probe_v21(model,cfg["keep_probe"])
    records=[];probes=[];calibration_inputs=[]
    order=read(artifact(old["order128"]))
    for i in range(8):
        ids,_=geo.orders[42].batch(i);require(ids==order["batches"][i],"V21_FROZEN_ORDER")
        samples=ds.load(ids)
        inputs=[dict(sample_id=s.sample_id,tensors_sha256=tensor_digest(s.depth_input,s.state,s.teacher_coeffs,s.teacher_durations),artifact=ds.rows[s.sample_id]["artifact"]) for s in samples]
        for item in inputs:artifact(item["artifact"])
        z=obj(model,samples,geo,old["old_weights"],TeacherGuidance(),0)
        row,_,_=audit_v21(model,z);row.update(ids=ids,inputs=inputs,selected=[a["selected"] for a in z["assignments"]],phase="UNCALIBRATED_UNIT_WEIGHTS")
        obj.assignment.rollback(0)
        require(row["components"]["keep"]["value"]==0 and row["components"]["keep"]["parameter_norm"]==0,"V21_INITIAL_KEEP_NOT_ZERO")
        pz=obj(probe,samples,geo,old["old_weights"],TeacherGuidance(),0)
        pr,_,_=audit_v21(probe,pz,strata_parameters=False);pr.update(ids=ids,probe=probe_meta,phase="PROBE_ONLY")
        obj.assignment.rollback(0)
        records.append(row);probes.append(pr);calibration_inputs.append(inputs)
        write(output/("TRAIN_BATCH_%02d.json"%i),row);write(output/("KEEP_PROBE_%02d.json"%i),pr)
        print("V21_TRAIN_AND_KEEP_AUDIT",i,flush=True)
    ids_flat=[sid for row in records for sid in row["ids"]]
    require(len(ids_flat)==64 and len(set(ids_flat))==62,"V21_EXPECTED_OCCURRENCES_UNIQUE")
    calibration=calibrate_v21(records,probes,cfg)
    ids,_=geo.orders[42].batch(0);samples=ds.load(ids)
    obj.weights=calibration["weights"]
    # Required stop/geom activation failures block qualification instead of silently disabling them.
    require(obj.weights["stop"]>0,"V21_STOP_NOT_CALIBRATABLE")
    final=obj(model,samples,geo,old["old_weights"],TeacherGuidance(),0)
    cpu,_,_=audit_v21(model,final);obj.assignment.rollback(0)
    require(all(p.grad is None for p in model.parameters()) and all(p.grad is None for p in obj.reference.model.parameters()),"V21_CPU0_GRAD_BUFFER_MUTATION")
    require(state_sha(model)==before and state_sha(obj.reference.model)==before
        and digest(obj.assignment.state_dict())==initial_history and torch.equal(torch.get_rng_state(),original_rng),"V21_CPU0_MUTATION")
    probe_active=all(p["components"]["keep"]["parameter_norm"]>0 for p in probes)
    pg2_again=pg2_forward(model,samples,geo,"query_minco_pg2_guided_v2_safe_v1",TeacherGuidance(),old["old_weights"])
    params=list(model.parameters())
    ga=flat(torch.autograd.grad(final["pg2"]["loss"],params,retain_graph=True,allow_unused=True),params)
    gb=flat(torch.autograd.grad(pg2_again["loss"],params,retain_graph=True,allow_unused=True),params)
    baseline=dict(loss_equal=torch.equal(final["pg2"]["loss"],pg2_again["loss"]),parameter_grad_equal=torch.equal(ga,gb),
        raw_grad_equal=torch.equal(torch.autograd.grad(final["pg2"]["loss"],final["raw"],retain_graph=True)[0],torch.autograd.grad(pg2_again["loss"],pg2_again["raw"],retain_graph=True)[0]))
    require(all(baseline.values()),"V21_PG2_BASELINE_DRIFT")
    write(output/"BASELINE_AUDIT.json",dict(baseline=baseline,code=code,pg2_sources={p:h for p,h in code["files"].items() if "/pg2_guided_v2/" in p}))
    audit=dict(schema="pg3_v2_1_complete_gradient_audit_v1",code=code,sources=snapshot(),configuration=ref(CONFIG),records=records,keep_probes=probes,
        keep_probe_identity=probe_meta,keep_probe_active=probe_active,calibration=calibration,training_inputs=calibration_inputs,
        calibration_occurrences=64,calibration_unique_samples=62,uses_development=False,formal_optimizer_steps=0)
    write(output/"GRADIENT_AUDIT_V2_1.json",audit)
    gate=calibration["passed"] and probe_active and cpu["ratios"]["R_D"]<=10 and cpu["ratios"]["R_base"]<=10
    cpu_doc=dict(status="PASS" if gate else "FAILED_GRADIENT_GUARD",ratios=cpu["ratios"],calibration_ratios=calibration["joint_ratios"],
        threshold=10,increment_limit=cfg["increment_limit"],code=code,sources=snapshot(),weights=calibration["weights"],
        weights_before=before,weights_after=state_sha(model),optimizer_created=False,optimizer_steps=0,exposure_unchanged=True,
        gradient_audit=ref(output/"GRADIENT_AUDIT_V2_1.json"),detail=cpu,blockers=calibration["blockers"])
    write(output/"CPU0_V2_1.json",cpu_doc)
    shadow=shadow_step(model,geo,ds,old,calibration["weights"],cfg,records,output)
    write(output/"SHADOW_OPTIMIZER_PROBE.json",shadow)
    write(output/"PREPARATION.json",dict(status="PENDING_FINAL_TESTS",code=code,sources=snapshot(),init=init,bindings=ref(output/"BINDINGS.json"),
        order=old["order128"],pg2_weights=old["old_weights"],historical_cpu0=ref(historical_cpu0),configuration=ref(CONFIG),cpu0=ref(output/"CPU0_V2_1.json"),
        gradient_audit=ref(output/"GRADIENT_AUDIT_V2_1.json"),shadow_probe=ref(output/"SHADOW_OPTIMIZER_PROBE.json")))
    print("V21_CPU0",cpu_doc["status"],cpu_doc["ratios"],"SHADOW_PROBE",shadow["status"],flush=True)


class ReportResult(unittest.TextTestResult):
    def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.rows={}
    def addSuccess(self,test):super().addSuccess(test);self.rows[test.id()]=dict(status="PASS")
    def addFailure(self,test,err):super().addFailure(test,err);self.rows[test.id()]=dict(status="FAIL",error=self._exc_info_to_string(err,test))
    def addError(self,test,err):super().addError(test,err);self.rows[test.id()]=dict(status="FAIL",error=self._exc_info_to_string(err,test))
    def addSkip(self,test,reason):super().addSkip(test,reason);self.rows[test.id()]=dict(status="NOT_RUN",reason=reason)


def test_report(output,evidence=None):
    if evidence:os.environ["PG3_V21_EVIDENCE"]=str(Path(evidence).resolve())
    suite=unittest.TestSuite()
    for pattern in ("test_pg2*.py","test_pg3*.py"):suite.addTests(unittest.defaultTestLoader.discover(str(ROOT/"tests"),pattern=pattern))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromName("experiments.pg3_feasible_mode_v1.test_fulltrain128"))
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2,resultclass=ReportResult).run(suite)
    report=dict(code_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=str(ROOT),text=True).strip(),total=result.testsRun,
        passed=sum(r["status"]=="PASS" for r in result.rows.values()),failed=sum(r["status"]=="FAIL" for r in result.rows.values()),
        not_run=sum(r["status"]=="NOT_RUN" for r in result.rows.values()),tests=result.rows,command=sys.argv,evidence=None if not evidence else str(evidence),log=stream.getvalue())
    report["T01_T19"]={name.split(".test_",1)[1][:3]:dict(test=name,**row) for name,row in result.rows.items() if ".test_T" in name and "V21Acceptance" in name}
    report["worktree_status"]=subprocess.check_output(["git","status","--porcelain","--untracked-files=all"],cwd=str(ROOT),text=True)
    report["package_source_hashes"]=source_files()
    write(output,report);print(stream.getvalue());return report


def finalize(output):
    output=Path(output);prep=read(output/"PREPARATION.json");verify_code(prep["code"])
    require(snapshot()==prep["sources"],"V21_FINAL_SOURCE_DRIFT")
    tests=read(output/"TEST_REPORT.json");cpu=read(output/"CPU0_V2_1.json");shadow=read(output/"SHADOW_OPTIMIZER_PROBE.json")
    require(tests["code_commit"]==prep["code"]["git_commit"],"V21_TEST_COMMIT")
    blockers=[]
    if tests["failed"] or tests["not_run"] or set(tests["T01_T19"])!={"T%02d"%i for i in range(1,20)}:blockers.append("FAILED_ACCEPTANCE_TESTS")
    if cpu["status"]!="PASS":blockers.append(cpu["status"])
    if shadow["status"]!="PASS":blockers.append(shadow["status"])
    status="QUALIFIED" if not blockers else "FAILED_QUALIFICATION"
    for arm in ("A","D"):
        spec=dict(schema=COMPOSE_ID_V21,run_id="A42_"+arm+"_PG3V21_CPU128_run001",arm=arm,
            method_id="query_minco_pg2_guided_v2_safe_v1" if arm=="A" else METHOD_ID_V21,code=prep["code"],sources=prep["sources"],
            initialization=prep["init"],bindings=prep["bindings"],order=prep["order"],configuration=prep["configuration"],pg2_weights=prep["pg2_weights"],historical_calibration=prep["historical_cpu0"],
            stop_mode="legacy" if arm=="A" else "replace_full_stop",weights={} if arm=="A" else cpu["weights"],
            cpu0=ref(output/"CPU0_V2_1.json"),tests=ref(output/"TEST_REPORT.json"),seed=42,device="cpu",threads=1,precision="FP32",batch_size=8,
            updates=128,optimizer=dict(name="AdamW",betas=[.9,.999],eps=1e-8,weight_decay=0.,clip_norm=10.),
            schedule="fulltrain128_cosine_2e-5_1e-6_endpoint_v1",status=status,approved=False,training="NOT_RUN",output_root=str(output.parent.parent/"runs"))
        write(output/("RUN_SPEC_"+arm+".json"),spec)
    qualification=dict(status=status,blockers=blockers,code=prep["code"],sources=prep["sources"],allow_uncommitted_sources=False,
        data=ref(output/"DATA_QUALIFICATION.json"),cpu0=ref(output/"CPU0_V2_1.json"),tests=ref(output/"TEST_REPORT.json"),
        gradient_audit=prep["gradient_audit"],shadow_probe=prep["shadow_probe"],runspecs={arm:ref(output/("RUN_SPEC_"+arm+".json")) for arm in ("A","D")},
        ready_to_request_training_authorization=not blockers,training_authorized=False)
    write(output/"QUALIFICATION.json",qualification)
    write(output/"IMPLEMENTATION_RESULT_V2_1.json",dict(status=status,blockers=blockers,qualification=ref(output/"QUALIFICATION.json"),
        optimizer_steps=0,diagnostic_shadow_optimizer_steps=shadow["optimizer_steps"],training="NOT_RUN",paired_eval="NOT_RUN",forest="NOT_RUN",shadow="NOT_RUN",flight="NOT_RUN",ready=False,may_request_training_authorization=not blockers))
    print("V21_FINAL_QUALIFICATION",status,blockers,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument("action",choices=["prepare","tests","finalize"])
    p.add_argument("--output",type=Path,required=True);p.add_argument("--expected-commit");p.add_argument("--historical-cpu0",type=Path,default=DEFAULT_OLD);p.add_argument("--evidence",type=Path)
    a=p.parse_args()
    if a.action=="prepare":
        require(bool(a.expected_commit),"V21_EXPECTED_COMMIT_REQUIRED");prepare(a.output,a.expected_commit,a.historical_cpu0)
    elif a.action=="tests":
        r=test_report(a.output,a.evidence)
        if r["failed"] or r["not_run"]:raise SystemExit(1)
    else:finalize(a.output)
if __name__=="__main__":main()
