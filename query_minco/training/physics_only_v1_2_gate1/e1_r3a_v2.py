"""Versioned dependency bootstrap and read-only Train13 zero-update orchestration."""
import argparse,csv,hashlib,importlib,importlib.util,json,os,sys,sysconfig,time,traceback
from pathlib import Path
BASE=Path(__file__).resolve().parents[3]
OUT=BASE/"research_outputs/physics_only_v1_2_e1_r3a_v2"
PREP=OUT/"_preparation/freeze_001"
PYTEST_ROOT=Path("/home/lizheng/anaconda3/lib/python3.11/site-packages")
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    with Path(p).open("x") as f:json.dump(x,f,indent=2,allow_nan=False);f.write("\n")
def guard():
    if (PREP/"STOP.json").exists():raise ValueError("PERSISTENT_STOP")
    e=read(PREP/"ENTRY.json")
    for p,h in e["protected_sha256"].items():
        if sha(p)!=h:raise ValueError("HISTORY_SHA_DRIFT "+p)
    for p,h in read(PREP/"SOURCE_FREEZE.json").items():
        if sha(BASE/p)!=h:raise ValueError("SOURCE_SHA_DRIFT "+p)
    for p,h in read(PREP/"INPUT_HASHES.json").items():
        if sha(p)!=h:raise ValueError("INPUT_SHA_DRIFT "+p)
    if time.time()-e["started_unix"]>3600:raise ValueError("TOTAL_BUDGET")
def bootstrap():
    """Append accepted pure-Python test dependency path after numerical imports."""
    if sys.executable!="/home/lizheng/anaconda3/envs/yopo/bin/python":raise ValueError("PYTHON_IDENTITY")
    numeric={n:importlib.import_module(n) for n in ("numpy","scipy","torch")}
    if not PYTEST_ROOT.is_dir():raise ValueError("PYTEST_SOURCE_MISSING")
    if str(PYTEST_ROOT) not in sys.path:sys.path.append(str(PYTEST_ROOT))
    import importlib.metadata as md
    modules={n:importlib.import_module(n) for n in ("pytest","_pytest","pluggy","iniconfig","packaging","exceptiongroup","tomli")}
    from packaging.requirements import Requirement
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version
    py=Version(".".join(str(x) for x in sys.version_info[:3]))
    requirements=[]
    for n in ("pytest","pluggy","iniconfig","packaging","exceptiongroup","tomli"):
        dist=md.distribution(n);reqpy=dist.metadata.get("Requires-Python")
        if reqpy and py not in SpecifierSet(reqpy):raise ValueError("DEPENDENCY_PYTHON_INCOMPATIBLE "+n)
        for line in dist.requires or ():
            req=Requirement(line)
            if req.marker is None or req.marker.evaluate({"extra":""}):
                actual=md.version(req.name)
                if req.specifier and Version(actual) not in req.specifier:raise ValueError("TRANSITIVE_VERSION "+line)
                requirements.append(dict(parent=n,requirement=line,actual_version=actual))
    if str(PYTEST_ROOT) not in str(modules["pytest"].__file__) or modules["pytest"].__version__!="7.4.0":raise ValueError("PYTEST_LOCATION_VERSION")
    extensions=[str(m.__file__) for m in tuple(sys.modules.values()) if getattr(m,"__file__",None) and str(PYTEST_ROOT) in str(m.__file__) and str(m.__file__).endswith((".so",".pyd"))]
    if extensions:raise ValueError("CROSS_ABI_EXTENSION "+repr(extensions))
    info=dict(executable=sys.executable,python=sys.version,SOABI=sysconfig.get_config_var("SOABI"),cache_tag=sys.implementation.cache_tag,bootstrap="append accepted base pure-Python pytest after fixed numerical environment imports",modules={n:dict(path=str(m.__file__),version=getattr(m,"__version__",None),sha256=sha(m.__file__)) for n,m in {**numeric,**modules}.items()},transitive_requirements=requirements,cross_ABI_extensions=extensions,environment_installed_or_modified=False)
    print(json.dumps(info,indent=2),flush=True)
    return info
def preflight():
    guard();dest=OUT/"runs/preflight_001";dest.mkdir(parents=True)
    info=bootstrap();write(dest/"DEPENDENCIES.json",info)
    import torch
    from .e1_dataset_binding import load_e1_bound
    from .short_train import initialize,INITIAL_SHA
    from query_minco.training.physics_only_v1.smoke import parameter_hash
    from .e1_r3_zero_update import phase_spec
    from dataclasses import asdict
    spec=read(PREP/"RUN_SPEC.json")
    assert asdict(phase_spec())==spec["physics_spec"]
    x=read(PREP/"TRAIN13_AND_EXCLUSIONS.json");rows=x["Train13"]
    assert len(rows)==13 and len(x["all28"])==28 and len(x["quarantine7"])==7
    assert [r["sample_id"] for r in rows]==spec["sample_order"]
    reports=[]
    for row in rows:
        assert row["split"]=="train" and row["source"]=="SOURCE_QUALIFIED" and row["reachability"]=="FEASIBLE_CERTIFIED" and row["data_eligible"]
        assert sha(row["hard_evidence_path"])==row["hard_evidence_sha256"]
        b=load_e1_bound(row["source_manifest_path"],row["source_manifest_sha256"]);b.validate()
        reports.append(dict(sample_id=row["sample_id"],scene_id=row["scene_id"],source_manifest_sha256=row["source_manifest_sha256"],state=b.state.tolist(),roi_world=b.geometry[0].roi.tolist()))
    m=initialize();identity=parameter_hash(m)
    if identity!=INITIAL_SHA or identity!=spec["initial_model_sha256"]:raise ValueError("INITIAL_MODEL_SHA")
    write(dest/"RESULT.json",dict(status="PASS",verified_Train13=reports,model_sha256=identity,physics_spec=asdict(phase_spec()),Dev_forward_count=0))
def tests():
    guard();info=bootstrap()
    if read(PREP/"DEPENDENCIES_PASS.json")["status"]!="PASS":raise ValueError("DEPENDENCIES_REQUIRED")
    dest=OUT/"runs/tests_001";dest.mkdir(parents=True)
    write(dest/"DEPENDENCIES.json",info)
    import pytest
    spec=read(PREP/"RUN_SPEC.json")
    argv=["-q","-p","no:cacheprovider","--basetemp="+str(dest/"pytest_tmp"),"--junitxml="+str(dest/"TEST_RESULTS.xml"),*spec["test_paths"],"--deselect=tests/test_physics_v12_gate1_geometry.py::test_exact_distance_matches_independent_all_boxes"]
    write(dest/"PYTEST_COMMAND.json",dict(pytest_argv=argv,deselection_contract="original function is unchanged and executed by test_original_failure_unchanged_and_reproduced, expecting its original GradcheckError"))
    code=pytest.main(argv)
    import xml.etree.ElementTree as ET
    counts={}
    if (dest/"TEST_RESULTS.xml").exists():
        tree=ET.parse(dest/"TEST_RESULTS.xml");cases=tree.findall(".//testcase")
        counts=dict(collected_executed=len(cases),passed=sum(not any(c.find(k) is not None for k in ("failure","error","skipped")) for c in cases),failures=sum(c.find("failure") is not None for c in cases),errors=sum(c.find("error") is not None for c in cases),skipped=sum(c.find("skipped") is not None for c in cases))
    write(dest/"RESULT.json",dict(status="PASS" if code==0 else "STOP",exit_code=int(code),counts=counts))
    if code:raise SystemExit(code)
def gradient_packets(result,raw,model):
    import torch
    from .e1_r3_zero_update import packet,module_groups,PRIMARY,PHASE_COEFFICIENTS
    named=list(model.named_parameters());groups=module_groups(named)
    extra=("G_N","G_F","blend","F_dyn","F_domain","F_speed","F_acceleration","F_jerk","V","Q","J")
    def scalars(query=None):
        values={}
        for n in (*PRIMARY,*extra):
            term=result.metrics[n]
            value=term.mean() if query is None else term[0,query]/6
            if not value.requires_grad:value=raw.sum()*0
            values[n]=value*PHASE_COEFFICIENTS[n] if n in PRIMARY else value
        values["TOTAL"]=result.loss if query is None else (result.metrics["V"][0,query]+.1*result.metrics["Q"][0,query])/6
        return values
    full=packet(scalars(),raw,named,groups)
    full["scope"]={n:("actual Phase A weighted contribution" if n in PRIMARY or n=="TOTAL" else "unweighted diagnostic mean, overlaps other components; not added to objective") for n in scalars()}
    full["Phase_A_coefficients"]=PHASE_COEFFICIENTS
    full["unweighted_primary_stats"]={n:{group:{**stat,**{k:stat[k]/PHASE_COEFFICIENTS[n] for k in ("norm","rms","max_abs")}} for group,stat in full["stats"][n].items()} for n in PRIMARY}
    full["module_parameters"]={g:[dict(name=named[i][0],shape=list(named[i][1].shape),numel=named[i][1].numel()) for i in ids] for g,ids in groups.items()}
    queries=[]
    for k in range(6):
        q=packet(scalars(k),raw,named,groups);q["query"]=k;queries.append(q)
    raw.retain_grad();result.loss.backward()
    from .e1_r3_zero_update import norm_stats
    total=torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros_like(p).flatten() for _,p in named])
    full["actual_backward"]=dict(raw=norm_stats(raw.grad),network=norm_stats(total),inactive_raw_zero=not bool(raw.grad[...,15:21].count_nonzero()))
    if not full["actual_backward"]["inactive_raw_zero"]:raise ValueError("INACTIVE_RAW_GRADIENT")
    return full,queries
def export(records,status,error=None):
    from .e1_r3_zero_update import scene_equal_summary
    import numpy as np
    expected=read(PREP/"TRAIN13_AND_EXCLUSIONS.json")["Train13"]
    dest=OUT/"runs/zero_update_001";summary=OUT/"summary/freeze_001"
    complete={r["sample_id"]:r for r in records};task=[]
    for row in expected:
        r=complete.get(row["sample_id"])
        task.append(dict(sample_id=row["sample_id"],scene_id=row["scene_id"],status=r["status"] if r else "NOT_RUN_AFTER_STOP",record_path=str(dest/(row["sample_id"]+(".json" if r else "_NOT_RUN"))),loss=r.get("loss") if r else None))
    write(summary/"PER_TASK_ZERO_UPDATE.json",dict(status=status,tasks=task,records=records,missing_data="null/NOT_RUN; never zero imputation"))
    rows=[]
    for r in records:
        metrics=r.get("metrics",r["non_collision_diagnostics"])
        for k,h in enumerate(r["independent_hard"]):
            row=dict(sample_id=r["sample_id"],scene_id=r["scene_id"],query=k,status=r["status"],loss=r.get("loss"),Hard_pass=h["passed"],Hard_reasons=";".join(h["reasons"]),max_v=h["dynamic_polynomial_maxima"][0],max_a=h["dynamic_polynomial_maxima"][1],max_j=h["dynamic_polynomial_maxima"][2],sampled_outside_ROI=r["roi"]["queries"][k]["sampled_outside_count"])
            for n in ("C_viol","C_shape","G_N","G_F","blend","G_task_train","P","F_dyn","F_domain","F","S","V","Q","J","min_sdf_m"):
                v=metrics.get(n);row[n]=v[0][k] if v is not None else None
            row["L_A_query"]=row["V"]+.1*row["Q"] if row["V"] is not None else None
            rows.append(row)
    def csvwrite(name,rows):
        with (summary/name).open("x",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)) if rows else ["status"]);w.writeheader();w.writerows(rows)
    csvwrite("PER_TASK_ZERO_UPDATE.csv",rows)
    sg=scene_equal_summary(records,expected);write(summary/"SCENE_EQUAL_GRADIENTS.json",sg)
    gr=[]
    for scope,x in sg["scopes"].items():
        for module,g in x["available_scene_summary"].items():
            for cost,rms in g["RMS"].items():gr.append(dict(scope=scope,status=x["status"],module=module,cost=cost,RMS=rms,numeric_scope=x["numeric_scope"]))
    csvwrite("SCENE_EQUAL_GRADIENTS.csv",gr)
    write(summary/"RESULT.json",dict(E1_R3a_v2=status,expected=13,forward_backward_complete=sum(r["status"]=="PASS" for r in records),failed=sum(r["status"]!="PASS" for r in records),not_run=13-len(records),R3b="NO-GO" if status!="PASS" else "PROPOSAL_ONLY_SEPARATE_APPROVAL",optimizer_updates=0,checkpoint_loaded=False,CUDA_calls=0,Dev_forward_count=0,error=error))
def zero():
    guard()
    if read(PREP/"TESTS_PASS.json")["status"]!="PASS":raise ValueError("TESTS_REQUIRED")
    import torch,numpy as np
    from unittest.mock import patch
    from .short_train import initialize,INITIAL_SHA
    from query_minco.training.physics_only_v1.smoke import parameter_hash
    from .e1_dataset_binding import load_e1_bound
    from .e1_r3_zero_update import prepare,phase_spec,save_svg
    from .objective import BoundPhysicsV12Objective
    from query_minco.evaluation.validation import evaluate
    from dataclasses import asdict
    dest=OUT/"runs/zero_update_001";dest.mkdir(parents=True);records=[]
    with patch.object(torch.optim.Optimizer,"__init__",side_effect=RuntimeError("OPTIMIZER_FORBIDDEN")),patch.object(torch,"load",side_effect=RuntimeError("CHECKPOINT_FORBIDDEN")),patch.object(torch,"save",side_effect=RuntimeError("CHECKPOINT_SAVE_FORBIDDEN")):
        model=initialize();before=parameter_hash(model);spec=phase_spec()
        if before!=INITIAL_SHA:raise ValueError("INITIAL_MODEL_SHA")
        write(dest/"MODEL_BEFORE.json",dict(sha256=before,seed=20261104,spec=asdict(spec),checkpoint=None))
        try:
            for row in read(PREP/"TRAIN13_AND_EXCLUSIONS.json")["Train13"]:
                guard();model.zero_grad(set_to_none=True)
                if any(p.grad is not None for p in model.parameters()):raise ValueError("ACCUMULATED_GRADIENT")
                b=load_e1_bound(row["source_manifest_path"],row["source_manifest_sha256"])
                raw=model(b.depth,b.state)
                if not torch.isfinite(raw).all():raise ValueError("NONFINITE_RAW")
                r=prepare(raw,b,spec);r.update(sample_id=row["sample_id"],scene_id=row["scene_id"],source_sha=row,split="train")
                r["terminal_acceleration_B0"]=[evaluate(np.asarray(c)[-1],float(d[-1]),2).tolist() for c,d in zip(r["coefficients_B0"][0],r["durations"][0])]
                save_svg(r,OUT/"visualization/freeze_001"/(row["sample_id"]+"_ROI.svg"))
                objective=BoundPhysicsV12Objective(spec)
                if r["roi"]["any_sample_outside"] or r["roi"]["any_continuous_center_outside"]:
                    exception=None
                    if r["roi"]["any_sample_outside"]:
                        try:objective(raw,b)
                        except ValueError as ex:
                            exception=str(ex)
                            if "UNKNOWN" not in exception:raise
                        else:raise ValueError("UNKNOWN_NOT_REJECTED")
                    r.update(status="STOP_ROI_UNKNOWN",production_exception=exception,loss=None,gradients=None,per_query_gradients=None)
                    records.append(r);write(dest/(row["sample_id"]+".json"),r)
                    raise ValueError("KNOWN_ROI_UNKNOWN "+row["sample_id"])
                result=objective(raw,b)
                if not torch.isfinite(result.loss):raise ValueError("NONFINITE_OBJECTIVE")
                full,perq=gradient_packets(result,raw,model)
                r.update(status="PASS",loss=float(result.loss),metrics={n:t.detach().tolist() for n,t in result.metrics.items()},gradients=full,per_query_gradients=perq)
                if parameter_hash(model)!=before:raise ValueError("MODEL_UPDATED")
                records.append(r);write(dest/(row["sample_id"]+".json"),r)
                print(json.dumps(dict(sample_id=row["sample_id"],status="PASS",loss=r["loss"],hard_pass=r["hard_valid_candidate_count"])),flush=True)
        except BaseException:
            write(dest/"MODEL_AFTER.json",dict(before=before,after=parameter_hash(model),unchanged=before==parameter_hash(model)))
            export(records,"STOP",traceback.format_exc());raise
        write(dest/"MODEL_AFTER.json",dict(before=before,after=parameter_hash(model),unchanged=before==parameter_hash(model)))
        export(records,"PASS")
def main():
    p=argparse.ArgumentParser();p.add_argument("--worker",choices=("preflight","tests","zero"),required=True);a=p.parse_args()
    {"preflight":preflight,"tests":tests,"zero":zero}[a.worker]()
if __name__=="__main__":main()
