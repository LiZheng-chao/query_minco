"""Failure/lifecycle regressions: no optimizer.step and no formal training."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import traceback
from unittest.mock import patch
from query_minco.training.io import write, read, require, sha
from query_minco.training.run_spec import ROOT, identity, source_files
from query_minco.training.audit import audit_run
from query_minco.training.isolation import launch

def main():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--completed-run",required=True)
    p.add_argument("--output",required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir();results=[]
    s=read(a.config);s["code"]["files"]=source_files();s["output_root"]=str(out)
    config=out/"current_config.json";write(config,s)
    def test(name,fn):
        try:details=fn() or {};r=dict(name=name,passed=True,details=details)
        except BaseException as e:r=dict(name=name,passed=False,error=repr(e),traceback=traceback.format_exc())
        write(out/(name+".json"),r);results.append(r)
    def rejects(fn):
        try:fn()
        except (ValueError,FileNotFoundError,FileExistsError):return
        raise AssertionError("INVALID_ACCEPTED")
    def wrong_source():
        cfg=copy.deepcopy(s);cfg["code"]["files"]["query_minco/models/planner.py"]="0"*64
        p=out/"wrong_source.json";write(p,cfg)
        from query_minco.training.run_spec import load_spec
        rejects(lambda:load_spec(p))
        return dict(refused=True)
    test("wrong_source_SHA",wrong_source)
    def fake_success():
        r=out/"exit0_incomplete";r.mkdir()
        for n,value in [("RUN_SPEC.json",s),("PROCESS_EXIT.json",{"returncode":0}),
                        ("PROTECTION_BEFORE.json",{"passed":True,"files":{}}),
                        ("PROTECTION_AFTER.json",{"passed":True,"files":{}}),
                        ("ISOLATION.json",{"passed":True}),("WORKER_RESULT.json",
                         {"status":"WORKER_FINISHED_AWAITING_AUDIT","completed_optimizer_updates":0})]:
            write(r/n,value)
        rejects(lambda:audit_run(r,candidate=True))
        return dict(exit0_does_not_mean_completed=True)
    test("exit0_without_budget_rejected",fake_success)
    def external_failure():
        # Synthetic patch changes only the comparator's second snapshot; no history file is touched.
        import query_minco.training.isolation as isolation
        original=isolation.protection;count=[0]
        def injected(paths):
            count[0]+=1;v=original(paths)
            if count[0]==2:v["injected_test_only"]="different"
            return v
        r=out/"external_failure"
        with patch.object(isolation,"protection",injected):
            rejects(lambda:launch(s,config,r,"preflight"))
        require(read(r/"PROCESS_EXIT.json")["returncode"]==0 and
                not read(r/"PROTECTION_AFTER.json")["passed"] and
                (r/"STOP.json").exists() and not (r/"RESULT.json").exists(),"EXTERNAL_FAILURE_NOT_STOPPED")
        return dict(optimizer_updates=0,worker_exit=0,parent_stopped=True)
    test("external_protection_failure_blocks_completion",external_failure)
    def real_worker_failure():
        cfg=copy.deepcopy(s);cfg["initialization"]["weights_sha256"]="0"*64
        p=out/"wrong_state.json";write(p,cfg)
        r=out/"worker_failure"
        rejects(lambda:launch(cfg,p,r,"preflight"))
        require(read(r/"PROCESS_EXIT.json")["returncode"]!=0 and (r/"STOP.json").exists() and
                (r/"stderr.log").stat().st_size>0 and read(r/"PROTECTION_AFTER.json")["passed"] and
                not (r/"RESULT.json").exists() and "INITIALIZATION_STATE_SHA" in (r/"stderr.log").read_text(),"FAILURE_SCENE")
        return dict(real_process_failed=True,stderr_preserved=True,external_after_ran=True,optimizer_updates=0)
    test("real_worker_failure_preserved",real_worker_failure)
    def trained_audit():
        return audit_run(a.completed_run)
    test("existing_disposable_run_still_valid",trained_audit)
    # Partial checkpoint refusal and ERROR output are exercised without running model forward.
    def loop_failure():
        from query_minco.training import loop
        r=out/"loop_failure";r.mkdir()
        fake=type("Binding",(),{"model_factory":None,"objective":staticmethod(lambda *x,**k:(_ for _ in ()).throw(
            RuntimeError("injected before model forward"))),"optimizer_factory":staticmethod(loop.optimizer)})()
        with patch.object(loop,"components",return_value=fake):
            try:loop.execute(s,r)
            except RuntimeError as e:require("injected" in str(e),"WRONG_FAILURE")
            else:raise AssertionError("INJECTED_FAILURE_NOT_RAISED")
        require(read(r/"ERROR.json")["completed_optimizer_updates"]==0 and
                (r/"FAILED_STATE_NOT_A_PRESET_CHECKPOINT.pt.json").exists() and
                not (r/"RESULT.json").exists(),"LOOP_ERROR_RECORD")
        return dict(actual_model_forward_calls=0,optimizer_updates=0,failed_state_retained=True)
    test("loop_exception_retains_actual_zero_updates",loop_failure)
    passed=all(r["passed"] for r in results)
    write(out/"RESULT.json",dict(passed=passed,tests=results,optimizer_updates=0))
    print(json.dumps(dict(passed=passed,tests=len(results))))
    return 0 if passed else 1
if __name__=="__main__":raise SystemExit(main())
