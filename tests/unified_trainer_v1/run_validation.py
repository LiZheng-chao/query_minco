"""Bounded validation orchestrator; exactly one disposable neural optimizer update."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from query_minco.training.io import write, read, sha, require, digest
from query_minco.training.run_spec import ROOT, identity

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir()
    commands=[]
    def command(name, argv):
        d=out/name;d.mkdir()
        write(d/"COMMAND.json",dict(argv=argv))
        with (d/"stdout.log").open("xb") as o,(d/"stderr.log").open("xb") as e:
            rc=subprocess.run(argv,cwd=str(ROOT),stdout=o,stderr=e,env={
                **os.environ,"PYTHONPATH":str(ROOT),"PYTHONDONTWRITEBYTECODE":"1","CUDA_VISIBLE_DEVICES":""}).returncode
        write(d/"EXIT.json",dict(returncode=rc))
        commands.append(dict(name=name,returncode=rc,path=str(d)))
        return rc,d
    def sandbox(name, mode, trained=None):
        writable=out/(name+"_workspace");writable.mkdir()
        (writable/"tmp").mkdir()
        cp=Path(read(config)["initialization"]["artifact"]["path"])
        fd=os.open(str(cp),os.O_RDONLY|os.O_CLOEXEC)
        argv=["/usr/bin/bwrap","--die-with-parent","--new-session","--unshare-net","--ro-bind","/","/",
              "--proc","/proc","--dev","/dev","--tmpfs","/home/lizheng/reach_planner",
              "--ro-bind-fd",str(fd),str(cp),"--remount-ro","/home/lizheng/reach_planner",
              "--bind",str(writable),str(writable),"--setenv","PATH","/usr/bin:/bin",
              "--setenv","HOME",str(writable),"--setenv","TMPDIR",str(writable/"tmp"),
              "--setenv","PYTHONPATH",str(ROOT),"--setenv","CUDA_VISIBLE_DEVICES","",
              "--chdir",str(ROOT),sys.executable,"-B","-s","tests/unified_trainer_v1/test_suite.py",
              "--config",str(config),"--output",str(writable/"result"),"--mode",mode]
        if trained:argv += ["--trained-run",str(trained)]
        d=out/name;d.mkdir();write(d/"COMMAND.json",dict(argv=argv,source=str(cp)))
        try:
            with (d/"stdout.log").open("xb") as o,(d/"stderr.log").open("xb") as e:
                rc=subprocess.run(argv,cwd=str(ROOT),stdout=o,stderr=e,pass_fds=(fd,),env={"PATH":"/usr/bin:/bin","LC_ALL":"C.UTF-8"}).returncode
        finally:os.close(fd)
        write(d/"EXIT.json",dict(returncode=rc))
        commands.append(dict(name=name,returncode=rc,path=str(d)))
        require(rc==0,"TEST_FAILED "+name)
        require(read(writable/"result/RESULT.json")["passed"],"CONTENT_TEST_FAILED")
    try:
        rc,d=command("prepare", [sys.executable,"-B","-s","tests/unified_trainer_v1/prepare.py",
                                "--output",str(out/"prepared")])
        require(rc==0,"PREPARE_FAILED")
        config=out/"prepared/config.json"
        sandbox("unit001","unit")
        sandbox("numerical001","numerical")
        cli=[sys.executable,"-B","-s","-m","query_minco.training.cli"]
        rc,d=command("plan",cli+["plan","--config",str(config)]);require(rc==0,"PLAN_FAILED")
        plan=json.loads((d/"stdout.log").read_text())
        require(not plan["training_started"],"PLAN_TRAINED")
        pf=out/"prepared/runs/preflight001"
        rc,d=command("preflight",cli+["preflight","--config",str(config),"--output",str(pf)])
        require(rc==0,"PREFLIGHT_FAILED")
        require(read(pf/"PREFLIGHT.json")["optimizer_updates"]==0,"PREFLIGHT_TRAINED")
        auth=plan["authorization_template"]
        denied=out/"denied_authorization.json";write(denied,auth)
        rc,d=command("reject_unapproved",cli+["train","--config",str(config),
                 "--authorization",str(denied),"--preflight",str(pf/"PREFLIGHT.json")])
        require(rc!=0 and not (out/"prepared/runs/disposable_step001").exists(),"UNAPPROVED_RUN_STARTED")
        auth.update(approved=True,authorized_by="user-authorized disposable one-step fixture")
        granted=out/"synthetic_one_step_authorization.json";write(granted,auth)
        rc,d=command("one_step",cli+["train","--config",str(config),"--authorization",str(granted),
                                   "--preflight",str(pf/"PREFLIGHT.json")])
        require(rc==0,"ONE_STEP_FAILED")
        run=out/"prepared/runs/disposable_step001"
        rc,d=command("audit",cli+["audit","--run-dir",str(run)]);require(rc==0,"AUDIT_FAILED")
        sandbox("restore001","restore",run)
        rc,d=command("reject_overwrite",cli+["train","--config",str(config),"--authorization",str(granted),
                                             "--preflight",str(pf/"PREFLIGHT.json")])
        require(rc!=0,"RUN_OVERWRITE_ACCEPTED")
        write(out/"RESULT.json",dict(passed=True,commands=commands,disposable_optimizer_updates=1,
                                    formal_training_updates=0,formal_evaluations=0))
    except BaseException as e:
        import traceback
        write(out/"RESULT.json",dict(passed=False,error=repr(e),traceback=traceback.format_exc(),commands=commands))
        raise
    print(json.dumps(read(out/"RESULT.json"),indent=2))
if __name__=="__main__":main()
