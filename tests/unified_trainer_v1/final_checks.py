"""Final bounded checks of current source. No optimizer updates."""
import argparse
import json
from pathlib import Path
import subprocess
import os
import sys
from query_minco.training.io import write, read, require
from query_minco.training.run_spec import ROOT

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",required=True);p.add_argument("--completed-run",required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir();commands=[]
    def command(name,argv,fd=None):
        d=out/name;d.mkdir();write(d/"COMMAND.json",dict(argv=argv))
        with (d/"stdout.log").open("xb") as o,(d/"stderr.log").open("xb") as e:
            rc=subprocess.run(argv,cwd=str(ROOT),stdout=o,stderr=e,
                              pass_fds=() if fd is None else (fd,),env={
                                  "PATH":"/usr/bin:/bin","LC_ALL":"C.UTF-8","PYTHONPATH":str(ROOT),
                                  "PYTHONDONTWRITEBYTECODE":"1","CUDA_VISIBLE_DEVICES":""}).returncode
        write(d/"EXIT.json",dict(returncode=rc));commands.append(dict(name=name,returncode=rc))
        require(rc==0,"FINAL_CHECK_FAILED "+name)
    command("prepare",[sys.executable,"-B","-s","tests/unified_trainer_v1/prepare.py","--output",str(out/"prepared")])
    config=out/"prepared/config.json";cp=read(config)["initialization"]["artifact"]["path"]
    for mode in ("unit","numerical"):
        work=out/(mode+"_workspace");work.mkdir()
        fd=os.open(cp,os.O_RDONLY|os.O_CLOEXEC)
        cmd=["/usr/bin/bwrap","--die-with-parent","--unshare-net","--ro-bind","/","/","--proc","/proc","--dev","/dev",
             "--tmpfs","/home/lizheng/reach_planner","--ro-bind-fd",str(fd),cp,"--remount-ro","/home/lizheng/reach_planner",
             "--bind",str(work),str(work),"--setenv","PYTHONPATH",str(ROOT),"--setenv","TMPDIR",str(work),
             "--setenv","HOME",str(work),"--chdir",str(ROOT),sys.executable,"-B","-s",
             "tests/unified_trainer_v1/test_suite.py","--config",str(config),"--output",str(work/"result"),"--mode",mode]
        try:command(mode,cmd,fd)
        finally:os.close(fd)
        require(read(work/"result/RESULT.json")["passed"],"TEST_CONTENT")
    pf=out/"prepared/runs/preflight001"
    command("preflight",[sys.executable,"-B","-s","-m","query_minco.training.cli","preflight",
                         "--config",str(config),"--output",str(pf)])
    require(read(pf/"PREFLIGHT.json")["gradient_gate"]["passed"] and
            read(pf/"PREFLIGHT.json")["optimizer_updates"]==0,"PREFLIGHT_CONTENT")
    command("failure_tests",[sys.executable,"-B","-s","tests/unified_trainer_v1/failure_tests.py",
                "--config",str(config),"--completed-run",a.completed_run,"--output",str(out/"failure_results")])
    require(read(out/"failure_results/RESULT.json")["passed"],"FAILURE_TEST_CONTENT")
    write(out/"RESULT.json",dict(passed=True,commands=commands,optimizer_updates=0,formal_runs=0))
    print(json.dumps(read(out/"RESULT.json"),indent=2))
if __name__=="__main__":main()
