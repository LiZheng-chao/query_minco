"""Protect every file predating P2C-1A, with no modification exceptions."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from query_minco.p2b.utf8_io import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument("--experiment-dir",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    method=Path(__file__).resolve().parents[1];root=method.parent
    a.experiment_dir.resolve().relative_to(method/"outputs");a.output.resolve().relative_to(a.experiment_dir.resolve())
    before=json.loads((a.experiment_dir/"audit/stage_before.json").read_text())
    changed=[];missing=[]
    for name,saved in before["files"].items():
        path=root/name
        if not path.is_file():missing.append(name);continue
        current={"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"size":path.stat().st_size,"mtime_ns":path.stat().st_mtime_ns}
        if current!=saved:changed.append({"path":name,"before":saved,"after":current})
    env=dict(os.environ,GIT_OPTIONAL_LOCKS="0")
    def git(*args):return subprocess.check_output(["git","-C",str(root)]+list(args),text=True,env=env)
    head=git("rev-parse","HEAD").strip();branch=git("branch","--show-current").strip()
    outside=lambda text:[line for line in text.splitlines() if not line.startswith("?? query_minco/")]
    result={"existing_query_files":len(before["files"]),"sha256_checked":len(before["files"]),
        "changed":changed,"missing":missing,"authorized_existing_modifications":[],
        "head":head,"branch":branch,"head_unchanged":head==before["head"],"branch_unchanged":branch==before["branch"],
        "existing_git_status_unchanged":outside(git("status","--porcelain=v1","-uall"))==outside(before["git_status"]),
        "tracked_diff":git("diff","--name-status"),"staged_diff":git("diff","--cached","--name-status")}
    result["passed"]=(not changed and not missing and result["head_unchanged"] and result["branch_unchanged"]
        and result["existing_git_status_unchanged"] and not result["tracked_diff"] and not result["staged_diff"])
    write_json(a.output,result);print(json.dumps(result,indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__=="__main__":main()
