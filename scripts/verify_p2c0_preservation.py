"""Read-only SHA256/size/mtime protection of all files predating P2C-0."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from query_minco.p2b.utf8_io import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();method=Path(__file__).resolve().parents[1];root=method.parent
    args.output.resolve().relative_to(method)
    before=json.loads((method/"_p2c0_artifacts/stage_before.json").read_text())
    missing=[];changed=[]
    for name,saved in before["files"].items():
        path=root/name
        if not path.is_file():missing.append(name);continue
        current={"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                 "size":path.stat().st_size,"mtime_ns":path.stat().st_mtime_ns}
        if current!=saved:changed.append({"path":name,"before":saved,"after":current})
    env=dict(os.environ,GIT_OPTIONAL_LOCKS="0")
    def git(*a):return subprocess.check_output(["git","-C",str(root)]+list(a),text=True,env=env)
    outside=lambda s:[line for line in s.splitlines() if not line.startswith("?? query_minco/")]
    head=git("rev-parse","HEAD").strip();branch=git("branch","--show-current").strip()
    status=git("status","--porcelain=v1","-uall")
    result={"existing_query_files":len(before["files"]),"sha256_checked":len(before["files"]),
            "changed":changed,"missing":missing,"head":head,"branch":branch,
            "head_unchanged":head==before["head"],"branch_unchanged":branch==before["branch"],
            "existing_git_status_outside_query_minco_unchanged":outside(status)==outside(before["git_status"]),
            "tracked_diff":git("diff","--name-status"),"staged_diff":git("diff","--cached","--name-status"),
            "authorized_existing_modifications":[]}
    result["passed"]=(not changed and not missing and result["head_unchanged"] and result["branch_unchanged"]
        and result["existing_git_status_outside_query_minco_unchanged"] and not result["tracked_diff"] and not result["staged_diff"])
    write_json(args.output,result);print(json.dumps(result,indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__=="__main__":main()
