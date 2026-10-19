"""Verify every P1 file and previous Git status; write a new P2A receipt."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    method=Path(__file__).resolve().parents[1]; root=method.parent
    args.output.resolve().relative_to(method/"_p2a_artifacts")
    before=json.loads((method/"_p2a_artifacts/p1_before.json").read_text())
    changes=[]
    for name,saved in before["p1_files"].items():
        p=root/name
        if not p.is_file():
            changes.append({"path":name,"reason":"missing"}); continue
        st=p.stat()
        if st.st_size!=saved["size"] or st.st_mtime_ns!=saved["mtime_ns"] or hashlib.sha256(p.read_bytes()).hexdigest()!=saved["sha256"]:
            changes.append({"path":name,"reason":"size/mtime/hash changed"})
    env=dict(os.environ,GIT_OPTIONAL_LOCKS="0")
    status=subprocess.check_output(["git","-C",str(root),"status","--porcelain=v1","-uall"],env=env,text=True)
    previous=set(before["git_status"].splitlines()); current=set(status.splitlines())
    removed=sorted(previous-current); added=sorted(current-previous)
    outside=[x for x in added if not x.startswith("?? query_minco/")]
    head=subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()
    result={"p1_files_checked":len(before["p1_files"]),"p1_changed":changes,
            "previous_git_entries_removed_or_changed":removed,"new_git_entries":added,
            "unexpected_git_entries":outside,"head_unchanged":head==before["head"]}
    result["passed"]=not changes and not removed and not outside and result["head_unchanged"]
    with args.output.open("x",encoding="utf-8") as f: json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!="new_git_entries"},indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__=="__main__":
    main()
