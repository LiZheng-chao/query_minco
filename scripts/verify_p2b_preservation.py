"""Verify frozen P1/P2A files and native data, allowing only documented P2A text repair."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from query_minco.p2b.utf8_io import write_json


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root):
    method=root/"query_minco"
    before=json.loads((method/"_p2b_artifacts/stage_before.json").read_text(encoding="utf-8"))
    repair=json.loads((method/"_p2b_artifacts/encoding_repair.json").read_text(encoding="utf-8"))
    allowed=before["authorized_document_repair"]
    changed=[];missing=[];authorized=[]
    for name,saved in before["query_files"].items():
        path=root/name
        if not path.is_file():
            missing.append(name);continue
        current={"sha256":digest(path),"size":path.stat().st_size,"mtime_ns":path.stat().st_mtime_ns}
        if current!=saved:
            backup=method/"_p2b_artifacts/encoding_backup/P2A_IMPLEMENTATION_REPORT.md.original_bytes"
            if (name==allowed and saved["sha256"]==repair["original_sha256"]
                    and current["sha256"]==repair["repaired_sha256"]
                    and digest(backup)==repair["original_sha256"]
                    and repair["round_trip_equal"] and repair["repaired_chinese_chars"]>0):
                authorized.append(name)
            else:
                changed.append({"path":name,"before":saved,"after":current})
    audit=json.loads((method/"_p2b_artifacts/data_audit_v1/audit_summary.json").read_text())
    changed_data=[name for name,sha in audit["input_sha256"].items()
                  if not (root/name).is_file() or digest(root/name)!=sha]
    env=dict(os.environ,GIT_OPTIONAL_LOCKS="0")
    def git(*args):
        return subprocess.check_output(["git","-C",str(root)]+list(args),text=True,env=env)
    status=git("status","--porcelain=v1","-uall")
    outside=lambda value: [line for line in value.splitlines() if not line.startswith("?? query_minco/")]
    head=git("rev-parse","HEAD").strip()
    result={"frozen_query_files":len(before["query_files"]),
            "unchanged_query_files":len(before["query_files"])-len(changed)-len(missing)-len(authorized),
            "authorized_document_repairs":authorized,"unexpected_changed":changed,"missing":missing,
            "native_source_hashes_checked":len(audit["input_sha256"]),
            "native_source_changed":changed_data,"head_before":before["head"],"head_after":head,
            "existing_git_status_unchanged":outside(status)==outside(before["git_status"]),
            "tracked_diff":git("diff","--name-status"),"staged_diff":git("diff","--cached","--name-status")}
    result["passed"]=(not changed and not missing and not changed_data
                      and result["existing_git_status_unchanged"] and head==before["head"]
                      and not result["tracked_diff"] and not result["staged_diff"])
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    method=Path(__file__).resolve().parents[1]
    args.output.resolve().relative_to(method)
    result=verify(method.parent)
    write_json(args.output,result)
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__=="__main__":main()
