"""Compare P1 pre-state with existing files; writes only a new method receipt."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    method = Path(__file__).resolve().parents[1]
    root = method.parent
    args.output.resolve().relative_to(method)
    with gzip.open(method / "_p1_artifacts/baseline_before.json.gz", "rt", encoding="utf-8") as stream:
        before = json.load(stream)
    changed, missing = [], []
    for name, saved in before["files"].items():
        path = root / name
        try:
            stat = path.lstat()
        except FileNotFoundError:
            missing.append(name)
            continue
        fields = []
        if stat.st_size != saved["size"]:
            fields.append("size")
        if stat.st_mtime_ns != saved["mtime_ns"]:
            fields.append("mtime_ns")
        if "link" in saved and os.readlink(path) != saved["link"]:
            fields.append("symlink_target")
        if "sha256" in saved and hashlib.sha256(path.read_bytes()).hexdigest() != saved["sha256"]:
            fields.append("sha256")
        if fields:
            changed.append({"path": name, "changed": fields})
    unexpected = []
    for base, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = [d for d in dirs if d != ".git" and not (Path(base) == root and d == "query_minco")]
        for name in names:
            relative = (Path(base) / name).relative_to(root).as_posix()
            if relative not in before["files"]:
                unexpected.append(relative)
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    def git(*arguments):
        return subprocess.check_output(["git", "-C", str(root)] + list(arguments), env=env, text=True)
    after_status = git("status", "--porcelain=v1", "-uall")
    existing_status = "".join(line for line in after_status.splitlines(keepends=True)
                              if not line.startswith("?? query_minco/"))
    result = {
        "head_before": before["head"], "head_after": git("rev-parse", "HEAD").strip(),
        "branch_before": before["branch"], "branch_after": git("branch", "--show-current").strip(),
        "existing_files": len(before["files"]),
        "sha256_checked": sum("sha256" in row for row in before["files"].values()),
        "hash_policy": before["hash_policy"], "changed": changed, "missing": missing,
        "unexpected_outside_query_minco": unexpected,
        "existing_git_status_unchanged": existing_status == before["git_status"],
        "tracked_diff": git("diff", "--name-status"),
        "staged_diff": git("diff", "--cached", "--name-status"),
        "new_method_files": sorted(p.relative_to(root).as_posix()
                                   for p in method.rglob("*") if p.is_file())}
    result["passed"] = (not changed and not missing and not unexpected
                        and result["head_before"] == result["head_after"]
                        and result["branch_before"] == result["branch_after"]
                        and result["existing_git_status_unchanged"])
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "new_method_files"}, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
