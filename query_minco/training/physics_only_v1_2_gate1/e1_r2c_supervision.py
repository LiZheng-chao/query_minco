"""Outer worker supervision covers Python qualification driver plus all descendants."""
import json
import hashlib
import time
import os
from pathlib import Path
from .e1_resource_supervisor import supervise

def supervised_worker(argv,folder,*,cwd,env,stop_path,budget_root,guard=None,timeout=900.,rss_limit=4294967296):
    # The Python driver is the monitored root, rather than an unsupervised caller.
    result=supervise(argv,folder,"driver",cwd=cwd,env=env,stop_path=stop_path,budget_root=budget_root,
        guard=guard,timeout=timeout,rss_limit=rss_limit,disk_limit=536870912,interval=.05)
    result_scope=dict(monitored_root="qualification Python worker",includes="worker imports, geometry construction, CPU checks and all observed descendants",
        driver_RSS_peak_included=True,sampled_not_continuous=True,child_CUDA_supervisor="existing unchanged GPU-capable supervisor retained; no CUDA in this audit")
    from .e1_resource_supervisor import write
    write(Path(folder)/"DRIVER_SCOPE.json",result_scope)
    return result


def guard_inputs(root):
    base=Path(__file__).resolve().parents[3];prep=Path(root)/"_preparation/review_001"
    if (prep/"STOP.json").exists():raise ValueError("PERSISTENT_STOP")
    e=json.loads((prep/"ENTRY.json").read_text())
    for p,h in e["protected_sha256"].items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h:raise ValueError("HISTORICAL_SOURCE_DRIFT")
    for p,h in json.loads((prep/"SOURCE_FREEZE.json").read_text()).items():
        if hashlib.sha256((base/p).read_bytes()).hexdigest()!=h:raise ValueError("NEW_SOURCE_DRIFT")
    if time.time()-e["started_unix"]>=1800:raise ValueError("TOTAL_WALL_BUDGET")
