"""The monitored root is the Python qualification driver, including its own RSS."""
import json
import os
from pathlib import Path
import sys
import time
import hashlib
import pytest
from query_minco.training.physics_only_v1_2_gate1.e1_r2c_supervision import supervised_worker,guard_inputs
from query_minco.training.physics_only_v1_2_gate1.e1_resource_supervisor import proc_records

def run(tmp_path,code,**kw):
    return supervised_worker([sys.executable,"-B","-c",code],tmp_path/"worker",cwd=tmp_path,
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1",CUDA_VISIBLE_DEVICES=""),stop_path=tmp_path/"STOP.json",budget_root=tmp_path,**kw)
def receipt(tmp_path):return json.loads((tmp_path/"worker/driver_EXIT.json").read_text())
def inactive(pid):
    x=proc_records().get(pid)
    return x is None or x["state"] in ("Z","X")

def test_driver_own_allocation_is_included_and_enforced(tmp_path):
    with pytest.raises(RuntimeError):
        run(tmp_path,"import os,time;print(os.getpid(),flush=True);x=bytearray(80*1024*1024);time.sleep(10)",rss_limit=48*1024*1024,timeout=3)
    r=receipt(tmp_path);pid=int((tmp_path/"worker/driver_stdout.log").read_text())
    assert r["stop"]=="CPU_TREE_RSS_BUDGET" and r["sampled_peak_RSS_bytes"]>48*1024*1024
    assert str(pid) in r["observed_processes"] and inactive(pid)
    assert not r["live_descendants_after_cleanup"] and (tmp_path/"STOP.json").exists()

def test_driver_and_descendant_are_both_measured(tmp_path):
    code="import subprocess,sys,time; x=bytearray(32*1024*1024);p=subprocess.Popen([sys.executable,'-c','import time;x=bytearray(32*1024*1024);time.sleep(10)']);print(p.pid,flush=True);time.sleep(10)"
    with pytest.raises(RuntimeError):run(tmp_path,code,rss_limit=64*1024*1024,timeout=5)
    r=receipt(tmp_path);pid=int((tmp_path/"worker/driver_stdout.log").read_text())
    assert r["stop"]=="CPU_TREE_RSS_BUDGET" and len(r["observed_processes"])>=2
    assert inactive(pid) and not r["live_descendants_after_cleanup"]

def test_success_scope_receipt_includes_driver(tmp_path):
    r=run(tmp_path,"import time;x=bytearray(8*1024*1024);time.sleep(.15)")
    assert r["status"]=="PASS"
    scope=json.loads((tmp_path/"worker/DRIVER_SCOPE.json").read_text())
    assert scope["driver_RSS_peak_included"] and scope["sampled_not_continuous"]

def test_driver_exception_timeout_and_stop_preserve_evidence(tmp_path):
    with pytest.raises(RuntimeError):run(tmp_path,"raise RuntimeError('driver_failure')")
    r=receipt(tmp_path);assert r["exit_code"]==1
    assert "driver_failure" in (tmp_path/"worker/driver_stderr.log").read_text()
    first=(tmp_path/"STOP.json").read_bytes()
    with pytest.raises(RuntimeError):
        supervised_worker([sys.executable,"-c","print('bad')"],tmp_path/"blocked",cwd=tmp_path,env=os.environ.copy(),
            stop_path=tmp_path/"STOP.json",budget_root=tmp_path)
    assert (tmp_path/"STOP.json").read_bytes()==first
    assert json.loads((tmp_path/"blocked/driver_EXIT.json").read_text())["child_started"] is False

def test_driver_timeout(tmp_path):
    with pytest.raises(RuntimeError):run(tmp_path,"import time;time.sleep(10)",timeout=.2)
    assert receipt(tmp_path)["stop"]=="SUBPROCESS_TIME_BUDGET"

def test_readonly_source_and_new_stop_guard(tmp_path):
    prep=tmp_path/"_preparation/review_001";prep.mkdir(parents=True)
    source=tmp_path/"immutable";source.write_bytes(b"old")
    (prep/"ENTRY.json").write_text(json.dumps(dict(started_unix=time.time(),protected_sha256={str(source):hashlib.sha256(b"old").hexdigest()})))
    (prep/"SOURCE_FREEZE.json").write_text("{}")
    guard_inputs(tmp_path)
    source.write_bytes(b"changed")
    with pytest.raises(ValueError,match="HISTORICAL"):guard_inputs(tmp_path)
    (prep/"STOP.json").write_text("{}")
    with pytest.raises(ValueError,match="PERSISTENT_STOP"):guard_inputs(tmp_path)
