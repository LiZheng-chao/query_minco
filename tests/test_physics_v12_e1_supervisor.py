"""Real subprocess resource/STOP tests with isolated expected-failure latches."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import pytest
from query_minco.training.physics_only_v1_2_gate1 import e1_resource_supervisor as s

def run(tmp_path,code,**kw):
    return s.supervise([sys.executable,"-B","-c",code],tmp_path/"run","child",cwd=tmp_path,env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1"),
       stop_path=tmp_path/"STOP.json",budget_root=tmp_path,**kw)
def receipt(tmp_path):return json.loads((tmp_path/"run/child_EXIT.json").read_text())
def inactive(pid):
    record=s.proc_records().get(pid)
    return record is None or record["state"] in ("Z","X")

def test_real_success_stdout_stderr_and_exit(tmp_path):
    r=run(tmp_path,"import sys;print('out');print('err',file=sys.stderr)")
    assert r["status"]=="PASS" and r["exit_code"]==0
    assert (tmp_path/"run/child_stdout.log").read_text()=="out\n"
    assert (tmp_path/"run/child_stderr.log").read_text()=="err\n"
    assert not (tmp_path/"STOP.json").exists()

def test_child_nonzero_latches_and_retry_never_runs(tmp_path):
    with pytest.raises(RuntimeError):run(tmp_path,"raise SystemExit(7)")
    r=receipt(tmp_path);assert r["exit_code"]==7 and r["stop"]=="CHILD_EXIT"
    before=(tmp_path/"STOP.json").read_bytes()
    with pytest.raises(RuntimeError):s.supervise([sys.executable,"-c","open('BAD','w').write('x')"],tmp_path/"blocked","child",
        cwd=tmp_path,env=os.environ.copy(),stop_path=tmp_path/"STOP.json",budget_root=tmp_path)
    assert not (tmp_path/"BAD").exists()
    assert json.loads((tmp_path/"blocked/child_EXIT.json").read_text())["child_started"] is False
    assert (tmp_path/"STOP.json").read_bytes()==before
    with pytest.raises(FileExistsError):run(tmp_path,"print('overwrite')")
    assert (tmp_path/"STOP.json").read_bytes()==before

def test_popen_failure_has_command_stderr_exit_and_stop(tmp_path):
    with pytest.raises(RuntimeError):
        s.supervise(["/missing_e1_r2a_executable"],tmp_path/"run","child",cwd=tmp_path,env=os.environ.copy(),stop_path=tmp_path/"STOP.json",budget_root=tmp_path)
    r=receipt(tmp_path)
    assert r["exit_code"] is None and r["child_started"] is False
    assert "FileNotFoundError" in (tmp_path/"run/child_stderr.log").read_text()
    assert (tmp_path/"run/child_COMMAND.json").is_file() and (tmp_path/"STOP.json").is_file()

def test_monitor_start_failure_has_receipts(tmp_path,monkeypatch):
    def fail():raise PermissionError("injected /proc unavailable")
    monkeypatch.setattr(s,"monitor_preflight",fail)
    with pytest.raises(RuntimeError):run(tmp_path,"print('must not execute')")
    assert receipt(tmp_path)["child_started"] is False
    assert "PermissionError" in (tmp_path/"run/child_stderr.log").read_text()
    assert (tmp_path/"STOP.json").is_file()

def test_real_timeout_kills_parent_and_descendant(tmp_path):
    code="import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(20)']);print(p.pid,flush=True);time.sleep(20)"
    with pytest.raises(RuntimeError):run(tmp_path,code,timeout=.5)
    r=receipt(tmp_path);assert r["stop"]=="SUBPROCESS_TIME_BUDGET"
    pid=int((tmp_path/"run/child_stdout.log").read_text().strip())
    assert inactive(pid) and not r["live_descendants_after_cleanup"]
    assert len(r["observed_processes"])>=2

def test_real_tree_rss_budget_counts_descendant(tmp_path):
    code="import subprocess,sys,time;p=subprocess.Popen([sys.executable,'-c','import time;x=bytearray(64*1024*1024);print(1,flush=True);time.sleep(20)'],stdout=subprocess.DEVNULL);print(p.pid,flush=True);time.sleep(20)"
    with pytest.raises(RuntimeError):run(tmp_path,code,rss_limit=48*1024*1024,timeout=5)
    r=receipt(tmp_path);assert r["stop"]=="CPU_TREE_RSS_BUDGET"
    assert r["sampled_peak_RSS_bytes"]>48*1024*1024 and len(r["observed_processes"])>=2
    pid=int((tmp_path/"run/child_stdout.log").read_text().strip());assert inactive(pid)

def test_parent_exit_does_not_leave_orphan(tmp_path):
    code="import subprocess,sys,time;p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(20)']);print(p.pid,flush=True);time.sleep(.15)"
    with pytest.raises(RuntimeError):run(tmp_path,code,timeout=5)
    assert receipt(tmp_path)["stop"]=="ORPHAN_DESCENDANTS"
    pid=int((tmp_path/"run/child_stdout.log").read_text().strip());assert inactive(pid)

def test_output_budget_stop(tmp_path):
    with pytest.raises(RuntimeError):run(tmp_path,"import time;open('large','wb').write(b'x'*500000);time.sleep(10)",disk_limit=100000)
    assert receipt(tmp_path)["stop"]=="OUTPUT_BUDGET"

def test_proc_and_disk_monitor_actual_semantics(tmp_path):
    r=s.monitor_preflight();assert r["self_RSS_bytes"]>0
    (tmp_path/"bytes").write_bytes(b"123")
    assert s.disk_bytes(tmp_path)==3
    me=s.proc_records()[os.getpid()]
    assert me["ppid"]>0 and me["start"]>0

def test_old_stop_is_a_real_guard():
    from query_minco.training.physics_only_v1_2_gate1 import e1_dataset_qualification as q
    import inspect
    source=inspect.getsource(q.protect)
    assert source.index("STOP.json")<source.index('e=read')
    with pytest.raises(ValueError,match="SEPARATE_R2B"):q.qualify()
