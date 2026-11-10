"""Fail-closed stdlib /proc supervisor. Resource samples are not continuous peaks."""
import errno
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

def write(path,value):
    with Path(path).open("x") as f:json.dump(value,f,indent=2,allow_nan=False);f.write("\n")

def stop_once(path,record):
    try:write(path,record)
    except FileExistsError:pass  # Immutable first failure; never erase the latch.

def check_stop(path):
    if Path(path).exists():raise RuntimeError("PERSISTENT_STOP "+str(path))

def proc_records():
    records={}
    for path in Path("/proc").iterdir():
        if not path.name.isdecimal():continue
        try:
            text=(path/"stat").read_text()
            fields=text.rsplit(")",1)[1].split()
            records[int(path.name)]=dict(state=fields[0],ppid=int(fields[1]),pgid=int(fields[2]),sid=int(fields[3]),
                                        start=int(fields[19]),rss=int(fields[21])*os.sysconf("SC_PAGE_SIZE"))
        except FileNotFoundError:continue
        except ProcessLookupError:continue
        # Other I/O/parse/permission errors fail closed; not a zero-RSS sample.
    return records

def monitor_preflight():
    records=proc_records();me=records.get(os.getpid())
    if not me or me["rss"]<=0:raise RuntimeError("PROC_MONITOR_UNAVAILABLE")
    if not hasattr(os,"killpg"):raise RuntimeError("PROCESS_GROUP_SIGNAL_UNAVAILABLE")
    return dict(backend="Linux /proc/stat; no psutil dependency",self_RSS_bytes=me["rss"],
                proc_processes=len(records),page_size=os.sysconf("SC_PAGE_SIZE"))

def tree_sample(root,seen):
    records=proc_records()
    ids={p for p,v in records.items() if p==root or v["sid"]==root or v["pgid"]==root or seen.get(p)==v["start"]}
    while True:
        more={p for p,v in records.items() if v["ppid"] in ids}
        if more<=ids:break
        ids|=more
    for pid in ids:seen[pid]=records[pid]["start"]
    live={p:records[p] for p in ids if records[p]["state"] not in ("Z","X")}
    return sum(v["rss"] for v in live.values()),live

def disk_bytes(path):
    # Reuse the already accepted math-contract disk traversal, unchanged.
    source=Path(__file__).resolve().parents[3]/"experiments/physics_only_v1_2_math_contract_v1/run_math_acceptance.py"
    spec=importlib.util.spec_from_file_location("_e1_frozen_disk_helper",source)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    return helper.disk_bytes(path)

def terminate(root,seen,grace=2.):
    try:os.killpg(root,signal.SIGKILL)
    except ProcessLookupError:pass
    deadline=time.monotonic()+grace
    while True:
        _,live=tree_sample(root,seen)
        for pid,v in live.items():
            if pid==os.getpid():raise RuntimeError("SUPERVISOR_SELF_SELECTED")
            try:os.kill(pid,signal.SIGKILL)
            except ProcessLookupError:pass
        if not live:return []
        if time.monotonic()>=deadline:return sorted(live)
        time.sleep(.02)

def supervise(argv,folder,name,*,cwd,env,stop_path,budget_root,timeout=300.,
              rss_limit=4294967296,disk_limit=2147483648,guard=None,expected_exit=(0,),interval=.05,gpu_monitor=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    # Establish receipts before dependency, monitor, guard, or Popen work.
    write(folder/(name+"_COMMAND.json"),dict(argv=[str(x) for x in argv],cwd=str(cwd),env={k:env[k] for k in ("PYTHONPATH","PYTHONDONTWRITEBYTECODE","CUDA_VISIBLE_DEVICES","OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","TMPDIR","LD_LIBRARY_PATH","QUERY_MINCO_E1_DATA_REVIEW") if k in env},
        timeout_s=timeout,RSS_limit_bytes=rss_limit,disk_limit_bytes=disk_limit,
        expected_exit_codes=list(expected_exit),backend="/proc",sample_interval_s=interval))
    proc=None;code=None;reason=None;error=None;peak=0;seen={};peak_pids=[];started=time.monotonic();left=[];gpu_samples=[]
    with (folder/(name+"_stdout.log")).open("x") as out,(folder/(name+"_stderr.log")).open("x") as err:
        try:
            check_stop(stop_path)
            monitor_preflight()
            if guard:guard()
            if disk_bytes(budget_root)>disk_limit:raise RuntimeError("OUTPUT_BUDGET")
            if gpu_monitor:gpu_samples.append(gpu_monitor())
            proc=subprocess.Popen([str(x) for x in argv],cwd=cwd,env=env,stdout=out,stderr=err,start_new_session=True)
            while True:
                if gpu_monitor:gpu_samples.append(gpu_monitor())
                rss,live=tree_sample(proc.pid,seen)
                if rss>peak:peak=rss;peak_pids=sorted(live)
                if rss>rss_limit:reason="CPU_TREE_RSS_BUDGET"
                if time.monotonic()-started>timeout:reason="SUBPROCESS_TIME_BUDGET"
                if disk_bytes(budget_root)>disk_limit:reason="OUTPUT_BUDGET"
                if guard:guard()
                code=proc.poll()
                if code is not None:
                    _,live=tree_sample(proc.pid,seen)
                    if live:reason=reason or "ORPHAN_DESCENDANTS"
                    break
                if reason:break
                time.sleep(interval)
            if reason:left=terminate(proc.pid,seen)
            code=proc.wait(timeout=3)
            if code not in expected_exit and reason is None:reason="CHILD_EXIT"
            if guard:guard()
            if gpu_monitor:gpu_samples.append(gpu_monitor())
        except BaseException:
            error=traceback.format_exc();err.write(error);err.flush()
            reason=reason or "SUPERVISOR_START_OR_MONITOR_FAILURE"
            if proc:
                try:left=terminate(proc.pid,seen);code=proc.wait(timeout=3)
                except BaseException:error+="\n"+traceback.format_exc()
        finally:
            if proc and proc.poll() is None:
                try:left=terminate(proc.pid,seen);code=proc.wait(timeout=3)
                except BaseException:error=(error or "")+"\n"+traceback.format_exc()
    passed=reason is None and error is None and code in expected_exit
    receipt=dict(status="PASS" if passed else "STOP",exit_code=code,stop=reason,supervisor_error=error,
                 child_started=proc is not None,elapsed_s=time.monotonic()-started,sampled_peak_RSS_bytes=peak,
                 peak_sample_pids=peak_pids,observed_processes=seen,live_descendants_after_cleanup=left,
                 expected_exit_codes=list(expected_exit),continuous_peak_claim=False,gpu_samples=gpu_samples,
                 sampled_peak_GPU_used_bytes=max((x["used_bytes"] for x in gpu_samples),default=None))
    write(folder/(name+"_EXIT.json"),receipt)
    if not passed:
        stop_once(stop_path,dict(reason=reason,command_name=name,exit_code=code,supervisor_error=error))
        raise RuntimeError("SUPERVISOR_STOP "+name+" "+str(reason))
    return receipt

def gpu_callback_acceptance(root,python):
    """CPU subprocesses only; simulated GPU telemetry tests failure/receipt wiring."""
    results=[]
    for case in ("normal","missing_before_launch","monitor_failure_during_child"):
        dest=Path(root)/("gpu_callback_"+case);dest.mkdir()
        calls=[0]
        def probe():
            calls[0]+=1
            if case=="missing_before_launch" or (case=="monitor_failure_during_child" and calls[0]>=2):
                raise RuntimeError("INJECTED_GPU_MONITOR_UNAVAILABLE")
            return dict(used_bytes=0,total_bytes=2147483648,simulated_no_GPU_workload=True)
        try:
            receipt=supervise([python,"-B","-c","import time;time.sleep(.15)"],dest,"probe",cwd=dest,
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1",CUDA_VISIBLE_DEVICES=""),stop_path=dest/"STOP.json",
                budget_root=dest,gpu_monitor=probe,timeout=3)
            if case!="normal" or calls[0]<3:raise AssertionError("GPU_CALLBACK_EXPECTED_FAILURE_NOT_OBSERVED")
        except RuntimeError:
            if case=="normal":raise
            receipt=json.loads((dest/"probe_EXIT.json").read_text())
            if receipt["status"]!="STOP" or receipt["live_descendants_after_cleanup"]:raise AssertionError("GPU_CALLBACK_FAILURE_NOT_CLOSED")
            if case=="missing_before_launch" and receipt["child_started"]:raise AssertionError("GPU_STARTED_WITHOUT_MONITOR")
        results.append(dict(case=case,status="PASS",callbacks=calls[0],receipt=receipt,
                            scope="simulated GPU telemetry; CPU subprocess, no CUDA call"))
    write(Path(root)/"GPU_CALLBACK_ACCEPTANCE.json",results)
    return results
