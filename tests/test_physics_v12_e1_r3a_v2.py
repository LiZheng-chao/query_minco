"""R3a-v2 bootstrap and complete parent+descendant resource contracts."""
import importlib.util,json,os,sys
from pathlib import Path
import pytest
from query_minco.training.physics_only_v1_2_gate1 import e1_r3a_v2 as v2
from query_minco.training.physics_only_v1_2_gate1.e1_resource_supervisor import proc_records
BASE=Path(__file__).resolve().parents[1]
def entry():
    s=importlib.util.spec_from_file_location("_r3a_v2_entry_test",BASE/"experiments/physics_only_v1_2_e1_r3a_v2/run_zero_update.py")
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def launch(tmp_path,code,**kw):
    return entry().launch([sys.executable,"-B","-c",code],tmp_path/"launch",dict(os.environ,PYTHONDONTWRITEBYTECODE="1",CUDA_VISIBLE_DEVICES=""),tmp_path/"STOP.json",tmp_path,**kw)
def receipt(tmp_path):return json.loads((tmp_path/"launch/EXIT.json").read_text())
def test_accepted_bootstrap_actual_compatibility():
    x=v2.bootstrap()
    assert x["modules"]["pytest"]["version"]=="7.4.0"
    assert "/envs/yopo/" in x["modules"]["torch"]["path"]
    assert "python3.11/site-packages/pytest/" in x["modules"]["pytest"]["path"]
    assert not x["cross_ABI_extensions"] and x["transitive_requirements"]
def test_full_parent_is_counted_before_child(tmp_path):
    own=proc_records()[os.getpid()]["rss"]
    with pytest.raises(RuntimeError,match="SUPERVISOR_STOP"):
        launch(tmp_path,"import time;time.sleep(5)",rss_limit=own-1)
    r=receipt(tmp_path)
    assert r["reason"]=="COMPLETE_TREE_RSS_BUDGET"
    assert r["sampled_peak_launcher_RSS_bytes"]>own-1
    assert str(os.getpid()) in r["observed_processes"]
    assert not r["live_descendants_after_cleanup"]
def test_success_includes_parent_and_descendant(tmp_path):
    code="import subprocess,sys,time;p=subprocess.Popen([sys.executable,'-c','import time;x=bytearray(8*1024*1024);time.sleep(.3)']);p.wait()"
    r=launch(tmp_path,code)
    assert r["status"]=="PASS" and len(r["observed_processes"])>=3
    assert r["sampled_peak_full_tree_RSS_bytes"]>r["sampled_peak_launcher_RSS_bytes"]
def test_timeout_persistent_stop_and_no_retry(tmp_path):
    with pytest.raises(RuntimeError):launch(tmp_path,"import time;time.sleep(5)",timeout=.2)
    assert receipt(tmp_path)["reason"]=="WALLTIME_BUDGET"
    first=(tmp_path/"STOP.json").read_bytes()
    m=entry()
    with pytest.raises(RuntimeError):
        m.launch([sys.executable,"-c","print('must not start')"],tmp_path/"blocked",os.environ.copy(),tmp_path/"STOP.json",tmp_path)
    assert (tmp_path/"STOP.json").read_bytes()==first
    assert not json.loads((tmp_path/"blocked/EXIT.json").read_text())["child_started"]
def test_dependency_bootstrap_failure_has_real_receipts(tmp_path):
    with pytest.raises(RuntimeError):
        launch(tmp_path,"import _intentionally_missing_r3a_v2_dependency")
    r=receipt(tmp_path)
    assert r["exit_code"]==1 and r["reason"]=="CHILD_EXIT"
    assert "ModuleNotFoundError" in (tmp_path/"launch/stderr.log").read_text()
    assert (tmp_path/"launch/COMMAND.json").is_file() and (tmp_path/"STOP.json").is_file()
def test_frozen_train_denominator_and_scene_order():
    x=v2.read(v2.PREP/"TRAIN13_AND_EXCLUSIONS.json");rows=x["Train13"]
    assert len(x["all28"])==28 and len(x["quarantine7"])==7
    assert len(rows)==13 and sum(r["sample_id"].endswith("_near") for r in rows)==7
    from collections import Counter
    assert sorted(Counter(r["scene_id"] for r in rows).values())==[1,4,4,4]
    assert all(r["source"]=="SOURCE_QUALIFIED" and r["reachability"]=="FEASIBLE_CERTIFIED" for r in rows)
