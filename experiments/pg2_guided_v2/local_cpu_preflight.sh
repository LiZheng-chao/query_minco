#!/usr/bin/env bash
set -Eeuo pipefail
trap 'rc=$?; echo "LOCAL_CPU_PREFLIGHT_FAILED line=$LINENO exit=$rc" >&2; exit "$rc"' ERR
ROOT=/home/lizheng/query_minco_mainline
PY=/home/lizheng/anaconda3/envs/yopo/bin/python
CPU0="${1:-$ROOT/research_outputs/pg2_guided_v2/runs/A42_C1_V2_CleanCPU0_run003}"
cd "$ROOT"
test "$(git branch --show-current)" = research/pg2-guided-v2
test -z "$(git status --porcelain --untracked-files=all)"
test -x "$PY"
export PYTHONPATH="$ROOT" PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PG2_PYTHON="$PY" PG2_CPU0="$CPU0"
"$PY" - <<'PY'
import os, pathlib, shutil, subprocess, torch
from query_minco.training.io import read, sha, digest, write, require
from query_minco.training.run_spec import ROOT, verify_code
from query_minco.training.pg2_guided_v2.spec import BASELINE, PREFIX, draft
root=ROOT.resolve()
cpu0=pathlib.Path(os.environ["PG2_CPU0"]).resolve()
prep=root/"research_outputs/pg2_guided_v2/_preparation"
plan=prep/"local_cpu_pilot32_preflight_run002"
runs=root/"research_outputs/pg2_guided_v2/runs"
torch.set_num_threads(1)
require(os.environ.get("CUDA_VISIBLE_DEVICES")=="" and not torch.cuda.is_available() and torch.get_num_threads()==1,"CPU_ENV")
mem=int(next(x.split()[1] for x in pathlib.Path("/proc/meminfo").read_text().splitlines() if x.startswith("MemAvailable:")))
require(mem>=3072*1024 and shutil.disk_usage(root).free>=4*1024**3,"CPU_RESOURCES")
result=read(cpu0/"RESULT.json");seal=read(cpu0/"RUN_SEAL.json");cal=read(cpu0/"CALIBRATION.json")
bp=cpu0/"audit/BINDINGS.json";b=read(bp)
require(result["numerical_passed"] is True and result["training_gradient_guard_passed"] is True,"CPU0_NUMERICAL_GRADIENT")
require(result["optimizer_created"] is False and result["optimizer_steps"]==0 and result["unique_tasks"]==384 and result["batches"]==96 and result["weights_before"]==result["weights_after"],"CPU0_ZERO_UPDATE_COVERAGE")
require(seal.get("purpose")=="CPU0_ONLY_NOT_TRAINING_AUTHORIZATION","CPU0_SEAL")
for name,checksum in seal["files"].items():
 rel=pathlib.Path(name)
 require(not rel.is_absolute() and ".." not in rel.parts and sha(cpu0/rel)==checksum,"CPU0_SEAL_FILE")
require(cal.get("unique_tasks")==384 and cal.get("exposures")==768 and cal.get("gradient_guard_passed") is True,"CPU0_CALIBRATION")
require(b["code"]["allow_uncommitted_sources"] is False,"SOURCE_FROZEN")
verify_code(b["code"])
binding=dict(path=str(bp),sha256=sha(bp))
calibration=dict(path=str(cpu0/"CALIBRATION.json"),sha256=sha(cpu0/"CALIBRATION.json"))
require(cal["code"]==b["code"] and cal["bindings"]==binding,"CALIBRATION_IDENTITY")
init=next(x for x in b["initializations"] if x["seed"]==42)
require(result["weights_before"]==init["weights_sha256"],"C1_WEIGHTS")
arms=[("A",BASELINE),("B",PREFIX+"safe_v1"),("C",PREFIX+"guidance_v1")]
require(not plan.exists(),"EXCLUSIVE_PLAN")
for arm,method in arms:
 require(not(runs/f"A42_{arm}_PG2v2_CPU_Pilot32_run002").exists(),"EXCLUSIVE_RUN")
 require(not(prep/f"local_cpu_pilot32_preflight_{arm}_run002").exists(),"EXCLUSIVE_PREFLIGHT")
plan.mkdir(parents=True,exist_ok=False)
summary=dict(schema="pg2_v2_local_cpu_preflight_summary_v1",code=b["code"],cpu0=dict(path=str(cpu0/"RUN_SEAL.json"),sha256=sha(cpu0/"RUN_SEAL.json")),arms=[],optimizer_steps=0)
for arm,method in arms:
 rid=f"A42_{arm}_PG2v2_CPU_Pilot32_run002"
 s=draft(binding,b["code"],calibration,method,rid,device="cpu")
 s.update(schema="pg2_v2_pilot32_spec_v1",status="READY_FOR_MANUAL_PILOT",cpu0=summary["cpu0"],output_root=str(runs),initialization=init)
 sp=plan/f"{arm}_RUN_SPEC_PRE_PREFLIGHT.json";write(sp,s)
 out=prep/f"local_cpu_pilot32_preflight_{arm}_run002"
 subprocess.run([os.environ["PG2_PYTHON"],"-u","-m","query_minco.training.pg2_guided_v2.cli","preflight","--spec",str(sp),"--output",str(out)],cwd=root,check=True,timeout=600)
 final_path=out/"RUN_SPEC_WITH_PREFLIGHT.json";final=read(final_path);receipt=read(out/"PREFLIGHT.json")
 require(receipt.get("passed") is True and receipt.get("device")=="cpu" and receipt.get("optimizer_created") is False and receipt.get("optimizer_steps")==0 and receipt.get("deterministic_repeat") is True,"CPU_PREFLIGHT")
 write(out/"AUTHORIZATION_TEMPLATE.json",dict(schema="pg2_v2_single_run_authorization_v1",run_id=rid,spec_sha256=digest(final),max_optimizer_updates=32,approved=False,authorized_by=""))
 summary["arms"].append(dict(arm=arm,method=method,run_id=rid,spec=dict(path=str(final_path),sha256=sha(final_path)),spec_identity=digest(final),preflight=dict(path=str(out/"PREFLIGHT.json"),sha256=sha(out/"PREFLIGHT.json"))))
 print("CPU_PREFLIGHT_PASS",arm,rid,digest(final),flush=True)
summary["status"]="CPU_PREFLIGHT_PASS_AUTHORIZATION_PENDING"
write(plan/"PREFLIGHT_SUMMARY.json",summary)
print("ALL_CPU_PREFLIGHTS_PASS",plan,flush=True)
PY
