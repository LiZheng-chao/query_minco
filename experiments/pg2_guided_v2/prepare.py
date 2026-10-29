"""Seven independently identified draft arms; no automatic training approval."""
import argparse
from pathlib import Path
from query_minco.training.run_spec import ROOT
from query_minco.training.io import read,sha,write,require
from query_minco.training.pg2_guided_v2.spec import METHODS,BASELINE,draft
def run(cpu0,output,device="cuda",arms="ABCDEFG",run_index=1):
 cpu0=Path(cpu0).resolve();output=Path(output).resolve()
 require(device in ("cpu","cuda") and arms and all(a in "ABCDEFG" for a in arms) and
  len(set(arms))==len(arms) and run_index>=1,"V2_PLAN_PROFILE")
 require(output.parent==ROOT/"research_outputs/pg2_guided_v2/_preparation","V2_PLAN_OUTPUT")
 output.mkdir(parents=True,exist_ok=False)
 r=read(cpu0/"RESULT.json");cal=read(cpu0/"CALIBRATION.json")
 require(r["numerical_passed"] and r["optimizer_steps"]==0,"V2_PLAN_CPU0")
 binding=dict(path=str(cpu0/"audit/BINDINGS.json"),sha256=sha(cpu0/"audit/BINDINGS.json"))
 b=read(cpu0/"audit/BINDINGS.json");init=next(x for x in b["initializations"] if x["seed"]==42)
 calibration=dict(path=str(cpu0/"CALIBRATION.json"),sha256=sha(cpu0/"CALIBRATION.json"));plans=[]
 method_by_arm=dict(zip("ABCDEFG",[BASELINE]+list(METHODS)))
 for arm in arms:
  method=method_by_arm[arm]
  s=draft(binding,b["code"],calibration,method,
   "A42_"+arm+"_PG2v2_"+device.upper()+"_Pilot32_run%03d"%run_index,device=device)
  s["initialization"]=init;s["cpu0"]=dict(path=str(cpu0/"RUN_SEAL.json"),sha256=sha(cpu0/"RUN_SEAL.json"))
  s["output_root"]=str(ROOT/"research_outputs/pg2_guided_v2/runs")
  write(output/(arm+"_RUN_SPEC_DRAFT.json"),s)
  plans.append(dict(arm=arm,method=method,run_id=s["run_id"],spec_file_sha256=sha(output/(arm+"_RUN_SPEC_DRAFT.json"))))
 write(output/"READINESS.json",dict(numerical_cpu0=True,gradient_guard=cal["gradient_guard_passed"],source_clean=not b["code"]["allow_uncommitted_sources"],
  device=device,status="DRAFT_NOT_QUALIFIED",approved=False,training_executed=False,plans=plans,
  blockers=["Clean CPU0 must be rerun on committed current source","device-specific zero-update preflight NOT_RUN",
   "per-run authorization NOT_GRANTED","negative-gradient safety counterexamples remain; Pilot measures efficacy, not safety certification"]))
 print("DRAFT_PLANS",output,len(plans),flush=True)
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--cpu0",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
 p.add_argument("--device",choices=("cpu","cuda"),default="cuda");p.add_argument("--arms",default="ABCDEFG");p.add_argument("--run-index",type=int,default=1)
 a=p.parse_args();run(a.cpu0,a.output,a.device,a.arms,a.run_index)
