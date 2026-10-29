"""Explicit CPU-only PG3 experiment commands. Never auto-expands a failed run."""
import argparse
from pathlib import Path
import torch
from query_minco.training.io import read,write,sha,digest,require
from query_minco.training.pg1.state_io import inspect_commit
from query_minco.training.pg3_feasible_mode_v1.execution import make_spec,authorization,run,BASELINE
from query_minco.training.pg3_feasible_mode_v1.spec import PREFIX
from experiments.pg3_feasible_mode_v1.common import OUT
def launch(kind,arm=None):
 prep=OUT/"preparation002";cal=read(prep/"CALIBRATION.json")
 if kind=="overfit128":method=PREFIX+"_route";rid="A42_PG3_Overfit128_run001";selected=cal["overfit_ids"]
 else:
  require(arm in "ABCD" and len(arm)==1,"PG3_PILOT_ARM")
  method=dict(A=BASELINE,B=PREFIX+"_mode",C=PREFIX+"_route",D=PREFIX+"_control")[arm]
  rid="A42_"+arm+"_PG3_CPU_Pilot32_run001";selected=None
  if arm!="A":
   previous="ABCD"["ABCD".index(arm)-1]
   result=read(OUT/"runs"/("A42_"+previous+"_PG3_CPU_Pilot32_run001")/"RESULT.json")
   require(result["status"]=="COMPLETED" and result["optimizer_updates"]==32,"PG3_PREVIOUS_ARM_INCOMPLETE")
 specs=OUT/"specs";specs.mkdir(exist_ok=True)
 s=make_spec(cal["bindings"],dict(path=str(prep/"CALIBRATION.json"),sha256=sha(prep/"CALIBRATION.json")),dict(path=str(prep/"CPU0.json"),sha256=sha(prep/"CPU0.json")),method,rid,kind,selected)
 sp=specs/(rid+"_SPEC.json");ap=specs/(rid+"_AUTHORIZATION.json")
 write(sp,s);write(ap,authorization(s,True,"LiZheng-chao"))
 run(sp,ap)
 rp=OUT/"runs"/rid;result=read(rp/"RESULT.json");n=result["optimizer_updates"]
 require(n==32 if kind=="paired32" else n in (32,128),"PG3_UPDATE_BUDGET")
 require(not(rp/"FAILURE_RECORD.json").exists(),"PG3_FAILURE")
 for i in range(n+1):
  c=inspect_commit(rp/"commits"/f"step{i:04d}",s)
  require(c["step"]==i,"PG3_COMMIT_CURSOR")
 blob=torch.load(rp/"commits"/f"step{n:04d}"/f"checkpoint_step{n:04d}.pt",map_location="cpu",weights_only=True)
 require(all(int(v["step"])==n for v in blob["optimizer_state_dict"]["state"].values()),"PG3_ACTUAL_ADAM_STEPS")
 write(rp/"RUN_SEAL.json",dict(schema="pg3_completed_run_seal_v1",files={str(p.relative_to(rp)):sha(p) for p in rp.rglob("*") if p.is_file()},actual_updates=n))
 print("PG3_RUN_ACCEPTED",rid,n,flush=True)
def main():
 p=argparse.ArgumentParser();sub=p.add_subparsers(dest="command",required=True)
 sub.add_parser("prepare");sub.add_parser("raw");sub.add_parser("overfit")
 a=sub.add_parser("pilot");a.add_argument("--arm",choices=list("ABCD"),required=True)
 args=p.parse_args()
 if args.command=="prepare":
  from .prepare import run as f;f()
 elif args.command=="raw":
  from .raw_learnability import run as f;f()
 elif args.command=="overfit":launch("overfit128")
 else:launch("paired32",args.arm)
if __name__=="__main__":main()
