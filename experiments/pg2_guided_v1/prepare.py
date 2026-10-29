"""Draft ablation contracts/data specification only, no training dispatch."""
import argparse,json
from pathlib import Path
from query_minco.training.io import read,write,sha,require,artifact
from query_minco.training.run_spec import ROOT,source_files
from query_minco.training.pg2_guided_v1.spec import METHODS,draft_spec

def prepare(cpu0,output):
 output=Path(output).resolve()
 root=ROOT/"research_outputs/pg2_guided_v1"
 require(output.parent==root/"_preparation","PG2_PLAN_SCOPE")
 output.mkdir(parents=True,exist_ok=False)
 result=read(Path(cpu0)/"RESULT.json")
 require(result["passed"] is True and result["optimizer_steps"]==0,"PG2_PLAN_CPU0")
 binding=dict(path=str(Path(cpu0).resolve()/"audit/CPU0_BINDINGS.json"),sha256=sha(Path(cpu0)/"audit/CPU0_BINDINGS.json"))
 b=read(artifact(binding));init=next(x for x in b["initializations"] if x["seed"]==42)
 require(source_files()==b["code"]["files"],"PG2_PLAN_SOURCE_DRIFT")
 plans=[]
 for method in METHODS:
  arm=method.removeprefix("query_minco_pg2_guided_").removesuffix("_v1") if hasattr(str,"removeprefix") else method[len("query_minco_pg2_guided_"):-3]
  spec=draft_spec(binding,b["code"],init,method,"A42_PG2_"+arm+"_CPU_Pilot32_run001")
  spec["cpu0_evidence"]=dict(path=str(Path(cpu0).resolve()/"RUN_SEAL.json"),sha256=sha(Path(cpu0)/"RUN_SEAL.json"))
  spec["blockers"]+=["shared run_engine score-row immutability must remain intact for PG1; PG2 score-row policy needs explicit reviewed extension",
   "complex-task teacher qualification and map-split evidence", "gradient budget candidate review"]
  write(output/(arm+"_RUN_SPEC_DRAFT.json"),spec)
  plans.append(dict(method=method,run_id=spec["run_id"],spec_file_sha256=sha(output/(arm+"_RUN_SPEC_DRAFT.json"))))
 write(output/"ABLATION_PLAN.json",dict(schema="pg2_staged_ablations_v1",baseline="query_minco_pg1_task_complete_v1",
  baseline_contract="new matched baseline with same current code and optimizer/order, not sealed historical identity reuse",
  arms=plans,approved=False,training_executed=False,lambda_search_on_open_dev8=False,
  order=["PG1 baseline","PG2 Safe","PG2 Safe+Guidance","PG2 Safe+Guidance+Control","PG2 Full"],
  stop_conditions=["nonfinite loss or gradients","two consecutive physical/imitation gradient ratios >10",
   "original Goal/Oracle/mode regression alarm","data/source identity drift","resource contract violation"]))
 write(output/"COMPLEX_TASK_DATA_SPEC.json",dict(schema="pg2_complex_tasks_design_v1",status="NOT_GENERATED_NOT_QUALIFIED",
  levels=[
   dict(level=0,geometry="certified empty or wide corridor",initial_speed_mps=[.8,1.7]),
   dict(level=1,geometry="single trusted AABB; at least one verified detour",initial_speed_mps=[.8,1.7]),
   dict(level=2,geometry="narrow passage or multiple independently feasible Teacher routes",initial_speed_mps=[.8,1.7]),
   dict(level=3,geometry="complex 3D + nonzero acceleration + near dynamic boundary",initial_speed_mps=[1.7,2.4])],
  original_frozen384="unchanged baseline; never regenerated",
  qualification=["explicit frame/camera/depth/state and hashes","Teacher provenance + original continuous hard validator + original Goal task validation",
   "Teacher search failure is UNKNOWN, not unreachable","train/dev/test map hashes and family split disjoint",
   "M1/M4 and matched/unmatched tracked; no homotopy claim from heuristic mode",
   "new high speed and acceleration states require independent qualified Teacher feasibility"],
  experiments="loss ablations use identical dataset; complexity/data change is a separate paired factor",
  hard_limits=dict(radius=.12,margin=.03,speed=3.,acceleration=8.,jerk=100.,total_time=[.8,2.2])))
 write(output/"READINESS.json",dict(status="NOT_READY_FOR_TRAINING",cpu0_passed=True,source_clean=False,
  optimizer_steps=0,approved=False,plans=plans))
 print(json.dumps(dict(output=str(output),plans=plans,status="NOT_READY_FOR_TRAINING"),allow_nan=False))
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--cpu0",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
 a=p.parse_args();prepare(a.cpu0,a.output)
