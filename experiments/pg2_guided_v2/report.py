"""V2 evidence-derived reports plus actual YOPO-arm CPU0, no optimizer."""
import os,json,csv,math,subprocess,collections,time
from pathlib import Path
import numpy as np
import torch
from query_minco.training.run_spec import ROOT,source_files
from query_minco.training.io import read,sha,artifact,require,digest
from query_minco.training.pg2_guided_v1.audit import write,resource_gate
from query_minco.training.pg1.spec import qualify_bindings
from query_minco.training.initialization import configure
from query_minco.models import QueryMincoPlanner
from query_minco.inference.identity import state_sha
from query_minco.training.pg2_guided_v1.geometry import TeacherGuidance
from query_minco.training.pg2_guided_v1.ranking import safety_priority_projection
from query_minco.training.pg1_v2_r12.diagnostics import flat
from query_minco.training.pg2_guided_v2.spec import COMPONENTS,METHODS,BASELINE,PREFIX,weights_for_method
from query_minco.training.pg2_guided_v2.gradients import budget_record,guard_pass
from query_minco.training.pg2_guided_v2.objective import forward
OUT=ROOT/"research_outputs/pg2_guided_v2"

def yopo_cpu0(cpu0):
 out=OUT/"audit/yopo_cpu0_001";out.mkdir(parents=True,exist_ok=False)
 cal=read(cpu0/"CALIBRATION.json");binding=dict(path=str(cpu0/"audit/BINDINGS.json"),sha256=sha(cpu0/"audit/BINDINGS.json"))
 ds,geo,init,q=qualify_bindings(binding,42);configure(dict(threads=1,seed=42,device="cpu"))
 model=QueryMincoPlanner();model.load_state_dict(torch.load(str(artifact(init["checkpoint"])),map_location="cpu",weights_only=True)["state_dict"],strict=True);model.train()
 before=state_sha(model);context=TeacherGuidance();weights=weights_for_method(cal["weights"],PREFIX+"yopo_rank_v1");pp=list(model.parameters())
 rows=[];covered=set();start=time.monotonic()
 with (out/"BATCHES.jsonl").open("x") as ledger:
  for i in range(96):
   resource_gate(start,600)
   ids,_=geo.orders[42].batch(i);z=forward(model,ds.load(ids),geo,PREFIX+"yopo_rank_v1",context,weights)
   gt=flat(torch.autograd.grad(z["loss"],pp,retain_graph=True,allow_unused=True),pp)
   gi=flat(torch.autograd.grad(z["terms"]["imitation"],pp,retain_graph=True,allow_unused=True),pp)
   gr=flat(torch.autograd.grad(z["terms"]["ranking"],pp,retain_graph=True,allow_unused=True),pp)
   require(bool(torch.isfinite(gt).all()) and float(gi.norm())>0,"V2_YOPO_FINITE")
   baseline=flat(torch.autograd.grad(z["base"]["loss"],pp,retain_graph=True,allow_unused=True),pp)
   baseline_ratio=float((baseline-gi).norm())/float(gi.norm())
   ratio=float((gt-gi).norm())/float(gi.norm())
   with np.load(cpu0/"arrays"/("batch%03d.npz"%i),allow_pickle=False) as a:require(np.array_equal(a["raw"],z["raw"].detach().numpy()),"V2_YOPO_SAME_RAW")
   row=dict(batch=i,ids=ids,ratio=ratio,baseline_ratio=baseline_ratio,ranking_parameter_norm=float(gr.norm()),loss=float(z["loss"].detach()),targets=z["ranking_diag"]["targets"])
   ledger.write(json.dumps(row,allow_nan=False)+"\n");ledger.flush();rows.append(row);covered.update(ids)
   if i%24==0:print("YOPO_CPU0",i+1,flush=True)
 require(len(covered)==384 and before==state_sha(model),"V2_YOPO_COVERAGE_WEIGHTS")
 result=dict(guard_passed=guard_pass([r["ratio"] for r in rows]),unique_tasks=384,batches=96,exposures=768,
  weights_before=before,weights_after=state_sha(model),calibration_sha256=sha(cpu0/"CALIBRATION.json"),code=source_files(),optimizer_steps=0,wall_s=time.monotonic()-start)
 write(out/"RESULT.json",result);return rows,result

def main(cpu0,output):
 os.environ["CUDA_VISIBLE_DEVICES"]="";torch.set_num_threads(1)
 cpu0=Path(cpu0).resolve();output=Path(output).resolve();require(output.parent==OUT/"summary","V2_REPORT_SCOPE")
 output.mkdir(parents=True,exist_ok=False)
 rows=[json.loads(l) for l in (cpu0/"BATCHES.jsonl").read_text().splitlines()]
 cal=read(cpu0/"CALIBRATION.json");result=read(cpu0/"RESULT.json")
 require(cal["code"]["files"]==source_files(),"V2_REPORT_SOURCE")
 yopo,yresult=yopo_cpu0(cpu0)
 def text(n,s):
  with (output/n).open("x",encoding="utf8") as f:f.write(s)
 def f(x):return "%.6g"%float(x)
 def quant(v):return [float(x) for x in np.quantile(v,[0,.5,.95,1])]
 table="|component|weight|unweighted parameter norm P50|weighted norm P50|\n|---|---:|---:|---:|\n"
 csvrows=[]
 for k in COMPONENTS:
  norms=[r["components"][k]["parameter_norm"] for r in rows];weight=cal["weights"][k]
  table+="|%s|%s|%s|%s|\n"%(k,f(weight),f(np.median(norms)),f(np.median(norms)*weight))
  for r in rows:
   v=r["components"][k];g=np.asarray(r["gram"]);j=COMPONENTS.index(k)
   def cosine_to(indices):
    vector=np.array([cal["weights"][c] if c in indices else 0. for c in COMPONENTS]);den=math.sqrt(max(0.,g[j,j])*max(0.,float(vector@g@vector)))
    return float(g[j]@vector)/den if den else None
   csvrows.append(dict(batch=r["batch"],component=k,value=v["value"],weight=weight,parameter_norm=v["parameter_norm"],
    weighted_parameter_norm=v["parameter_norm"]*weight,raw_norm=v["raw_norm"],
    cosine_goal=cosine_to(["original_task","dense_guidance"]),cosine_control=cosine_to(["acceleration","jerk","time"]),
    **{"raw_time"+str(i):v["time15_19"][i-15] for i in range(15,19)},
    **{s+"_norm":v["strata"][s]["parameter_norm"] for s in ["matched_pass","matched_fail","unmatched_pass","unmatched_fail"]}))
 with (output/"GRADIENT_COMPONENTS.csv").open("x",newline="") as fcsv:
  writer=csv.DictWriter(fcsv,fieldnames=list(csvrows[0]));writer.writeheader();writer.writerows(csvrows)
 armgates={}
 for method in METHODS:
  if method==PREFIX+"yopo_rank_v1":ratios=[r["ratio"] for r in yopo]
  else:
   w=weights_for_method(cal["weights"],method);ratios=[budget_record(r["gram"],w)["ratio"] for r in rows]
  armgates[method]=dict(guard_passed=guard_pass(ratios),ratio_min_p50_p95_max=quant(ratios),actual_or_gram="actual_all96" if method==PREFIX+"yopo_rank_v1" else "same_frozen_gradient_gram_all96")
 armgates[BASELINE]=dict(guard_passed=guard_pass([r["baseline_ratio"] for r in yopo]),ratio_min_p50_p95_max=quant([r["baseline_ratio"] for r in yopo]),actual_or_gram="actual_original_baseline_all96")
 write(output/"PER_METHOD_CPU0_GATES.json",armgates)
 status=collections.Counter(s["status"] for r in rows for ss in r["safety_records"] for s in ss)
 retreat=collections.Counter(s["retreat"] for r in rows for ss in r["safety_records"] for s in ss)
 unknown_pass=sum(s["status"]=="CHORD_UNPROVEN" and r["hard_pass"][b][q] for r in rows for b,ss in enumerate(r["safety_records"]) for q,s in enumerate(ss))
 ratios=[r["ratio"] for r in cal["budget_records"]];qualityconf=sum(r["safe_quality_cosine"]<0 for r in cal["budget_records"])
 safe=np.median([r["safe_norm"] for r in cal["budget_records"]]);quality=np.median([r["quality_norm"] for r in cal["budget_records"]])
 before=read(OUT/"_preparation/implementation001/PROTECTION_BEFORE.json")
 protected=[n for n,h in before["files"].items() if sha(ROOT/n)!=h];require(not protected,"V2_OLD_ASSET_DRIFT")
 new=[]
 for root in ["query_minco/training/pg2_guided_v2","experiments/pg2_guided_v2","docs/pg2_guided_v2"]:
  new.extend(str(p.relative_to(ROOT)) for p in (ROOT/root).rglob("*") if p.is_file() and p.suffix in (".py",".md",".sh"))
 new.append("tests/test_pg2_guided_v2.py");new=sorted(new)
 write(output/"SOURCE_SHA256.json",{n:sha(ROOT/n) for n in new})
 with (output/"SOURCE.diff").open("x") as target:
  for n in new:
   r=subprocess.run(["git","diff","--no-index","--","/dev/null",n],text=True,capture_output=True);require(r.returncode in (0,1),"V2_DIFF");target.write(r.stdout)
 write(output/"PROTECTION_AFTER.json",dict(passed=True,files_checked=len(before["files"]),drift=protected,head=before["head"],optimizer_steps=0))
 v1=read(ROOT/"research_outputs/pg2_guided_v1/summary/cpu0_run001/SOURCE_SHA256.json")
 write(output/"V1_SYNC_INVENTORY.json",{n:dict(exists=(ROOT/n).is_file(),sha256=sha(ROOT/n),matches_v1_seal=sha(ROOT/n)==h,tracked=subprocess.run(["git","ls-files","--error-unmatch",n],capture_output=True).returncode==0) for n,h in v1.items()})
 text("IMPLEMENTATION_REPORT.md","# PG2-Guided-v2 implementation\n\n"
 "Branch research/pg2-guided-v2; base "+before["head"]+". All existing v1/PG1/R0/Rtail/Teacher/MINCO/validator files were preserved. V1's 12 source/document hashes all match; existing untracked v1 documentation/experiment entries were NOT staged, moved or overwritten. V1_SYNC_INVENTORY.json records their actual Git status.\n\n"
 "Implemented: three-way safety evidence; confirmed-only Retreat; bounded braking increment with uncapped requirement telemetry; confirmed/unknown-weighted Teacher repair; unchanged numerical task penalties plus separately gated dense guidance; differentiable lateral acceleration; separate QA/QJ/QT; training-set gradient calibration; three Ranking modes; per-method score-row execution policy. Existing atomic checkpoint primitives are reused; PG1 run_engine/assertions are unmodified.\n\n"
 "Source edits are isolated. No optimizer was constructed by CPU0 or the readonly diagnostic. Training adapter code was NOT exercised with optimizer.step. No GPU or ROS was started.\n\n"
 "Safety statuses across 4608 exposures: "+json.dumps(dict(status))+". CHORD_UNPROVEN has "+str(unknown_pass)+" original-hard-PASS cases, demonstrating that uncertainty is not collision. Its Retreat is zero. First-node reasons: "+json.dumps(dict(retreat))+". MARGIN_WITH_ANCHOR labels the FIRST violation; a trajectory may penetrate later, as the signed-clearance rows show.\n")
 text("LOSS_MATH_AND_CONTRACT.md","# Loss mathematics and contracts\n\n"
 "All physical time samples retain the original 64 nodes/segment and normalized trapezoid mass. Radius=.12m and hard margin=.03m; hard limits remain 3m/s,8m/s²,100m/s³. New proxies are dimensionless.\n\n"
 "Base clearance r=eps*softplus((.15-SDF)/eps)/.15; physical-time weighted LSE, eps=.05m,tau=.025/.15. For closing c=max(-v·stopgrad(normal),0), full requirement b_req=c²/(2*4). cap=min(.15m,.1s*c); used reserve=cap*(1-exp(-b_req/max(cap,1e-8))). Braking loss is LSE(r_with_reserve)-LSE(r_base). This bounded REACTION proxy is NOT full stopping proof; required/used reserves and cap exceedances remain logged.\n\n"
 "CONFIRMED_VIOLATION means an actual sampled SDF<.15 witness. CHORD_UNPROVEN is chord coverage uncertainty WITHOUT such a witness. NO_DISCRETE_FAILURE is not continuous safety. Strong Retreat uses only a confirmed first violation and a verified earlier anchor; absent/invalid-start anchors do not imply safety. Unproven-only samples get no Retreat and only .1 corridor repair weight.\n\n"
 "Teacher trajectories are independently rechecked by the original validator; matched assignments persist, unmatched fill least-used verified modes. Free-ball targets/radii/assignments are detached. Corridor applies to confirmed violations and weakly to uncertainty, leaving already safe alternatives alone. M1 cannot invent other modes; ball chains do NOT certify predicted curves.\n\n"
 "Goal = original task VALUE + weighted dense guidance. Original Near/Far position/speed/progress/reduction/direction/lateral hinges are unchanged. Confirmed margin/domain violations detach direct task derivatives while retaining numerical penalties; detached Teacher-target guidance supplies repair. Trusted infeasibility gating remains. Hard Goal criteria are unchanged.\n\n"
 "Lateral square a_lat²=||v×a||²/(||v||²+(.1m/s)²); loss=mean_time ReLU(a_lat²/(4m/s²)²-1)². Both v and a remain differentiable, finite at zero velocity. Original Dynamic Limit and Domain remain separate.\n\n"
 "QA=integral||a||²dt/(8²*2.2), QJ=integral||jerk||²dt/(100²*2.2), QT=T/2.2. SI energies: m²/s³ and m²/s⁵; time seconds. Positive 4-node Gauss quadrature is polynomial-exact and checked against independent antiderivatives/finite differences. Each has a separate calibrated coefficient. Flatness is diagnostic-only with explicit gravity/frame/zero-drag/fixed-yaw assumptions; not Controller validation.\n\n"
 "PG2 Ranking uses detached physical AND Goal labels, then quality selection, with NONE rank target on all-invalid cases and BCE invalid labels. YOPO-style mode reproduces its feasible-first target rule and least-bad fallback; no-ranking disables both score losses. Raw19/20 are not calibrated safety probabilities. GT never participates in online selection.\n")
 text("GRADIENT_CALIBRATION.md","# Training-set-only calibration\n\n"
 "384 unique tasks; original Seed42 FrozenOrder, 96 batches/768 exposures, identical C1 step2880 weights. Each loss is measured separately before weighting. Targets were fixed before reading development diagnostics. Candidate weight = predeclared budget / median positive gradient-norm ratio to Imitation; a common scalar then bounds the sum of NEW component norm ratios by2 on these training batches. Original task=1, Dynamic Limit=.1, Domain=1 remain fixed. This is a frozen-state candidate, not an optimum or a guarantee for later updates.\n\n"+table+
 "\nCombined non-Imitation/Imitation min/P50/P95/max = "+"/".join(f(x) for x in quant(ratios))+
 ". Original >10 for two consecutive updates rule is unchanged. V1 P50 was ~803. Weighted Safe norm P50="+f(safe)+", Quality norm P50="+f(quality)+
 ". Conflict remains in "+str(qualityconf)+"/96 batches (v1 96/96). Multiple v2 changes prevent attributing this reduction to one component.\n\n"
 "Three actual weighted-gradient recomputations match the independent Gram reconstruction. All seven methods have frozen gradient-budget gates; baseline and YOPO mode additionally ran full96 actual gradients. Details in PER_METHOD_CPU0_GATES.json. Per-component raw gradients, raw15:19, four matched/pass strata and layer norms are preserved. Projection is OFF in recipes; optional projection checks only a first-order dot product, not AdamW or hard safety.\n")
 deep=read(OUT/"audit/deep001/DEEP_RESULTS.json")
 t="|component|proxy down|clearance up >1µm|clearance down >1µm|hard PASS|\n|---|---:|---:|---:|---:|\n"
 for k in ["v1_safe","v2_safe","v2_corridor","QA","QJ","QT"]:
  rr=[r for r in deep["directions"] if r["component"]==k and r["status"]=="OFFLINE_RAW_DIRECTION"]
  t+="|%s|%d/14|%d/14|%d/14|%d/14|\n"%(k,sum(r["proxy_after"]<r["proxy_before"] for r in rr),sum(r["clearance_after"]-r["clearance_before"]>1e-6 for r in rr),sum(r["clearance_after"]-r["clearance_before"] < -1e-6 for r in rr),sum(r["hard_pass"] for r in rr))
 text("DEEP_QUERY_COMPARISON.md","# Fourteen historical unmatched penetrations\n\n"
 "Readonly sealed Rtail step32, same initial raw/geometry; fixed raw-output L2 perturbation .001. No network weights changed; no optimizer/AdamW update. All84 trials, including negative results, are in DEEP_RESULTS.json. Signed minima use2049 nodes/segment numerical diagnostics; original Bernstein and polynomial extrema determine hard acceptance.\n\n"+t+
 "\nV2 Safe decreases on14/14 proxies but only6 improve clearance and8 worsen; hard PASS remains0/14. Repair is NOT established. Corridor-only still has6 adverse-clearance examples. QA/QJ can sacrifice clearance; QT often has only sub-micrometre effects here. V1/V2 raw Safe gradient cosine P50="+f(np.median([r["v1_v2_raw_gradient_cosine"] for r in deep["rows"]]))+
 ". Direction changed; changed direction alone is not evidence of a better feasible route. No hyperparameter was selected using this set.\n")
 # Grade existing scores without feeding labels into selection.
 oracle=sum(sum(r["ranking"]["oracle_at6"]) for r in rows)
 selected=sum(sum(r["ranking"]["complete_selected_at6"]) for r in rows)
 phys=sum(sum(r["ranking"]["score_selected_at6"]) for r in rows)
 gsel=sum(sum(r["ranking"]["goal_selected_at6"]) for r in rows)
 invalid=sum(sum(r["ranking"]["all_invalid_truth"]) for r in rows)
 rejects=sum(sum(r["ranking"]["classifier_rejected"]) for r in rows)
 brier=np.mean([r["ranking"]["brier"] for r in rows])
 text("RANKING_BASELINE_COMPARISON.md","# Ranking is an auxiliary baseline\n\n"
 "Inspected local /home/lizheng/YOPO/YOPO/policy/yopo_trainer.py and the public YOPO-MINCO branch:\n"
 "https://github.com/TJU-Aerial-Robotics/YOPO/blob/YOPO-MINCO/YOPO/policy/yopo_trainer.py\n\n"
 "The target CE offsets colliding trajectories above feasible ones; all-colliding images select least-bad. Its generation also uses detached score weighting and inner/radius regularization. Our rank_yopo_style copies only the TARGET RULE on six current Queries, not the complete YOPO training system. Ordinary CE is not a novelty claim.\n\n"
 "PG2 adds explicit physical+Goal eligibility and quality ordering; all-ineligible means NONE for ranking and negative validity labels. No mode is calibrated or trained yet. Metrics below grade existing C1 scores, identical before any learning across modes: they are NOT PG2 training improvements. Oracle uses GT only for offline grading; score argmax does not.\n\n"
 "Task exposures=768; Oracle@6="+str(oracle)+", score-selected physical="+str(phys)+", score-selected Goal="+str(gsel)+", complete selected="+str(selected)+
 ", all-invalid truth="+str(invalid)+", classifier rejects="+str(rejects)+", Brier="+f(brier)+". Goal-selected means Goal success of the score-selected Query, not an Oracle-selected candidate. Threshold .5 is uncalibrated and does not grant execution permission.\n\n"
 "The independent v2 lifecycle reuses atomic checkpoint primitives, leaving the old PG1 lifecycle untouched. Baseline/no-rank freeze19/20; YOPO rank permits19 only; PG2 rank permits19/20. CPU policy counterexamples pass; actual optimizer execution and server validation are NOT_RUN.\n")
 text("TEST_AND_CPU0_RESULTS.md","# Real test and CPU0 results\n\n"
 "44 new tests +34 v1 +17 R1.2 +17 R1 =112 passed. Test commands, exit codes and original logs are under _preparation/implementation001. Includes coefficient/duration FD, low-speed lateral gradients, braking/narrow-channel counterexamples, uncertain chord semantics, task regression, Teacher assignment, ranking NONE and immutable old score contract.\n\n"
 "Main full Frozen384:96 batches,768 task exposures,384 unique tasks,4608 candidate exposures. Numerical gradients finite, original task values/Dynamic Limit bitwise unchanged, old-v1 raw and hard labels bitwise identical. Wall="+f(result["wall_s"])+"s; peakRSS="+f(result["peak_rss_mib"])+"MiB. Weights before/after="+result["weights_before"]+". Additional full96 YOPO-rank CPU0 completed without updates.\n\n"
 "Original continuous hard checks are saved perbatch; signed clearance is a separately labeled numerical diagnostic. No dropped samples, no threshold change. Matrix recombination was independently checked on actual gradients at batches0/47/95. Strict JSON, NPZ, SHA and seal checks are in final validation.\n")
 text("TRAINING_READINESS.md","# NOT_READY_FOR_SERVER_TRAINING\n\n"
 "PASS: math, finite gradients, original contracts, complete Frozen384 coverage and calibrated initial-state gradient budget. Ranking score policy is implemented and tested separately from PG1. This is not a completed training lifecycle acceptance.\n\n"
 "Required before manual Pilot: review and freeze the new branch (existing untracked v1 files need explicit Git disposition); rerun --require-clean with that commit; provide approved server root/interpreter and qualified relocated bindings; run actual deployment zero-update preflight; issue separate exact-spec authorization for each run. None has been substituted by this dirty-source CPU0.\n\n"
 "Scientific limitation:45/96 quality/safety gradient conflicts and8/14 adverse-clearance v2 Safe direction probes remain. A calibrated norm is not safety efficacy. Any decision to test a Safe-only Pilot must explicitly accept this unresolved research risk and retain hard safety/Goal/mode regression stops. No user approval is inferred.\n\n"
 "The seven plans are DRAFT/approved=false. GPU resource/preflight and actual scored optimizer lifecycle are NOT_RUN. New Level2-3 data are correctly deferred as an independent later factor.\n")
 text("PILOT_RUN_COMMANDS.md",'''# Training commands: BLOCKED pending actual deployment identity

No directly executable SERVER training command is issued: approved absolute server root,
interpreter, transported bindings, Clean CPU0 and deployment preflight are not available.
Do not run DRAFT specs or reuse old R0/Rtail/Tail-weight authorizations.

The implemented entry points are real:
- python -m query_minco.training.pg2_guided_v2.cli preflight --spec ABS_READY_SPEC --output ABS_EXCLUSIVE_PREFLIGHT
- python -m query_minco.training.pg2_guided_v2.cli validate --spec ABS_READY_SPEC --authorization ABS_EXACT_AUTH
- bash experiments/pg2_guided_v2/run_pilot.sh ABS_READY_SPEC ABS_EXACT_AUTH ABS_EXCLUSIVE_LAUNCH
These are interface descriptions, NOT qualified concrete server commands.

After source review/commit only, the actual local Clean CPU0 command is:

```bash
cd /home/lizheng/query_minco_mainline
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH=/home/lizheng/query_minco_mainline
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
timeout --signal=TERM --kill-after=15s 1830s /home/lizheng/anaconda3/envs/yopo/bin/python -u -m query_minco.training.pg2_guided_v2.cli cpu0 --require-clean --output /home/lizheng/query_minco_mainline/research_outputs/pg2_guided_v2/runs/A42_C1_V2_CleanCPU0_run002
```

Current dirty worktree will correctly reject --require-clean.
Manual launcher limits each approved run to32 updates,3600s contract and3615s external timeout,
step0/8/16/32 checkpoints, independent optimizer/authorization/output. It never starts
a following arm after failure. Seven intended arms A-G use same C1/FrozenOrder/B8/FP32/AdamW/cosine.
All GPU commands belong to the user. No GPU/ROS/training was launched here.
''')
 text("FAILURE_AND_LIMITATIONS.md","# Failures and unrun work\n\n"
 "No hidden test/sample failures were erased. 112 tests pass; CPU0 completed with zero updates. Calibration is an initial-state training-set diagnostic; later gradients still face the unchanged >10/two-update guard. Fixed candidate weights are not optimal.\n\n"
 "Proxy/hard inconsistency remains:8/14 v2 Safe directions worsen signed clearance despite lower proxy; no hard failure-to-pass conversion. Quality directions also have adverse examples. Projection is first-order only and disabled in recipes.\n\n"
 "Signed minima are densely sampled diagnostics, not exact interval minima. CHORD_UNPROVEN is not collision; absence of a safe anchor is not safety. Teacher point balls cannot certify whole predicted curves. M1 cannot promise mode diversity beyond known verified routes.\n\n"
 "Training, ranking efficacy, CUDA preflight, actual new optimizer execution, server resource/transport qualification and independent new-scene generalization: NOT_RUN. Existing untrained score metrics are not evidence of Ranking improvement. Flatness has no real Controller validation.\n\n"
 "The old PG1 score-row contract, old source/checkpoints/receipts, YOPO and reach_planner were not modified. V1 docs/experiment files were complete by hash but retain their original untracked Git state. New source is uncommitted, so clean-source training qualification is intentionally false.\n")
 write(output/"STATUS.json",dict(math_passed=True,tests_passed=112,cpu0_numerical_passed=True,calibrated_gradient_guard_passed=cal["gradient_guard_passed"],
  yopo_actual_cpu0_passed=yresult["guard_passed"],training_ready=False,training_executed=False,optimizer_steps=0,
  safe_direction_consistency_not_proven=True,source_files=len(new),per_method_gates=armgates))
 print("REPORTS_DONE",output,flush=True)
if __name__=="__main__":
 import argparse
 p=argparse.ArgumentParser();p.add_argument("--cpu0",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();main(a.cpu0,a.output)
