"""Real-state functional AdamW counterfactual; no neural optimizer.step or saved trial weights."""
import os,copy,random,importlib.util,csv,subprocess
from pathlib import Path
import numpy as np,torch
from query_minco.training.io import read,write,sha,artifact,require,digest
from query_minco.training.run_spec import ROOT,source_files
from query_minco.training.pg1.spec import qualify_bindings
from query_minco.training.pg3_feasible_mode_v1.objective import forward,augment
from query_minco.training.pg3_feasible_mode_v1.feasible_section import RegionBottleneck
from query_minco.training.pg3_feasible_mode_v1.spec import PREFIX,recipe
from query_minco.training.pg3_feasible_mode_v1.teacher_labels import TeacherBank
from query_minco.training.pg3_feasible_mode_v1.adaptive_assignment import AdaptiveAssignment
from query_minco.training.pg3_feasible_mode_v1.gradients import audit
from query_minco.training.pg3_feasible_mode_v1.head_coordination import actual_repair_vector,functional_candidate,masks
from query_minco.training.pg2_guided_v1.geometry import TeacherGuidance
from query_minco.training.pg3_feasible_mode_v1.state_io import lr_at
from query_minco.training.losses.imitation import cost_matrix
from query_minco.inference.identity import state_sha
from experiments.pg3_feasible_mode_v1.common import OUT,model_from,query_metrics
R=OUT/"feasible_section001";P=OUT/"adaptive_assignment001"
def cos(a,b):
 a=a.double();b=b.double();den=float(a.norm()*b.norm());return float(a@b)/den if den else None
def csvout(name,rows):
 with (R/name).open("x",newline="") as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def restore_rng(blob):
 r=blob["rng"];n=r["numpy"];random.setstate(r["python"]);torch.set_rng_state(r["torch"]);np.random.set_state((n["kind"],n["keys"].numpy().astype(np.uint32),n["position"],n["has_gauss"],n["cached_gaussian"]))
def run():
 torch.set_num_threads(1);require(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CF_CPU_ONLY")
 require(not subprocess.check_output(["git","status","--porcelain"],text=True).strip(),"CF_CLEAN")
 cal=read(P/"preparation002/CPU0.json");binding=read(artifact(cal["bindings"]));binding["code"]=dict(git_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),files=source_files(),allow_uncommitted_sources=False)
 write(R/"BINDINGS.json",binding);ref=dict(path=str(R/"BINDINGS.json"),sha256=sha(R/"BINDINGS.json"));ds,geo,init,q=qualify_bindings(ref,42);write(R/"QUALIFICATION.json",q)
 samples=ds.load(cal["ids"]);fields=geo.load(samples);rp=next((P/"runs").glob("*"));history_spec=read(rp/"RUN_SPEC.json");records=[];rows=[]
 helper=OUT/"section_alignment001/diagnose_readonly.py";hs=importlib.util.spec_from_file_location("sealed_section_helper",helper);hd=importlib.util.module_from_spec(hs);hs.loader.exec_module(hd)
 write(R/"HELPER_BINDING.json",dict(path=str(helper),sha256=sha(helper),role="read-only implicit section diagnostics; never training loss"))
 pre=dict(steps=[1,16,32],eps=1e-12,clip=10.,min_improvement_m=1e-6,max_reverse_motion_m=1e-5,
  required_improved_steps=0,matched_cost_relative_upper=1.05,matched_cost_absolute_slack=1e-6,hard_safe_query_regressions_allowed=0,goal_regressions_allowed=0,
  score_rows_changed_allowed=0,outside_head_clipped_delta_must_be_equal=False,
  requirement="Region must move Q0 toward legal region vs its pre-update state; point-vs-region superiority is measured by Pilot, not preselected",projection_enabled=False,
  note="fixed before inspecting counterfactual results; no retuning to pass")
 write(R/"COUNTERFACTUAL_PREREG.json",pre)
 def checkpoint(step):
  p=rp/"commits"/f"step{step:04d}"/f"checkpoint_step{step:04d}.pt";require(sha(p)==read(str(p)+".json")["sha256"],"CF_CHECKPOINT_SHA")
  return torch.load(p,map_location="cpu",weights_only=True)
 for step in pre["steps"]:
  before=checkpoint(step-1);actual=checkpoint(step);update=read(rp/"commits"/f"step{step:04d}"/"UPDATE.json")
  m=model_from(init);m.load_state_dict(before["state_dict"],strict=True);m.train();source_weight_sha=state_sha(m);restore_rng(before)
  policy=AdaptiveAssignment(state=update["adaptive_state_before"]);bank=TeacherBank(recipe())
  z=forward(m,samples,geo,PREFIX+"_route",cal["old_weights"],cal["new_weights"],bank,TeacherGuidance(),step-1,assignment_policy=policy)
  require(z["assignments"]==update["assignments"] and float(z["loss"].detach())==update["loss"],"CF_CAUSAL_REPLAY")
  rec,vec,total=audit(m,z,details=True);im=vec["imitation"];require(z["base"]["weights"]["imitation"]==1.,"CF_IMITATION_WEIGHT");repair=actual_repair_vector(z,vec);names=list(m.named_parameters())
  region_policy=RegionBottleneck()
  rz=augment(z["base"],samples,bank,PREFIX+"_route",cal["new_weights"],step-1,assignment_policy=AdaptiveAssignment(state=update["adaptive_state_before"]),bottleneck_policy=region_policy)
  require(rz["assignments"]==z["assignments"],"REGION_CF_CHANGED_ASSIGNMENT")
  rrec,rvec,routed=audit(m,rz,details=True)
  coord=dict(target="POINT_TO_CERTIFIED_REGION",projection_enabled=False,region_gradient_ratio=rrec["physical_imitation_ratio"])
  head,score=masks(names,total.device)
  ratios=dict(original=float((total-im).norm()/im.norm()),candidate=float((routed-im).norm()/im.norm()))
  before_scores={n:p.detach()[19:21].clone() for n,p in names if n.startswith("trajectory_head.") and p.shape[0]==21}
  actual_delta=torch.cat([(actual["state_dict"][n]-before["state_dict"][n]).flatten() for n,p in names])
  outcomes={};deltas={};function_records={};matched_query_costs={}
  target=bank.get(samples[0],fields[0])["teachers"][0];labels=target["bottlenecks"]
  section=hd.section_loss(z["output"].trajectory.coeffs[0,0],z["output"].parameters.durations[0,0],labels)
  from query_minco.training.pg1_v2_r12.diagnostics import flat
  prepoint,_=hd.true_point(z["output"].trajectory.coeffs[0,0],z["output"].parameters.durations[0,0],labels[0])
  remaining_before=max(float(prepoint[2].detach())+.43,0.)
  section_g=flat(torch.autograd.grad(section,list(m.parameters()),retain_graph=True,allow_unused=True),list(m.parameters()))
  for arm,gradient in (("A",total),("B",routed)):
   weights,simstate,delta,simrec=functional_candidate(names,before["optimizer_state_dict"],gradient,lr_at(history_spec["schedule"],step-1))
   full=copy.deepcopy(before["state_dict"]);full.update(weights)
   baseline_weights_equal=all(torch.equal(full[k],v) for k,v in actual["state_dict"].items()) if arm=="A" else None
   if arm=="A":
    require(baseline_weights_equal,"CF_BASELINE_NOT_EXACT")
    for pid,state in actual["optimizer_state_dict"]["state"].items():
     require(all(torch.equal(v,simstate["state"][pid][key]) for key,v in state.items()),"CF_BASELINE_ADAM_STATE")
   require(all(torch.equal(weights[n][19:21],old) for n,old in before_scores.items()),"CF_SCORE_ROWS")
   evalmodel=model_from(init);evalmodel.load_state_dict(full,strict=True);evalmodel.eval()
   with torch.no_grad():
    out=__import__("query_minco.trajectory",fromlist=["TrajectoryPostprocessor"]).TrajectoryPostprocessor()(evalmodel(depth_input=torch.stack([s.depth_input for s in samples]),state=torch.stack([s.state for s in samples])),torch.stack([s.state for s in samples]))
   costlist=[];matched_query_costs[arm]=[]
   for b,sample in enumerate(samples):
    cost,_=cost_matrix(out,sample,b);match=z["base"]["base"]["matches"][b]
    for qid,tid in zip(match["queries"],match["teachers"]):matched_query_costs[arm].append(float(cost[int(qid),int(tid)]))
   # query_metrics expects a one-row output; explicitly slice using postprocessor again, no fake sample index.
   ppout=__import__("query_minco.trajectory",fromlist=["TrajectoryPostprocessor"]).TrajectoryPostprocessor()
   armrows=[]
   for b,sample in enumerate(samples):
    if sample.sample_id in {x["sample_id"] for x in armrows}:continue
    with torch.no_grad():single=ppout(out.raw[b:b+1] if hasattr(out,"raw") else evalmodel(depth_input=sample.depth_input[None],state=sample.state[None]),sample.state[None])
    single_cost,_=cost_matrix(single,sample,0)
    original_match=z["base"]["base"]["matches"][b];matched_map={int(q):int(t) for q,t in zip(original_match["queries"],original_match["teachers"])}
    for qid in range(6):
     met=query_metrics(single,sample,fields[b],qid)
     row=dict(step=step,arm=arm,sample_id=sample.sample_id,query=qid,M=len(sample.teacher_coeffs),matched_before=bool(z["matched"][b,qid]),hard_before=bool(z["labels"]["passed"][b,qid]),hard_pass=met["hard_pass"],hard_reasons=";".join(met["hard_reasons"]),goal_pass=met["goal"]["passed"],goal_mode=met["goal"]["mode"],matched_teacher_before=matched_map.get(qid),fixed_matched_imitation_cost=float(single_cost[qid,matched_map[qid]]) if qid in matched_map else None,nearest_teacher_cost=float(single_cost[qid].min()),goal_progress=met["goal"]["projected_progress_m"],signed_clearance=met["signed"]["minimum_signed_surface_clearance_m"] if met["signed"] else None,vmax=met["vmax"],amax=met["amax"],jmax=met["jmax"],time_s=met["time_s"],qa=met["acceleration_energy"],qj=met["jerk_energy"])
     if b==0 and qid==0:
      point,event=hd.true_point(single.trajectory.coeffs[0,0],single.parameters.durations[0,0],labels[0]);proxy,bd=hd.bottleneck_loss(single.trajectory.coeffs[0,0],single.parameters.durations[0,0],labels)
      row.update(section_y=float(point[1]),section_z=float(point[2]),remaining_boundary_m=max(float(point[2])+.43,0),true_section_loss=float(hd.section_loss(single.trajectory.coeffs[0,0],single.parameters.durations[0,0],labels)),bottleneck_loss=float(proxy),region_loss=float(region_policy(single.trajectory.coeffs[0,0],single.parameters.durations[0,0],target,bank.get(samples[0],fields[0]))[0]),section_time=event["time"],inner_waypoints=single.parameters.inner_pts[0,0].detach().tolist())
     armrows.append(row)
   outcomes[arm]=armrows;rows+=armrows;deltas[arm]=delta;function_records[arm]=simrec
   function_records[arm].update(baseline_reproduces_historical_weights=baseline_weights_equal,temporary_weights_sha256=state_sha(evalmodel),actual_delta_vs_negative_section=cos(delta,-section_g),head_delta_vs_negative_repair=cos(delta[head],-repair[head]),outside_delta_vs_historical=cos(delta[~head],actual_delta[~head]),gradient_ratio=ratios["original" if arm=="A" else "candidate"])
   del evalmodel,full,weights,simstate
  A=next(x for x in outcomes["A"] if x["sample_id"]==samples[0].sample_id and x["query"]==0);B=next(x for x in outcomes["B"] if x["sample_id"]==samples[0].sample_id and x["query"]==0)
  safe_regressions=[(b["sample_id"],b["query"]) for a,b in zip(outcomes["A"],outcomes["B"]) if b["hard_before"] and not b["hard_pass"]]
  goal_regressions=[(b["sample_id"],b["query"]) for a,b in zip(outcomes["A"],outcomes["B"]) if a["goal_pass"] and not b["goal_pass"]]
  costA=np.asarray(matched_query_costs["A"]);costB=np.asarray(matched_query_costs["B"])
  costok=bool(np.all(costB<=1.05*costA+1e-6))
  outside_equal=torch.equal(deltas["A"][~head],deltas["B"][~head])
  records.append(dict(step=step,target_change=coord,gradient_ratios=ratios,functions=function_records,
   q0_remaining_before=remaining_before,q0_remaining_A=A["remaining_boundary_m"],q0_remaining_B=B["remaining_boundary_m"],q0_boundary_improvement_m=A["remaining_boundary_m"]-B["remaining_boundary_m"],
   q0_true_loss_A=A["true_section_loss"],q0_true_loss_B=B["true_section_loss"],q0_true_loss_improved=B["true_section_loss"]<A["true_section_loss"],
   matched_imitation_mean_A=float(costA.mean()),matched_imitation_mean_B=float(costB.mean()),matched_imitation_max_relative_change=float(np.max((costB-costA)/(costA+1e-12))),matched_imitation_guard_passed=costok,
   safe_regressions=safe_regressions,goal_regressions=goal_regressions,outside_head_actual_delta_bitwise_equal=outside_equal,
   source_model_weights_unchanged=state_sha(m)==source_weight_sha,formal_optimizer_steps=0))
  require(state_sha(m)==source_weight_sha,"CF_SOURCE_MODEL_MUTATION")
  print("COUNTERFACTUAL",step,records[-1]["q0_boundary_improvement_m"],costok,safe_regressions,flush=True)
 improved=sum(x["q0_boundary_improvement_m"]>1e-6 and x["q0_true_loss_improved"] for x in records)
 safe=all(not x["safe_regressions"] and not x["goal_regressions"] and x["matched_imitation_guard_passed"]  and x["gradient_ratios"]["candidate"]<=10 and x["q0_remaining_B"]<=x["q0_remaining_before"]+1e-5 for x in records)
 gate=dict(passed=safe and read(R/"preparation002/CPU0.json")["region_cpu0_passed"],improved_steps=improved,safety_and_task_checks_passed=safe,pilot_status="ELIGIBLE_PENDING_TRAINER_BINDING" if safe else "NOT_RUN_COUNTERFACTUAL_FAILED",formal_optimizer_steps=0,temporary_functional_updates=6)
 # CSV has Q0-only columns; normalize missing fields explicitly.
 keys=list(dict.fromkeys(k for row in rows for k in row))
 with (R/"COUNTERFACTUAL_QUERY_METRICS.csv").open("x",newline="") as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
 write(R/"COUNTERFACTUAL_RESULTS.json",records);write(R/"COUNTERFACTUAL_GATE.json",gate);print("REGION_COUNTERFACTUAL_GATE",gate,flush=True)
if __name__=="__main__":run()
