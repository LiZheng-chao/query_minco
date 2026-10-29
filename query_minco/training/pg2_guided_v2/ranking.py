"""Explicit auxiliary selection baselines; all GT targets are detached."""
import torch
import torch.nn.functional as F
from query_minco.training.io import require

MODES=("rank_none","rank_yopo_style","rank_pg2_feasible_goal_quality")
def losses(raw,physical,goal,quality,utility,mode):
 require(mode in MODES and raw.shape[-2:]==(6,21),"V2_RANK_MODE")
 require(physical.dtype==goal.dtype==torch.bool and physical.shape==goal.shape==raw.shape[:2],"V2_RANK_LABELS")
 eligible=(physical&goal).detach();quality=quality.detach();utility=utility.detach()
 zero=raw[:,:,19].sum(-1)*0;rank=[];targets=[]
 for b in range(len(raw)):
  if mode=="rank_none":rank.append(zero[b]);targets.append(None);continue
  if mode=="rank_yopo_style":
   # Literal local YOPO target rule: feasible first, least-bad fallback.
   penalized=utility[b]+(utility.max()+1.)*(~physical[b]).to(raw.dtype)
   target=penalized.argmin()
  else:
   ids=torch.nonzero(eligible[b],as_tuple=False).flatten()
   if not len(ids):rank.append(zero[b]);targets.append(None);continue
   target=ids[quality[b,ids].argmin()]
  rank.append(F.cross_entropy(raw[b,:,19][None],target[None]));targets.append(int(target))
 rank=torch.stack(rank)
 valid=F.binary_cross_entropy_with_logits(raw[:,:,20],eligible.to(raw.dtype),reduction="none").mean(-1) if mode=="rank_pg2_feasible_goal_quality" else raw[:,:,20].sum(-1)*0
 return dict(ranking=rank,validity=valid),dict(mode=mode,physically_feasible=physical.detach(),goal_satisfied=goal.detach(),
  eligible=eligible,quality_cost=quality,targets=targets,all_invalid=~eligible.any(-1),training_only=True)

def metrics(raw,physical,goal):
 # Scores choose WITHOUT consulting GT. GT is used only to grade that choice.
 score=raw[:,:,19].detach().argmax(-1);valid=raw[:,:,20].detach().sigmoid()
 choose=valid>=.5;all_invalid=~(physical&goal).any(-1)
 score_physical=physical.gather(1,score[:,None])[:,0]
 score_goal=goal.gather(1,score[:,None])[:,0]
 eligible=physical&goal
 return dict(oracle_at6=eligible.any(-1).tolist(),score_selected_at6=score_physical.tolist(),
  goal_selected_at6=score_goal.tolist(),complete_selected_at6=(score_physical&score_goal).tolist(),
  classifier_rejected=(~choose.any(-1)).tolist(),all_invalid_truth=all_invalid.tolist(),
  brier=float((valid-eligible.to(raw.dtype)).square().mean()),threshold=.5,
  calibration_status="UNCALIBRATED_NOT_EXECUTION_PERMISSION")
