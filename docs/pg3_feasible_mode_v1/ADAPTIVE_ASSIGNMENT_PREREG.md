# M1 Adaptive Assignment v1: fixed before training
Research candidate, not optimality claim. Training only. No network/loss/Teacher/data/threshold change.

Eligible q = unmatched AND detached original continuous hard FAIL. Existing Hungarian reservations occupy capacity2. Only certified representable Teachers receive extra slots.

For each current training query and Teacher:
r=clip((0.03-min sampled sphere-surface clearance)/0.30,0,1).
p=RMS(concat(two decoded inner waypoint residuals, endpoint residual))/1m.
b=mean Teacher-time-aligned bottleneck lateral error/1m.
d=.75 clip(p,0,1)+.25 clip(b,0,1); if no certified bottleneck, d=clip(p,0,1) and b is null.
age=consecutive eligible training updates without direct supervision; h=direct_count/max(eligible_count,1).
P = 1*r - .5*d + .5*clip(age/5,0,1) - .1*h.
If age>=5 add100+age as explicit starvation safeguard. Expanded-slot min-cost assignment maximizes P, with sorted query/Teacher IDs and deterministic scipy solver.

All inputs and choices detached. Physical refs are fixed from existing PG3 sigma_W=1m, two*(radius+margin)=.30m, and six-query count; no development fitting. Risk is 64-node proxy, NOT continuous safety evidence. Original hard checker is used only to determine training eligibility and final results.

Duplicate sample rows share pre-step state and commit union once per distinct sample/update. History commits only after actual optimizer update. Matched/safe exposure resets consecutive waiting age. No eligible/no certified/no capacity is explicit. State is bound in immutable UPDATE/COMMIT, not an unversioned global cache.

Only C is trained if exact old/new A/B loss and full-parameter gradient match on the actual initial batch and historical sealed common contracts match. Reuse historical A/B evidence, do not rerun32 updates unnecessarily. C method=query_minco_pg3_feasible_mode_v1_m1_adaptive_assignment_v1. CPU FP32/threads1, C1step2880,Seed42,same four training tasks repeatedBatch8,AdamW,original cosine2e-5->1e-6,32 updates only,checkpoints0/8/16/32. Clip10,ratio>10consecutive2, original dev alarms, wall1800s/RSS4096MiB/MemAvailable3072MiB/disk4GiB. No128 extension, no retry or new training scenes.
