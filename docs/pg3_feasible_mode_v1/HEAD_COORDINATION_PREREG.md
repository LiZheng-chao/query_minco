# Shared trajectory-head coordination: counterfactual qualification
Only a conventional, optional projection of actual weighted Imitation on actual assigned hard-fail Mode+Bottleneck+Corridor repair gradient, within trajectory_head parameters.
gI_new=gI-min(0,dot(gI,gR))/(norm(gR)^2+1e-12)*gR.
g_total_new=g_total+head_mask*(gI_new-gI). All other Loss derivatives remain exactly once. Score rows19/20 must remain zero. Assignment/weights/Teacher/data/architecture/bottleneck formula unchanged.

Counterfactual uses original sealed Adaptive pre-update checkpoints and full AdamW state at steps1,16,32, restoring exact RNG/assignment. Existing clip10 and registered learning-rate sequence unchanged. PyTorch functional AdamW runs only on detached temporary tensors; no neural Optimizer.step and no trial checkpoint saved. Original branch must reproduce historical full weights and moments bitwise.

Gate fixed before results: at least2/3 steps improve Q0 remaining midplane distance by>1e-6m and true section loss; none worsens distance by>1e-4m; no originally safe Query hard regressions, no Goal regressions, each fixed matched Teacher cost <=baseline*1.05+1e-6, candidate gradient ratio<=10, finite values and score freeze. Head-outside parameter deltas must equal baseline in this qualification. The global clipping algorithm remains original; if a changed global norm causes other-layer deltas, report it and fail this gate rather than changing clipping.

Only after this passes, integrate and preregister an independent CPU32 method, same Adaptive allocation, C1init/Seed42/four tasks/Batch8/FP32/AdamW/cosine2e-5->1e-6/checkpoints0/8/16/32, all old guards. No128, GPU, ROS or arbitrary tuning. Reuse sealed Adaptive control only after numerical/source/spec checks.
