"""New geometry/task contract, frozen independently of all historical protocols."""
CONFIG={
 "stage":"P2C-2C1A","seed":621019,"family_splits":{"family00":"train","family01":"train","family02":"development","family03":"development"},
 "families":[[1.40,.26,.26,.14,2.05,4.65],[1.55,.29,.28,.18,2.22,4.88],[1.48,.27,.29,.16,2.38,5.11],[1.67,.30,.26,.20,2.54,5.34]],
 "layouts":["free","four_way","left_only","right_only","up_only","down_only","narrow3d","multi3d","blocked"],
 "goals_train":[[2.40,0,0],[2.80,.45,.45],[3.10,-.45,-.45],[5.60,0,0],[6.10,1.0,.9],[5.80,-1.0,-.9]],
 "goals_development":[[2.55,.15,.10],[2.95,.55,-.35],[3.05,-.55,.40],[5.90,.25,-.20],[6.30,1.1,1.0],[5.50,-1.1,-1.0]],
 "velocities_train":[[.8,0,0],[1.7,.12,-.08]],"velocities_development":[[1.,-.08,.06],[1.8,.10,0]],
 "goal_semantics":"local rolling reference in raw B0 metres; deterministic near-arrival / far-progress evaluation, not global hard terminal equality",
 "near_distance_threshold_m":3.4,"near_endpoint_tolerance_m":.15,"near_terminal_speed_max":.25,
 "far_min_distance_reduction_m":1.0,"far_min_projected_progress_m":1.0,"far_min_direction_cosine":.5,"far_max_terminal_ray_offset_m":.8,
 "expert_far_endpoint_radius_m":2.85,"expert_far_terminal_speed_mps":1.2,"expert_near_terminal_speed_mps":0.,
 "body_radius_m":.12,"required_margin_m":.03,"speed_limit":3.,"acceleration_limit":8.,"jerk_limit":100.,
 "min_total_time_s":.8,"max_total_time_s":2.2,"min_segment_s":.12,
 "seed_amplitudes_m":[.60,.75,.48],"seed_total_times_s":[2.18,2.02],
 "max_optimization_steps":140,"optimizer_lr":.025,"optimizer_sample_points_per_piece":49,
 "verification_points_per_piece":1025,"signed_witness_points_per_piece":2049,
 "max_experts":6,"dedup_common_physical_time_RMS_m":.25,
 "camera":{"K":[[100.,0.,79.5],[0.,100.,63.5],[0.,0.,1.]],"optical_to_B0_rotation":[[0.,0.,1.],[-1.,0.,0.],[0.,-1.,0.]],"origin_B0":[0.,0.,0.],"depth_range_m":[.04,20.],"normalization":"optical Z /20 + valid mask"},
 "small_overfit":"family00 free all6 goals velocity0 + family00 four_way goal0 and goal3 velocity0; exact8 train IDs",
 "unresolved":"M=0 search failure is unknown, NEVER proven infeasible or a pseudo-negative teacher",
 "physical_infeasibility":"only analytic full-cross-section separating slab proof for exact same task contract",
 "max_scene_count":432,"network_optimizer_steps":0,"new_generator_modules":False,"S1_modified":False,"blind_test":False,"P2C2C1B_started":False
}
