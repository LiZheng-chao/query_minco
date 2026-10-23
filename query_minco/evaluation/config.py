"""Frozen controlled-data geometry and kinematic contract (not flight certification)."""
CONFIG = {
    "version": "p2c0_v1", "source": "synthetic_controlled_aabb_minco",
    "seed": 42000, "image_height": 128, "image_width": 160,
    "K": [[100.,0.,79.5],[0.,100.,63.5],[0.,0.,1.]],
    "optical_to_B0_rotation": [[0.,0.,1.],[-1.,0.,0.],[0.,-1.,0.]],
    "camera_origin_B0_m": [0.,0.,0.], "depth_range_m": [.04,20.],
    "state": [1.4,0.,0.,0.,0.,0.,2.8,0.,0.,0.,0.,-1.],
    "total_time_s": 2., "min_segment_time_s": .12,
    "position_decoder_bounds_m": [[0.,-4.,-2.],[10.,4.,2.]],
    "flight_center_domain_m": [[0.,-1.65,-.45],[10.,1.65,.45]],
    "body_radius_m": .12, "required_surface_margin_m": .03,
    "max_speed_mps": 3., "max_acceleration_mps2": 8., "max_jerk_mps3": 100.,
    "boundary_atol_pva": [.0002,.002,.01],
    "continuity_atol_orders_0_to_4": [.0002,.002,.01,.1,1.],
    "max_certificate_depth": 18, "dedup_rms_m": .25,
    "route_amplitudes_m": [0.,.5,-.5,.6,-.6,.7,-.7],
    "parameter_groups": {
        "train": [[1.35,.26,.16],[1.40,.28,.18],[1.45,.30,.20],
                  [1.50,.32,.16],[1.55,.34,.18],[1.60,.36,.20]],
        "val": [[1.28,.27,.17],[1.32,.31,.19],[1.58,.35,.21]],
        "test": [[1.22,.29,.15],[1.26,.33,.22],[1.62,.37,.24]]},
    "held_out_layout": "staggered_double", "training_authorized": False
}
