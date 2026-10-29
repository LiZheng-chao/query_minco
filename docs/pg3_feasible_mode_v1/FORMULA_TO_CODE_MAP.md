# PG3 proposed mathematics -> implementation
All hypotheses remain unproven until actual experiments. Architecture is unchanged.

| Contract | Code | Gradient / units / reduction |
|---|---|---|
| original Hungarian | pg1.objective.forward -> training.losses.imitation.cost_matrix -> matching.hungarian.match | original cost and matched mean preserved, never double-counted |
| capacity assignment D | pg3_feasible_mode_v1/assignment.py:assign | detached physical W/P squared distances (reference 1m); residual capacity2 includes original Hungarian; rotating subset of bad unmatched; safe unmatched remains -1 |
| L_unmatched_mode | losses.py:mode_query + objective.py:augment | decoded two W, tailP, tailV; component Huber mean xyz, SUM two W; alphaV=.1; per-selected-query mean per sample then batch mean; sigmaW/P=1m,V=1m/s |
| waypoint diagnostic | losses.py:mode_query second return | two-inner mean; NOT added twice |
| Teacher labels | teacher_labels.py:TeacherBank,representability,inverse_raw | original hard-certified Teachers, immutable state+geometry+tensor digest; no labels from search failures |
| bottleneck | teacher_labels.py:landmarks; bottleneck.py:bottleneck_loss,at_progress | authenticated obstacle x-midplane, forward non-tangent Teacher crossings within .5m; Teacher progress detached; prediction p(s*T) differentiable c/T; perp Huber plus .1 axial Huber, per-query event mean then sample/batch mean |
| power -> Bernstein | corridor.py:bernstein | B_k=sum_i<=k comb(k,i)/comb(5,i)*c_i*T^i; xyz/degree axes tested; c/T gradient |
| local convex region | corridor.py:polytope,verify_polytope,build_corridor | full flight-domain halfspaces plus one separating axis face for EVERY AABB inflated by .15m+1e-6; original Teacher Bernstein hull must be contained; recursively split to depth4 |
| convex penalty | corridor.py:penalty + objective.py:augment | De Casteljau exact corresponding Teacher segment/subfraction; normalized unit normals, Huber(ReLU(residual)/.15m); mean all valid control-point*planes per Query, then Query/sample mean; missing proof gives zero and explicit missing reason, never a certificate |
| continuous_safe | existing pg2_guided_v2 objective/losses unchanged | clearance/braking/retreat and old discrete Teacher-ball term retained as separate weak baseline; fixed old train-calibrated weights |
| Goal/dynamics/domain | unchanged PG2 Safe computation | old task numerical values and blocked-gradient policy retained; v3/a8/j100; body.12/margin.03 |
| conditional control | losses.py:conditional_control | original exact four-node polynomial Ea/Ej; qa=Ea/(8^2*2.2), qj=Ej/(100^2*2.2), qt=T/2.2; detached hard+Goal gate (gamma0); sum gate*term / max(1,sumgate) |
| lateral quality | losses.py:conditional_control separate lateral term | ||v cross a||²/(||v||²+.1²)/4², physical-time mean; separately weighted, not silently included in QA/QJ/QT |
| optional diversity | losses.py:diversity | only TWO certified distinct face signatures at same obstacle section; delta=.5 Teacher transverse separation; M1/unknown identity ->zero; not interpreted as homotopy |
| gradients / protection | gradients.py:audit; execution.py:run | raw, coeff, duration, parameter layers and four classes; strict finite, raw19/20 frozen; original ratio>10 two consecutive stops, clip10 |
| method/authorization | spec.py + execution.py:make_spec,validate,authorization | independent canonical RunSpec identity, frozen source/data/Teacher lineage, exclusive claims/outputs; CPU only |
Direct Mode bypasses MINCO coefficients: partial derivative dL_mode/dcoeff is structurally None, while dL_mode/dW/raw/model is nonzero. Bottleneck/corridor route gradients pass through coefficients/durations to raw/model. Teacher and hard labels never get gradients.

Bottleneck timing is a Teacher-progress proxy, not an implicit spatial root gradient. Multiple crossings use reproducible first forward root; no crossing/tangent/coincident segment disables that event. Original hard checker is still authoritative. Different Teacher indices alone do not establish different topologies.
