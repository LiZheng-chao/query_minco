from pathlib import Path
import json,hashlib,subprocess
R=Path.cwd();O=R/'research_outputs/g4cp4_realtime_20261112_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
e=json.loads((O/'ENTRY.json').read_text());q=json.loads((O/'QUALIFICATION.json').read_text());s=json.loads((O/'SAFETY.json').read_text())
source_changes=[p for p,h in e['source'].items() if sha(p)!=h];history_changes=[p for p,h in e['history'].items() if sha(p)!=h]
git={n:subprocess.check_output(['git']+a,text=True) for n,a in [('head',['rev-parse','HEAD']),('diff',['diff']),('cached',['diff','--cached'])]}
assert not source_changes and not history_changes and all(git[k]==e['git'][k] for k in git)
protection=dict(status='PASS',protected_source_files=len(e['source']),protected_P3_and_E0_files=len(e['history']),source_changes=source_changes,history_changes=history_changes,git_unchanged=git,
 scope='All preexisting mainline source and selected P3/E0 evidence checked. Other historical outputs and outside repository assets not rehashed this round;no writes issued there.')
(O/'HISTORICAL_PROTECTION.json').write_text(json.dumps(protection,indent=2))
reports={
'P4_BOOTSTRAP_SAFETY_DESIGN.md':"""# Bootstrap: NO-GO
Actual E0:101frames,identical depth,213/213 initial sphere cells UNKNOWN. Repeating E0 is not prescribed.
Forward camera cannot observe rear hemisphere,near plane or occluded near cells. Software cannot infer FREE from absence of evidence.
External evidence schema implemented in observed_space_v2/bootstrap.py. This validates bytes,finite bounds,source references,expiry and ball containment only.
It explicitly returns motion_authority=False and NEVER inserts FREE into the map.
No independent measurement device/data/calibration is installed or qualified in this workspace. Initial physical safety remains FAIL.

Proposed independent measurement contract (DESIGN_ONLY):
calibrated near-field multi-direction range scanner or physically measured launch enclosure;world registration bound,range/angular uncertainty,scan time and stationary-scene seal.
Measure a bounded ball at the actual launch pose;require measured radius minus total error >=0.30m. Example0.40m radius with0.02m bound is a schema test,NOT a real measurement.
Evidence must identify raw scans,calibration,device,operator/reviewer,frame transform,scene epoch and monotonic/wall-time binding.
Validity at most60s in current schema;actual acceptable lifetime must be preregistered based on controlled static environment.
Movement of objects,scanner/vehicle or scene epoch invalidates the independent certificate.
No current algorithm combines this schema result into online permission. A truthful independently qualified measurement authority is still missing.
Never expand this initial volume to a corridor. Outside it requires real online depth;unknown rejects.

Alternatives:
ring/near-field cameras:requires physical extrinsics,overlapping near coverage,synchronization and conservative ray certification.
Independently rotating camera:requires real actuator encoder/poses and a mount that can scan without arming the vehicle;native body-mounted forward renderer cannot simulate this by fake odom.
No such hardware/native sensor mechanism currently exists here. Do not rotate the vehicle before proving its safe start.
Paper capability would be 'planning with independently measured launch region and depth-observed space',not arbitrary blind-start autonomy.
Failure tests cover empty evidence,expiry,outside-volume,corridor scope,changed raw bytes. SHA authenticates bytes,not measurement truth.
""",
'P4_OBSERVED_SPACE_CODE_AUDIT.md':"""# Code audit
New package observed_space_v2 reuses frozen v1 carving,MINCO,polynomial evaluator,controller and Hard. No old implementation overwritten.
core.sample_batch evaluates original polynomial coefficients through authoritative evaluation.validation.evaluate,with original piece-boundary search(side=right),durations,world rotation and origin.
PVA equivalence tested over129 points and exact segment ends to1e-11.
Prepared caches the129 original-grid values for Snapshot.certify,which retains radius,tracking tube,sample count and continuous speed padding.
Repeated-content frames still have distinct source stamps and epochs. Only unchanged pose/quaternion/depth identity reuses frame masks,minfilters and hit tree.
Frame construction still performs some hit preprocessing before cache reuse:remaining optimization opportunity,not claimed eliminated.
Changed obstacle/content takes fresh geometry;rollback/expiry behavior inherited.
No GT imports in the new online modules. External measured evidence never modifies occupancy.
Current implementation is reusable backend code,NOT a completed ROS E1/E2 port. Existing frozen v5 remains untouched.
""",
'P4_REALTIME_OPTIMIZATION.md':"""# Measured offline performance
See QUALIFICATION.json for full target-machine P50/P95/P99/max.
Paired101-frame actual E0 replay,original and optimized use identical raw arrays and exact recorded poses.
Original map P50/P95/P99/max ms:56.232/60.522/65.256/65.519.
Optimized map:5.659/6.771/7.535/52.263;101/101 accepted,0 map150ms deadline violations in both runs.
Final state arrays identical. Per-frame full state equivalence and long moving-history performance remain NOT_RUN.
129-point PVA original P50/P95/P99/max ms:25.021/27.405/28.657/28.963.
Batch:0.501/0.574/0.602/0.610 over30 iterations.
The earlier56.50ms single sample was a different machine load/environment;not contradicted or erased.
Current run uses pinned single-thread BLAS. These numbers do not establish scheduling under ROS/CUDA/control contention.
Full live sensor-to-map-to-plan-to-authorization-to-control P50/P95/P99/max:NOT_RUN.
Actual5Hz/10Hz success rate,deadline counts and100Hz continuity:NOT_RUN.
Initial safety is blocked so no actual motion performance can be inferred from this benchmark.
""",
'P4_SCHEDULING_AND_WATCHDOG.md':"""# Scheduling
AsyncCertificates implements one worker and at most one outstanding task. Pending,failed,wrong-plan,expired or mismatched-epoch results never authorize.
The consumer only reads completed proof and current epoch/expiry;no map reconstruction or polynomial solve in consume.
Prepared creation is included in certificate50ms budget;map retains150ms. No limits increased.
Snapshot/Frame use unchanged conservative geometry. Updated content immediately advances epoch even when worker is running.
Thread shutdown waits for worker;SciPy/NumPy deadlines remain cooperative,not hard preemption.
A map update can retain the inherited ledger RLock during bounded reconstruction;ingestion/control callback lock separation in ROS remains unimplemented.
The worker is NOT wired into the frozen v5 supervisor/reference. No claim of100Hz control-path isolation in a real process.
A production port must ingest bounded source frames outside control locks,immediately fence permission on ingress/new obstacle,serialize current proof+publication atomically,and terminate Plant on absent/failing proof.
Do not hold the last permission while worker catches up. Timeout/watchdog/Plant termination and callback teardown require new physical fault qualification.
Static tests verify actual completed proof,wrong plan,expiry and new epoch revocation;missing-map worker failure also rejects.
""",
'P4_V5_PHYSICS_REQUALIFICATION.md':"""# V5/new backend requalification
V4 C six cases and D0.4m PHYSICAL_PASS remain historical evidence with their original binding.
V5/new backend real qualification:NOT_RUN.
No ROS safety/permission/trajectory publisher source was changed this round;new backend is not an authorized physical execution chain.
Future adapter changes proof scheduling,ingress invalidation and publication race behavior. Old V4 PASS cannot authorize that port.
Required:static race tests,recorded timestamp replays,then affected request timeout,depth loss,rollback,NaN,planner/controller exit under real armed feedback;actual Plant stop must be confirmed.
After same-identity fault qualification,run same-identity D with original progress/error/speed limits,then E1.
No real ARM or fault injection was executed by Codex.
""",
'P4_FOREST_E1_E2_READINESS.md':"""# E1/E2 readiness
NO-GO. Initial independent measurement unavailable;new ROS scheduler integration and same-source physics qualification incomplete.
Existing versioned v5 E1/E2 code is preserved,not promoted to runnable qualified P4.
This delivery supplies optimized reusable map/sampling/async interfaces,not a complete executable ROS P4 E1/E2 system.
E1 needs real native depth/odom,at least one actual replanning/switch,100Hz control,independent collision monitor and confirmed stop on rejected proof.
E2 requires E1 plus>=20 valid replans,>=2m,<=1m/s and unchanged continuous limits.
All actual E1/E2 outcomes,tracking,clearance and switch continuity remain NOT_RUN.
No movement commands are released before missing prerequisites pass.
""",
'P4_TEST_MATRIX.md':"""# Tests
logs/qualify and logs/safety preserve argv,cwd,environment,stdout/stderr,true child exit and sampled process-tree resources.
qualify:exit0,7 tests plus paired101-frame replay and30x sampling benchmarks.
safety:exit0,2 tests;bounded external certificate schema and async proof epoch/expiry.
STATIC_PASS9 total. No fake clock used for performance figures;synthetic logical times only in contract fixtures.
Actual initial independent measurement:NOT_RUN. E0 evidence review confirms initial UNKNOWN FAIL.
Actual ROS5Hz/10Hz/100Hz:NOT_RUN.
Native compile:NOT_RUN;Python modules compiled/imported by tests.
Full existing K1/K6/Hard regression:NOT_RUN,untouched files protected by SHA.
Long moving-history and adversarial depth-noise qualification:NOT_RUN.
No network or optimizer update;no ARM.
""",
'P4_MANUAL_COMMANDS.md':"""# Next executable validation
Do not rerun stationary E0:it cannot add directional coverage.
Next hardware task:inventory an independent near-field/surround measurement device or measured launch enclosure,provide its calibration,raw scans and world registration. No such device evidence exists here.
Next software task:port the tested AsyncCertificates backend into a new ROS execution version with source-bound ingress invalidation and atomic proof/publication;then replay sensor/control timing before manual fault tests.
Offline replay command (new exclusive log directory is required;qualification JSON output already exists,so do not overwrite it):
use existing logs/qualify and logs/safety for this completed run. No repeat is needed.
No E1/E2 command is provided because initial safety and same-version ROS/physics qualification have not passed.
No fabricated CLI,ROS topic or nonexistent certificate is offered as ready.
""",
'P4_FINAL_GO_NO_GO.md':"""# Final NO-GO
Initial sensor observability:FAIL. Independent measured evidence:NOT_RUN/no installed qualified device.
Batch math:STATIC_PASS. Static cache/fail-closed/async/schema tests:STATIC_PASS9.
Paired stationary E0 replay:STATIC_PASS,significant measured speedup;live end-to-end realtime:NOT_RUN.
P4 ROS integration:INCOMPLETE. Same-version V5 physical C/D:NOT_RUN.
E1/E2:NOT_RUN. No actual ARM,network update or optimizer update.
Historical protection:PASS for checked mainline sources and P3/E0 files;scope and counts in HISTORICAL_PROTECTION.json.
This is a partial backend implementation and measured optimization delivery,not complete runnable forest motion infrastructure.
Next concrete prerequisite is independent startup measurement evidence,alongside new ROS scheduling integration. Keep NO-GO;do not infer real control success from offline latency.
"""
}
for name,text in reports.items():(O/name).write_text(text)
files=[p for a in e['allowed'] for p in (R/a).rglob('*') if p.is_file() and p.name!='FILES_AND_SHA.json']
(O/'FILES_AND_SHA.json').write_text(json.dumps({str(p.relative_to(R)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in files},indent=2))
print('DELIVERY',len(reports),'reports',q['status'],s['status'],protection['status'])
