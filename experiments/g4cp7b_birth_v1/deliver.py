from pathlib import Path
import json,hashlib
r=Path.cwd();o=r/'research_outputs/g4cp7b_birth_execution_20261112_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
e=json.loads((o/'ENTRY.json').read_text());assert all(sha(p)==h for p,h in e['source'].items())
b=json.loads((o/'SOURCE_BINDING_v3.json').read_text());assert all(sha(p)==h for p,h in b.items())
(o/'HISTORICAL_PROTECTION.json').write_text(json.dumps(dict(status='PASS',old_source_files=len(e['source']),scope='preexisting mainline source SHA;outside/historical outputs full rehash NOT_RUN,no writes issued'),indent=2))
reports={
'BIRTH_AUTHORIZATION_AND_SCOPE.md':"""# Explicit finite-volume engineering premise
Researcher authorized finite birth ball with recorded volume/use.
Preregistered center[0,0,2]m,radius0.60m,world frame,native seed3/maze5/tree_dist4,resolution0.1.
Issued only while HELD and zero integration,from independent native /mock_map clearance lower bound>=0.60m and birth position agreement<=0.005m.
Observer subtracts full0.1*sqrt(3) voxel extent reserve;point cloud finite AABB bounds enforced. Raw map bytes SHA in every heartbeat binds certificate.
Valid30s,only current episode/map identity. Object/map change invalidates,never expands volume.
No GT supplied to network/search. Deterministic world+X goal fixed0.28m;all centerline+0.30m body/tracking tube+original speed interpolation padding must fit this ball.
For this narrow E1-A experiment GT establishes a bounded launch volume,NOT online obstacle avoidance or P4 strict sensing proof.
The source depth/odom remain real and freshness guarded;depth does not provide the launch ball itself.
Outside birth volume is UNKNOWN and rejected. E1-B/E2 cannot use this permission to create a corridor.
No actual scene-clearance confirmation occurred this turn. Radius0.60 is not tuned after simulation success;insufficient clearance rejects without reseeding.
""",
'FOREST_EXECUTION_IMPLEMENTATION.md':"""# Forest E1-A execution assembled
Private master11363;original ~/YOPO/Simulator native CUDA renderer with actual native topics/remaps and correct cwd.
Native position nodelet from unchanged P6 original-position-controller extraction,unchanged gains/mass/dynamics.
Versioned gated native Host supplies producer-stamped IMU,HELD/ACK/ARM,RPM/final-command ACK and native integration. Prespun equilibrium_hover explicitly retained,not zero-RPM takeoff.
Owner/native truth/reference/admission are versioned P7b;P6/P7 evidence untouched.
Map/KD processing in bounded worker,not odom journal lock. Identical map skips geometric rebuild;new map latches failure and fences the launch premise.
Truth ready+fresh+birth clearance required BEFORE ARM. During motion truth failure/staleness routes to same owned Plant freeze/kill and actual exit confirmation.
Reference fixes target once at anchor+.28X,builds original3-piece T2 MINCO;replans once after1.8s from actual PVA to same target.
Admission independently checks original MINCO/PVA/time/switch/dynamic limits plus birth tube at each request,including hold.
D acceptance adds>=2 actual plans to unchanged old progress>=.25,error<=.10,max tracking<=.15,terminal speed<=.10,v<=1,a<=8,j<=100.
Actual tracking/switch/clearance/latency NOT_RUN;no physical success claimed.
Native map point cap2M may reject original scene. Map extent/projection correspondence require actual stream validation. No silent point subsampling.
ROS1 trusted local graph assumed;not authentication against malicious publishers.
""",
'SAFETY_REQUALIFICATION.md':"""# New-chain gate
C phase now seven cases:original6 plus truth_failure.
Last case injects independent observer failure only after real ARMED feedback;owner must detect matching failure,confirm Plant stop within existing bound and cleanly exit.
Controlled truth FAIL after injection is expected in that case only;prior truth failures cannot pass.
Specific supervisor BIRTH_TRUTH_INVALID rejection also accepted for same injected case;arbitrary REQUEST failures not accepted as truth-stop proof.
D refuses old P6 six-case C and source mismatches. Requires7 SHA-bound episodes with actual physical feedback and PASS.
Static tests5 PASS and no-ROS check-only PASS in tests_v3/logs/qualify_v3. Covers sphere escape,expiry,map change,ideal PVA switch and specific fault mapping.
No actual new C/D/forest run performed.
Original P4 certification remains FAIL,older P6 physical PASS remains genuine but not substitute for new scene/safety chain.
""",
'FINAL_GO_NO_GO.md':"""# Status
FINITE_BIRTH_VOLUME_AUTHORIZED=YES
BIRTH_CLEARANCE_LIVE=NOT_RUN
NATIVE_FOREST_EXECUTION_CODE=IMPLEMENTED
NEW_CHAIN_C7_STATIC_PREFLIGHT=PASS
NEW_CHAIN_C7_PHYSICAL=NOT_RUN
E1_A_MOTION=CONDITIONAL_ON_C7_AND_LIVE_BIRTH_CHECK
E1_A_RESULT=NOT_RUN
E1_B_AVOIDANCE=NOT_IMPLEMENTED_NO_GO
E2=NO_GO
P4_STRICT_CERTIFICATION=FAIL_UNCHANGED
NEW_NETWORK_UPDATES=0
NEW_PHYSICS_RUNS=0
HISTORICAL_SOURCE_PROTECTION=PASS
Commands are supplied for researcher execution,not evidence of success.
"""
}
reports['MANUAL_COMMANDS.md']="""# Manual native forest phases
No physical process launched by Codex. Use same shell,do not automatically chain motion after fault tests.
C7 performs real hover/faults in fixed native forest. It will refuse ARM if the0.60m birth sphere is not established.
No ROS setup is required in caller;owned launcher sets environment and refuses occupied master11363.

Phase1:new native scene seven-case safety qualification:
~~~bash
cd /home/lizheng/query_minco_mainline
PY=/home/lizheng/anaconda3/envs/yopo/bin/python
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export P7B_SAFE="research_outputs/p7b_native_safety_$(date +%Y%m%d_%H%M%S)_$$"
"$PY" -B -m query_minco.audit.run_provenance_v1 --output "$P7B_SAFE/process" --cwd "$PWD" --timeout-s 600 -- "$PY" -B experiments/g4cp7b_birth_v1/manual.py --stage C --allow-physics RESEARCHER_MANUAL_C_ONLY --output "$P7B_SAFE/run"
echo "$P7B_SAFE/run/RESULT.json"
~~~

Review final PASS,all7 episode status/STOP_CONFIRMED,truth heartbeat,BIRTH_CERTIFICATE,map_id/process exits. Do not proceed on failure.
After researcher review,Phase2:E1-A bounded launch-ball0.28m tracking with actual second plan:
~~~bash
cd /home/lizheng/query_minco_mainline
test -n "$P7B_SAFE" || { echo "Set P7B_SAFE to actual successful new safety run"; exit 2; }
PY=/home/lizheng/anaconda3/envs/yopo/bin/python
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
P7B_E1="research_outputs/p7b_native_E1A_$(date +%Y%m%d_%H%M%S)_$$"
"$PY" -B -m query_minco.audit.run_provenance_v1 --output "$P7B_E1/process" --cwd "$PWD" --timeout-s 120 -- "$PY" -B experiments/g4cp7b_birth_v1/manual.py --stage D --allow-physics RESEARCHER_MANUAL_D_ONLY --prior "$P7B_SAFE/run/RESULT.json" --output "$P7B_E1/run"
echo "$P7B_E1/run/RESULT.json"
~~~
CLI internal C/D denote safety/tracking engine phases;D here is E1-A in a native forest birth ball,not old P6 open0.4m task.
Do not edit source between phases;latest frozen binding SOURCE_BINDING_v3.json is authoritative. Earlier bindings retained as development snapshots.
E1-B/E2 commands not offered. Failed birth/map-size/deadline/switch rejects must be preserved,not corrected by reducing radius/tracking envelope or changing seed.
"""
for name,text in reports.items():(o/name).write_text(text)
files=[p for a in e['allowed']+[str(o.relative_to(r))] for p in (r/a).rglob('*') if p.is_file()]
(o/'FILES_AND_SHA.json').write_text(json.dumps({str(p.relative_to(r)):sha(p) for p in files},indent=2))
print('P7b manual C7/E1-A delivered;all physical NOT_RUN;source checks PASS')
