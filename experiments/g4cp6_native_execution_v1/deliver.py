from pathlib import Path
import json,hashlib
r=Path.cwd();o=r/'research_outputs/g4cp6_native_execution_20261112_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
e=json.loads((o/'ENTRY.json').read_text());assert all(sha(p)==h for p,h in e['source'].items())
binding=json.loads((o/'SOURCE_BINDING.json').read_text());assert all(sha(p)==h for p,h in binding.items())
(o/'HISTORICAL_PROTECTION.json').write_text(json.dumps(dict(status='PASS',protected_source_files=len(e['source']),scope='mainline and YOPO source SHA;historical outputs full rehash NOT_RUN;no old writes'),indent=2))
host=r/'research_outputs/g4cp1_startup_control_20261111_v1/build_equilibrium_v3'
(o/'HOST_IDENTITY.json').write_text(json.dumps({str(p):sha(p) for p in [host/'gated_host',host/'gated_host.cpp',host/'BUILD_IDENTITY.json',host/'DERIVATION.json',host/'plant_gate.hpp']},indent=2))
reports={
'P6_NATIVE_SHADOW_RESULTS.md':"""# Native live status
Original uninstrumented S0/Shadow NOT_RUN;no ROS physics started.
Installed ROS definitions loaded without ROS master,types/MD5/slots in tests_v2/MESSAGE_TYPES.json. Live connection/rates NOT_RUN.
Native original IMU producer omits stamp. Reused versioned Host fixes it at actual state publication with imu.header.stamp=tnow,same sample as odom.
No receipt-time restamp or finite-difference acceleration is used for executable A/B.
Host is explicitly instrumented original dynamics,not untouched native executable.
""",
'P6_EXECUTION_BRIDGE.md':"""# Executable A/B workflow
manual.py implements owned master11359/process launch,HELD/PREPARE/ACK/ARMED,real reference/admission/controller/SO3/Plant ACK,truth monitoring,Plant-first stop and cleanup.
Reuses frozen v4 lifecycle/readiness/admission/reference/fault relay/truth monitor and original MINCO.
Controller launch extracts original simulator_position_control.launch SO3ControlNodelet with every non-init parameter unchanged;only namespace and initial coordinates resolved.
Combined original launch is split to make exact Plant process exclusively owned. No second dynamics/controller created.
Original dynamics/mass/gains unchanged;Host adds startup gate,source IMU stamp,watchdog,RPM/force/final-command ACK and disclosed equilibrium_hover initialization.
CLI C=S1-A six fault episodes after healthy real hover feedback;D=S1-B fixed0.4m. D requires matching new C binding;old v4 PASS rejected.
No ordinary P5 candidate subscriber;requests enter original Admission,raw SO3 passes readiness before final topic.
ROS1 graph is trusted local infrastructure,not secure against hostile publishers.
A/B use frozen native open fixture,not original random forest seed3. Forest stage disabled.
""",
'P6_NATIVE_CONTROL_AUDIT.md':"""# Audited failures and fixes
Native stamp0:reuse producer-fixed Host,do not patch original YOPO or restamp at callback.
Stale command reuse:original nodelet unchanged;external readiness+Host watchdog+owned freeze/kill stops integration. Stopping publication alone is not braking.
Cold start:first integration prohibited until five distinct ACK and explicit permit. equilibrium_hover is pre-spun,not zero-RPM takeoff.
Rolling target:P5 shadow goal followed pose;executable reference anchors target once to initial state+[.4,0,0],verified by regression.
Original Admission checks independent MINCO,continuous1/8/100 limits,initial PVA,plan age,command race,switch continuity and envelope.
Host ACK exposes actual final command stamp,world force,RPM and integration count;new live consumption NOT_RUN.
Truth A/B uses declared bounded open floor fixture and conservative inter-sample bound,no GT route oracle.
Random forest truth/observed-space integration remains incomplete.
Owned process group/family/start-time accounting and parent-death stop reused;nonphysical actual sleeper process stop unit PASS.
No old source/controller gain/Hard threshold changed.
""",
'P6_SAFETY_STOP_TESTS.md':"""# Tests
Final tests_v2:31 STATIC_PASS,logs/qualification_v2 true exit0.
Includes original runtime/curve contracts and native-controller param equality,actual owned sleeper stop,fixed target,IMU producer stamp and staged owner.
Initial test failed only on spaces around C++ assignment;corrected whitespace-normalized assertion;failed log retained.
PositionCommand installed MD5 4712f0609ca29a79af79a35ca3e3967a;SO3Command a466650b2633e768513aa3bf62383c86.
Other types/MD5 in MESSAGE_TYPES.json. Live ROS connection NOT_RUN.
Python/XML validation and no-ROS check-only PASS.
No fresh C++ compile claimed:immutable built Host reused with compiler/build/derivation provenance and SHA.
Actual Plant faults,control rate,latency and cleanup NOT_RUN for P6. Old historical-output-writing test excluded.
""",
'P6_MINCO_TRACKING_READINESS.md':"""# Manual readiness
Code/preflight ready for researcher S1-A;physical PASS NOT_RUN.
S1-B only after new exact-binding C campaign PASS;old v4 C is not authority.
Original3-piece MINCO,T2,actual source-stamped native world PVA,constant0.4m target.
Thresholds unchanged:progress>=.25m,terminal error<=.10m,max tracking error<=.15m,terminal speed<=.10m/s,v<=1,a<=8,j<=100.
Do not edit source/config/binary between A and B. Source identity verified before launch.
""",
'P6_FOREST_E1_RUN_PLAN.md':"""# Forest NO-GO
S1-C/S2 disabled until actual new A/B PASS and forest monitor/admission integration.
P5 fixed collision-free birth is an engineering premise,not P4 strict observed-space certification.
P4 FAIL retained,without using missing independent measurement as the sole blocker for engineering A/B.
Random-map identity/truth observer,sensor-only observed-space admission and switch-aware replanning still need same-version integration/qualification.
No UNKNOWN->FREE,GT online route,network update or new anchor.
S1-C>=1 actual switch;S2>=20 valid replans and>=2m,<=1m/s. All outcomes NOT_RUN.
""",
'P6_FINAL_GO_NO_GO.md':"""# Decision
S1_A_MANUAL_PREFLIGHT=PASS
S1_A_PHYSICAL=NOT_RUN
S1_B_CODE_IMPLEMENTED=YES
S1_B_AUTHORITY=REQUIRES_MATCHING_P6_A_PASS
S1_B_PHYSICAL=NOT_RUN
S1_C_FOREST=NO_GO_NOT_ENABLED
S2=NOT_RUN
ORIGINAL_UNINSTRUMENTED_S0=NOT_RUN
HOST=VERSIONED_INSTRUMENTED_NATIVE_PRESPUN
DYNAMICS_AND_GAINS=UNCHANGED
STATIC_TESTS=PASS31
HISTORICAL_SOURCE_PROTECTION=PASS
NEW_NETWORK_UPDATES=0
NEW_PHYSICS_RUNS=0
A/B manual executable delivery is complete;full forest integration is not.
"""
}
reports['P6_MANUAL_COMMANDS.md']="""# Researcher manual physics commands
Use one shell. Never automatically chain B after A.
A includes real physics and six faults under explicit authority,with equilibrium_hover pre-spun initial condition.
Master11359 refuses occupied port. Do not kill unrelated ROS. A/B use frozen open native-rendered fixture,not forest.

S1-A:
~~~bash
cd /home/lizheng/query_minco_mainline
PY=/home/lizheng/anaconda3/envs/yopo/bin/python
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export P6_S1A="research_outputs/p6_manual_A_$(date +%Y%m%d_%H%M%S)_$$"
"$PY" -B -m query_minco.audit.run_provenance_v1 --output "$P6_S1A/process" --cwd "$PWD" --timeout-s 600 -- "$PY" -B experiments/g4cp6_native_execution_v1/manual.py --stage C --allow-physics RESEARCHER_MANUAL_C_ONLY --output "$P6_S1A/run"
printf 'S1_A_RESULT=%s/run/RESULT.json\n' "$P6_S1A"
~~~

Review run/RESULT.json,six episode RESULT/STOP_CONFIRMED and process logs. Only final PASS with same SHA allows B.
In another shell set P6_S1A explicitly to the actual printed new directory;never use old v4 C.

S1-B:
~~~bash
cd /home/lizheng/query_minco_mainline
test -n "$P6_S1A" || { echo "Set P6_S1A to successful P6 A directory"; exit 2; }
PY=/home/lizheng/anaconda3/envs/yopo/bin/python
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
P6_S1B="research_outputs/p6_manual_B_$(date +%Y%m%d_%H%M%S)_$$"
"$PY" -B -m query_minco.audit.run_provenance_v1 --output "$P6_S1B/process" --cwd "$PWD" --timeout-s 120 -- "$PY" -B experiments/g4cp6_native_execution_v1/manual.py --stage D --allow-physics RESEARCHER_MANUAL_D_ONLY --prior "$P6_S1A/run/RESULT.json" --output "$P6_S1B/run"
printf 'S1_B_RESULT=%s/run/RESULT.json\n' "$P6_S1B"
~~~

No physical command above was run by Codex. On any failure retain output and do not advance.
Do not remap candidate topics to controller. Forest motion commands not released.
"""
for name,body in reports.items():(o/name).write_text(body)
files=[p for folder in e['allowed']+[str(o.relative_to(r))] for p in (r/folder).rglob('*') if p.is_file()]
(o/'FILES_AND_SHA.json').write_text(json.dumps({str(p.relative_to(r)):sha(p) for p in files},indent=2))
print('8 reports;manual A/B executable;forest NO_GO;protected sources',len(e['source']))
