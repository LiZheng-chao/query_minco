from pathlib import Path
import json,hashlib
r=Path.cwd();o=r/'research_outputs/stop_latency_fix_20261116_v1';e=json.loads((o/'ENTRY.json').read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
changes=[p for p,h in e['source'].items() if sha(p)!=h]
assert not changes
old=r/'isolated_worktrees/e1b_fix2_20261112_v1/research_outputs/singleclock_C_20261112_144517_1276/run/RESULT.json'
d=json.loads(old.read_text());assert all(sha(p)==h for p,h in d['source_binding'].items())
b=json.loads((o/'SOURCE_BINDING_v2.json').read_text());assert all(sha(p)==h for p,h in b.items())
(o/'HISTORICAL_PROTECTION.json').write_text(json.dumps(dict(status='PASS',protected_source_files=len(e['source']),old_failed_C_binding='PASS',changes=changes,scope='preexisting mainline and accepted execution-worktree Python source;other historical outputs/outside repository full rehash NOT_RUN,no writes issued'),indent=2))
reports={
'STOP_LATENCY_ROOT_CAUSE.md':"""# Actual failed C,not qualification PASS
request_timeout:injection->Supervisor latch202.274ms,latch->owner stop1.671ms,stop->exit confirmation2.637ms,total206.582ms.
depth_loss:injection->Supervisor latch181.379ms,latch->owner stop51.315ms,stop->exit confirmation174.463ms,total407.157ms.
232.693ms is injection->owner stop request,not the first Supervisor latch.
Both expected fault type,actual termination-9 and cleanupPASS are preserved. Depth case exceeds350ms and remainsFAIL;five cases NOT_RUN.
Old SIGSTOP-send/pause/SIGKILL/waitpid timings NOT_RECORDED. We cannot infer whether174.5ms was stop scheduling,kernel exit or polling.
Confirmed blocking code:owner enumerates /proc for every child before reading heartbeat. This may delay owner reaction;no claim all51.3ms is uniquely attributable.
Popen.wait(timeout=3) has timeout polling/backoff;avoiding it removes a possible exit-observation delay,not proof kernel reaping is now always faster.
Baseline12 nonmotion64MiB children under2 CPU workers max8.056ms;fixed same workload max8.152ms. Historical174.5ms NOT_REPRODUCED.
Thus no measured stress speedup claimed and no native350ms success claimed.
""",
'STOP_CODE_AND_TESTS.md':"""# Minimal changes
New owned_stop_v2:verify own group leader/PID start ticks;trace STOP_REQUEST,SIGSTOP_BEGIN/SENT,actual stopped waitpid state,SIGKILL_BEGIN/SENT,pidfd exit-ready,WAITPID_BEGIN/REAPED,EXIT_CONFIRMED.
Linux x86_64 pidfd waits for actual exit readiness;other/unsupported kernels use bounded waitpid polling. Actual exit/reap remains criterion,not pause.
Original100ms maximum stop-state observation and3s exit error bound retained;350ms total acceptance unchanged. Overrun never becomesPASS.
Events persist on guard;owner saves PLANT_STOP_EVENTS including errors/fallback indication.
New owner records abort receipt,stop entry,Plant exit and each child cleanup begin/end.
Family discovery moved to background tracker,stopped only after Plant stop request;final synchronous family scan happens AFTER actual Plant exit,not on stop critical path.
Cleanup/orphan checks retained;tracker shutdown errors causeFAIL.
Static4 tests PASS,actual ordinary-process SIGSTOP+SIGKILL+reap tested,12 baseline+12 fixed stress cases saved.
No ROS/Plant movement launched. Native timing remains to be measured.
""",
'NEW_SOURCE_AND_S_GATE.md':"""# New identity
experiments/stop_latency_fix_v1/manual.py and query_minco/integration/owned_stop_v2/stop.py are new versioned source;accepted e1b_fix2 worktree and failed C sources untouched.
Current binding SOURCE_BINDING_v2.json includes actual mainline runtime sources and unchanged native resources;source mapping is explicit,no old S relabeling.
New S check-only PASS exit0,Python compile PASS. Need researcher-run actual non-ARM S because execution stop/owner identity changed.
No watchdog/dynamics/radius/error/350ms criterion changes.
If real stop still exceeds350ms,remainNO-GO and use per-event evidence. Do not replace exit-confirm metric with pause timestamp.
""",
'E1A_READINESS.md':"""# E1-A NO-GO
Failed C7 cannot authorize movement. New S then new seven physical faults mustPASS.
Existing target0.28m,three-piece MINCO,T2,>=2plans/one switch,1m/s and all original tracking criteria preserved by new owner.
No E1-A command released until actual new C7 allPASS;no automatic ARM.
""",
'E1B_PROGRESS.md':"""# Independent E1-B progress
Worktree isolated_worktrees/e1b_ros_20261112_v1;active old execution worktree untouched.
Existing real birth-authority loader,shared Depth Reference/Admission,5Hz plan scheduling and per-frame recertification retained.
New execution.py adapter connects proof publication failure/exception to identity-bound owned Plant stop;arm observation requires native ARMED,DYNAMICS_INTEGRATED,positive step count and physical constants.
It does not issue ARM or create another simulator/controller/MINCO. Only actually owned Popen process can be stopped.
8 isolated tests PASS including actual nonmotion child termination when proof unavailable. No owner_verified boolean in production path.
Live ROS authority callback into actuator publisher remains unqualified;candidate node is still noncontrolling by default.
Outside-birth observed channel feasibility,actual5Hz/100Hz timing,switch continuity and native fail-stop tests NOT_RUN.
These are actual remaining deployment/qualification gaps,not solved by a unit-test fake ACK. No E1-B physical success claim.
""",
'FINAL_GO_NO_GO.md':"""# Decision
OLD_C7=FAIL_PRESERVED
STOP_INTERNAL_TIMELINE_OLD=UNRECORDED
NONMOTION_BASELINE_AND_FIXED_STRESS=PASS_NOT_NATIVE_PROOF
HISTORICAL174MS_DELAY_REPRODUCED=NO
NEW_STOP_TRACE_AND_OWNER_PATCH=IMPLEMENTED
STATIC_TESTS=PASS4
E1B_ISOLATED_TESTS=PASS8
NEW_S_CHECK_ONLY=PASS
NEW_REAL_S=NOT_RUN
NEW_REAL_C7=NOT_RUN
E1_A=NO_GO
E1_B=NO_GO_PENDING_LIVE_INTEGRATION
WATCHDOG=0.2_UNCHANGED
TOTAL_FAULT_TO_ACTUAL_EXIT_LIMIT=0.350_UNCHANGED
NEW_NETWORK_UPDATES=0
NEW_PHYSICS_RUNS=0
HISTORICAL_SOURCE_PROTECTION=PASS
"""
}
reports['MANUAL_COMMANDS.md']="""# New identity:non-ARM S first
No actual ROS/physics execution was run by Codex.
Private master11375 refuses occupied port. Run only this first;S never sends PREPARE/ARM.
~~~bash
cd /home/lizheng/query_minco_mainline
PY=/home/lizheng/anaconda3/envs/yopo/bin/python
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export STOPFIX_S="research_outputs/stopfix_S_$(date +%Y%m%d_%H%M%S)_$$"
"$PY" -B -m query_minco.audit.run_provenance_v1 --output "$STOPFIX_S/process" --cwd "$PWD" --timeout-s 120 -- "$PY" -B experiments/stop_latency_fix_v1/manual.py --stage S --allow-physics RESEARCHER_MANUAL_S_ONLY --output "$STOPFIX_S/run"
echo "$PWD/$STOPFIX_S/run/RESULT.json"
~~~
Return entire new run. Review sensor/truth/birth,zero integration and actual stop trace,not only process exit.
Only after S PASS may researcher separately run current owner stageC with --prior pointing to this exact S;old S/failed C do not match.
No C7 or E1-A auto-chaining and no E1-A movement command released now.
For C7 later inspect each case PLANT_STOP_EVENTS.json,OWNER_STOP_TIMELINE.json,Supervisor EVENTS and final RESULT:
injection,latch,owner receipt,SIGSTOP begin/sent,paused observation,SIGKILL begin/sent,pidfd readiness,waitpid/reap,actual exit,all child cleanup.
Fallbacks/errors must remain visible and fail qualification. Keep exit confirmation as350ms endpoint.
"""
for name,text in reports.items():(o/name).write_text(text)
files=[p for folder in e['allowed']+[str(o.relative_to(r))] for p in (r/folder).rglob('*') if p.is_file()]
w=r/'isolated_worktrees/e1b_ros_20261112_v1'
for folder in ['query_minco/integration/e1b_trial_v1','query_minco/integration/owned_stop_v2','tests/e1b_ros_v1']:
 files.extend(p for p in (w/folder).glob('*.py'))
(o/'FILES_AND_SHA.json').write_text(json.dumps({str(p):sha(p) for p in files},indent=2))
print('new S ready;old evidence preserved;native timeliness still NOT_RUN')
