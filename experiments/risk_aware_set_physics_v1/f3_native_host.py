from pathlib import Path
BASE=Path('/home/lizheng/query_minco_mainline/experiments/risk_aware_set_physics_v1/f3_committed_host.py')
s=BASE.read_text().replace('f3_execution004/native_ref001','f3_execution005/native_ref001')
s=s.replace("'f3_execution004_native_ref001'","'f3_execution005_native_ref001'")
old="arbiter=launch('arbiter',[PYTHON,'-B','-s','-u','-m','query_minco.inference.ros_committed_reference','--run',str(HOST),'--reference-only'],ROOT)"
assert old in s
s=s.replace(old,"arbiter=launch('arbiter',[str(OUT/'runtime/native_reference_publisher'),str(OUT/'preparation/NATIVE_REFERENCES.bin'),str(HOST),'--reference-only'],ROOT)")
s=s.replace('lastres=0;swapactive=0;','stress_started=False;lastres=0;swapactive=0;')
s=s.replace("  if (HOST/'ARBITER_RESULT.json').exists():","  if not stress_started and (HOST/'CONTROL_STAGE.jsonl').exists():\n   try:\n    stages=[json.loads(x) for x in (HOST/'CONTROL_STAGE.jsonl').read_text().splitlines()]\n    if any(x['stage']==2 for x in stages):\n     job=HOST/'stress_JOB.json';job.write_text(json.dumps({'kind':'stress'}))\n     launch('background_cpu',[PYTHON,'-B','-s',str(ROOT/'experiments/risk_aware_set_physics_v1/prepare_committed_bench.py'),'--out',str(HOST),'--job',str(job)],ROOT)\n     stress_started=True\n   except ValueError:pass\n  if (HOST/'ARBITER_RESULT.json').exists():")
s=s.replace("  if any(p.poll() is not None for n,p in spawned):","  if now-start>15 and 'odom' not in latest:raise RuntimeError('NO_NATIVE_ODOM')\n  if any(p.poll() is not None for n,p in spawned if n!='background_cpu'):")
exec(compile(s,str(BASE)+'[native-transport-overlay]','exec'),{'__name__':'__main__','__file__':str(BASE)})
