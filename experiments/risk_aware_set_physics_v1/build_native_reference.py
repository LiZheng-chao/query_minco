import json,shlex,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'research_outputs/rsps_f3_low_speed_v1/f3_execution005'
flags=Path('/home/lizheng/YOPO/Controller/build/so3_quadrotor_simulator/CMakeFiles/quadrotor_simulator_so3.dir/flags.make').read_text()
d={}
for l in flags.splitlines():
 if ' = 'in l:
  k,v=l.split(' = ',1);d[k]=shlex.split(v)
link=shlex.split(Path('/home/lizheng/YOPO/Controller/build/so3_quadrotor_simulator/CMakeFiles/quadrotor_simulator_so3.dir/link.txt').read_text())
obj=OUT/'runtime/native_reference_publisher.o'
cmd=['/usr/bin/c++','-std=c++14','-O2','-Wall']+d['CXX_DEFINES']+d['CXX_INCLUDES']+['-c',str(ROOT/'query_minco/inference/native_reference_publisher.cpp'),'-o',str(obj)]
link[link.index('-o')+1]=str(OUT/'runtime/native_reference_publisher')
link=[str(obj) if x.endswith('quadrotor_simulator_so3.cpp.o') else x for x in link]
with (OUT/'BUILD_COMMANDS.json').open('w') as f:json.dump([cmd,link],f,indent=2)
with (OUT/'logs/BUILD_NATIVE.log').open('a') as f:
 for c in [cmd,link]:
  r=subprocess.run(c,stdout=f,stderr=subprocess.STDOUT)
  if r.returncode:raise RuntimeError('BUILD_FAILED')
print('native reference-only transport built')
