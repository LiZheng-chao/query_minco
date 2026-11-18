"""Exclusive out-of-tree build and seal; YOPO inputs are strictly read-only."""
from pathlib import Path
import os,sys,json,hashlib,subprocess,argparse
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'research_outputs/f0_controller_consumption_20261118_v1'
PKG=ROOT/'ros/f0_controller_audit_pkg'
YOPO=Path('/home/lizheng/YOPO')
UP=YOPO/'Controller/src/so3_control'
EXPECTED={'src/so3_control_nodelet.cpp':'fae6d2a9206b8f5f37665b1bec1de72a15dcc5ba4fdba7f5ea92a1ad3b5ea7ed',
'src/SO3Control.cpp':'3cdcde195d4a8fc19e80ec16ea473d8e458e8d523338514042b7ec535623cda9',
'include/so3_control/SO3Control.h':'86ad2fb732f60ffce40e2831e6c1b8ca2a83c39879d18e8a06acf791bbec099e'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def environment():
    return dict(os.environ,CMAKE_PREFIX_PATH=str(OUT/'devel_v2')+':'+str(YOPO/'Controller/devel')+':/opt/ros/noetic',
        ROS_PACKAGE_PATH=str(ROOT/'ros')+':'+str(YOPO/'Controller/src')+':/opt/ros/noetic/share',
        LD_LIBRARY_PATH=str(OUT/'devel_v2/lib')+':'+str(YOPO/'Controller/devel/lib')+':/opt/ros/noetic/lib',
        PYTHONPATH='/opt/ros/noetic/lib/python3/dist-packages:'+str(YOPO/'Controller/devel/lib/python3/dist-packages'),
        ROS_HOME=str(OUT/'ros_home'),ROS_LOG_DIR=str(OUT/'ros_logs'),PYTHONDONTWRITEBYTECODE='1',
        PATH='/opt/ros/noetic/bin:'+os.environ['PATH'],OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
def run(name,argv,env=None):
    log=OUT/'audit';log.mkdir(exist_ok=True)
    for suffix in ['stdout.log','stderr.log','command.json']:
        if (log/(name+'.'+suffix)).exists():raise FileExistsError(name)
    with (log/(name+'.stdout.log')).open('x') as so,(log/(name+'.stderr.log')).open('x') as se:
        p=subprocess.run(list(map(str,argv)),cwd=ROOT,env=env or environment(),stdout=so,stderr=se)
    (log/(name+'.command.json')).write_text(json.dumps(dict(argv=list(map(str,argv)),cwd=str(ROOT),exit_code=p.returncode),indent=2)+'\n')
    print(name,'exit',p.returncode,flush=True)
    if p.returncode:raise RuntimeError('COMMAND_FAILED '+name)
def verify_upstream():
    for rel,h in EXPECTED.items():
        if sha(UP/rel)!=h:raise ValueError('UPSTREAM_SHA_DRIFT '+rel)
def build():
    verify_upstream()
    generated=OUT/'generated';generated.mkdir(exist_ok=True)
    run('apply_patch_v2',['patch','--batch','--fuzz=0','-o',generated/'audit_nodelet_v2.cpp',UP/'src/so3_control_nodelet.cpp',PKG/'nodelet_audit.patch'])
    run('cmake_v2',['cmake','-S',PKG,'-B',OUT/'build_v2','-DCMAKE_BUILD_TYPE=Release',
        '-DCATKIN_DEVEL_PREFIX='+str(OUT/'devel_v2'),'-DCMAKE_INSTALL_PREFIX='+str(OUT/'install_v2'),
        '-DPYTHON_EXECUTABLE=/usr/bin/python3','-DF0_GENERATED_NODELET='+str(generated/'audit_nodelet_v2.cpp')])
    run('compile_v2',['cmake','--build',OUT/'build_v2','--parallel','2'])
    # Independently linked original library provides a numerical reference.
    run('compile_original_reference_v2',['g++','-std=c++11','-O3',ROOT/'tests/f0_controller_consumption_v1/controller_equivalence.cpp',
        '-I'+str(UP/'include'),'-I/usr/include/eigen3','-L'+str(YOPO/'Controller/devel/lib'),
        '-Wl,-rpath,'+str(YOPO/'Controller/devel/lib'),'-lSO3Control','-o',OUT/'build_v2/original_control_equivalence'])
    verify_upstream()
def seal():
    verify_upstream()
    runtime=OUT/'runtime';runtime.mkdir()
    old=ROOT/'research_outputs/f0_admission_20261117_v1/runtime'
    binding=json.loads((old/'SOURCE_BINDING.json').read_text())
    allowed={str(ROOT/p) for p in ['ros/g4cp7b_fix2_v1/supervisor.py','experiments/stop_latency_fix_v1/manual.py','query_minco/integration/f0_continuity_v1/execution_evidence.py']}
    for path,expected in binding.items():
        if path not in allowed and sha(path)!=expected:raise ValueError('UNAPPROVED_BINDING_DRIFT '+path)
    for folder in [PKG,ROOT/'experiments/f0_controller_consumption_v1',ROOT/'tests/f0_controller_consumption_v1',OUT/'generated',OUT/'devel_v2/lib']:
        for p in folder.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:binding[str(p.resolve())]=sha(p)
    for rel in EXPECTED:binding[str(UP/rel)]=sha(UP/rel)
    # Compiler-generated dependency lists bind the headers actually used.
    for dep in (OUT/'build_v2').rglob('*.o.d'):
        tokens=dep.read_text().replace('\\\n',' ').split()
        for token in tokens:
            p=Path(token)
            if p.is_absolute() and p.is_file():binding[str(p.resolve())]=sha(p)
    for p in [OUT/'build_v2/CMakeCache.txt']+list((OUT/'build_v2').rglob('link.txt')):
        binding[str(p)]=sha(p)
    maps=(OUT/'audit/ros_bench_v4/diagnostic_on.maps').read_text()
    for line in maps.splitlines():
        path=line.split()[-1]
        if path.startswith('/') and '.so' in Path(path).name and Path(path).is_file():binding[str(Path(path).resolve())]=sha(path)
    binding[str(OUT/'BUILD_IDENTITY_v2.json')]=sha(OUT/'BUILD_IDENTITY_v2.json')
    for p in [UP/'CMakeLists.txt',UP/'package.xml',UP/'nodelet_plugin.xml',YOPO/'Controller/src/utils/quadrotor_msgs/msg/PositionCommand.msg',ROOT/'ros/g4cp7b_fix2_v1/supervisor.py']:
        binding[str(p)]=sha(p)
    for name in ['SCENES.json','BIRTH_SELECTION.json','LEGACY_B_PHYSICAL_WINDOW.json']:
        (runtime/name).write_bytes((old/name).read_bytes());binding[str(runtime/name)]=sha(runtime/name)
    # Refresh the three explicitly approved existing sources, leaving old manifests unchanged.
    for p in list(binding):binding[p]=sha(p)
    (runtime/'SOURCE_BINDING.json').write_text(json.dumps(binding,sort_keys=True,indent=2)+'\n')
    identity=json.loads((OUT/'BUILD_IDENTITY_v2.json').read_text())
    if identity['library_sha256']!=sha(OUT/'devel_v2/lib/libf0_audit_nodelet.so'):raise ValueError('BUILD_IDENTITY_DRIFT')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--build',action='store_true');p.add_argument('--seal',action='store_true');a=p.parse_args()
    if a.build:build()
    if a.seal:seal()
