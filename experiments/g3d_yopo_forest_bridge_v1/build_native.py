"""Standalone native generator build; source/config remain read-only."""
from pathlib import Path
import sys,json,os
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from query_minco.audit.run_provenance_v1 import run_logged
from query_minco.evaluation.mirror_holdout_v1 import sha,write
R=Path(__file__).resolve().parents[2];Y=R.parent/'YOPO';O=R/'research_outputs/g3d_yopo_forest_bridge_20261110_v1'
def main():
    b=O/'build';b.mkdir();env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',LD_LIBRARY_PATH='/opt/ros/noetic/lib')
    inc=['-I'+str(Y/'Simulator/src/include'),'-I'+str(Y/'Simulator/src/src'),'-I/usr/include/opencv4','-I/usr/include/pcl-1.10','-I/usr/include/eigen3','-I/opt/ros/noetic/include']
    sources=[R/'query_minco/evaluation/forest_generator_host_v1.cpp',Y/'Simulator/src/src/maps.cpp',Y/'Simulator/src/src/perlinnoise.cpp']
    objects=[]
    for i,p in enumerate(sources):
        obj=b/('source%d.o'%i);objects.append(obj)
        q=run_logged(['/usr/bin/g++-9','-std=c++17','-O2']+inc+['-c',str(p),'-o',str(obj)],b/('compile%d'%i),cwd=R,env=env,runspec=O/'PREREGISTRATION.json',timeout_s=180)
        if q['exit_code']!=0:raise RuntimeError('COMPILATION_FAILED '+str(i))
    binary=b/'forest_generate_only'
    libs=['-L/opt/ros/noetic/lib','-Wl,-rpath,/opt/ros/noetic/lib','-lopencv_core','-lopencv_imgproc','-lopencv_calib3d','-lpcl_common','-lpcl_io','-lyaml-cpp','-lrosconsole','-lrostime','-lcpp_common','-lroscpp']
    q=run_logged(['/usr/bin/g++-9']+[str(x) for x in objects]+['-o',str(binary)]+libs,b/'link',cwd=R,env=env,runspec=O/'PREREGISTRATION.json',timeout_s=120)
    if q['exit_code']!=0:raise RuntimeError('LINK_FAILED')
    write(b/'BUILD_IDENTITY.json',dict(binary=str(binary),binary_sha256=sha(binary),source_sha256={str(p):sha(p) for p in sources},read_only_yopo=True,no_ros_node=True))
    print('BUILD_PASS',binary,sha(binary))
if __name__=='__main__':main()
