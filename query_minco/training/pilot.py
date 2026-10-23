"""Gate E CLI extension. GPU execution and training are researcher-run only."""
import argparse,json,os,sys,traceback
from pathlib import Path
import torch,numpy,scipy,query_minco
from .io import read,write,require,digest,sha,artifact
from .pilot_spec import load_spec,qualify,authorize,OUT
from .pilot_audit import audit_run

def worker_command(command,s):
    command=list(command)
    i=command.index('query_minco.training.cli');command[i]='query_minco.training.pilot'
    extra=['--setenv','OMP_NUM_THREADS','1','--setenv','MKL_NUM_THREADS','1','--setenv','PYTHONHASHSEED','42']
    if s['device']=='cuda':
        devices=[Path(x) for x in ['/dev/dxg','/dev/nvidia0','/dev/nvidiactl','/dev/nvidia-uvm','/dev/nvidia-uvm-tools'] if Path(x).exists()]
        require(devices,'CUDA_DEVICE_NODES_MISSING')
        for p in devices:extra+=['--dev-bind',str(p),str(p)]
        if Path('/usr/lib/wsl/lib').is_dir():extra+=['--setenv','LD_LIBRARY_PATH','/usr/lib/wsl/lib']
    i=command.index('--chdir');command[i:i]=extra
    return command

def main(argv=None):
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest='command',required=True)
    for name in ['plan','preflight','train']:
        x=sub.add_parser(name);x.add_argument('--config',required=True)
        if name=='preflight':x.add_argument('--output',required=True)
        if name=='train':
            x.add_argument('--authorization',required=True);x.add_argument('--preflight',required=True);x.add_argument('--review',required=True)
    x=sub.add_parser('audit');x.add_argument('--run-dir',required=True)
    x=sub.add_parser('_worker');x.add_argument('--config',required=True);x.add_argument('--run-dir',required=True)
    x.add_argument('--action',choices=['preflight','train'],required=True);x.add_argument('--authorization');x.add_argument('--preflight')
    a=p.parse_args(argv)
    if a.command=='audit':result=audit_run(a.run_dir)
    else:
        s=load_spec(a.config)
        if a.command=='plan':
            _,_,gate=qualify(s)
            result=dict(profile=s['profile'],spec_sha256=digest(s),data=gate,optimizer_updates=0,
                        ready_for_bounded_pilot=False,ready_for_formal_training=False)
        elif a.command=='_worker':
            from .isolation import worker_probe
            from .pilot_worker import preflight,execute
            run=Path(a.run_dir);worker_probe(run,s)
            write(run/'ENVIRONMENT.json',dict(executable=sys.executable,python=sys.version,torch=torch.__version__,
                numpy=numpy.__version__,scipy=scipy.__version__,query_minco_file=query_minco.__file__,
                device=s['device'],seed=s['seed'],threads=1,strict_determinism=True,AMP=False,TF32=False,
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG')))
            if a.action=='preflight':
                try:
                    preflight(s,run)
                except BaseException as error:
                    from .pilot_worker import snapshot_failure
                    snapshot_failure(run,error,s)
                    raise
            else:
                require(a.authorization is not None and a.preflight is not None,'MANUAL_TRAIN_AUTH')
                authorize(s,a.authorization)
                require(read(a.preflight+'.seal.json')['sha256']==sha(a.preflight),'PREFLIGHT_SEAL')
                execute(s,run,a.authorization,a.preflight)
            result=dict(worker_finished=True)
        else:
            from .isolation import launch
            extension=True
            if a.command=='train':
                authorize(s,a.authorization)
                review=read(artifact(dict(path=str(Path(a.review).resolve()),sha256=read(a.review+'.seal.json')['sha256'])))
                require(review['passed'] and review['spec_sha256']==digest(s) and review['scope']=='GATE_E2_REVIEWED_GPU_PREFLIGHT'
                    and review['preflight_sha256']==sha(a.preflight),'E2_HUMAN_REVIEW_REQUIRED')
                receipt=read(a.preflight)
                require(receipt['passed'] and receipt['device']=='cuda' and receipt['optimizer_updates']==0
                    and receipt['deterministic_repeat'] and receipt['batch_audit']['passed'],'CUDA_PREFLIGHT_REQUIRED')
                result=launch(s,a.config,OUT/s['run_id'],'train',a.authorization,a.preflight,extension=extension)
            else:
                dest=Path(a.output)
                require(dest.is_absolute() and dest.parent==OUT and dest.name!=s['run_id'],'PREFLIGHT_DESTINATION')
                result=launch(s,a.config,dest,'preflight',extension=extension)
    print(json.dumps(result,indent=2,allow_nan=False))
    return 0
if __name__=='__main__':raise SystemExit(main())
