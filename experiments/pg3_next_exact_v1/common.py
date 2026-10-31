import copy,csv,hashlib,subprocess
from pathlib import Path
from query_minco.training.io import read,write,sha,digest,artifact,require
from query_minco.training.run_spec import ROOT,source_files,verify_code
from experiments.pg3_v2_no_corridor.qualification_v21 import snapshot,ref
from experiments.pg3_v2_no_corridor.ablation_v21 import source_identity,bind_context,load_context
BASE='10e1d9e4a30b6330d194d0f35a18fd24a0fa25d7'
ORIGINAL=Path('/home/lizheng/query_minco_mainline')
PAIRED=ORIGINAL/'research_outputs/pg3_v21_paired128_v1'
ABLATION=ORIGINAL/'research_outputs/pg3_v21_ablation_v1'
DEFAULT=ORIGINAL/'research_outputs/pg3_next_exact_v1'
ARMS=('A','S','M','G','K')
def runpath(arm):return PAIRED/'runs'/('A42_'+('D' if arm=='K' else 'A')+'_PG3V21_formal_run001') if arm in ('A','K') else ABLATION/'runs'/('A42_'+arm+'_PG3V21_formal_run001')
def table(path):
    with Path(path).open() as f:return list(csv.DictReader(f))
def csvwrite(path,rows):
    require(bool(rows),'NEXT_EMPTY_TABLE')
    with Path(path).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def prepare_context(out):
    folder=out/'CONTEXT';folder.mkdir(exist_ok=False);binding=bind_context(folder);prep,context=load_context(binding);code=source_identity();write(folder/'DATA_QUALIFICATION.json',context[3]);write(folder/'SOURCE.json',dict(code=code,sources=snapshot(),bindings=binding,base=BASE))
    return prep,context,code

def verify_history(out):
    source=read(DEFAULT/'BASELINE_SOURCE.json');changes=[]
    for name,h in source['files'].items():
        p=ROOT/name
        if name.startswith(('experiments/pg3_feasible_mode_v1/exposure_','experiments/pg3_feasible_mode_v1/test_exposure_','query_minco/training/pg3_feasible_mode_v1/exposure_','query_minco/training/pg3_feasible_mode_v1/test_exposure_')) and not p.exists():continue
        require(p.is_file() and sha(p)==h,'NEXT_EXISTING_SOURCE_CHANGED:'+name)
    inventory={}
    for arm in ARMS:
        run=runpath(arm);seal=read(run/'RUN_SEAL.json')
        require(all(sha(run/k)==v for k,v in seal['files'].items()),'NEXT_HISTORICAL_OUTPUT_CHANGED:'+arm);inventory[arm]=ref(run/'RUN_SEAL.json')
    write(out/'HISTORICAL_IMMUTABILITY.json',dict(status='PASS',source_base=BASE,runs=inventory))
    return inventory
