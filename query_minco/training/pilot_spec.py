"""Gate E profile: immutable Seed42 32-update manual pilot, separate from reference."""
import math
from pathlib import Path
from .io import require, exact, artifact, read, digest, sha
from .run_spec import ROOT, verify_code
from .qualification import qualify as reference_qualify
from .datasets.geometry_sidecar import GeometrySidecar

PROFILE='rsps_pilot32_v1'
IDS={a:'query_minco_rsps_pilot32_'+a.lower()+'_v1' for a in ['E0','E1','E2','E3']}
OUT=ROOT/'research_outputs/rsps_gate_e_v1/manual_runs'

def recipe(arm):
    require(arm in IDS,'PILOT_ARM')
    return dict(arm=arm,imitation='original_B_set_six_terms',sampler='fixed_segment_trapezoid_v1',
        points_per_segment=64,eps=.05,radius=.12,margin=.03,beta=0. if arm in ['E0','E1'] else 1.,
        v_ref=1.,w_max=3.,physical_scale=0. if arm=='E0' else 1.,matched_strength=1.,
        unmatched_strength=2. if arm=='E3' else 1.,max_updates=32,checkpoints=[0,8,16,32],
        schedule='cosine_local_index_0_to_31_2e-5_to_1e-6_v1',weight_decay=0.,clip_norm=10.)

def lr_at(index):
    require(type(index) is int and 0<=index<32,'PILOT_LR_INDEX')
    return 1e-6+.5*(2e-5-1e-6)*(1+math.cos(math.pi*index/31))

def validate(s):
    exact(s,['schema','run_id','training_method_id','reference_method_id','profile','seed','experiment_arm',
        'objective_arm','device','threads','initialization','dataset','order','output_root','code',
        'sidecar','all_orders','recipe','recipe_sha256','development','resources'],'PilotSpec')
    require(s['schema']==2 and s['profile']==PROFILE and s['seed']==42 and s['threads']==1,'PILOT_PROFILE')
    require(s['training_method_id']==IDS.get(s['objective_arm']) and s['experiment_arm']=='C1_orthogonal'
        and s['reference_method_id']=='query_minco_b2c1_c1_e1_v1','PILOT_IDENTITY')
    require(s['recipe']==recipe(s['objective_arm']) and s['recipe_sha256']==digest(s['recipe']),'FIXED_RECIPE')
    require(s['device'] in ['cpu','cuda'],'PILOT_DEVICE')
    require(s['resources']==dict(max_wall_s=900,max_gpu_mib=6144,max_gpu_fraction=.75,min_free_gpu_mib=512),'RESOURCE_BUDGET')
    rid=s['run_id'];require(isinstance(rid,str) and rid and all(c.isalnum() or c in '_-' for c in rid),'RUN_ID')
    require(Path(s['output_root'])==OUT and OUT.is_dir(),'PILOT_OUTPUT_ROOT')
    require(s['initialization']['origin']==dict(kind='C1_step2880',seed=42,step=2880),'C1_ORIGIN')
    require(s['initialization']['artifact']['sha256']=='2352de6dd1e4a839b78ec12c115b5319b34da3c61cada105dbdeb24a4faec6d1'
        and s['initialization']['weights_sha256']=='68570b63e15f21ae9e9d512393391c9a4f3cf8569dfc730cfbeb9cb8f436d63b','C1_INIT')
    verify_code(s['code'])
    for b in [s['initialization']['artifact'],s['sidecar'],s['development']]:artifact(b)
    return s

def load_spec(path):return validate(read(path))

def qualify(s):
    validate(s)
    base=dict(s,profile='b2c1_reference_v1')
    ds,order,gate=reference_qualify(base)
    geo=GeometrySidecar(s['sidecar'],ds)
    require(s['all_orders']==geo.manifest['orders'],'THREE_ORDER_BINDING')
    require(s['order']==next(x['artifact'] for x in s['all_orders'] if x['seed']==42),'SEED42_ORDER')
    paths=[geo.path]
    for x in geo.manifest['orders']:paths.append(artifact(x['artifact']))
    for row in geo.rows.values():
        for b in [row['scene_source'],row['contract_source'],row['anchor']['artifact']]:paths.append(artifact(b))
        if row['anchor']['seal']:paths.append(artifact(row['anchor']['seal']))
    from .pilot_evaluation import qualify_development
    paths+=qualify_development(s['development'],set(ds.rows),{x['scene_id'] for x in geo.rows.values()})
    ds.files+=paths;ds.geometry=geo
    return ds,order,dict(gate,geometry_rows=len(geo.rows),development_qualified=True)

def authorize(s,path):
    a=read(path)
    exact(a,['schema','scope','run_id','spec_sha256','max_updates','authorized_by','approved'],'PilotAuthorization')
    require(s['device']=='cuda' and s['code']['allow_uncommitted_sources'] is False,'MANUAL_CUDA_ONLY')
    require(a['schema']==1 and a['scope']=='RSPS_MANUAL_PILOT32' and a['run_id']==s['run_id'] and
        a['spec_sha256']==digest(s) and a['max_updates']==32 and a['approved'] is True and
        bool(a['authorized_by']),'PILOT_AUTHORIZATION')
    return a
