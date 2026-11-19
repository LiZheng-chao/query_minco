"""Researcher-authorized, native simulation only."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

R=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(R))
OUT=R/'research_outputs/sim_dev_oracle_forest10m_v1'
BUNDLES={'p0':'runtime_bundle_v5_p0','p0-p1':'runtime_bundle_v5'}
MODE='UNQUALIFIED_SIM_DEV_DEBUG'

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--variant',choices=BUNDLES,default='p0-p1')
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--mode',choices=[MODE],required=True)
    parser.add_argument('--allow-physics')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    B=OUT/BUNDLES[args.variant]
    if Path.cwd().resolve()!=R:
        raise ValueError('EXPECTED_REPOSITORY_CWD')
    if not args.check_only and args.allow_physics!='RESEARCHER_MANUAL_FOREST10M_ONLY':
        raise PermissionError('EXPLICIT_FOREST10M_AUTHORIZATION_REQUIRED')
    if not args.check_only and (args.output is None or args.output.exists() or
                                args.output.resolve().parent!=B/'runs'):
        raise ValueError('UNIQUE_FOREST10M_RUN_REQUIRED')
    scene=json.loads((B/'runtime_v2/SCENES.json').read_text())['forest']
    cfg=json.loads(Path(__file__).with_name('config.json').read_text())
    if scene['forest10m']!=cfg or scene['goal']!=cfg['goal']:
        raise ValueError('FOREST10M_CONFIG_DRIFT')
    native_plant=R/'research_outputs/g4cp1_startup_control_20261111_v1/build_equilibrium_v3/gated_host'
    native_renderer=R/'research_outputs/forest_dev_sparse_seed3_dist6_fixedbirth_v1/build/native_renderer'
    if Path(scene['plant_binary'])!=native_plant or Path(scene['sensor_binary'])!=native_renderer:
        raise PermissionError('NATIVE_SIM_ONLY')
    owner=B/'generated_v2/debug_owner.py'
    spec=importlib.util.spec_from_file_location('forest10m_owner',owner)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.argv=[str(owner),'--stage','SIM_DEV_ORACLE','--allow-physics',
              'RESEARCHER_MANUAL_SIM_DEV_ORACLE_ONLY','--prior',str(OUT/'TARGETED_P0_P1_TESTS.json'),
              '--output',str(args.output or B/'runs/CHECK_ONLY_NO_CREATE')]
    if args.check_only:
        sys.argv.append('--check-only')
    module.main(config_root=B/'runtime_v2', binding_name='SOURCE_BINDING.json',
                audit_build=R/'research_outputs/f0_controller_consumption_20261118_v1')

if __name__=='__main__':main()
