"""Create exclusive admission binding, preserving all old qualification bytes."""
from pathlib import Path
import hashlib
import json
import sys
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'research_outputs/f0_admission_20261117_v1'
OLD = ROOT/'research_outputs/c7_episode_lifecycle_20261117_v1/runtime'
PRIOR = ROOT/'research_outputs/episode_master_C_20261117_142001_1241/run/RESULT.json'
CHANGED = ROOT/'experiments/stop_latency_fix_v1/manual.py'
BASE_HEAD = 'b8d2969fd6e5dacf999247bf80610f969d392ca7'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def binding_with_additions(old, additions, changed=CHANGED):
    # The only approved existing source change is explicitly recorded, never hidden.
    for path, expected in old.items():
        if Path(path) != changed and sha(path) != expected:
            raise ValueError('UNEXPECTED_BASELINE_DRIFT '+path)
    return {str(p): sha(p) for p in sorted(set(map(Path, old)) | set(additions))}

def main():
    import subprocess
    import importlib.metadata as metadata
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError('EXPECTED_REPOSITORY_CWD')
    if OUT.exists():
        raise FileExistsError('EXCLUSIVE_OUTPUT_EXISTS '+str(OUT))
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    if head != BASE_HEAD:
        raise ValueError('BASE_HEAD_CHANGED')
    old = json.loads((OLD/'SOURCE_BINDING.json').read_text())
    original = subprocess.check_output(['git', 'show', BASE_HEAD+':experiments/stop_latency_fix_v1/manual.py'])
    if hashlib.sha256(original).hexdigest() != old[str(CHANGED)]:
        raise ValueError('BASELINE_LAUNCHER_IDENTITY')
    prior = json.loads(PRIOR.read_text())
    if prior.get('status') != 'PASS' or prior.get('source_binding') != old:
        raise ValueError('OLD_C7_IDENTITY')
    additions = set()
    for folder in ['query_minco/integration/observed_space_v1',
                   'query_minco/integration/observed_space_v2',
                   'experiments/f0_admission_v1', 'tests/f0_admission_v1']:
        additions.update((ROOT/folder).glob('*.py'))
    additions.add(ROOT/'query_minco/integration/f0_continuity_v1/execution_evidence.py')
    binding = binding_with_additions(old, additions)
    # Copy frozen configuration bytes into the new exclusive preparation only.
    configs = {name: (OLD/name).read_bytes() for name in
               ['SCENES.json', 'BIRTH_SELECTION.json', 'LEGACY_B_PHYSICAL_WINDOW.json']}
    for name, raw in configs.items():
        binding[str(OUT/'runtime'/name)] = hashlib.sha256(raw).hexdigest()
    report = dict(status='PREREQUISITE_PENDING_NO_ROS_NO_ARM', base_head=head,
                  old_binding_sha256=sha(OLD/'SOURCE_BINDING.json'),
                  prior_sha256=sha(PRIOR), old_c7_preserved=True,
                  qualification_reuse=False, reason='NEW_SOURCE_BINDING_REQUIRES_NEW_QUALIFICATION',
                  changed={str(CHANGED):dict(before=old[str(CHANGED)], after=sha(CHANGED))},
                  added=sorted(set(binding)-set(old)), binding_count=len(binding),
                  controller_plan_lineage='UNKNOWN', physical_ready=False,
                  python=dict(executable=sys.executable, sha256=sha(sys.executable)),
                  packages={p:metadata.version(p) for p in ['numpy','scipy','torch','PyYAML']},
                  environment_binding_scope='VERSION_METADATA_NOT_FULL_TRANSITIVE_BINARY_LOCK')
    OUT.mkdir()
    runtime = OUT/'runtime'
    runtime.mkdir()
    for name, raw in configs.items():
        (runtime/name).write_bytes(raw)
    (runtime/'SOURCE_BINDING.json').write_text(json.dumps(binding, indent=2, sort_keys=True)+'\n')
    (OUT/'PREPARATION.json').write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
