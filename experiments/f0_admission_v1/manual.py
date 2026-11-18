"""Nonphysical admission entry. Physical qualification requires separate approval."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

def main():
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError('EXPECTED_REPOSITORY_CWD')
    if '--check-only' not in sys.argv[1:]:
        raise PermissionError('F0_ADMISSION_PHYSICAL_EXECUTION_NOT_AUTHORIZED')
    from experiments.stop_latency_fix_v1.manual import main as shared_main
    shared_main(config_root=ROOT/'research_outputs/f0_admission_20261117_v1/runtime',
                binding_name='SOURCE_BINDING.json')

if __name__ == '__main__':
    main()
