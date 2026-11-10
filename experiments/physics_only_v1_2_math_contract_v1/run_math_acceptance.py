"""Thin CPU0 receipt/resource supervisor. Never trains or overwrites run receipts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "research_outputs/physics_only_v1_2_math_contract_v1/audit"
PYTHON = "/home/lizheng/anaconda3/envs/yopo/bin/python"
PYTEST_SITE = "/home/lizheng/anaconda3/lib/python3.11/site-packages"
LIMIT_SECONDS = 120.
LIMIT_RSS = 2 * 1024**3
LIMIT_DISK = 128 * 1024**2


def write(path, value):
    with path.open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write("\n")


def disk_bytes(path):
    total = 0
    # pytest may remove temporary directories while the monitor traverses them.
    for directory, _, names in os.walk(str(path)):
        for name in names:
            try:
                total += (Path(directory) / name).stat().st_size
            except FileNotFoundError:
                continue
    return total


def process_tree_rss(pid):
    records = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = {}
            for line in (entry / "status").read_text().splitlines():
                key, _, value = line.partition(":")
                fields[key] = value.strip()
            records[int(entry.name)] = (int(fields["PPid"]), int(fields.get("VmRSS", "0 kB").split()[0])*1024)
        except (OSError, KeyError, ValueError):
            continue
    included = {pid}
    while True:
        extra = {p for p, (parent, _) in records.items() if parent in included}
        if extra <= included:
            break
        included |= extra
    return sum(records.get(p, (0, 0))[1] for p in included)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", required=True, choices=("v12", "regression", "smoke"))
    parser.add_argument("--case", required=True)
    args = parser.parse_args()
    if not re.fullmatch(args.kind + r"_\d{3}", args.case):
        raise ValueError("case name mismatch")
    destination = OUTPUT / args.case
    destination.mkdir()
    temporary = destination / "tmp"; temporary.mkdir()
    artifact = destination / "pytest_artifacts"; artifact.mkdir()
    config = ROOT / "experiments/physics_only_v1_2_math_contract_v1/zero_update_config.json"
    if args.kind == "smoke":
        command = [PYTHON, "-B", "-m", "query_minco.training.physics_only_v1_2.smoke",
                   "--config", str(config), "--output", str(destination / "zero_update.json")]
    else:
        selected = ["query_minco/training/physics_only_v1_2/tests"] if args.kind == "v12" else [
            "query_minco/training/physics_only_v1/tests", "tests/test_pg1_phase1.py",
            "tests/test_pg1_v2_r1.py", "tests/test_set_physical_risk_v1.py", "tests/test_reference_contracts.py"]
        pytest_args = ["-q", "-p", "no:cacheprovider", "--basetemp", str(temporary / "pytest")] + selected
        code = (
            "import sys,torch,numpy,scipy;sys.path.append(" + repr(PYTEST_SITE) + ");import pytest;"
            "print('ENV',sys.version,torch.__version__,numpy.__version__,scipy.__version__,pytest.__version__);"
            "print('IMPORTS',torch.__file__,numpy.__file__,scipy.__file__,pytest.__file__);"
            "assert torch.__version__.startswith('2.4.') and numpy.__version__.startswith('1.22.') "
            "and scipy.__version__.startswith('1.10.');"
            "raise SystemExit(pytest.main(" + repr(pytest_args) + "))")
        command = [PYTHON, "-B", "-c", code]
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(ROOT),
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
               TMPDIR=str(temporary), TEMP=str(temporary), TMP=str(temporary),
               PG1_TEST_OUTPUT=str(artifact))
    env.pop("PYTEST_ADDOPTS", None)
    source = list((ROOT / "query_minco/training/physics_only_v1_2").rglob("*.py")) + [
        Path(__file__).resolve(), config]
    write(destination / "COMMAND.json", dict(
        argv=command, cwd=str(ROOT), env={k: env[k] for k in (
            "CUDA_VISIBLE_DEVICES", "PYTHONDONTWRITEBYTECODE", "PYTHONPATH", "OMP_NUM_THREADS",
            "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
            "TMPDIR", "TEMP", "TMP", "PG1_TEST_OUTPUT")},
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source},
        wall_seconds_limit=LIMIT_SECONDS, process_tree_rss_bytes_limit=LIMIT_RSS,
        output_total_bytes_limit=LIMIT_DISK, python=PYTHON))
    started = time.monotonic()
    max_rss, reason = 0, None
    with (destination / "stdout.log").open("xb") as out, (destination / "stderr.log").open("xb") as err:
        child = subprocess.Popen(command, cwd=str(ROOT), env=env, stdout=out, stderr=err, start_new_session=True)
        try:
            while child.poll() is None:
                rss = process_tree_rss(child.pid); max_rss = max(max_rss, rss)
                if time.monotonic()-started > LIMIT_SECONDS:
                    reason = "WALL_LIMIT"
                elif rss > LIMIT_RSS:
                    reason = "RSS_LIMIT"
                elif disk_bytes(OUTPUT.parent) > LIMIT_DISK:
                    reason = "DISK_LIMIT"
                if reason:
                    os.killpg(child.pid, signal.SIGKILL)
                    break
                time.sleep(.1)
            code = child.wait()
        except BaseException:
            failure = traceback.format_exc()
            if child.poll() is None: os.killpg(child.pid, signal.SIGKILL)
            code = child.wait()
            write(destination / "PROCESS_EXIT.json", dict(
                exit_code=code, supervisor_exit_code=1, stop_reason="SUPERVISOR_EXCEPTION",
                traceback=failure, elapsed_seconds=time.monotonic()-started,
                sampled_peak_process_tree_rss_bytes=max_rss, optimizer_updates=0, episodes=0))
            raise
    receipt = dict(exit_code=code, stop_reason=reason, elapsed_seconds=time.monotonic()-started,
                   sampled_peak_process_tree_rss_bytes=max_rss, output_total_bytes=disk_bytes(OUTPUT.parent),
                   optimizer_updates=0, episodes=0, resources_sampling_interval_s=.1,
                   source_environment="CPU only; no network or dependency installation")
    write(destination / "PROCESS_EXIT.json", receipt)
    print(json.dumps(dict(case=args.case, **receipt)))
    return 0 if code == 0 and reason is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
