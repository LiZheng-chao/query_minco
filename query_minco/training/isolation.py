"""Full-view coordinator and masked read-only Worker have distinct audit responsibilities."""
import errno
import os
from pathlib import Path
import subprocess
import sys
import traceback
from .io import write, read, sha, require, artifact
from .run_spec import ROOT, identity, run_path, authorize, verify_code
from .qualification import qualify

HISTORICAL_ROOT = Path("/home/lizheng/reach_planner")

def protection(paths):
    return {str(Path(p).resolve()): sha(p) for p in sorted(set(paths), key=str)}

def worker_probe(run, spec):
    require(os.environ.get("QUERY_MINCO_ISOLATED_WORKER") == "1", "WORKER_ISOLATION_REQUIRED")
    probes = []
    for p in (ROOT / "query_minco/models/planner.py", ROOT / "query_minco/training/loop.py",
              artifact(spec["initialization"]["artifact"]), artifact(spec["dataset"]), artifact(spec["order"])):
        try:
            fd = os.open(str(p), os.O_WRONLY); os.close(fd)
            raise ValueError("SOURCE_WRITABLE")
        except OSError as e:
            require(e.errno == errno.EROFS, "SOURCE_NOT_READONLY")
            probes.append(str(p))
    require(not list(HISTORICAL_ROOT.glob("YOPO/*")), "OLD_ALGORITHM_VISIBLE")
    # The parent output directory must remain read-only. Only this Run is writable.
    try:
        fd = os.open(str(run.parent / ("WRITE_PROBE_" + run.name)), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(fd)
        raise ValueError("SIBLING_OUTPUT_WRITABLE")
    except OSError as e:
        require(e.errno == errno.EROFS, "OUTPUT_WHITELIST")
    write(run / "ISOLATION.json", dict(passed=True, source_probes=probes,
          project_writable_path=str(run), historical_tree_masked=True, network_unshared=True))

def launch(spec, config_path, destination, action, authorization=None, preflight=None, extension=None):
    """No caller-provided arbitrary command. Output mkdir is exclusive; no resume."""
    if extension is not None:
        require(extension is True and spec['profile'] == 'rsps_pilot32_v1', 'UNREGISTERED_LIFECYCLE_EXTENSION')
        from . import pilot as extension
    ds, order, qualified = (qualify if extension is None else extension.qualify)(spec)
    paths = [Path(config_path).resolve(), ds.path, order.path, artifact(spec["initialization"]["artifact"])]
    paths += ds.files
    if spec["profile"] == "b2c1_reference_v1":
        origin = spec["initialization"]["origin"]
        paths += [artifact(origin["identity_receipt"]), artifact(origin["original_checkpoint"])]
    paths += [ROOT / n for n in spec["code"]["files"]]
    paths += [Path(__file__).resolve()]
    if authorization:
        (authorize if extension is None else extension.authorize)(spec, authorization); paths.append(Path(authorization).resolve())
    if preflight:
        p = read(preflight)
        require(p["passed"] and p["spec_sha256"] == identity(spec), "PREFLIGHT_RECEIPT")
        require(read(str(preflight) + ".seal.json")["sha256"] == sha(preflight), "PREFLIGHT_SEAL")
        paths.extend([Path(preflight).resolve(), Path(str(preflight) + ".seal.json").resolve()])
    before = protection(paths)
    require(not destination.is_symlink() and destination.parent.resolve() == destination.parent, "OUTPUT_SYMLINK")
    destination.mkdir()  # FileExistsError deliberately propagated; never enter an old Run.
    runtime = destination / "runtime"; runtime.mkdir()
    for n in ("home", "tmp", "cache"): (runtime / n).mkdir()
    write(destination / "RUN_SPEC.json", spec)
    write(destination / "PROTECTION_BEFORE.json", dict(passed=True, files=before, scope="explicit input and package source bytes"))
    write(destination / "LIFECYCLE_RESERVED.json", dict(action=action, spec_sha256=identity(spec)))
    fds = []
    code = None
    try:
        command = ["/usr/bin/bwrap", "--die-with-parent", "--new-session", "--unshare-net",
                   "--ro-bind", "/", "/", "--proc", "/proc", "--dev", "/dev"]
        masked = [p for p in (HISTORICAL_ROOT, Path("/mnt"), ROOT / "research_outputs") if p.is_dir()]
        # Additional external cache roots receive the same exact-file view, never their eval/blind siblings.
        for root in ds.roots:
            if ROOT not in root.parents and not any(p == root or p in root.parents for p in masked):
                require(root not in (Path("/"), Path("/home"), Path("/home/lizheng")) and ROOT not in root.parents
                        and root not in ROOT.parents, "OVERBROAD_CACHE_ROOT")
                masked.append(root)
        for root in masked:
            command += ["--tmpfs", str(root)]
        # Re-expose only required, pre-opened true source files under hidden trees.
        for p in sorted(set(paths), key=str):
            if any(root == p or root in p.parents for root in masked):
                fd = os.open(str(p), os.O_RDONLY | os.O_CLOEXEC)
                fds.append(fd)
                command += ["--ro-bind-fd", str(fd), str(p)]
        command += ["--bind", str(destination), str(destination)]
        for root in masked:
            command += ["--remount-ro", str(root)]
        env = dict(PATH="/usr/bin:/bin", HOME=str(runtime / "home"), TMPDIR=str(runtime / "tmp"),
                   XDG_CACHE_HOME=str(runtime / "cache"), PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1",
                   QUERY_MINCO_ISOLATED_WORKER="1", CUBLAS_WORKSPACE_CONFIG=":4096:8",
                   CUDA_VISIBLE_DEVICES="" if spec["device"] == "cpu" else os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
                   LC_ALL="C.UTF-8")
        for k, v in env.items(): command += ["--setenv", k, v]
        command += ["--chdir", str(ROOT), sys.executable, "-B", "-s", "-m", "query_minco.training.cli",
                    "_worker", "--config", str(Path(config_path).resolve()), "--run-dir", str(destination),
                    "--action", action]
        if action == "train":
            command += ["--authorization", str(Path(authorization).resolve()), "--preflight", str(Path(preflight).resolve())]
        if extension is not None:
            command = extension.worker_command(command, spec)
        write(destination / "COMMAND.json", dict(argv=command, executable=sys.executable,
              fd_sources=[str(p) for p in sorted(set(paths), key=str)
                          if any(root == p or root in p.parents for root in masked)],
              authorization=read(authorization) if authorization else None))
        with (destination / "stdout.log").open("xb") as out, (destination / "stderr.log").open("xb") as err:
            code = subprocess.run(command, stdout=out, stderr=err, pass_fds=tuple(fds), env=env, **({} if extension is None else {"timeout":spec["resources"]["max_wall_s"]})).returncode
        write(destination / "PROCESS_EXIT.json", dict(returncode=code))
    except BaseException as e:
        write(destination / "LAUNCH_ERROR.json", dict(error=repr(e), traceback=traceback.format_exc(), returncode=code))
        raise
    finally:
        for fd in fds: os.close(fd)
        after = protection(paths)
        write(destination / "PROTECTION_AFTER.json", dict(passed=before == after, files=after,
              scope="explicit input and package source bytes", historical_baseline_failure_status="UNCHANGED_NOT_RECLASSIFIED"))
    if before != after or code != 0:
        write(destination / "STOP.json", dict(reason="EXTERNAL_PROTECTION_AFTER" if before != after else "WORKER_NONZERO_EXIT",
              worker_returncode=code, no_resume=True))
    require(before == after, "EXTERNAL_PROTECTION_AFTER")
    require(code == 0, "WORKER_NONZERO_EXIT " + str(code))
    if action == "train":
        from .audit import audit_run
        try:
            result = (audit_run if extension is None else extension.audit_run)(destination, candidate=True)
        except BaseException as e:
            write(destination / "STOP.json", dict(reason="CONTENT_AUDIT_FAILED", error=repr(e),
                  traceback=traceback.format_exc(), worker_returncode=code, no_resume=True))
            raise
        write(destination / "CONTENT_AUDIT.json", result)
        write(destination / "RESULT.json", dict(status="COMPLETED", content_audited=True,
              profile=spec["profile"], synthetic_disposable=spec["profile"] == "disposable_single_step_v1"))
        write(destination / "RUN_SEAL.json", dict(files={str(p.relative_to(destination)): sha(p)
              for p in destination.rglob("*") if p.is_file()}))
        (audit_run if extension is None else extension.audit_run)(destination)
        return result
    result = read(destination / "PREFLIGHT.json")
    write(str(destination / "PREFLIGHT.json") + ".seal.json", dict(sha256=sha(destination / "PREFLIGHT.json")))
    return result
