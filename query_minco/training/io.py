"""Small strict, exclusive, side-effect-free artifact helpers."""
import hashlib
import json
import os
from pathlib import Path

def require(condition, message):
    if not condition:
        raise ValueError(message)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

def read(path):
    def pairs(items):
        obj = {}
        for k, v in items:
            require(k not in obj, "DUPLICATE_JSON_KEY " + k)
            obj[k] = v
        return obj
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError("NONFINITE_JSON")))

def write(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())

def exact(obj, keys, label):
    require(isinstance(obj, dict) and set(obj) == set(keys), "SCHEMA " + label)

def artifact(binding):
    exact(binding, ["path", "sha256"], "artifact")
    p = Path(binding["path"])
    require(p.is_absolute() and p.is_file() and not p.is_symlink(), "ARTIFACT_PATH")
    require(sha(p) == binding["sha256"], "ARTIFACT_SHA " + str(p))
    return p.resolve()
