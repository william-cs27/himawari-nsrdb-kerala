from __future__ import annotations
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import numpy as np
import pandas as pd


def merge(base, override):
    out = dict(base)
    for k, v in override.items():
        out[k] = (
            merge(out[k], v)
            if isinstance(v, dict) and isinstance(out.get(k), dict)
            else v
        )
    return out


def load_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    if "extends" in config:
        config = merge(load_config(path.parent / config.pop("extends")), config)
    config.update(_root=str(path.parent.parent), _config=str(path))
    return config


def resolve(config, path):
    return (Path(config["_root"]) / path).resolve()


def utc(value):
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise ValueError("Explicit timezone required")
    return stamp.tz_convert("UTC")


def nanos(index):
    return pd.DatetimeIndex(index).as_unit("ns").asi8


def fingerprint(config):
    return hashlib.sha256(
        json.dumps(
            {k: v for k, v in config.items() if not k.startswith("_")}, sort_keys=True
        ).encode()
    ).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def array_fingerprint(arrays):
    h = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        h.update(name.encode())
        h.update(str(value.dtype).encode())
        h.update(str(value.shape).encode())
        h.update(value.tobytes())
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")
    os.replace(temp, path)


def versions():
    info = {"python": platform.python_version(), "platform": platform.platform()}
    for name in [
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "pvlib",
        "satpy",
        "pyresample",
        "torch",
    ]:
        try:
            info[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            info[name] = None
    return info


def new_run(config, name):
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("Run name must be a simple directory name")
    path = resolve(config, "outputs") / name
    path.mkdir(parents=True, exist_ok=False)
    write_json(path / "config.json", config)
    write_json(path / "environment.json", versions())
    return path
