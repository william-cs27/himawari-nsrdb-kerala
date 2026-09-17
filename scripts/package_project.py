"""Package reusable files only; exclude dependencies, raw disks and runtime caches."""

from pathlib import Path
import argparse
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--output", default=str(ROOT.parent / "himawari_nsrdb_project.zip"))
args = p.parse_args()
folders = [
    "src",
    "configs",
    "scripts",
    "tests",
    "docs",
    "notebooks",
    "data/nsrdb",
    "data/processed",
    "data/patches/smoke",
    "outputs",
]
paths = [
    ROOT / name
    for name in ["README.md", "RESULTS.md", "pyproject.toml", ".gitignore"]
    if (ROOT / name).exists()
]
for folder in folders:
    paths.extend(
        f
        for f in (ROOT / folder).rglob("*")
        if f.is_file()
        and "__pycache__" not in f.parts
        and "archive" not in f.parts
        and not any(part.endswith(".egg-info") for part in f.parts)
        and not f.name.endswith(".tmp")
    )
paths = sorted(set(paths))
manifest = {
    str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
}
out = Path(args.output)
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in paths:
        archive.write(path, Path(ROOT.name) / path.relative_to(ROOT))
    archive.writestr(
        ROOT.name + "/MANIFEST.json", json.dumps(manifest, indent=2) + "\n"
    )
with zipfile.ZipFile(out) as archive:
    assert archive.testzip() is None
print(
    json.dumps(
        {
            "file": str(out),
            "files": len(paths),
            "bytes": out.stat().st_size,
            "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        }
    )
)
