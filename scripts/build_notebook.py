from pathlib import Path
import json
import textwrap

ROOT = Path(__file__).resolve().parents[1]
cells = []


def cell(kind, text):
    c = {
        "cell_type": kind,
        "metadata": {},
        "source": textwrap.dedent(text).strip().splitlines(keepends=True),
        "id": f"step-{len(cells):02d}",
    }
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    cells.append(c)


cell(
    "markdown",
    """
# Himawari–NSRDB: first reproducible experiment
Upload the **complete project ZIP** below. It includes the NSRDB CSV and real infrared crops, so this example needs no satellite downloads or credentials. Use Colab → Runtime → Change runtime type → GPU if available. CPU also works.

The January 1/3/5 example is an engineering check, not research evidence. Validation only is evaluated. The scientific November–December holdout remains untouched.
""",
)
cell(
    "code",
    """
from pathlib import Path
import os, sys, zipfile, subprocess, json
try:
    from google.colab import files
    IN_COLAB=True
except ImportError:
    IN_COLAB=False
if IN_COLAB:
    project=Path('/content/himawari_nsrdb')
    if not (project/'pyproject.toml').exists():
        uploaded=files.upload()
        archives=[Path(n) for n in uploaded if n.lower().endswith('.zip')]
        if len(archives)!=1: raise ValueError('Upload exactly one complete project ZIP')
        destination=Path('/content').resolve()
        with zipfile.ZipFile(archives[0]) as archive:
            for entry in archive.infolist():
                if destination not in (destination/entry.filename).resolve().parents: raise ValueError('Unexpected archive path')
            archive.extractall(destination)
else:
    project=Path.cwd()
    if project.name=='notebooks': project=project.parent
assert (project/'pyproject.toml').exists()
os.chdir(project)
print('Project:',project)
""",
)
cell(
    "markdown",
    """
## Install and verify
Decoder dependencies are optional because the crops are bundled. If a binary import conflict appears after installation, restart the runtime and rerun these cells. Exact versions used for delivered results are recorded under `docs/` and `outputs/`.
""",
)
cell(
    "code",
    """
subprocess.run([sys.executable,'-m','pip','install','-q','-e','.[neural,dev]'],check=True)
import torch
print('PyTorch:',torch.__version__,'| Device:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
subprocess.run([sys.executable,'-m','pytest','-q'],check=True)
""",
)
cell(
    "markdown",
    """
## Prepare and align
UTC joins, complete scan times, missing-sequence rejection and strict chronological splits are enforced. Read the protocol for upstream retrospective interpolation and shared Himawari/NSRDB provenance limitations.
""",
)
cell(
    "code",
    """
from datetime import datetime,timezone
CONFIG='configs/smoke.json'
PREFIX=datetime.now(timezone.utc).strftime('colab_%Y%m%dT%H%M%SZ')
BASE=[sys.executable,'-m','weather_fusion','--config',CONFIG]
def run(*args):
    result=subprocess.run(BASE+list(args),text=True,capture_output=True)
    if result.returncode:
        print(result.stdout); print(result.stderr); result.check_returncode()
    return result.stdout
prepared=json.loads(run('prepare'))
print('Rows:',prepared['rows'],'| Before image matching:',prepared['sample_counts'])
print(run('alignment'))
print('Run prefix:',PREFIX)
""",
)
cell(
    "markdown",
    """
## Train the same-cohort models
Each neural model runs three epochs. Tiny-sample rankings do not establish scientific skill. Each run saves predictions, metadata and resumable checkpoints in a new directory.
""",
)
cell(
    "code",
    """
print('Fitting matched baselines...',flush=True)
run('baselines','--matched-images','--run-name',PREFIX+'_baselines')
for mode in ['tabular','image','fusion']:
    print('Training:',mode,flush=True)
    log=run('train','--mode',mode,'--matched-images','--seed','13','--run-name',PREFIX+'_'+mode+'_s13','--device','auto')
    print('\\n'.join(line for line in log.splitlines() if line.startswith('epoch=')))
""",
)
cell(
    "code",
    """
import pandas as pd
rows=[]
for suffix in ['baselines','tabular_s13','image_s13','fusion_s13']:
    rows.extend(json.loads((Path('outputs')/(PREFIX+'_'+suffix)/'metrics.json').read_text()))
metrics=pd.DataFrame(rows)
display(metrics[metrics.horizon_minutes==60][['model','split','n','mae_k','mae_ghi_wm2']])
comparison=f'outputs/{PREFIX}_comparison.json'
run('compare',f'outputs/{PREFIX}_baselines/predictions.csv',f'outputs/{PREFIX}_fusion_s13/predictions.csv',
    '--baseline-model','smart_persistence','--candidate-model','fusion_none','--output',comparison)
print(Path(comparison).read_text())
print('Engineering sample: too few independent blocks for inference.')
""",
)
cell(
    "markdown",
    """
## Optional full NSRDB benchmark
Set the flag below to train January–July and validate August–October without imagery. This is a different cohort, so do not compare its scores directly to the engineering example. It does not evaluate November–December.
""",
)
cell(
    "code",
    """
RUN_FULL_TABULAR=False
if RUN_FULL_TABULAR:
    study=[sys.executable,'-m','weather_fusion','--config','configs/study.json']
    subprocess.run(study+['prepare'],check=True)
    subprocess.run(study+['baselines','--run-name',PREFIX+'_full_nsrdb'],check=True)
""",
)
cell(
    "markdown",
    """
## More satellite data
See `docs/RESOURCE_PLAN.md`: the original B13 sample requires about 1 GiB of full-disk transfer before cropping. A three-band full year may approach 3 TB. The study config intentionally has no download windows.

Install `.[satellite]` only to decode new files. Set explicit windows/new patch directory/byte cap, run `satellite-plan`, inspect its inventory, then use `satellite-fetch`. The notebook does not automatically download images.

## Export results before ending Colab
The next cell saves this run's results/checkpoints and small source/crop/prepared datasets. Keep the original project ZIP for the implementation and protocol. Colab session storage is temporary.
""",
)
cell(
    "code",
    """
export=project.parent/(PREFIX+'_results.zip')
paths=set()
for path in Path('outputs').glob(PREFIX+'*'):
    paths.update(path.rglob('*') if path.is_dir() else [path])
for folder in ['configs','data/nsrdb','data/patches/smoke','data/processed']:
    paths.update(Path(folder).rglob('*'))
with zipfile.ZipFile(export,'w',zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(paths):
        if path.is_file(): archive.write(path,Path('himawari_nsrdb')/path)
print('Saved:',export)
if IN_COLAB: files.download(str(export))
""",
)
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "name": "python3",
            "display_name": "Python 3",
            "language": "python",
        },
        "language_info": {"name": "python"},
        "colab": {"name": "colab_start.ipynb"},
    },
    "cells": cells,
}
output = ROOT / "notebooks/colab_start.ipynb"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(notebook, indent=2) + "\n")
print(output)
