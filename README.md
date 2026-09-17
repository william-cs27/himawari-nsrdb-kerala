# Himawari–NSRDB Kerala forecasting research

Standalone Python/Colab implementation for retrospective 10/30/60/120-minute irradiance forecasting at 9.96°N, 76.25°E. The question is whether regional Himawari infrared cloud histories improve predictions beyond local NSRDB weather, cloud type and recent irradiance.

Start with `notebooks/colab_start.ipynb` and the complete project ZIP. It uses the bundled NSRDB file and small real Himawari crops; no credentials or paid compute are required. The small satellite example tests execution, not scientific skill. See `RESULTS.md` for actual results and `docs/EXPERIMENT_PROTOCOL.md` for the larger study.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[neural,dev]"
python -m weather_fusion --config configs/smoke.json prepare
python -m weather_fusion --config configs/smoke.json alignment
python scripts/run_experiments.py --config configs/smoke.json --stage smoke --prefix my_first_run --execute
```

Use Python 3.10–3.12. Colab's provided PyTorch normally suffices. GPU setup on your computer is hardware-dependent; consult the [official PyTorch installer](https://pytorch.org/get-started/locally/). CPU works for the example. Decoder dependencies are optional: `python -m pip install -e ".[satellite]"`.

## Scientific baseline and satellite acquisition

```bash
python -m weather_fusion --config configs/study.json prepare
python -m weather_fusion --config configs/study.json baselines --run-name my_nsrdb_baselines
python -m weather_fusion --config configs/smoke.json satellite-plan
# Inspect outputs/engineering_smoke_inventory.json before fetching.
python -m weather_fusion --config configs/smoke.json satellite-fetch
```

The default study has no satellite acquisition windows and cannot silently download a year. New configurations belong inside `configs/`, inherit `study.json`, and set explicit UTC `satellite.windows`, a new patch directory and a deliberate byte cap. Arrays replace inherited arrays. Rerun prepare/inventory after config changes. Anonymous AWS access needs no account. Cached crops are verified and reused. Missing timestamps invalidate crossing histories; no temporal filling.

Historical AWS files are compressed full disks. Cropping saves retained storage, not transfer: the 44-frame one-band example needs about 1 GiB before cropping, while a three-band full year could approach 3 TB. The ZIP bundles the small crops. Do not request a full year on free Colab without a revised resource plan.

## Models and experiments

Implemented baselines: raw GHI persistence, clear-sky-index smart persistence, training-only calendar climatology, histogram gradient boosting. Neural models: causal tabular TCN, image CNN–GRU, and their fusion. Six fusion ablations remove image history, image ordering, spatial structure, irradiance, cloud categories, or moisture inputs.

```bash
# Print the planned commands without executing:
python scripts/run_experiments.py --config configs/study.json --stage core --prefix pilot
python scripts/run_experiments.py --config configs/study.json --stage ablations --prefix ablation
# Add --execute only after preparing a suitable satellite cohort.
```

Core and ablation stages use seeds 13, 29, 47 and identical image-matched cohorts. They do not download data or evaluate test. New runs require new names. `--evaluate-test` explicitly unlocks the final held-out split only after model/protocol selection. Default evaluation is validation only.

```bash
python -m weather_fusion --config configs/smoke.json compare outputs/my_first_run_baselines/predictions.csv outputs/my_first_run_fusion_s13/predictions.csv --baseline-model smart_persistence --candidate-model fusion_none --output outputs/my_comparison.json
python -m weather_fusion --config configs/smoke.json train --run-name my_first_run_fusion_s13 --mode fusion --matched-images --resume outputs/my_first_run_fusion_s13/last.pt
```

Resume requires unchanged config/seed/data/cohort, and both last/best checkpoints. Checkpoints preserve optimizer and random state; only load trusted project `.pt`/`.pkl` files. Save/download the run directory before Colab disconnects.

## Interpretation

- This NSRDB product already derives GHI/cloud information from Himawari: imagery and labels are not independent sensors. Independent ground irradiance remains needed for physical validation.
- The supplied CSV lacks clear-sky GHI and fill flags. A deterministic Haurwitz reference is used, not NSRDB REST2. Retrospective weather products may involve upstream interpolation using later observations.
- The site is seen at roughly 73° satellite zenith. Nominal 2 km infrared sampling applies at the sub-satellite point, not Kerala. Regridding does not restore resolution; parallax is uncorrected.
- Scan completion is checked, with an assumed minimum 10-minute latency. Actual real-time delivery/ancillary-feature availability has not been established.
- 2026 VOCI observations cannot be joined to 2018 and are not used. The present implementation forecasts irradiance and derives ramp/transition diagnostics; a calibrated cloud-transition classifier is a future extension.

`configs/study.json`: Jan–Jul training, Aug–Oct validation, Nov–Dec untouched test. `configs/smoke.json`: Jan 1/3/5 engineering splits. Six input points span 50 minutes; the pressure-change feature adds a further 30-minute raw-data lookback when purging split boundaries.

All source data, processing metadata, experiment settings, predictions and checkpoints are organized under `data/`, `configs/` and `outputs/`. See `docs/SOURCES.md` for official references and `docs/RESOURCE_PLAN.md` for scaling costs. The original uploads are unchanged.
