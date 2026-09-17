# Experiment protocol v0.1

## Question, estimand and hypotheses

Does regional Himawari infrared cloud evolution improve **retrospective NSRDB irradiance forecasts** beyond recent local weather, cloud type and irradiance at 9.96°N, 76.25°E? This is a single-site prediction study, not independent ground validation or causal inference.

Define k(t) = GHI(t) / Haurwitz_GHI(t). Use six tabular samples at t−50,...,t and six completed satellite frames nominally at t−60,...,t−10 minutes. Predict k at 10, 30, 60 and 120 minutes. Multiply by deterministic future Haurwitz GHI to recover W/m². Floor negative predictions at zero; retain values above one. The primary estimand is paired 60-minute index MAE improvement on identical out-of-time forecast timestamps.

- **H1, primary:** Fusion improves 60-minute index MAE over the strongest matched NSRDB-only baseline selected on validation. The practical target is ≥5% relative improvement. Evidence for any improvement and evidence for ≥5% improvement are different claims.
- **H2, diagnostic:** Ordered regional image histories outperform current-only, shuffled-history and spatial-mean controls. These test use of history/spatial structure, not necessarily physical advection.
- **H3, exploratory:** Removing moisture, cloud categories or recent irradiance changes performance, particularly on cloud-changing or ramp subsets. These ablations measure predictive contribution, not atmospheric causal effects.

## Data and eligibility

The user selected the 2016–2020 Asia/Australia/Pacific Himawari-based NSRDB product. The supplied 2018 point CSV has 52,560 ten-minute rows and a numeric `Time Zone` of zero. Do not interpret its separate `Local Time Zone` field as the timestamp offset. Preserve the two metadata rows and source checksum. The 2026 VOCI data do not overlap and are excluded.

Features are temperature, RH, dew point, pressure, precipitable water, wind speed/vector components, ozone, dew-point depression, backward 30-minute pressure change, GHI, past normalized GHI, one-hot cloud categories, solar geometry and cyclic calendar features. Recognized unknown cloud codes remain categorical. Reindexed missing rows remain missing; crossing histories are excluded, without interpolation or forward/backward filling.

Issuance and every requested target must have solar elevation >5° and Haurwitz GHI >50 W/m², with finite features/targets. Using the same four-horizon cohort excludes some late-afternoon forecasts. Haurwitz is a reproducible calendar/site reference, not NSRDB REST2; the CSV lacks supplied clear-sky GHI and fill flags. Completeness does not prove that upstream observations were unfilled or unsmoothed.

Satellite features are calibrated brightness temperature in Kelvin. The engineering sample uses B13 (10.4 μm), 32×32 pixels. The planned study uses B08 (6.2 μm), B13 and B15 (12.4 μm), 64×64 pixels. Both cover a 256×256 km azimuthal-equidistant region centered on the site. Nearest-neighbor resampling uses an 18 km radius; invalid pixels outside 150–350 K are masked, and crops with >1% missing pixels fail. Fixed scaling is (K−250)/50 with explicit mask channels. Source keys, file hashes, scan completion, geometry and crop hashes are saved.

Image availability is the later of scan completion and nominal time +10 minutes. All six nominal times must be contiguous and available at issue time. This is an explicit latency assumption, not a measured delivery guarantee. A later 20-minute-latency sensitivity is needed. At about 73° satellite zenith, distorted resolution and cloud parallax matter: nominal 2 km IR sampling applies at the sub-satellite point, and regridding does not restore it. No parallax correction is applied in v0.1.

## Splits and leakage controls

| Split | UTC interval, end exclusive | Purpose |
|---|---|---|
| Train | 2018-01-01 → 2018-08-01 | Fit models, normalization, climatology, thresholds |
| Validation | 2018-08-01 → 2018-11-01 | Early stopping and predeclared model selection |
| Test | 2018-11-01 → 2019-01-01 | Final evaluation after freezing choices |

The complete raw feature history, image history and furthest target must stay in their assigned block. Six tabular steps span 50 minutes, but the earliest pressure-change feature needs another 30 minutes; split purging includes both. Standardization is fitted on training only, using float64 moment accumulation and unit scale for constant features before conversion back to float32. There are no random row splits. Overlapping forecasts within a split are still dependent.

The January 1/3/5 smoke configuration is a separate, deliberately tiny engineering exercise. Its scores cannot support H1–H3. Validation only is evaluated in the first delivery; neither scientific nor engineering test scores are used for model choice.

Season and split are confounded in one year. November–December is not an all-season test. A future 2019 point-year and matched imagery, with product/version provenance checked, would provide stronger temporal evaluation. More sites/years require an explicit new holdout protocol.

## Experiment progression and controls

1. **Engineering gate:** parse/audit, acquire a bounded real image sample, calibrate/crop, align completed scans, train all three neural modes, serialize predictions and verify interrupted-run recovery.
2. **Acquisition feasibility:** choose contiguous blocks without inspecting target events/errors, cover seasons, and inventory bytes before transfer. A free budget may support only the engineering gate; do not label three days a research study.
3. **Core matrix:** raw and smart persistence, training climatology, gradient boosting, tabular TCN, image CNN–GRU and fusion. Rerun every local baseline on the exact same image-matched training and validation cohort. The all-NSRDB benchmark is supplemental.
4. **Ablations:** current-only image repeated to preserve sequence shape; deterministic shuffled historical order with the current frame fixed; spatial-mean image; zeroed standardized GHI plus derived k; zeroed cloud one-hots; zeroed RH/dew point/PW/dew-point depression. One change per experiment. See `experiment_matrix.csv`.
5. **Freeze:** record all runs, choose the strongest matched local comparator using validation, freeze preprocessing/architecture/thresholds and report any null result. Do not select the best random seed.
6. **Final test:** explicitly use `--evaluate-test` once after freezing choices. Any later tuning needs a new holdout.

Neural seeds are 13, 29, 47. The main configuration uses AdamW, learning rate 0.001, weight decay 0.0001, batch 32, hidden width 32, up to 50 epochs and patience 7. Train with MAE averaged over all horizons; select checkpoints by validation 60-minute index MAE. The smoke uses width16/batch8/three epochs. Record parameters, samples, device, environment, epoch count and duration.

Identical optimization settings do not imply identical capacity or compute. Before attributing a fusion win to modality, add a wider tabular capacity control. Longer histories, bands, spatial extent, parallax correction and latency sensitivity are future controlled experiments, not test-tuned settings.

## Outcomes and inference

Primary: 60-minute index MAE. Secondary: index/GHI MAE and RMSE at all horizons, cloud-changing subsets, and derived ramp ranking/decision diagnostics. A ramp is endpoint absolute k change above the **training-only 90th percentile**, independently per horizon. This is not the largest transient change within the interval. Average precision uses predicted absolute change as a ranking score; precision/recall use the same threshold. These are not calibrated probabilities. Report prevalence and distinct days/episodes, not merely overlapping positive rows.

Coarse cloud groups: clear/probably-clear {0,1}; liquid/fog {2,3,4}; ice/cirrus/overlap/overshooting {6,7,8,9}. Mixed {5}, dust/smoke/other {10,11,12} and unknown {−15} are excluded from this transition diagnostic. Grouping sensitivity is needed; the current code does not train a cloud-transition classifier.

Paired uncertainty uses non-overlapping seven-day calendar blocks and 5,000 bootstrap resamples, aggregating error sums/counts within sampled blocks. Report 95% intervals for MAE difference and relative skill. At least eight represented blocks are needed for the software to compute an interval; this is a minimum guard, not a power guarantee. Check 1/3/14-day block sensitivity. A 61-day final test affords few blocks.

Report paired seed-specific comparisons and mean/spread across all three seeds as training variability. Do not pool seeds as independent weather observations. Secondary outcomes/ablations are exploratory; no uncorrected family of confirmatory significance claims. Estimate block-level variance on pilot/validation data before choosing a larger acquisition target. The raw row count does not establish effective sample size or power.

## Evidence limits

NSRDB labels and Himawari imagery share satellite lineage. Improvements can reflect richer spatial context or reconstruction of the product's generating process. Independent contemporaneous pyranometer observations are required for claims about measured irradiance. VOCI surface weather is not such a reference.

Retrospective meteorological fields may use temporally interpolated or reanalyzed information. Downstream causal windows do not prove upstream real-time availability. Restrict any operational claim to sources whose release/assimilation latencies are audited. Document null results, missing scans, corruption, all attempted configurations and compute. This is a versioned implementation-time analysis plan, not external preregistration or a verified novelty claim.
