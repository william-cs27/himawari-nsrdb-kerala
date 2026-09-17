from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score


def prediction_frame(data, indices, prediction, model, seed, thresholds):
    prediction = np.maximum(np.asarray(prediction), 0.0)
    if (
        prediction.shape != data["y"][indices].shape
        or not np.isfinite(prediction).all()
    ):
        raise ValueError("Invalid predictions")
    frames = []
    for j, h in enumerate(data["horizons"]):
        frames.append(
            pd.DataFrame(
                {
                    "issue_time": pd.to_datetime(data["time_ns"][indices], utc=True),
                    "target_time": pd.to_datetime(
                        data["target_ns"][indices, j], utc=True
                    ),
                    "split": data["split"][indices],
                    "horizon_minutes": int(h),
                    "model": model,
                    "seed": seed,
                    "observed_k": data["y"][indices, j],
                    "predicted_k": prediction[:, j],
                    "observed_ghi": data["target_ghi"][indices, j],
                    "predicted_ghi": prediction[:, j]
                    * data["target_clear"][indices, j],
                    "k_now": data["k_now"][indices],
                    "ramp_threshold": float(thresholds[j]),
                    "cloud_now": data["cloud_now"][indices],
                    "cloud_future": data["cloud_future"][indices, j],
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def summarize(predictions):
    result = []
    for (model, seed, split, horizon), g in predictions.groupby(
        ["model", "seed", "split", "horizon_minutes"]
    ):
        error = g.predicted_k - g.observed_k
        ge = g.predicted_ghi - g.observed_ghi
        event = (g.observed_k - g.k_now).abs() > g.ramp_threshold
        score = (g.predicted_k - g.k_now).abs()
        decision = score > g.ramp_threshold
        changing = (
            (g.cloud_now != g.cloud_future) & (g.cloud_now >= 0) & (g.cloud_future >= 0)
        )
        result.append(
            {
                "model": model,
                "seed": int(seed),
                "split": split,
                "horizon_minutes": int(horizon),
                "n": len(g),
                "mae_k": float(error.abs().mean()),
                "rmse_k": float(np.sqrt((error**2).mean())),
                "mae_ghi_wm2": float(ge.abs().mean()),
                "rmse_ghi_wm2": float(np.sqrt((ge**2).mean())),
                "ramp_prevalence": float(event.mean()),
                "ramp_events_rows": int(event.sum()),
                "ramp_average_precision": (
                    float(average_precision_score(event, score))
                    if event.any() and (~event).any()
                    else None
                ),
                "ramp_precision": float(
                    precision_score(event, decision, zero_division=0)
                ),
                "ramp_recall": float(recall_score(event, decision, zero_division=0)),
                "cloud_changing_mae_k": (
                    float(error[changing].abs().mean()) if changing.any() else None
                ),
            }
        )
    return result


def ramp_thresholds(data):
    train = data["split"] == "train"
    return np.quantile(
        np.abs(data["y"][train] - data["k_now"][train, None]), 0.9, axis=0
    )


def compare(
    left, right, horizon=60, block_days=7, resamples=5000, seed=13, practical_skill=0.05
):
    if block_days < 1 or resamples < 1:
        raise ValueError("Positive block length/resamples required")
    keys = ["issue_time", "target_time", "horizon_minutes", "split"]
    a, b = (
        left[left.horizon_minutes == horizon],
        right[right.horizon_minutes == horizon],
    )
    for frame in [a, b]:
        if (
            not len(frame)
            or frame.duplicated(keys).any()
            or frame.model.nunique() != 1
            or frame.seed.nunique() != 1
            or frame.split.nunique() != 1
        ):
            raise ValueError(
                "Select one model/seed/split/horizon; no duplicate forecasts"
            )
    both = a.merge(
        b, on=keys, suffixes=("_baseline", "_candidate"), validate="one_to_one"
    )
    if len(both) != len(a) or len(both) != len(b):
        raise ValueError("Cohort mismatch; rerun using --matched-images")
    if not np.allclose(both.observed_k_baseline, both.observed_k_candidate, atol=1e-7):
        raise ValueError("Targets differ")
    eb = (both.predicted_k_baseline - both.observed_k_baseline).abs().to_numpy()
    ec = (both.predicted_k_candidate - both.observed_k_candidate).abs().to_numpy()
    dates = pd.to_datetime(both.issue_time, utc=True)
    block = np.floor(
        (dates - dates.min().normalize()).dt.total_seconds() / (block_days * 86400)
    ).astype(int)
    blocks = (
        pd.DataFrame({"b": block, "baseline": eb, "candidate": ec, "count": 1})
        .groupby("b")
        .sum()
    )
    base, cand = float(eb.mean()), float(ec.mean())
    out = {
        "horizon_minutes": horizon,
        "n": len(both),
        "blocks": len(blocks),
        "block_days": block_days,
        "baseline_mae": base,
        "candidate_mae": cand,
        "relative_skill": 1 - cand / base if base else None,
        "difference_mae_baseline_minus_candidate": base - cand,
        "practical_skill_threshold": practical_skill,
    }
    if len(blocks) < 8:
        out.update(status="descriptive_only_too_few_blocks", confidence_interval=None)
        return out
    if base == 0 or (blocks.baseline <= 0).any():
        out.update(
            status="descriptive_only_zero_baseline_error", confidence_interval=None
        )
        return out
    rng = np.random.default_rng(seed)
    sums = blocks[["baseline", "candidate", "count"]].to_numpy()
    samples = sums[rng.integers(0, len(sums), size=(resamples, len(sums)))].sum(axis=1)
    difference = (samples[:, 0] - samples[:, 1]) / samples[:, 2]
    skill = 1 - samples[:, 1] / samples[:, 0]
    out.update(
        status="block_bootstrap",
        confidence_interval=np.quantile(difference, [0.025, 0.975]).tolist(),
        skill_confidence_interval=np.quantile(skill, [0.025, 0.975]).tolist(),
        supports_any_improvement=bool(np.quantile(difference, 0.025) > 0),
        supports_practical_improvement=bool(
            np.quantile(skill, 0.025) >= practical_skill
        ),
    )
    return out
