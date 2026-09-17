from __future__ import annotations
import pickle
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from .common import array_fingerprint, new_run, write_json
from .data import load_prepared, subset
from .evaluation import prediction_frame, ramp_thresholds, summarize
from .satellite import align_patches


def run(cfg, name, matched=False, evaluate_test=False):
    started = time.monotonic()
    data = load_prepared(cfg)
    if matched:
        keep, _, _ = align_patches(cfg, data)
        data = subset(data, keep)
    train = np.flatnonzero(data["split"] == "train")
    val = np.flatnonzero(data["split"] == "validation")
    if not len(train) or not len(val):
        raise ValueError("Need training and validation samples")
    dest = new_run(cfg, name)
    thresholds = ramp_thresholds(data)
    flat = data["X"].reshape(len(data["X"]), -1)
    models = []
    with threadpool_limits(limits=cfg["training"]["cpu_threads"]):
        for j in range(len(data["horizons"])):
            model = HistGradientBoostingRegressor(
                loss="absolute_error",
                early_stopping=False,
                random_state=cfg["training"]["seed"],
                **cfg["gradient_boosting"],
            )
            model.fit(flat[train], data["y"][train, j])
            models.append(model)
    climatologies = []
    for j in range(len(models)):
        stamp = pd.to_datetime(data["target_ns"][train, j], utc=True)
        table = pd.DataFrame(
            {
                "month": stamp.month,
                "slot": stamp.hour * 6 + stamp.minute // 10,
                "y": data["y"][train, j],
            }
        )
        climatologies.append(
            (
                table.groupby(["month", "slot"]).y.mean().to_dict(),
                table.groupby("slot").y.mean().to_dict(),
                table.y.mean(),
            )
        )
    result = []
    for split in ["validation"] + (["test"] if evaluate_test else []):
        idx = np.flatnonzero(data["split"] == split)
        if not len(idx):
            continue
        predictions = {
            "smart_persistence": np.repeat(
                data["k_now"][idx, None], len(models), axis=1
            ),
            "raw_ghi_persistence": data["ghi_now"][idx, None]
            / data["target_clear"][idx],
        }
        with threadpool_limits(limits=cfg["training"]["cpu_threads"]):
            predictions["hist_gradient_boosting"] = np.column_stack(
                [m.predict(flat[idx]) for m in models]
            )
        clim = []
        for j, (month_slot, slot_map, overall) in enumerate(climatologies):
            times = pd.to_datetime(data["target_ns"][idx, j], utc=True)
            clim.append(
                [
                    month_slot.get(
                        (t.month, t.hour * 6 + t.minute // 10),
                        slot_map.get(t.hour * 6 + t.minute // 10, overall),
                    )
                    for t in times
                ]
            )
        predictions["calendar_climatology"] = np.column_stack(clim)
        for model, prediction in predictions.items():
            result.append(
                prediction_frame(
                    data, idx, prediction, model, cfg["training"]["seed"], thresholds
                )
            )
    frame = pd.concat(result, ignore_index=True)
    frame.to_csv(dest / "predictions.csv", index=False)
    metrics = summarize(frame)
    write_json(dest / "metrics.json", metrics)
    with open(dest / "baselines.pkl", "wb") as f:
        pickle.dump(
            {
                "boosting": models,
                "climatologies": climatologies,
                "thresholds": thresholds,
            },
            f,
        )
    write_json(
        dest / "run.json",
        {
            "matched_images": matched,
            "training_samples": len(train),
            "validation_samples": len(val),
            "wall_seconds": time.monotonic() - started,
            "test_evaluated": evaluate_test,
            "prepared_config_fingerprint": str(data["config_fingerprint"]),
            "dataset_sha256": array_fingerprint(data),
        },
    )
    return {"output": str(dest), "metrics": metrics}
