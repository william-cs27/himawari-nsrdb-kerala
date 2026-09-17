from __future__ import annotations
import csv
import numpy as np
import pandas as pd
from .common import fingerprint, nanos, resolve, sha256, utc, write_json

FIELDS = [
    "Temperature",
    "Relative Humidity",
    "Dew Point",
    "Pressure",
    "Precipitable Water",
    "Wind Speed",
    "Ozone",
]
CLOUD_CODES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, -15]


def read_nsrdb(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        metadata = dict(zip(next(reader), next(reader)))
    df = pd.read_csv(path, skiprows=2)
    needed = [
        "Year",
        "Month",
        "Day",
        "Hour",
        "Minute",
        "GHI",
        "Wind Direction",
        "Cloud Type",
        *FIELDS,
    ]
    if set(needed) - set(df):
        raise ValueError(f"Missing columns: {set(needed)-set(df)}")
    if "Time Zone" not in metadata:
        raise ValueError(
            "Time Zone metadata required; do not infer from Local Time Zone"
        )
    local = pd.to_datetime(df[["Year", "Month", "Day", "Hour", "Minute"]])
    times = pd.DatetimeIndex(
        local - pd.Timedelta(hours=float(metadata["Time Zone"]))
    ).tz_localize("UTC")
    if times.has_duplicates or not times.is_monotonic_increasing:
        raise ValueError("Duplicate or unsorted timestamps")
    df.index = times
    df.index.name = "time"
    for name in ["GHI", "Wind Direction", "Cloud Type", *FIELDS]:
        df[name] = pd.to_numeric(df[name], errors="coerce")
    return df.replace([-999, -9999], np.nan), metadata


def solar_reference(index, cfg):
    import pvlib

    pos = pvlib.solarposition.get_solarposition(
        index,
        cfg["latitude"],
        cfg["longitude"],
        altitude=cfg["elevation_m"],
        pressure=101325,
        temperature=12,
    )
    if cfg["clearsky_model"] != "haurwitz":
        raise ValueError("Pilot locks Haurwitz normalization")
    return pvlib.clearsky.haurwitz(pos.apparent_zenith).ghi.to_numpy().astype(
        np.float32
    ), pos.apparent_elevation.to_numpy().astype(np.float32)


def audit(cfg):
    path = resolve(cfg, cfg["csv"])
    df, meta = read_nsrdb(path)
    for key, field in [("latitude", "Latitude"), ("longitude", "Longitude")]:
        if abs(float(meta[field]) - cfg[key]) > 0.02:
            raise ValueError("Site mismatch")
    _, elev = solar_reference(df.index, cfg)
    return {
        "rows": len(df),
        "start_utc": df.index[0].isoformat(),
        "end_utc": df.index[-1].isoformat(),
        "metadata": meta,
        "source_sha256": sha256(path),
        "expected_interval_minutes": cfg["interval_minutes"],
        "nonconforming_intervals": int(
            (np.diff(nanos(df.index)) / 60e9 != cfg["interval_minutes"]).sum()
        ),
        "missing_by_column": {k: int(v) for k, v in df.isna().sum().items()},
        "cloud_counts": {
            str(k): int(v)
            for k, v in df["Cloud Type"].value_counts().sort_index().items()
        },
        "ghi_above_10_at_solar_elevation_below_minus5": int(
            ((elev < -5) & (df.GHI.to_numpy() > 10)).sum()
        ),
        "fill_flag_available": "Fill Flag" in df,
        "nsrdb_clearsky_available": "Clearsky GHI" in df,
        "clearsky_reference": "pvlib Haurwitz; not NSRDB REST2",
        "warnings": [
            "GHI/cloud labels share Himawari provenance with imagery.",
            "No independent ground irradiance is supplied.",
            "Retrospective ancillary fields may use upstream temporal interpolation.",
            "Completeness does not establish unfilled observations.",
        ],
    }


def cloud_group(values):
    arr = np.asarray(values)
    out = np.full(arr.shape, -1, dtype=np.int64)
    for group, codes in enumerate([[0, 1], [2, 3, 4], [6, 7, 8, 9]]):
        out[np.isin(arr, codes)] = group
    return out


def features(df, cfg, clear, elev):
    data = {c: df[c].to_numpy(dtype=float) for c in FIELDS}
    angle = np.deg2rad(df["Wind Direction"].to_numpy())
    speed = df["Wind Speed"].to_numpy()
    data.update(wind_u=-speed * np.sin(angle), wind_v=-speed * np.cos(angle))
    data["dewpoint_depression"] = (df.Temperature - df["Dew Point"]).to_numpy()
    data["pressure_change_30m"] = df.Pressure.diff(3).to_numpy()
    data.update(GHI=df.GHI.to_numpy(), clear_reference=clear, elevation=elev)
    day = (elev > cfg["solar_elevation_min"]) & (clear > cfg["clearsky_ghi_min"])
    k = np.divide(
        df.GHI.to_numpy(), clear, out=np.full(len(df), np.nan), where=clear > 0
    )
    data.update(k_history=np.where(day, k, 0.0), daylight=day.astype(float))
    minutes = df.index.hour.to_numpy() * 60 + df.index.minute.to_numpy()
    for name, phase in [
        ("clock", minutes / 1440),
        ("year", (df.index.dayofyear.to_numpy() - 1) / 365.25),
    ]:
        data[name + "_sin"] = np.sin(2 * np.pi * phase)
        data[name + "_cos"] = np.cos(2 * np.pi * phase)
    for code in CLOUD_CODES:
        data[f"cloud_{code}"] = (df["Cloud Type"].to_numpy() == code).astype(float)
    x = np.column_stack(list(data.values())).astype(np.float32)
    invalid = (
        df[[*FIELDS, "GHI", "Wind Direction", "Cloud Type"]]
        .isna()
        .any(axis=1)
        .to_numpy(copy=True)
    )
    invalid |= (df.GHI.to_numpy() < 0) | ~np.isin(df["Cloud Type"], CLOUD_CODES)
    x[invalid] = np.nan
    return x, list(data), k.astype(np.float32), day


def assign_splits(
    index, history_steps, interval, horizons, windows, feature_lookback_minutes=30
):
    left = index - pd.Timedelta(
        minutes=(history_steps - 1) * interval + feature_lookback_minutes
    )
    right = index + pd.Timedelta(minutes=max(horizons))
    out = np.full(len(index), "excluded", dtype="U10")
    intervals = sorted((utc(w["start"]), utc(w["end"])) for w in windows)
    if any(a[1] > b[0] for a, b in zip(intervals, intervals[1:])):
        raise ValueError("Split windows overlap")
    for w in windows:
        if w["split"] not in {"train", "validation", "test"}:
            raise ValueError("Unknown split")
        out[(left >= utc(w["start"])) & (right < utc(w["end"]))] = w["split"]
    return out


def prepare(cfg):
    if cfg["interval_minutes"] != 10:
        raise ValueError("This prototype is locked to 10-minute input")
    report = audit(cfg)
    df, _ = read_nsrdb(resolve(cfg, cfg["csv"]))
    dt = cfg["interval_minutes"]
    full = pd.date_range(df.index[0], df.index[-1], freq=f"{dt}min")
    if not df.index.isin(full).all():
        raise ValueError("Off-grid timestamps")
    df = df.reindex(full)
    clear, elev = solar_reference(df.index, cfg)
    x, names, k, day = features(df, cfg, clear, elev)
    if any(h <= 0 or h % dt for h in cfg["horizons_minutes"]):
        raise ValueError("Horizons must be positive cadence multiples")
    steps = np.asarray(cfg["horizons_minutes"]) // dt
    length = cfg["history_steps"]
    anchors = np.arange(length - 1, len(x) - max(steps))
    hist = anchors[:, None] - np.arange(length - 1, -1, -1)[None, :]
    targets = anchors[:, None] + steps[None, :]
    splits = assign_splits(
        df.index[anchors], length, dt, cfg["horizons_minutes"], cfg["windows"]
    )
    keep = (
        np.isfinite(x[hist]).all(axis=(1, 2))
        & day[anchors]
        & day[targets].all(axis=1)
        & np.isfinite(k[targets]).all(axis=1)
        & (splits != "excluded")
    )
    anchors, hist, targets, splits = (
        anchors[keep],
        hist[keep],
        targets[keep],
        splits[keep],
    )
    if not (splits == "train").any():
        raise ValueError("No training samples")
    arrays = dict(
        X=x[hist],
        y=k[targets],
        k_now=k[anchors],
        ghi_now=df.GHI.to_numpy()[anchors],
        target_ghi=df.GHI.to_numpy()[targets],
        target_clear=clear[targets],
        time_ns=nanos(df.index[anchors]),
        history_ns=nanos(df.index)[hist],
        target_ns=nanos(df.index)[targets],
        split=splits,
        cloud_now=cloud_group(df["Cloud Type"].to_numpy()[anchors]),
        cloud_future=cloud_group(df["Cloud Type"].to_numpy()[targets]),
        feature_names=np.asarray(names),
        horizons=np.asarray(cfg["horizons_minutes"]),
        config_fingerprint=np.asarray(fingerprint(cfg)),
    )
    path = resolve(cfg, cfg["prepared"])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    report.update(
        sample_counts={
            s: int((splits == s).sum()) for s in ["train", "validation", "test"]
        },
        feature_names=names,
        config_fingerprint=fingerprint(cfg),
    )
    write_json(path.with_suffix(".json"), report)
    return report


def load_prepared(cfg):
    with np.load(resolve(cfg, cfg["prepared"]), allow_pickle=False) as f:
        data = {k: f[k] for k in f.files}
    if str(data["config_fingerprint"]) != fingerprint(cfg):
        raise ValueError("Config mismatch; rerun prepare")
    return data


def subset(data, keep):
    keys = {
        "X",
        "y",
        "k_now",
        "ghi_now",
        "target_ghi",
        "target_clear",
        "time_ns",
        "history_ns",
        "target_ns",
        "split",
        "cloud_now",
        "cloud_future",
    }
    return {k: v[keep] if k in keys else v for k, v in data.items()}
