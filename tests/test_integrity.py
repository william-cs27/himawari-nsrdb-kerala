import csv
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from weather_fusion.common import (
    load_config,
    nanos,
    sha256,
    write_json,
    fingerprint,
    array_fingerprint,
)
from weather_fusion.data import FIELDS, assign_splits, features, read_nsrdb
from weather_fusion.evaluation import compare
from weather_fusion.satellite import (
    acquire,
    align_patches,
    patch_signature,
    select_objects,
)


def test_utc_uses_time_zone_not_local_metadata(tmp_path):
    path = tmp_path / "source.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Time Zone", "Local Time Zone"])
        w.writerow([0, 5])
        w.writerow(
            [
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
        )
        w.writerow([2018, 1, 1, 6, 0, 300, 90, 0, *([1] * len(FIELDS))])
    data, _ = read_nsrdb(path)
    assert data.index[0] == pd.Timestamp("2018-01-01T06:00Z")


def test_split_purges_complete_raw_feature_history_and_targets():
    times = pd.date_range("2018-01-01", periods=432, freq="10min", tz="UTC")
    windows = [
        dict(split="train", start="2018-01-01T00:00Z", end="2018-01-02T00:00Z"),
        dict(split="validation", start="2018-01-02T00:00Z", end="2018-01-04T00:00Z"),
    ]
    labels = assign_splits(times, 6, 10, [10, 30, 60, 120], windows)
    assert (times[labels == "train"] + pd.Timedelta(hours=2)).max() < (
        times[labels == "validation"] - pd.Timedelta(minutes=80)
    ).min()
    assert labels[times.get_loc(pd.Timestamp("2018-01-01T23:00Z"))] == "excluded"
    assert labels[times.get_loc(pd.Timestamp("2018-01-02T01:00Z"))] == "excluded"
    assert labels[times.get_loc(pd.Timestamp("2018-01-02T01:20Z"))] == "validation"


def test_overlapping_splits_fail():
    windows = [
        dict(split="train", start="2018-01-01T00:00Z", end="2018-03-01T00:00Z"),
        dict(split="test", start="2018-02-01T00:00Z", end="2018-04-01T00:00Z"),
    ]
    with pytest.raises(ValueError, match="overlap"):
        assign_splits(
            pd.date_range("2018-01-01", periods=2, tz="UTC"), 6, 10, [60], windows
        )


def test_future_measurements_do_not_change_past_features():
    index = pd.date_range("2018-01-01T04:00Z", periods=20, freq="10min")
    df = pd.DataFrame(
        {n: np.ones(20) for n in [*FIELDS, "Wind Direction", "GHI", "Cloud Type"]},
        index=index,
    )
    cfg = {"solar_elevation_min": 5, "clearsky_ghi_min": 50}
    before = features(df, cfg, np.full(20, 500), np.full(20, 40))[0]
    df.iloc[12:, df.columns.get_loc("GHI")] = 1000
    df.iloc[12:, df.columns.get_loc("Pressure")] = 2000
    after = features(df, cfg, np.full(20, 500), np.full(20, 40))[0]
    np.testing.assert_allclose(before[:12], after[:12], equal_nan=True)
    df.iloc[7, df.columns.get_loc("GHI")] = np.nan
    assert np.isnan(features(df, cfg, np.full(20, 500), np.full(20, 40))[0][7]).all()


def test_ahi_selection_does_not_mix_resolutions():
    rows = [
        {"key": "HS_H08_20180101_0000_B13_FLDK_R20_S0101.DAT.bz2", "bytes": 1},
        {"key": "HS_H08_20180101_0000_B13_FLDK_R10_S0101.DAT.bz2", "bytes": 2},
    ]
    assert select_objects(rows, ["B13"]) == [rows[0]]
    with pytest.raises(ValueError, match="Missing band"):
        select_objects(rows, ["B08"])


def test_incomplete_segments_fail():
    with pytest.raises(ValueError, match="Incomplete"):
        select_objects(
            [{"key": "HS_H08_20180101_0000_B13_FLDK_R20_S0110.DAT.bz2"}], ["B13"]
        )


def config_at(tmp_path):
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs/smoke.json")
    cfg["_root"] = str(tmp_path)
    return cfg


def test_unfinished_scan_not_available(tmp_path):
    cfg = config_at(tmp_path)
    folder = tmp_path / cfg["satellite"]["patch_dir"]
    folder.mkdir(parents=True)
    for stamp in pd.date_range("2018-01-01T04:00Z", periods=8, freq="10min"):
        path = folder / (stamp.strftime("%Y%m%dT%H%MZ") + ".npz")
        np.savez(path, image=np.ones((1, 2, 2)), valid=np.ones((1, 2, 2), dtype=bool))
        available = stamp + pd.Timedelta(minutes=10)
        if stamp == pd.Timestamp("2018-01-01T05:00Z"):
            available += pd.Timedelta(minutes=10)
        write_json(
            path.with_suffix(".json"),
            {
                "nominal_time": stamp.isoformat(),
                "available_time": available.isoformat(),
                "patch_signature": patch_signature(cfg),
                "patch_sha256": sha256(path),
            },
        )
    times = pd.date_range("2018-01-01T05:00Z", periods=3, freq="10min")
    data = {
        "time_ns": nanos(times),
        "target_ns": nanos(times + pd.Timedelta(hours=2))[:, None],
        "split": np.array(["train"] * 3),
    }
    keep, indices, _ = align_patches(cfg, data)
    assert keep.tolist() == [True, False, True]
    assert indices.shape == (2, 6)


def test_budget_stops_before_network(tmp_path, monkeypatch):
    cfg = config_at(tmp_path)
    cfg["satellite"]["byte_budget_gib"] = 0.00001
    plan = {
        "config_fingerprint": fingerprint(cfg),
        "incomplete_frames": 0,
        "frames": [
            {
                "nominal_time": "2018-01-01T04:00Z",
                "objects": [{"key": "test", "bytes": 10000000}],
            }
        ],
    }
    monkeypatch.setattr(
        "weather_fusion.satellite.download_object",
        lambda *a: pytest.fail("Network before budget check"),
    )
    with pytest.raises(ValueError, match="exceeds budget"):
        acquire(cfg, plan)


def test_mismatched_prediction_cohort_rejected():
    frame = pd.DataFrame(
        {
            "issue_time": ["2018-01-01T04:00Z", "2018-01-02T04:00Z"],
            "target_time": ["2018-01-01T05:00Z", "2018-01-02T05:00Z"],
            "horizon_minutes": 60,
            "split": "validation",
            "model": "baseline",
            "seed": 13,
            "observed_k": [0.5, 0.7],
            "predicted_k": [0.6, 0.8],
        }
    )
    with pytest.raises(ValueError, match="Cohort mismatch"):
        compare(frame, frame.iloc[:1])
    assert compare(frame, frame)["status"] == "descriptive_only_too_few_blocks"


def test_ghi_ablation_also_removes_derived_index():
    pytest.importorskip("torch")
    from weather_fusion.training import feature_keep

    assert feature_keep(["GHI", "k_history", "Temperature"], "remove_ghi").tolist() == [
        False,
        False,
        True,
    ]


def test_temporal_block_is_causal():
    torch = pytest.importorskip("torch")
    from weather_fusion.models import CausalBlock

    torch.manual_seed(0)
    block = CausalBlock(3, 2)
    x = torch.randn(2, 3, 8)
    y = x.clone()
    y[..., 5:] += 100
    torch.testing.assert_close(block(x)[..., :5], block(y)[..., :5])


def test_constant_float32_feature_has_unit_scale():
    pytest.importorskip("torch")
    from weather_fusion.training import fit_standardizer

    train = np.full((10, 6, 1), 4.1, dtype=np.float32)
    mean, scale = fit_standardizer(train)
    np.testing.assert_array_equal(scale, [1.0])
    np.testing.assert_array_equal((train - mean) / scale, np.zeros_like(train))
    assert abs(float((np.float32(4.3) - mean[0]) / scale[0])) < 0.21


def test_resume_fingerprint_tracks_values_not_dict_order():
    x = {"a": np.array([1, 2]), "b": np.array(["training"])}
    y = {"b": x["b"], "a": x["a"].copy()}
    assert array_fingerprint(x) == array_fingerprint(y)
    y["a"][0] = 3
    assert array_fingerprint(x) != array_fingerprint(y)
