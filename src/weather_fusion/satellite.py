from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from .common import fingerprint, nanos, resolve, sha256, utc, write_json

NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
PATTERN = re.compile(r"_B(\d{2})_FLDK_R20_S(\d{2})(\d{2})\.DAT\.bz2$")


def get(url):
    error = None
    for attempt in range(3):
        try:
            return urllib.request.urlopen(
                urllib.request.Request(
                    url, headers={"User-Agent": "himawari-nsrdb-research/0.1"}
                ),
                timeout=60,
            )
        except (OSError, TimeoutError) as exc:
            error = exc
            time.sleep(attempt + 1)
    raise OSError(f"Request failed: {url}: {error}")


def list_prefix(bucket, prefix):
    rows = []
    token = None
    while True:
        params = {"list-type": "2", "prefix": prefix, "max-keys": 1000}
        if token:
            params["continuation-token"] = token
        with get(
            f"https://{bucket}.s3.amazonaws.com/?" + urllib.parse.urlencode(params)
        ) as response:
            tree = ET.fromstring(response.read())
        for item in tree.findall("s:Contents", NS):
            rows.append(
                {
                    "key": item.find("s:Key", NS).text,
                    "bytes": int(item.find("s:Size", NS).text),
                    "etag": item.find("s:ETag", NS).text.strip('"'),
                }
            )
        more = tree.find("s:IsTruncated", NS)
        if more is None or more.text != "true":
            break
        token = tree.find("s:NextContinuationToken", NS).text
    return rows


def select_objects(rows, bands):
    result = []
    for band in bands:
        matches = [
            (r, PATTERN.search(r["key"])) for r in rows if f"_{band}_" in r["key"]
        ]
        matches = [(r, m) for r, m in matches if m]
        whole = [(r, m) for r, m in matches if m.group(3) == "01"]
        selected = whole or matches
        if not selected:
            raise ValueError(f"Missing band {band}")
        total = int(selected[0][1].group(3))
        if sorted(int(m.group(2)) for _, m in selected) != list(range(1, total + 1)):
            raise ValueError(f"Incomplete or duplicate segments for {band}")
        result.extend(r for r, _ in sorted(selected, key=lambda z: z[0]["key"]))
    return result


def frame_times(cfg):
    times = []
    for w in cfg["satellite"]["windows"]:
        times.extend(
            pd.date_range(
                utc(w["start"]), utc(w["end"]), freq="10min", inclusive="left"
            )
        )
    if not times:
        raise ValueError(
            "Set satellite.windows explicitly. No automatic full-year download."
        )
    result = pd.DatetimeIndex(sorted(set(times)))
    if any(t.minute % 10 or t.second or t.microsecond for t in result):
        raise ValueError("Images must be on the 10-minute grid")
    return result


def inventory(cfg):
    sat = cfg["satellite"]

    def inspect(stamp):
        rows = list_prefix(sat["bucket"], stamp.strftime("AHI-L1b-FLDK/%Y/%m/%d/%H%M/"))
        try:
            return {
                "nominal_time": stamp.isoformat(),
                "objects": select_objects(rows, sat["bands"]),
                "status": "complete",
            }
        except ValueError as exc:
            return {
                "nominal_time": stamp.isoformat(),
                "objects": [],
                "status": "missing",
                "reason": str(exc),
            }

    with ThreadPoolExecutor(max_workers=4) as pool:
        frames = list(pool.map(inspect, frame_times(cfg)))
    total = sum(o["bytes"] for f in frames for o in f["objects"])
    result = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "bucket": sat["bucket"],
        "config_fingerprint": fingerprint(cfg),
        "frames": frames,
        "total_bytes": total,
        "total_gib": total / 2**30,
        "incomplete_frames": sum(f["status"] != "complete" for f in frames),
    }
    write_json(resolve(cfg, f"outputs/{cfg['name']}_inventory.json"), result)
    return result


def download_object(bucket, item, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    check = destination.with_suffix(destination.suffix + ".sha256")
    if (
        destination.exists()
        and destination.stat().st_size == item["bytes"]
        and check.exists()
        and sha256(destination) == check.read_text().strip()
    ):
        return {**item, "sha256": check.read_text().strip(), "transferred_bytes": 0}
    part = destination.with_suffix(destination.suffix + ".part")
    sha = hashlib.sha256()
    md5 = hashlib.md5()
    size = 0
    url = f"https://{bucket}.s3.amazonaws.com/" + urllib.parse.quote(
        item["key"], safe="/"
    )
    with get(url) as source, open(part, "wb") as target:
        while block := source.read(2**20):
            size += len(block)
            if size > item["bytes"]:
                raise ValueError("Transfer exceeded listed size")
            target.write(block)
            sha.update(block)
            md5.update(block)
    if size != item["bytes"]:
        raise ValueError("Incomplete S3 object")
    if (
        re.fullmatch(r"[0-9a-fA-F]{32}", item["etag"])
        and md5.hexdigest() != item["etag"].lower()
    ):
        raise ValueError("ETag checksum mismatch")
    part.replace(destination)
    check.write_text(sha.hexdigest() + "\n")
    return {**item, "sha256": sha.hexdigest(), "transferred_bytes": size}


def view_zenith(latitude, longitude):
    cospsi = np.cos(np.deg2rad(latitude)) * np.cos(np.deg2rad(longitude - 140.7))
    r, orbit = 6371.0, 42164.0
    distance = np.sqrt(orbit**2 + r**2 - 2 * orbit * r * cospsi)
    return float(np.rad2deg(np.arccos((orbit * cospsi - r) / distance)))


def patch_signature(cfg):
    sat = cfg["satellite"]
    settings = {
        k: sat[k]
        for k in [
            "bands",
            "patch_size",
            "extent_km",
            "availability_lag_minutes",
            "max_missing_fraction",
        ]
    }
    settings.update(latitude=cfg["latitude"], longitude=cfg["longitude"])
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()


def crop_files(cfg, paths, nominal):
    from satpy import Scene
    from pyresample.geometry import AreaDefinition
    import dask

    sat = cfg["satellite"]
    n = sat["patch_size"]
    radius = sat["extent_km"] * 500
    area = AreaDefinition(
        "kerala",
        "Kerala crop",
        "local_aeqd",
        {
            "proj": "aeqd",
            "lat_0": cfg["latitude"],
            "lon_0": cfg["longitude"],
            "datum": "WGS84",
            "units": "m",
        },
        n,
        n,
        (-radius, -radius, radius, radius),
    )
    scene = Scene(filenames=[str(p) for p in paths], reader="ahi_hsd")
    scene.load(sat["bands"], calibration="brightness_temperature")
    ends = []
    for band in sat["bands"]:
        attrs = scene[band].attrs
        end = attrs.get("time_parameters", {}).get(
            "observation_end_time", attrs.get("end_time")
        )
        if end is None:
            raise ValueError("Cannot establish scan completion time")
        end = pd.Timestamp(end)
        ends.append(
            end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
        )
    with dask.config.set(scheduler="single-threaded"):
        local = scene.resample(area, resampler="nearest", radius_of_influence=18000)
        image = np.stack([local[b].values for b in sat["bands"]]).astype(np.float32)
    valid = np.isfinite(image) & (image >= 150) & (image <= 350)
    if (~valid).mean() > sat["max_missing_fraction"]:
        raise ValueError("Insufficient valid crop pixels")
    image[~valid] = np.nan
    nominal = utc(nominal)
    available = max(
        max(ends), nominal + pd.Timedelta(minutes=sat["availability_lag_minutes"])
    )
    return (
        image,
        valid,
        {
            "nominal_time": nominal.isoformat(),
            "observation_end": max(ends).isoformat(),
            "available_time": available.isoformat(),
            "availability_kind": "assumed latency; not measured delivery time",
            "valid_fraction": float(valid.mean()),
            "bands": sat["bands"],
            "units": "K",
            "shape": list(image.shape),
            "extent_km": sat["extent_km"],
            "grid_spacing_km": sat["extent_km"] / n,
            "approximate_view_zenith_degrees": view_zenith(
                cfg["latitude"], cfg["longitude"]
            ),
            "parallax_corrected": False,
            "patch_signature": patch_signature(cfg),
        },
    )


def acquire(cfg, plan):
    if plan["config_fingerprint"] != fingerprint(cfg):
        raise ValueError("Inventory/config mismatch")
    sat = cfg["satellite"]
    folder = resolve(cfg, sat["patch_dir"])
    folder.mkdir(parents=True, exist_ok=True)
    pending = []
    for frame in plan["frames"]:
        if frame.get("status", "complete") != "complete":
            continue
        tag = utc(frame["nominal_time"]).strftime("%Y%m%dT%H%MZ")
        path = folder / (tag + ".npz")
        meta_path = folder / (tag + ".json")
        if path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if (
                meta["patch_signature"] != patch_signature(cfg)
                or sha256(path) != meta["patch_sha256"]
            ):
                raise ValueError("Cached patch mismatch")
        else:
            pending.append((frame, tag))
    requested = sum(o["bytes"] for f, _ in pending for o in f["objects"])
    if requested > sat["byte_budget_gib"] * 2**30:
        raise ValueError(f"Download plan exceeds budget: {requested/2**30:.3f} GiB")
    transferred = 0
    started = time.monotonic()
    raw = resolve(cfg, "data/raw_himawari")

    def fetch(frame):
        paths = [raw / Path(o["key"]).name for o in frame["objects"]]
        with ThreadPoolExecutor(max_workers=3) as pool:
            objects = list(
                pool.map(
                    lambda pair: download_object(sat["bucket"], *pair),
                    zip(frame["objects"], paths),
                )
            )
        return paths, objects

    with ThreadPoolExecutor(max_workers=2) as prefetch:
        queue = {
            i: prefetch.submit(fetch, pending[i][0])
            for i in range(min(2, len(pending)))
        }
        for number, (frame, tag) in enumerate(pending):
            paths, objects = queue.pop(number).result()
            if number + 2 < len(pending):
                queue[number + 2] = prefetch.submit(fetch, pending[number + 2][0])
            image, valid, meta = crop_files(cfg, paths, frame["nominal_time"])
            path = folder / (tag + ".npz")
            temporary = folder / (tag + ".tmp.npz")
            np.savez_compressed(temporary, image=image, valid=valid)
            temporary.replace(path)
            meta.update(
                objects=objects, bucket=sat["bucket"], patch_sha256=sha256(path)
            )
            write_json(folder / (tag + ".json"), meta)
            transferred += sum(o["transferred_bytes"] for o in objects)
            print(
                f"Cropped {tag}: {meta['valid_fraction']:.1%} valid; {transferred/2**20:.1f} MiB",
                flush=True,
            )
            if not sat["keep_raw"]:
                for path in paths:
                    path.unlink()
                    path.with_suffix(path.suffix + ".sha256").unlink(missing_ok=True)
    result = {
        "frames_new": len(pending),
        "frames_cached": len(plan["frames"]) - plan["incomplete_frames"] - len(pending),
        "missing_frames": plan["incomplete_frames"],
        "transferred_bytes": transferred,
        "wall_seconds": time.monotonic() - started,
        "raw_cache_retained": sat["keep_raw"],
    }
    write_json(folder / "acquisition.json", result)
    return result


def align_patches(cfg, data):
    rows = []
    for path in sorted(resolve(cfg, cfg["satellite"]["patch_dir"]).glob("*.json")):
        if path.name == "acquisition.json":
            continue
        meta = json.loads(path.read_text())
        if "nominal_time" not in meta:
            continue
        patch = path.with_suffix(".npz")
        if (
            meta["patch_signature"] != patch_signature(cfg)
            or sha256(patch) != meta["patch_sha256"]
        ):
            raise ValueError("Patch settings/checksum mismatch")
        rows.append((utc(meta["nominal_time"]), utc(meta["available_time"]), patch))
    if not rows:
        raise ValueError("No crops; run satellite-plan and satellite-fetch")
    rows.sort()
    stamps = nanos([r[0] for r in rows])
    available = nanos([r[1] for r in rows])
    if len(np.unique(stamps)) != len(stamps):
        raise ValueError("Duplicate image time")
    step = cfg["interval_minutes"] * 60 * 10**9
    length = cfg["history_steps"]
    lag = cfg["satellite"]["availability_lag_minutes"] * 60 * 10**9
    indices = np.full((len(data["time_ns"]), length), -1, dtype=np.int64)
    for i, issue in enumerate(data["time_ns"]):
        last = np.searchsorted(stamps, issue - lag, side="right") - 1
        if last < length - 1:
            continue
        ids = np.arange(last - length + 1, last + 1)
        if (
            (np.diff(stamps[ids]) == step).all()
            and (available[ids] <= issue).all()
            and issue - stamps[last] == lag
        ):
            indices[i] = ids
    keep = (indices >= 0).all(axis=1)
    for i in np.flatnonzero(keep):
        first = stamps[indices[i, 0]]
        if not any(
            w["split"] == data["split"][i]
            and utc(w["start"]).value <= first
            and data["target_ns"][i].max() < utc(w["end"]).value
            for w in cfg["windows"]
        ):
            keep[i] = False
    return keep, indices[keep], [str(r[2]) for r in rows]
