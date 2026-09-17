from __future__ import annotations
from functools import lru_cache
import hashlib
from pathlib import Path
import random
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from .common import array_fingerprint, fingerprint, new_run, sha256, write_json
from .data import load_prepared, subset
from .evaluation import prediction_frame, ramp_thresholds, summarize
from .models import ForecastNet
from .satellite import align_patches

ABLATIONS = [
    "none",
    "current_only",
    "shuffle_order",
    "spatial_mean",
    "remove_ghi",
    "remove_cloud",
    "remove_moisture",
]


def fit_standardizer(values):
    """Training-only moments in float64; constant features must keep unit scale."""
    stable = np.asarray(values, dtype=np.float64)
    mean = stable.mean(axis=(0, 1))
    scale = stable.std(axis=(0, 1))
    scale = np.where(scale > 1e-6, scale, 1.0)
    return mean.astype(np.float32), scale.astype(np.float32)


def feature_keep(names, ablation):
    banned = set()
    if ablation == "remove_ghi":
        banned = {"GHI", "k_history"}
    elif ablation == "remove_cloud":
        banned = {n for n in names if n.startswith("cloud_")}
    elif ablation == "remove_moisture":
        banned = {
            "Relative Humidity",
            "Dew Point",
            "Precipitable Water",
            "dewpoint_depression",
        }
    return np.asarray([n not in banned for n in names])


class Windows(Dataset):
    def __init__(
        self, data, ids, mean, scale, keep, image_ids=None, paths=None, ablation="none"
    ):
        self.data, self.ids, self.mean, self.scale, self.keep = (
            data,
            ids,
            mean,
            scale,
            keep,
        )
        self.image_ids, self.paths, self.ablation = image_ids, paths, ablation

    def __len__(self):
        return len(self.ids)

    @lru_cache(maxsize=128)
    def image(self, key):
        with np.load(self.paths[key], allow_pickle=False) as f:
            values, mask = f["image"].astype(np.float32), f["valid"]
        return np.concatenate(
            [np.where(mask, (values - 250.0) / 50.0, 0.0), mask.astype(np.float32)]
        )

    def __getitem__(self, item):
        i = self.ids[item]
        tab = (self.data["X"][i] - self.mean) / self.scale
        tab[:, ~self.keep] = 0
        image = np.zeros((1, 1, 1, 1), dtype=np.float32)
        if self.image_ids is not None:
            image = np.stack([self.image(int(j)) for j in self.image_ids[i]])
            if self.ablation == "current_only":
                image = np.repeat(image[-1:], len(image), axis=0)
            elif self.ablation == "shuffle_order":
                rng = np.random.default_rng(int(self.data["time_ns"][i]) % (2**32))
                image[:-1] = image[rng.permutation(len(image) - 1)]
            elif self.ablation == "spatial_mean":
                image = np.broadcast_to(
                    image.mean(axis=(-2, -1), keepdims=True), image.shape
                ).copy()
        return (
            torch.from_numpy(tab.astype(np.float32)),
            torch.from_numpy(image),
            torch.from_numpy(self.data["y"][i]),
        )


def rng_state():
    return {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state):
    torch.set_rng_state(state["torch"].cpu())
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda"]])


def train(
    cfg,
    name,
    mode="fusion",
    ablation="none",
    matched=False,
    seed=None,
    device="auto",
    resume=None,
    evaluate_test=False,
):
    if ablation not in ABLATIONS or (mode != "fusion" and ablation != "none"):
        raise ValueError("Invalid ablation/model")
    settings = cfg["training"]
    seed = settings["seed"] if seed is None else seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(settings["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    chosen = (
        ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    )
    data = load_prepared(cfg)
    image_ids, paths = None, None
    if mode != "tabular" or matched:
        keep, image_ids, paths = align_patches(cfg, data)
        data = subset(data, keep)
        if mode == "tabular":
            image_ids = None
    ids = {
        s: np.flatnonzero(data["split"] == s) for s in ["train", "validation", "test"]
    }
    if any(not len(ids[s]) for s in ["train", "validation"]):
        raise ValueError("No matched training/validation windows")
    mean, scale = fit_standardizer(data["X"][ids["train"]])
    keep_features = feature_keep(data["feature_names"], ablation)
    thresholds = ramp_thresholds(data)
    datasets = {
        s: Windows(
            data, indices, mean, scale, keep_features, image_ids, paths, ablation
        )
        for s, indices in ids.items()
    }

    def loader(split, epoch=0):
        return DataLoader(
            datasets[split],
            batch_size=settings["batch_size"],
            shuffle=split == "train",
            num_workers=0,
            generator=torch.Generator().manual_seed(seed + epoch),
            pin_memory=chosen.startswith("cuda"),
        )

    model = ForecastNet(
        data["X"].shape[-1],
        len(cfg["satellite"]["bands"]),
        len(data["horizons"]),
        settings["hidden"],
        mode,
    ).to(chosen)
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
    )
    criterion = torch.nn.L1Loss()
    primary = list(data["horizons"]).index(cfg["primary_horizon_minutes"])
    digest = hashlib.sha256(data["time_ns"].tobytes())
    if image_ids is not None:
        for path in paths:
            digest.update(sha256(path).encode())
    contract = {
        "fingerprint": fingerprint(cfg),
        "mode": mode,
        "ablation": ablation,
        "seed": seed,
        "matched": matched or mode != "tabular",
        "cohort_sha256": digest.hexdigest(),
        "dataset_sha256": array_fingerprint(data),
    }
    start, best, stale, history = 0, float("inf"), 0, []
    if resume:
        checkpoint = Path(resume).resolve()
        dest = checkpoint.parent
        # Load only trusted project checkpoints: optimizer/RNG state uses pickle.
        state = torch.load(checkpoint, map_location=chosen, weights_only=False)
        if state["contract"] != contract:
            raise ValueError("Resume configuration/seed/data mismatch")
        if not np.array_equal(state["mean"], mean) or not np.array_equal(
            state["scale"], scale
        ):
            raise ValueError(
                "Resume normalization mismatch; do not mix preprocessing versions"
            )
        model.load_state_dict(state["model"])
        optim.load_state_dict(state["optimizer"])
        start, best, stale, history = (
            state["epoch"] + 1,
            state["best"],
            state["stale"],
            state["history"],
        )
        restore_rng(state["rng"])
    else:
        dest = new_run(cfg, name)
    started = time.monotonic()

    def predict(split):
        model.eval()
        result = []
        with torch.no_grad():
            for tab, image, target in loader(split):
                result.append(model(tab.to(chosen), image.to(chosen)).cpu().numpy())
        return np.maximum(np.concatenate(result), 0.0)

    for epoch in range(start, settings["epochs"]):
        if stale >= settings["patience"]:
            break
        model.train()
        total, count = 0.0, 0
        for tab, image, target in loader("train", epoch):
            optim.zero_grad(set_to_none=True)
            prediction = model(tab.to(chosen), image.to(chosen))
            loss = criterion(prediction, target.to(chosen))
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            total += loss.item() * len(tab)
            count += len(tab)
        val = predict("validation")
        mae = float(
            np.abs(val[:, primary] - data["y"][ids["validation"], primary]).mean()
        )
        improved = mae < best
        best, stale = (mae, 0) if improved else (best, stale + 1)
        history.append(
            {
                "epoch": epoch,
                "train_mae_all_horizons": total / count,
                "validation_mae_primary": mae,
            }
        )
        state = {
            "model": model.state_dict(),
            "optimizer": optim.state_dict(),
            "epoch": epoch,
            "best": best,
            "stale": stale,
            "history": history,
            "contract": contract,
            "mean": mean,
            "scale": scale,
            "feature_names": data["feature_names"],
            "feature_keep": keep_features,
            "thresholds": thresholds,
            "rng": rng_state(),
        }
        for filename in ["last.pt"] + (["best.pt"] if improved else []):
            temp = dest / (filename + ".tmp")
            torch.save(state, temp)
            temp.replace(dest / filename)
        write_json(dest / "history.json", history)
        print(
            f"epoch={epoch+1} train_MAE={total/count:.4f} validation_MAE_{cfg['primary_horizon_minutes']}m={mae:.4f}",
            flush=True,
        )
    best_state = torch.load(dest / "best.pt", map_location=chosen, weights_only=False)
    model.load_state_dict(best_state["model"])
    frames = []
    for split in ["validation"] + (["test"] if evaluate_test else []):
        if len(ids[split]):
            frames.append(
                prediction_frame(
                    data,
                    ids[split],
                    predict(split),
                    f"{mode}_{ablation}",
                    seed,
                    thresholds,
                )
            )
    frame = pd.concat(frames, ignore_index=True)
    frame.to_csv(dest / "predictions.csv", index=False)
    metrics = summarize(frame)
    write_json(dest / "metrics.json", metrics)
    info = {
        **contract,
        "device": chosen,
        "parameters": sum(p.numel() for p in model.parameters()),
        "samples": {k: len(v) for k, v in ids.items()},
        "epochs_completed": len(history),
        "session_wall_seconds": time.monotonic() - started,
        "test_evaluated": evaluate_test,
        "checkpoint_selection": f"validation {cfg['primary_horizon_minutes']}-minute MAE",
        "engineering_smoke": cfg["name"] == "engineering_smoke",
    }
    write_json(dest / "run.json", info)
    return {"output": str(dest), "run": info, "metrics": metrics}
