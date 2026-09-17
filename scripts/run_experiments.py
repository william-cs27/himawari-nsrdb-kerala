"""Print or execute a fixed experiment matrix; no downloads or test evaluation."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def commands(config, stage, prefix, device):
    common = [sys.executable, "-m", "weather_fusion", "--config", str(config)]
    seeds = [13] if stage == "smoke" else [13, 29, 47]
    runs = []
    if stage in {"smoke", "core"}:
        runs.append(
            common
            + ["baselines", "--run-name", prefix + "_baselines", "--matched-images"]
        )
        for seed in seeds:
            for mode in ["tabular", "image", "fusion"]:
                runs.append(
                    common
                    + [
                        "train",
                        "--mode",
                        mode,
                        "--matched-images",
                        "--seed",
                        str(seed),
                        "--device",
                        device,
                        "--run-name",
                        f"{prefix}_{mode}_s{seed}",
                    ]
                )
    else:
        for seed in seeds:
            for ablation in [
                "current_only",
                "shuffle_order",
                "spatial_mean",
                "remove_ghi",
                "remove_cloud",
                "remove_moisture",
            ]:
                runs.append(
                    common
                    + [
                        "train",
                        "--mode",
                        "fusion",
                        "--matched-images",
                        "--ablation",
                        ablation,
                        "--seed",
                        str(seed),
                        "--device",
                        device,
                        "--run-name",
                        f"{prefix}_{ablation}_s{seed}",
                    ]
                )
    return runs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/smoke.json")
    p.add_argument("--stage", choices=["smoke", "core", "ablations"], default="smoke")
    p.add_argument(
        "--prefix", default=datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    )
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    if (
        not args.prefix
        or Path(args.prefix).name != args.prefix
        or args.prefix in {".", ".."}
    ):
        p.error("Simple prefix required")
    matrix = commands(
        (ROOT / args.config).resolve(), args.stage, args.prefix, args.device
    )
    print(f"{len(matrix)} runs; validation only; no downloads", flush=True)
    for command in matrix:
        print(shlex.join(command), flush=True)
        if args.execute:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
