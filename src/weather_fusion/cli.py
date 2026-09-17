from __future__ import annotations
import argparse
import json
import os
from .common import load_config, resolve, write_json


def main():
    parser = argparse.ArgumentParser(
        description="Himawari–NSRDB retrospective research"
    )
    parser.add_argument("--config", default="configs/study.json")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["audit", "prepare", "satellite-plan", "satellite-fetch", "alignment"]:
        sub.add_parser(name)
    base = sub.add_parser("baselines")
    base.add_argument("--run-name", required=True)
    base.add_argument("--matched-images", action="store_true")
    base.add_argument("--evaluate-test", action="store_true")
    neural = sub.add_parser("train")
    neural.add_argument("--run-name", required=True)
    neural.add_argument(
        "--mode", choices=["tabular", "image", "fusion"], default="fusion"
    )
    neural.add_argument("--ablation", default="none")
    neural.add_argument("--matched-images", action="store_true")
    neural.add_argument("--evaluate-test", action="store_true")
    neural.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    neural.add_argument("--seed", type=int)
    neural.add_argument("--resume")
    comparison = sub.add_parser("compare")
    comparison.add_argument("baseline_csv")
    comparison.add_argument("candidate_csv")
    comparison.add_argument("--baseline-model", required=True)
    comparison.add_argument("--candidate-model", required=True)
    comparison.add_argument("--seed", type=int, default=13)
    comparison.add_argument(
        "--split", choices=["validation", "test"], default="validation"
    )
    comparison.add_argument("--block-days", type=int, default=7)
    comparison.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.command in {"audit", "prepare"}:
        from .data import audit, prepare

        result = prepare(cfg) if args.command == "prepare" else audit(cfg)
        write_json(resolve(cfg, f"outputs/{cfg['name']}_audit.json"), result)
    elif args.command == "satellite-plan":
        from .satellite import inventory

        result = {k: v for k, v in inventory(cfg).items() if k != "frames"}
    elif args.command == "satellite-fetch":
        from .satellite import acquire

        result = acquire(
            cfg,
            json.loads(
                resolve(cfg, f"outputs/{cfg['name']}_inventory.json").read_text()
            ),
        )
    elif args.command == "alignment":
        from .data import load_prepared
        from .satellite import align_patches

        data = load_prepared(cfg)
        keep, _, paths = align_patches(cfg, data)
        result = {
            "available_patches": len(paths),
            "matched_samples": int(keep.sum()),
            "splits": {
                s: int(((data["split"] == s) & keep).sum())
                for s in ["train", "validation", "test"]
            },
            "image_latency_minutes": cfg["satellite"]["availability_lag_minutes"],
        }
        write_json(resolve(cfg, f"outputs/{cfg['name']}_alignment.json"), result)
    elif args.command == "baselines":
        from .baselines import run

        result = run(cfg, args.run_name, args.matched_images, args.evaluate_test)
    elif args.command == "train":
        from .training import train

        result = train(
            cfg,
            args.run_name,
            args.mode,
            args.ablation,
            args.matched_images,
            args.seed,
            args.device,
            args.resume,
            args.evaluate_test,
        )
    else:
        import pandas as pd
        from .evaluation import compare

        a, b = pd.read_csv(args.baseline_csv), pd.read_csv(args.candidate_csv)
        a = a[
            (a.model == args.baseline_model)
            & (a.seed == args.seed)
            & (a.split == args.split)
        ]
        b = b[
            (b.model == args.candidate_model)
            & (b.seed == args.seed)
            & (b.split == args.split)
        ]
        result = compare(
            a,
            b,
            cfg["primary_horizon_minutes"],
            args.block_days,
            cfg["analysis"]["bootstrap_resamples"],
            args.seed,
            cfg["analysis"]["practical_skill"],
        )
        write_json(args.output, result)
    print(json.dumps(result, indent=2, default=str, allow_nan=False))


if __name__ == "__main__":
    main()
