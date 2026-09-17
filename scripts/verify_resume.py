"""Interrupt after epoch 1, resume, and compare with an uninterrupted smoke run."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import pandas as pd
from weather_fusion.common import load_config, resolve, write_json
from weather_fusion import training


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/smoke.json")
    p.add_argument("--reference", default="outputs/smoke_fusion_s13")
    p.add_argument("--run-name", default="smoke_resume_check")
    args = p.parse_args()
    cfg = load_config(args.config)
    if cfg["name"] != "engineering_smoke":
        raise ValueError("Smoke configuration only")
    original = training.write_json

    def interrupt(path, data):
        original(path, data)
        if Path(path).name == "history.json":
            raise InterruptedError("Deliberate interruption")

    training.write_json = interrupt
    try:
        training.train(
            cfg, args.run_name, mode="fusion", matched=True, seed=13, device="cpu"
        )
        raise AssertionError("Interruption did not occur")
    except InterruptedError:
        pass
    finally:
        training.write_json = original
    dest = resolve(cfg, "outputs") / args.run_name
    training.train(
        cfg,
        args.run_name,
        mode="fusion",
        matched=True,
        seed=13,
        device="cpu",
        resume=dest / "last.pt",
    )
    before = pd.read_csv(resolve(cfg, args.reference) / "predictions.csv")
    after = pd.read_csv(dest / "predictions.csv")
    assert before[["issue_time", "target_time", "horizon_minutes"]].equals(
        after[["issue_time", "target_time", "horizon_minutes"]]
    )
    np.testing.assert_array_equal(before.predicted_k, after.predicted_k)
    import torch

    reference_state = torch.load(
        resolve(cfg, args.reference) / "last.pt", map_location="cpu", weights_only=False
    )
    resumed_state = torch.load(dest / "last.pt", map_location="cpu", weights_only=False)
    for key in reference_state["model"]:
        torch.testing.assert_close(
            reference_state["model"][key], resumed_state["model"][key], rtol=0, atol=0
        )
    result = {
        "status": "passed",
        "check": "interruption after epoch 1; resumed final weights and predictions exactly match uninterrupted CPU run",
        "prediction_rows": len(after),
        "test_evaluated": False,
    }
    write_json(dest / "resume_verification.json", result)
    print(result)


if __name__ == "__main__":
    main()
