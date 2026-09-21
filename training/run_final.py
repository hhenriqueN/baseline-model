#!/usr/bin/env python3
"""Final training (spec section 18): after the CV-frozen architecture/hyperparameters,
train each network once on ALL valid baseline_core demonstrations (the
final_all_valid scaler's normalized dataset, which includes the fixed
diagnostic holdout session -- never used for offline model selection, only
here). Epoch count = median best-epoch observed during cross-validation
(provisional; final closed-loop rollouts are the real evaluation, spec
section 24 -- out of scope here).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import paths
from training.dataset import load_split, NETWORK_SPECS
from training.models import LateralMLP, LongitudinalMLP, ARCHITECTURE_CONFIG
from training.train_core import train_model_no_val, TRAINING_CONFIG

ROOT = paths.ROOT
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
MODEL_CLASSES = {"lateral": LateralMLP, "longitudinal": LongitudinalMLP}


def median_best_epoch(network: str) -> int:
    df = pd.read_csv(REPORTS_DIR / "training" / network / "cv_fold_results.csv")
    return int(round(float(np.median(df["best_epoch"])))) + 1  # +1: best_epoch is 0-indexed


def main():
    final_csv = paths.SCALERS_FINAL_DIR / "baseline_core_normalized_all_valid.csv"
    scaler_path = paths.SCALERS_FINAL_DIR / "scaler_final_all_valid.json"

    for network in ("lateral", "longitudinal"):
        n_epochs = median_best_epoch(network)
        tr = load_split(final_csv, network)
        model = MODEL_CLASSES[network]()
        result = train_model_no_val(model, tr.X, tr.y, n_epochs=n_epochs, cfg=TRAINING_CONFIG)

        out_dir = MODELS_DIR / network / "final"
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(result.best_state_dict, out_dir / "checkpoint.pt")
        with open(out_dir / "config.json", "w") as f:
            json.dump({
                "network": network,
                "architecture": ARCHITECTURE_CONFIG[network],
                "feature_order": NETWORK_SPECS[network]["feature_order"],
                "label_order": NETWORK_SPECS[network]["label_order"],
                "training_config": TRAINING_CONFIG,
                "n_epochs_trained": n_epochs,
                "n_epochs_source": "median CV best_epoch (0-indexed) + 1, over 13 LOSO folds",
                "dataset_version": "baseline_core (final_all_valid, includes fixed diagnostic holdout session)",
                "scaler_path": str(scaler_path.relative_to(ROOT)),
                "train_csv": str(final_csv.relative_to(ROOT)),
                "n_train_rows": len(tr.X),
                "note": "No held-out validation for this final fit -- CV (run_cv.py) already selected "
                        "architecture/hyperparameters/epoch count; official evaluation is closed-loop "
                        "FlightGear rollouts (future work), not this training run.",
            }, f, indent=2)
        with open(out_dir / "history.json", "w") as f:
            json.dump(result.history, f, indent=2)
        print(f"[final] {network}: trained {n_epochs} epochs on {len(tr.X)} rows -> {out_dir}")


if __name__ == "__main__":
    main()
