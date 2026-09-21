#!/usr/bin/env python3
"""Cross-validation training for both baseline networks over all LOSO folds
(spec sections 17-19). For each fold and each network:
  1. load the fold's already-normalized train/val CSVs (data/splits/normalized_folds),
  2. train with early stopping on validation MSE,
  3. save the best checkpoint + config + history,
  4. evaluate the best checkpoint on the full validation episode(s),
  5. save predictions with episode/timestamp metadata and metrics (global,
     per-output, per episode, per scenario, per phase).
Aggregates fold metrics (mean/std) into reports/training/<network>/cv_summary.csv.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import paths
from pipeline.features import LATERAL_LABEL_ORDER, LONGITUDINAL_LABEL_ORDER
from training.dataset import load_split, NETWORK_SPECS
from training.models import LateralMLP, LongitudinalMLP, ARCHITECTURE_CONFIG
from training.train_core import train_model, TRAINING_CONFIG
from training.metrics import evaluate_predictions, evaluate_by_group

ROOT = paths.ROOT
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

SAT_BOUNDS = {
    "lateral": [(-1.0, 1.0), (-1.0, 1.0)],
    "longitudinal": [(-1.0, 1.0), (0.0, 1.0)],
}
MODEL_CLASSES = {"lateral": LateralMLP, "longitudinal": LongitudinalMLP}
LABEL_ORDERS = {"lateral": LATERAL_LABEL_ORDER, "longitudinal": LONGITUDINAL_LABEL_ORDER}


def fold_ids():
    files = sorted((paths.SPLITS_DIR / "normalized_folds").glob("fold_*_train.csv"))
    return sorted({int(f.stem.split("_")[1]) for f in files})


def run_one_fold_network(fold_id: int, network: str):
    train_csv = paths.SPLITS_DIR / "normalized_folds" / f"fold_{fold_id:02d}_train.csv"
    val_csv = paths.SPLITS_DIR / "normalized_folds" / f"fold_{fold_id:02d}_val.csv"
    scaler_path = next(paths.SCALERS_FOLDS_DIR.glob(f"fold_{fold_id:02d}_*.json"))

    tr = load_split(train_csv, network)
    va = load_split(val_csv, network)

    model = MODEL_CLASSES[network]()
    result = train_model(model, tr.X, tr.y, va.X, va.y, TRAINING_CONFIG)

    model.load_state_dict(result.best_state_dict)
    model.eval()
    with torch.no_grad():
        y_pred = model(va.X).numpy()
    y_true = va.y.numpy()

    label_order = LABEL_ORDERS[network]
    sat_bounds = SAT_BOUNDS[network]
    global_metrics = evaluate_predictions(va.meta, y_true, y_pred, label_order, sat_bounds)
    per_episode = evaluate_by_group(va.meta, y_true, y_pred, label_order, sat_bounds, "episode_id")
    per_phase = evaluate_by_group(va.meta, y_true, y_pred, label_order, sat_bounds, "phase")
    per_scenario = evaluate_by_group(va.meta, y_true, y_pred, label_order, sat_bounds, "scenario_id")

    out_dir = MODELS_DIR / network / "folds" / f"fold_{fold_id:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(result.best_state_dict, out_dir / "checkpoint.pt")
    with open(out_dir / "config.json", "w") as f:
        json.dump({
            "network": network, "fold_id": fold_id,
            "architecture": ARCHITECTURE_CONFIG[network],
            "feature_order": NETWORK_SPECS[network]["feature_order"],
            "label_order": label_order,
            "sat_bounds": sat_bounds,
            "training_config": TRAINING_CONFIG,
            "dataset_version": "baseline_core",
            "scaler_path": str(scaler_path.relative_to(ROOT)),
            "train_csv": str(train_csv.relative_to(ROOT)),
            "val_csv": str(val_csv.relative_to(ROOT)),
            "best_epoch": result.best_epoch,
            "best_val_loss": result.best_val_loss,
            "n_train_rows": len(tr.X), "n_val_rows": len(va.X),
        }, f, indent=2)
    with open(out_dir / "history.json", "w") as f:
        json.dump(result.history, f, indent=2)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"global": global_metrics, "per_episode": per_episode,
                    "per_phase": per_phase, "per_scenario": per_scenario}, f, indent=2)

    preds_df = va.meta.copy()
    for i, name in enumerate(label_order):
        preds_df[f"true__{name}"] = y_true[:, i]
        preds_df[f"pred__{name}"] = y_pred[:, i]
    preds_df.to_csv(out_dir / "val_predictions.csv", index=False)

    return {
        "fold_id": fold_id, "network": network,
        "best_epoch": result.best_epoch, "best_val_loss": result.best_val_loss,
        "n_train_rows": len(tr.X), "n_val_rows": len(va.X),
        **{f"{name}__{k}": v for name in label_order for k, v in global_metrics["per_output"][name].items()
           if k in ("mse", "rmse", "mae", "r2", "pearson_r")},
    }


def main():
    all_rows = {"lateral": [], "longitudinal": []}
    for fold_id in fold_ids():
        for network in ("lateral", "longitudinal"):
            print(f"[cv] training {network} fold {fold_id:02d} ...", flush=True)
            row = run_one_fold_network(fold_id, network)
            all_rows[network].append(row)
            print(f"[cv] {network} fold {fold_id:02d}: best_epoch={row['best_epoch']} "
                  f"best_val_loss={row['best_val_loss']:.5f}", flush=True)

    for network in ("lateral", "longitudinal"):
        df = pd.DataFrame(all_rows[network])
        out_dir = REPORTS_DIR / "training" / network
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / "cv_fold_results.csv", index=False)
        summary = df.drop(columns=["fold_id", "network"]).agg(["mean", "std"]).T
        summary.to_csv(out_dir / "cv_summary_mean_std.csv")
        print(f"\n[{network}] CV summary (mean +/- std across {len(df)} folds):")
        print(summary.to_string())


if __name__ == "__main__":
    main()
