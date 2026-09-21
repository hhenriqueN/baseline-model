#!/usr/bin/env python3
"""Builds the plots and flattened metric tables required by spec section 19-20,
from the artifacts already written by training/run_cv.py:
  - loss curves (train/val vs epoch) per fold/network
  - predicted-vs-true scatter + residual histograms per fold/network
  - per-validation-episode time-series plots (expert vs predicted, all 4 commands)
  - flattened per-fold / per-flight / per-phase / per-scenario metric CSVs
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import paths

ROOT = paths.ROOT
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

LABEL_COLORS = {"aileron_command": "tab:blue", "rudder_command": "tab:orange",
                 "elevator_effective": "tab:green", "throttle_command": "tab:red"}


def fold_ids(network):
    return sorted(int(p.name.split("_")[1]) for p in (MODELS_DIR / network / "folds").glob("fold_*"))


def plot_loss_curves(network):
    out_dir = REPORTS_DIR / "training" / network / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    for fid in fold_ids(network):
        fold_dir = MODELS_DIR / network / "folds" / f"fold_{fid:02d}"
        hist = json.load(open(fold_dir / "history.json"))
        cfg = json.load(open(fold_dir / "config.json"))
        df = pd.DataFrame(hist)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(df["epoch"], df["train_loss"], label="train")
        ax.plot(df["epoch"], df["val_loss"], label="val")
        ax.axvline(cfg["best_epoch"], color="black", linestyle="--", linewidth=0.8, label="best epoch (checkpoint)")
        ax.set_xlabel("epoch"); ax.set_ylabel("MSE loss"); ax.set_yscale("log")
        ax.set_title(f"{network} fold {fid:02d} loss curve")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"fold_{fid:02d}_loss_curve.png", dpi=110)
        plt.close(fig)


def plot_scatter_and_residuals(network):
    out_dir = REPORTS_DIR / "training" / network / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for fid in fold_ids(network):
        fold_dir = MODELS_DIR / network / "folds" / f"fold_{fid:02d}"
        frames.append(pd.read_csv(fold_dir / "val_predictions.csv"))
    all_preds = pd.concat(frames, ignore_index=True)

    true_cols = [c for c in all_preds.columns if c.startswith("true__")]
    labels = [c[len("true__"):] for c in true_cols]

    fig, axes = plt.subplots(1, len(labels), figsize=(5 * len(labels), 5))
    if len(labels) == 1:
        axes = [axes]
    for ax, name in zip(axes, labels):
        t = all_preds[f"true__{name}"]; p = all_preds[f"pred__{name}"]
        ax.scatter(t, p, s=3, alpha=0.15, color=LABEL_COLORS.get(name, "tab:blue"))
        lo, hi = min(t.min(), p.min()), max(t.max(), p.max())
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)
        ax.set_xlabel(f"expert {name}"); ax.set_ylabel(f"predicted {name}")
        ax.set_title(name)
    fig.suptitle(f"{network}: predicted vs. true (all CV validation folds pooled)")
    fig.tight_layout()
    fig.savefig(out_dir / "scatter_pred_vs_true_all_folds.png", dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(labels), figsize=(5 * len(labels), 4))
    if len(labels) == 1:
        axes = [axes]
    for ax, name in zip(axes, labels):
        resid = all_preds[f"pred__{name}"] - all_preds[f"true__{name}"]
        ax.hist(resid, bins=60, color=LABEL_COLORS.get(name, "tab:blue"))
        ax.set_xlabel(f"residual (pred - true) {name}"); ax.set_ylabel("count")
    fig.suptitle(f"{network}: residual distributions (all CV validation folds pooled)")
    fig.tight_layout()
    fig.savefig(out_dir / "residuals_all_folds.png", dpi=110)
    plt.close(fig)
    return all_preds


def plot_per_episode_timeseries():
    out_dir = REPORTS_DIR / "training" / "timeseries"
    out_dir.mkdir(parents=True, exist_ok=True)
    lat_frames, lon_frames = [], []
    for fid in fold_ids("lateral"):
        lat_frames.append(pd.read_csv(MODELS_DIR / "lateral" / "folds" / f"fold_{fid:02d}" / "val_predictions.csv"))
    for fid in fold_ids("longitudinal"):
        lon_frames.append(pd.read_csv(MODELS_DIR / "longitudinal" / "folds" / f"fold_{fid:02d}" / "val_predictions.csv"))
    lat_all = pd.concat(lat_frames, ignore_index=True)
    lon_all = pd.concat(lon_frames, ignore_index=True)

    episodes = sorted(set(lat_all.episode_id) | set(lon_all.episode_id))
    for eid in episodes:
        lat_ep = lat_all[lat_all.episode_id == eid].sort_values("sim_time")
        lon_ep = lon_all[lon_all.episode_id == eid].sort_values("sim_time")
        fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
        if len(lat_ep):
            axes[0].plot(lat_ep.sim_time, lat_ep["true__aileron_command"], label="expert", color="black")
            axes[0].plot(lat_ep.sim_time, lat_ep["pred__aileron_command"], label="predicted", color="tab:blue", alpha=0.8)
            axes[1].plot(lat_ep.sim_time, lat_ep["true__rudder_command"], label="expert", color="black")
            axes[1].plot(lat_ep.sim_time, lat_ep["pred__rudder_command"], label="predicted", color="tab:orange", alpha=0.8)
        if len(lon_ep):
            axes[2].plot(lon_ep.sim_time, lon_ep["true__elevator_effective"], label="expert", color="black")
            axes[2].plot(lon_ep.sim_time, lon_ep["pred__elevator_effective"], label="predicted", color="tab:green", alpha=0.8)
            axes[3].plot(lon_ep.sim_time, lon_ep["true__throttle_command"], label="expert", color="black")
            axes[3].plot(lon_ep.sim_time, lon_ep["pred__throttle_command"], label="predicted", color="tab:red", alpha=0.8)
        for ax, name in zip(axes, ["aileron", "rudder", "elevator_effective", "throttle"]):
            ax.set_ylabel(name); ax.legend(fontsize=7)
        axes[-1].set_xlabel("sim_time (s)")
        fig.suptitle(f"Expert vs. predicted commands (held-out validation): {eid}")
        fig.tight_layout()
        fig.savefig(out_dir / f"{eid}.png", dpi=110)
        plt.close(fig)


def flatten_metric_tables(network):
    rows_episode, rows_phase, rows_scenario = [], [], []
    for fid in fold_ids(network):
        fold_dir = MODELS_DIR / network / "folds" / f"fold_{fid:02d}"
        m = json.load(open(fold_dir / "metrics.json"))
        for group_name, group_dict in (("per_episode", rows_episode), ("per_phase", rows_phase),
                                         ("per_scenario", rows_scenario)):
            for key, val in m[group_name].items():
                for out_name, out_metrics in val["per_output"].items():
                    row = {"fold_id": fid, "group": key, "output": out_name}
                    row.update(out_metrics)
                    group_dict.append(row)
    out_dir = REPORTS_DIR / "training" / network
    pd.DataFrame(rows_episode).to_csv(out_dir / "metrics_per_flight.csv", index=False)
    pd.DataFrame(rows_phase).to_csv(out_dir / "metrics_per_phase.csv", index=False)
    pd.DataFrame(rows_scenario).to_csv(out_dir / "metrics_per_scenario.csv", index=False)


def main():
    for network in ("lateral", "longitudinal"):
        plot_loss_curves(network)
        plot_scatter_and_residuals(network)
        flatten_metric_tables(network)
        print(f"[reports] wrote figures + flattened metric tables for {network}")
    plot_per_episode_timeseries()
    print("[reports] wrote per-validation-episode time-series plots")


if __name__ == "__main__":
    main()
