"""Evaluation metrics required by spec section 19."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return float("nan")  # degenerate (near-constant) group -- correlation undefined
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.corrcoef(a, b)[0, 1]
    return float(r) if np.isfinite(r) else float("nan")


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-12:
        return float("nan")  # caution: near-constant target, R^2 is not meaningful
    return float(1.0 - ss_res / ss_tot)


def per_output_metrics(y_true: np.ndarray, y_pred: np.ndarray, sat_lo: float, sat_hi: float) -> dict:
    """y_true/y_pred: 1D arrays for one output column."""
    err = y_pred - y_true
    abs_err = np.abs(err)
    y_pred_sat = np.clip(y_pred, sat_lo, sat_hi)
    sat_rate = float(np.mean((y_pred < sat_lo) | (y_pred > sat_hi)))
    return {
        "mse": float(np.mean(err ** 2)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(abs_err)),
        "r2": _r2(y_true, y_pred),
        "pearson_r": _pearson(y_true, y_pred),
        "max_abs_error": float(np.max(abs_err)) if len(abs_err) else float("nan"),
        "p95_abs_error": float(np.percentile(abs_err, 95)) if len(abs_err) else float("nan"),
        "saturation_rate_after_clip": sat_rate,
        "mse_after_saturation": float(np.mean((y_pred_sat - y_true) ** 2)),
        "n": int(len(y_true)),
    }


def command_rate_mae(meta: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, output_idx: int) -> float:
    """Mean absolute error of the CAUSAL sample-to-sample command rate
    (finite backward difference, never a centered/future-looking difference),
    computed independently per episode and concatenated (never diffs across an
    episode boundary)."""
    meta = meta.reset_index(drop=True)  # positions must match y_true/y_pred exactly
    errs = []
    for eid, idx in meta.groupby("episode_id").groups.items():
        idx = np.asarray(idx)
        order = np.argsort(meta.loc[idx, "sim_time"].to_numpy())
        idx = idx[order]
        if len(idx) < 2:
            continue
        true_rate = np.diff(y_true[idx, output_idx])
        pred_rate = np.diff(y_pred[idx, output_idx])
        errs.append(np.abs(true_rate - pred_rate))
    if not errs:
        return float("nan")
    return float(np.mean(np.concatenate(errs)))


def evaluate_predictions(meta: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray,
                           label_order: list[str], sat_bounds: list[tuple[float, float]]) -> dict:
    """Returns {"global": {...}, "per_output": {name: {...}}}"""
    out = {"per_output": {}}
    for i, name in enumerate(label_order):
        lo, hi = sat_bounds[i]
        m = per_output_metrics(y_true[:, i], y_pred[:, i], lo, hi)
        m["command_rate_mae"] = command_rate_mae(meta, y_true, y_pred, i)
        out["per_output"][name] = m
    return out


def evaluate_by_group(meta: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray,
                        label_order: list[str], sat_bounds: list[tuple[float, float]],
                        group_col: str) -> dict:
    """Metrics broken out by an arbitrary grouping column (episode_id, scenario_id, phase, ...)."""
    result = {}
    for g, idx in meta.groupby(group_col).groups.items():
        idx = np.asarray(idx)
        sub_meta = meta.loc[idx]
        result[str(g)] = evaluate_predictions(sub_meta, y_true[idx], y_pred[idx], label_order, sat_bounds)
    return result
