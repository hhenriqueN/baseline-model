"""Loads normalized fold/final CSVs into (X, y, metadata) tensors for the two
baseline networks (spec sections 15-16). Pure data plumbing -- no scaler
fitting happens here (that is stage 09/10 of the pipeline); this module only
reads the already-normalized CSVs it is given.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from pipeline.features import (
    LATERAL_FEATURE_ORDER, LATERAL_LABEL_ORDER,
    LONGITUDINAL_FEATURE_ORDER, LONGITUDINAL_LABEL_ORDER,
)

NETWORK_SPECS = {
    "lateral": {"feature_prefix": "lat__", "label_prefix": "laty__",
                 "feature_order": LATERAL_FEATURE_ORDER, "label_order": LATERAL_LABEL_ORDER},
    "longitudinal": {"feature_prefix": "lon__", "label_prefix": "lony__",
                       "feature_order": LONGITUDINAL_FEATURE_ORDER, "label_order": LONGITUDINAL_LABEL_ORDER},
}

META_COLS = ["episode_id", "session_id", "source_file", "scenario_id", "runway_id",
             "sim_time", "frame_10hz", "phase"]


@dataclass
class FoldTensors:
    X: torch.Tensor
    y: torch.Tensor
    meta: pd.DataFrame


def load_split(csv_path, network: str) -> FoldTensors:
    spec = NETWORK_SPECS[network]
    df = pd.read_csv(csv_path)
    feat_cols = [spec["feature_prefix"] + f for f in spec["feature_order"]]
    label_cols = [spec["label_prefix"] + l for l in spec["label_order"]]
    missing = [c for c in feat_cols + label_cols if c not in df.columns]
    if missing:
        raise KeyError(f"{csv_path}: missing expected columns {missing}")
    X = torch.tensor(df[feat_cols].to_numpy(dtype=np.float32))
    y = torch.tensor(df[label_cols].to_numpy(dtype=np.float32))
    assert np.isfinite(X.numpy()).all(), f"{csv_path}: non-finite values in inputs"
    assert np.isfinite(y.numpy()).all(), f"{csv_path}: non-finite values in labels"
    meta = df[[c for c in META_COLS if c in df.columns]].copy()
    return FoldTensors(X=X, y=y, meta=meta)
