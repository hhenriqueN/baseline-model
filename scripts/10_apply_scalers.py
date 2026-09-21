#!/usr/bin/env python3
"""Stage 10: apply each fold's scaler to its own train/val baseline_core rows
and write normalized per-fold datasets (train.csv / val.csv), plus a
final_all_valid-normalized full dataset. Purely a convenience materialization
of stage 9's scalers -- nothing here recomputes statistics.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import paths
from pipeline.scalers import apply_scaler


def main():
    episodes = pd.read_csv(paths.MANIFESTS_DIR / "episodes.csv")
    eligible = episodes[episodes.eligible_for_training]
    episode_to_session = dict(zip(eligible.episode_id, eligible.session_id))

    assignment = pd.read_csv(paths.SPLITS_DIR / "cv_folds_session_assignment.csv")
    normalized_dir = paths.SPLITS_DIR / "normalized_folds"
    normalized_dir.mkdir(exist_ok=True)

    for fold_id, grp in assignment.groupby("fold_id"):
        val_session = grp.loc[grp.role == "val", "session_id"].iloc[0]
        scaler_path = next(paths.SCALERS_FOLDS_DIR.glob(f"fold_{fold_id:02d}_*.json"))
        scaler = json.load(open(scaler_path))

        for role in ("train", "val"):
            sessions = grp.loc[grp.role == role, "session_id"].tolist()
            eids = [eid for eid, sid in episode_to_session.items() if sid in sessions]
            dfs = [pd.read_csv(paths.FEATURES_BASELINE_CORE_DIR / f"{eid}.csv") for eid in eids]
            if not dfs:
                continue
            df = pd.concat(dfs, ignore_index=True)
            df_norm = apply_scaler(df, scaler)
            out = normalized_dir / f"fold_{fold_id:02d}_{role}.csv"
            df_norm.to_csv(out, index=False)
        print(f"[normalize] fold {fold_id:02d} (val={val_session}) -> wrote train/val normalized CSVs")

    # final_all_valid normalized (whole eligible baseline_core dataset)
    final_scaler = json.load(open(paths.SCALERS_FINAL_DIR / "scaler_final_all_valid.json"))
    all_eids = list(episode_to_session.keys())
    df_all = pd.concat([pd.read_csv(paths.FEATURES_BASELINE_CORE_DIR / f"{eid}.csv") for eid in all_eids], ignore_index=True)
    df_all_norm = apply_scaler(df_all, final_scaler)
    df_all_norm.to_csv(paths.SCALERS_FINAL_DIR / "baseline_core_normalized_all_valid.csv", index=False)
    print(f"\nWrote final_all_valid normalized dataset: {len(df_all_norm)} rows.")


if __name__ == "__main__":
    main()
