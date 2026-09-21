#!/usr/bin/env python3
"""Stage 9: session-grouped LOSO CV folds, fixed diagnostic holdout, and
per-fold Z-score scalers (computed only on each fold's TRAIN sessions'
baseline_core rows) + a final_all_valid scaler over all eligible sessions.
"""
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import paths
from pipeline.splits import select_fixed_diagnostic_holdout, build_loso_folds
from pipeline.scalers import compute_scaler, save_scaler, ZSCORE_VARIABLES


def load_baseline_core(session_ids, episode_to_session):
    frames = []
    for eid, sid in episode_to_session.items():
        if sid in session_ids:
            frames.append(pd.read_csv(paths.FEATURES_BASELINE_CORE_DIR / f"{eid}.csv"))
    return frames


def main():
    paths.ensure_dirs()
    episodes = pd.read_csv(paths.MANIFESTS_DIR / "episodes.csv")
    eligible = episodes[episodes.eligible_for_training]
    episode_to_session = dict(zip(eligible.episode_id, eligible.session_id))

    holdout_session = select_fixed_diagnostic_holdout(eligible, scenario="wind_16kt")
    print(f"Fixed diagnostic holdout session: {holdout_session} (deterministic sha256 pick among wind_16kt sessions)")

    cv_session_ids = sorted(s for s in eligible.session_id.unique() if s != holdout_session)
    folds = build_loso_folds(cv_session_ids)
    print(f"{len(cv_session_ids)} sessions in the LOSO rotation -> {len(folds)} folds "
          f"(holdout session '{holdout_session}' excluded from all folds).")

    fold_rows = []
    for fold in folds:
        train_dfs = load_baseline_core(fold["train_sessions"], episode_to_session)
        val_dfs = load_baseline_core([fold["val_session"]], episode_to_session)

        scaler = compute_scaler(train_dfs, fold["train_sessions"], purpose="cv_fold_training")
        scaler_path = paths.SCALERS_FOLDS_DIR / f"fold_{fold['fold_id']:02d}_{fold['val_session']}.json"
        save_scaler(scaler, scaler_path)

        n_train_rows = sum(len(d) for d in train_dfs)
        n_val_rows = sum(len(d) for d in val_dfs)
        scenario_val = eligible.loc[eligible.session_id == fold["val_session"], "scenario_id"].iloc[0]
        fold_rows.append({
            "fold_id": fold["fold_id"], "val_session": fold["val_session"], "val_scenario": scenario_val,
            "n_train_sessions": len(fold["train_sessions"]), "n_val_sessions": 1,
            "n_train_rows_baseline_core": n_train_rows, "n_val_rows_baseline_core": n_val_rows,
            "scaler_path": str(scaler_path.relative_to(paths.ROOT)),
            "n_degenerate_std_variables": len(scaler["degenerate_std_variables"]),
        })
        print(f"[fold {fold['fold_id']:02d}] val={fold['val_session']:32s} scenario={scenario_val:22s} "
              f"train_rows={n_train_rows:6d} val_rows={n_val_rows:5d} degenerate_std={len(scaler['degenerate_std_variables'])}")

    fold_summary = pd.DataFrame(fold_rows)
    fold_summary.to_csv(paths.SPLITS_DIR / "cv_folds_summary.csv", index=False)

    # explicit per-session assignment table (spec: manifests of CV grouped by session)
    assign_rows = []
    for fold in folds:
        for sid in fold["train_sessions"]:
            assign_rows.append({"fold_id": fold["fold_id"], "session_id": sid, "role": "train"})
        assign_rows.append({"fold_id": fold["fold_id"], "session_id": fold["val_session"], "role": "val"})
    pd.DataFrame(assign_rows).to_csv(paths.SPLITS_DIR / "cv_folds_session_assignment.csv", index=False)

    holdout_info = {
        "holdout_session": holdout_session,
        "scenario": "wind_16kt",
        "selection_method": "deterministic: sha256(session_id) over all wind_16kt sessions, ascending, first",
        "other_wind_16kt_sessions_remaining_in_training_rotation": cv_session_ids and [
            s for s in cv_session_ids if eligible.loc[eligible.session_id == s, "scenario_id"].iloc[0] == "wind_16kt"
        ],
        "purpose": "offline diagnostic only; NEVER used in any CV fold's train or val; official baseline test is closed-loop FlightGear evaluation",
    }
    with open(paths.SPLITS_DIR / "fixed_diagnostic_holdout.yaml", "w") as f:
        yaml.safe_dump(holdout_info, f, sort_keys=False)
    print(f"\nWrote {paths.SPLITS_DIR / 'fixed_diagnostic_holdout.yaml'}: {holdout_info}")

    # ---- final_all_valid scaler: ALL eligible sessions' baseline_core rows ----
    all_sessions = sorted(eligible.session_id.unique())
    all_train_dfs = load_baseline_core(all_sessions, episode_to_session)
    final_scaler = compute_scaler(all_train_dfs, all_sessions, purpose="final_training_after_model_selection")
    save_scaler(final_scaler, paths.SCALERS_FINAL_DIR / "scaler_final_all_valid.json")
    print(f"Wrote final_all_valid scaler over {len(all_sessions)} sessions "
          f"(includes the fixed diagnostic holdout session -- this scaler is NOT for offline validation, "
          f"see purpose field).")


if __name__ == "__main__":
    main()
