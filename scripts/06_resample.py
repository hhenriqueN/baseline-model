#!/usr/bin/env python3
"""Stage 6: causal anti-alias filter + resample every episode to 10 Hz.

Eligible episodes -> data/processed/10hz/{episode_id}.csv
voo-inicio-teste (excluded)  -> data/processed/excluded/{episode_id}.csv
"""
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import paths, columns as C
from pipeline.resample import resample_episode

KEEP_ROLES = {"metadata", "neural_input", "label", "derived_feature_source", "flight_manager_only"}
META_COLS = ["episode_id", "session_id", "source_file", "scenario_id", "runway_id", "eligible_for_training"]


def main():
    paths.ensure_dirs()
    with open(paths.CONFIG_DIR / "preprocessing.yaml") as f:
        cfg = yaml.safe_load(f)
    episodes = pd.read_csv(paths.MANIFESTS_DIR / "episodes.csv")
    catalog = pd.read_csv(paths.MANIFESTS_DIR / "columns_catalog.csv")

    keep_cols = catalog.loc[catalog.role.isin(KEEP_ROLES), "column_name"].tolist()
    keep_cols = [c for c in keep_cols if c not in (C.FRAME, C.SIM_TIME)]  # handled separately by resampler
    print(f"Carrying {len(keep_cols)} raw property columns forward into processed/10hz "
          f"(roles: {sorted(KEEP_ROLES)}). Remaining columns stay available in data/validated/.")

    diag_rows = []
    for _, ep in episodes.iterrows():
        val_path = paths.VALIDATED_DIR / f"{ep['episode_id']}.csv"
        df = pd.read_csv(val_path)

        resampled, diag = resample_episode(df, keep_cols, META_COLS, cfg)
        diag["episode_id"] = ep["episode_id"]
        diag_rows.append(diag)

        out_dir = paths.PROCESSED_10HZ_DIR if ep["eligible_for_training"] else paths.PROCESSED_EXCLUDED_DIR
        out_path = out_dir / f"{ep['episode_id']}.csv"
        resampled.to_csv(out_path, index=False)
        print(f"[resample] {ep['episode_id']}: {diag['n_raw_samples']} @ {diag['effective_fs_raw_hz']:.2f}Hz "
              f"-> {diag['n_resampled_samples']} @ {diag['target_hz']}Hz "
              f"(group delay ~{diag['group_delay_ms_at_cutoff']:.1f} ms) -> {out_path.relative_to(paths.ROOT)}")

    diag_df = pd.DataFrame(diag_rows)
    diag_df.to_csv(paths.MANIFESTS_DIR / "resampling_diagnostics.csv", index=False)

    # sanity: verify 10Hz within tolerance
    dt_check = 1.0 / diag_df["target_hz"].iloc[0]
    for _, ep in episodes.iterrows():
        out_dir = paths.PROCESSED_10HZ_DIR if ep["eligible_for_training"] else paths.PROCESSED_EXCLUDED_DIR
        df = pd.read_csv(out_dir / f"{ep['episode_id']}.csv")
        dt = df[C.SIM_TIME].diff().dropna()
        assert (abs(dt - dt_check) < 1e-6).all(), f"{ep['episode_id']}: resampled grid not uniform 10Hz"
    print("\nAll resampled episodes verified as a uniform 10 Hz grid.")


if __name__ == "__main__":
    main()
