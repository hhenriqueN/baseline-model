#!/usr/bin/env python3
"""Stage 8b: populate data/processed/full_episodes/ (10Hz data + phase/mask
columns, full duration, pre-feature-engineering) and
data/processed/initialization_transients/ (the isolated transient-only rows
for the 14 standardized-start episodes, preserved for future study -- never
silently discarded).
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import paths, columns as C
from pipeline.features import build_phase_and_masks

def main():
    paths.ensure_dirs()
    episodes = pd.read_csv(paths.MANIFESTS_DIR / "episodes.csv")
    boundaries = pd.read_csv(paths.MANIFESTS_DIR / "episode_phase_boundaries.csv").set_index("episode_id")
    row_phase = pd.read_csv(paths.MANIFESTS_DIR / "landing_fsm_row_phase.csv")

    n_transient_files = 0
    for _, ep in episodes.iterrows():
        eid = ep["episode_id"]
        is_eligible = bool(ep["eligible_for_training"])
        src_dir = paths.PROCESSED_10HZ_DIR if is_eligible else paths.PROCESSED_EXCLUDED_DIR
        df10 = pd.read_csv(src_dir / f"{eid}.csv")

        if eid in boundaries.index:
            b = boundaries.loc[eid].to_dict()
            is_standardized_start = bool(b["is_standardized_start"])
            has_am_fa = pd.notna(b.get("approach_maneuvering_s"))
            overlay = None
            ep_overlay = row_phase[row_phase.episode_id == eid]
            if len(ep_overlay):
                overlay_by_frame = dict(zip(ep_overlay.frame_10hz, ep_overlay.row_phase))
                overlay = df10["frame"].map(overlay_by_frame)
            masks = build_phase_and_masks(df10, b, is_standardized_start, has_am_fa, overlay)
            full_ep = pd.concat([df10, masks], axis=1)
        else:
            full_ep = df10.copy()
            full_ep["phase"] = "UNSEGMENTED"
            full_ep["initialization_transient"] = 0
            full_ep["baseline_core"] = 0
            full_ep["full_approach"] = 0
            full_ep["recovery_ablation"] = 0
            is_standardized_start = False

        out_dir = paths.PROCESSED_FULL_EPISODES_DIR if is_eligible else paths.PROCESSED_EXCLUDED_DIR
        full_ep.to_csv(out_dir / f"{eid}_full_episode.csv", index=False)

        if is_standardized_start and full_ep["initialization_transient"].sum() > 0:
            transient_only = full_ep[full_ep["initialization_transient"] == 1]
            transient_only.to_csv(paths.PROCESSED_INIT_TRANSIENTS_DIR / f"{eid}_transient.csv", index=False)
            n_transient_files += 1
            print(f"[processed] {eid}: full_episode={len(full_ep)} rows, transient_only={len(transient_only)} rows")
        else:
            print(f"[processed] {eid}: full_episode={len(full_ep)} rows, no transient segment")

    print(f"\nWrote {len(episodes)} files to {paths.PROCESSED_FULL_EPISODES_DIR.relative_to(paths.ROOT)} "
          f"and {n_transient_files} files to {paths.PROCESSED_INIT_TRANSIENTS_DIR.relative_to(paths.ROOT)}")


if __name__ == "__main__":
    main()
