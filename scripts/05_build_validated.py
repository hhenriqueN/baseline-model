#!/usr/bin/env python3
"""Stage 5: slice each raw file down to its episode boundaries, sort by
sim_time, drop exact duplicate timestamps, and write to data/validated/
at the ORIGINAL (~60Hz) frequency. No resampling, no column dropping here.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import paths, columns as C
from pipeline.io_utils import read_flight_csv


def main():
    paths.ensure_dirs()
    episodes = pd.read_csv(paths.MANIFESTS_DIR / "episodes.csv")
    raw_manifest = pd.read_csv(paths.MANIFESTS_DIR / "raw_files.csv").set_index("source_file")

    rows_report = []
    file_cache = {}
    for _, ep in episodes.iterrows():
        src = ep["source_file"]
        if src not in file_cache:
            raw_path = raw_manifest.loc[src, "raw_path"]
            df, _fmt = read_flight_csv(raw_path)
            df = df.sort_values(C.SIM_TIME, kind="mergesort").reset_index(drop=True)
            n_before = len(df)
            df = df.drop_duplicates(subset=[C.SIM_TIME], keep="first").reset_index(drop=True)
            n_after = len(df)
            file_cache[src] = (df, n_before, n_after)
        df, n_before, n_after = file_cache[src]

        lo, hi = int(ep["raw_row_start"]), int(ep["raw_row_end"])
        lo = min(lo, len(df) - 1)
        hi = min(hi, len(df) - 1)
        ep_df = df.iloc[lo:hi + 1].copy().reset_index(drop=True)

        ep_df.insert(0, "episode_id", ep["episode_id"])
        ep_df.insert(1, "session_id", ep["session_id"])
        ep_df.insert(2, "source_file", src)
        ep_df.insert(3, "scenario_id", ep["scenario_id"])
        ep_df.insert(4, "runway_id", ep["runway_id"])
        ep_df.insert(5, "eligible_for_training", ep["eligible_for_training"])
        ep_df.insert(6, "orig_row_index", range(lo, hi + 1))

        is_monotonic = bool(ep_df[C.SIM_TIME].is_monotonic_increasing)
        n_dup_ts_removed = n_before - n_after

        out_path = paths.VALIDATED_DIR / f"{ep['episode_id']}.csv"
        ep_df.to_csv(out_path, index=False)

        rows_report.append({
            "episode_id": ep["episode_id"], "n_rows_validated": len(ep_df),
            "is_monotonic": is_monotonic,
            "n_duplicate_timestamps_removed_from_source_file": n_dup_ts_removed,
            "t_start_s": float(ep_df[C.SIM_TIME].iloc[0]), "t_end_s": float(ep_df[C.SIM_TIME].iloc[-1]),
        })
        print(f"[validated] {ep['episode_id']}: rows={len(ep_df)} monotonic={is_monotonic} dup_ts_removed={n_dup_ts_removed}")

    report = pd.DataFrame(rows_report)
    report.to_csv(paths.MANIFESTS_DIR / "validated_report.csv", index=False)
    assert report.is_monotonic.all(), "Non-monotonic sim_time found after validation!"
    print(f"\nWrote {len(report)} validated episode files to {paths.VALIDATED_DIR}")


if __name__ == "__main__":
    main()
