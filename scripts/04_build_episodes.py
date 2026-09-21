#!/usr/bin/env python3
"""Stage 4: build data/manifests/episodes.csv — one row per training episode.

13 straight-landing files -> 1 episode each.
circuito-completo.csv -> 2 episodes (landing-01, landing-02), same session_id.
voo-inicio-teste.csv -> 1 episode, eligible_for_training=False.
"""
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import paths, columns as C
from pipeline.io_utils import read_flight_csv
from pipeline.scenario import scenario_from_filename, confirm_scenario_signals


def main():
    paths.ensure_dirs()
    manifest = pd.read_csv(paths.MANIFESTS_DIR / "raw_files.csv")
    with open(paths.CONFIG_DIR / "episode_overrides.yaml") as f:
        overrides = yaml.safe_load(f)

    rows = []
    for _, r in manifest.iterrows():
        src_name = r["source_file"]
        df, _fmt = read_flight_csv(r["raw_path"])
        df = df.sort_values(C.SIM_TIME, kind="mergesort").reset_index(drop=True)
        n = len(df)
        signals = confirm_scenario_signals(df)
        scenario = scenario_from_filename(src_name)
        stem = Path(src_name).stem

        if stem == "circuito-completo":
            ov = overrides["circuito-completo"]
            l1, l2 = ov["landing_01"], ov["landing_02"]
            rows.append({
                "episode_id": l1["episode_id"], "session_id": ov["session_id"],
                "source_file": src_name, "scenario_id": scenario,
                "eligible_for_training": True, "exclusion_reason": "",
                "runway_id": l1["runway_id"],
                "raw_row_start": l1["start_idx"], "raw_row_end": l1["complete_idx"],
                "n_samples_raw": l1["complete_idx"] - l1["start_idx"] + 1,
                "duration_s_raw": l1["end_t_s"] - l1["start_t_s"],
                "full_approach_start_idx_in_raw": l1["start_idx"],
                "baseline_core_start_idx_in_raw": l1["start_idx"],  # transient handled later (stage 5/6)
                "has_known_initialization_transient": True,
                "notes": l1["note"],
                **signals,
            })
            # NOTE: the episode's stored range starts at approach_maneuvering_start_idx,
            # NOT the airborne-gap boundary used only internally by stage 3 to split
            # the landing-01/landing-02 touchdown search windows. gap_end_idx falls
            # near short final (~14s before touchdown) -- well AFTER
            # approach_maneuvering_start_idx and final_approach_established_idx -- so
            # using it here would silently truncate away both of those phases.
            l2_episode_start_idx = min(l2["approach_maneuvering_start_idx"], l2["full_episode_start_idx"])
            rows.append({
                "episode_id": l2["episode_id"], "session_id": ov["session_id"],
                "source_file": src_name, "scenario_id": scenario,
                "eligible_for_training": True, "exclusion_reason": "",
                "runway_id": l2["runway_id"],
                "raw_row_start": l2_episode_start_idx, "raw_row_end": l2["complete_idx"],
                "n_samples_raw": l2["complete_idx"] - l2_episode_start_idx + 1,
                "duration_s_raw": l2["end_t_s"] - float(df[C.SIM_TIME].iloc[l2_episode_start_idx]),
                "full_approach_start_idx_in_raw": l2["approach_maneuvering_start_idx"],
                "baseline_core_start_idx_in_raw": l2["final_approach_established_idx"],
                "has_known_initialization_transient": False,
                "notes": l2["note"] + " " + l2["runway_id_note"],
                **signals,
            })
        else:
            episode_id = stem
            eligible = bool(r["eligible_for_training"])
            rows.append({
                "episode_id": episode_id, "session_id": episode_id,
                "source_file": src_name, "scenario_id": scenario,
                "eligible_for_training": eligible,
                "exclusion_reason": r["exclusion_reason"],
                "runway_id": "02",
                "raw_row_start": 0, "raw_row_end": n - 1,
                "n_samples_raw": n,
                "duration_s_raw": float(df[C.SIM_TIME].iloc[-1] - df[C.SIM_TIME].iloc[0]),
                "full_approach_start_idx_in_raw": 0,
                "baseline_core_start_idx_in_raw": None,  # determined in stage 5 (transient detection)
                "has_known_initialization_transient": eligible,  # standardized ~3km/800ft start
                "notes": "Standardized ~3km/800ft start (per pilot description); initialization_transient boundary detected in stage 5." if eligible else "Pilot familiarization flight; excluded from training/normalization/splits per spec.",
                **signals,
            })
        print(f"[episodes] {src_name}: scenario={scenario} wind_mean={signals['qa_wind_speed_kt_mean']:.2f}kt "
              f"turb_max={signals['qa_turbulence_magnitude_max']:.3f}")

    episodes = pd.DataFrame(rows)
    out_path = paths.MANIFESTS_DIR / "episodes.csv"
    episodes.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(episodes)} episodes)")
    print(episodes[["episode_id", "session_id", "scenario_id", "eligible_for_training", "runway_id", "n_samples_raw"]].to_string(index=False))

    n_sessions = episodes.session_id.nunique()
    print(f"\n{len(episodes)} episodes across {n_sessions} sessions.")
    assert episodes.loc[episodes.episode_id.str.startswith("circuito-completo"), "session_id"].nunique() == 1


if __name__ == "__main__":
    main()
