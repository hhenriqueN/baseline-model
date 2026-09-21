#!/usr/bin/env python3
"""Stage 3: segment circuito-completo.csv into landing-01 and landing-02,
detect approach-phase boundaries for landing-02, plot diagnostics, and write
data/config/episode_overrides.yaml.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import paths, columns as C
from pipeline.io_utils import read_flight_csv
from pipeline.segment_circuit import find_airborne_gap, segment_one_landing
from pipeline.runway import RunwayConfig, compute_runway_features


def detect_final_approach_established(df, cfg, feats, search_lo, search_hi):
    fa = cfg["final_approach_detection"]
    t = df[C.SIM_TIME].to_numpy()
    hdg_err = feats["heading_error_deg"].to_numpy()
    cross = feats["cross_track_error_m"].to_numpy()
    along = feats["along_track_m"].to_numpy()
    vs = df[C.VERTICAL_SPEED_FPS].to_numpy() * 60.0  # fps -> fpm

    cond = (
        (np.abs(hdg_err) <= fa["heading_error_max_deg"]) &
        (np.abs(cross) <= fa["cross_track_max_m"]) &
        (along <= fa["along_track_max_m"]) &
        (vs <= fa["descending_min_fpm"])
    )
    k = search_lo
    while k <= search_hi:
        if cond[k]:
            run_start = k
            while k <= search_hi and cond[k]:
                k += 1
            run_end = k - 1
            if t[run_end] - t[run_start] >= fa["min_sustained_s"]:
                return run_start
        else:
            k += 1
    return None


def detect_approach_maneuvering_start(df, cfg, feats, search_lo, search_hi):
    am = cfg["approach_maneuvering_detection"]
    t = df[C.SIM_TIME].to_numpy()
    hdg_err = feats["heading_error_deg"].to_numpy()
    vs = df[C.VERTICAL_SPEED_FPS].to_numpy()
    flaps = df[C.FLAPS_CMD].to_numpy()
    flaps0 = flaps[search_lo]

    cond = (np.abs(hdg_err) <= am["heading_error_max_deg"]) & ((vs < -0.5) | (flaps > flaps0 + 0.02))
    k = search_lo
    while k <= search_hi:
        if cond[k]:
            run_start = k
            while k <= search_hi and cond[k]:
                k += 1
            run_end = k - 1
            if t[run_end] - t[run_start] >= am["min_sustained_s"]:
                return run_start
        else:
            k += 1
    return None


def main():
    paths.ensure_dirs()
    with open(paths.CONFIG_DIR / "preprocessing.yaml") as f:
        cfg = yaml.safe_load(f)
    runway_cfg = RunwayConfig.from_yaml(paths.CONFIG_DIR / "runway.yaml")

    src = paths.RAW_DIR / "circuito-completo.csv"
    df, fmt = read_flight_csv(src)
    df = df.sort_values(C.SIM_TIME, kind="mergesort").reset_index(drop=True)

    feats = compute_runway_features(
        df, runway_cfg,
        lat_col=C.LAT, lon_col=C.LON, heading_col=C.HEADING_DEG,
        alt_msl_col=C.ALT_MSL_FT, airspeed_col=C.AIRSPEED_KT, agl_col=C.ALT_AGL_FT,
    )
    # landing-02 touches down on a DIFFERENT physical runway (BIKF RWY29, not
    # RWY02/20) -- see data/config/runway_29.yaml for the full evidence. Its
    # approach-phase detection must use RWY29 geometry, not RWY02/20.
    runway29_cfg = RunwayConfig.from_yaml(paths.CONFIG_DIR / "runway_29.yaml")
    feats29 = compute_runway_features(
        df, runway29_cfg,
        lat_col=C.LAT, lon_col=C.LON, heading_col=C.HEADING_DEG,
        alt_msl_col=C.ALT_MSL_FT, airspeed_col=C.AIRSPEED_KT, agl_col=C.ALT_AGL_FT,
    )

    seg_cfg = cfg["landing_segmentation"]
    gap = find_airborne_gap(df, cfg["gear"]["main_indices"], seg_cfg["airborne_gap_min_s"], seg_cfg["airborne_gap_agl_ft"])
    if gap is None:
        raise RuntimeError("Could not find the airborne gap separating the two circuito-completo landings.")
    gap_start_idx, gap_end_idx, gap_duration = gap
    print(f"Airborne gap (circuit) detected: idx [{gap_start_idx},{gap_end_idx}], "
          f"t=[{df[C.SIM_TIME].iloc[gap_start_idx]:.1f},{df[C.SIM_TIME].iloc[gap_end_idx]:.1f}]s, duration={gap_duration:.1f}s")

    n = len(df)
    window1 = (0, gap_start_idx)
    window2 = (gap_end_idx, n - 1)

    result1, events1 = segment_one_landing(df, window1, cfg)
    result2, events2 = segment_one_landing(df, window2, cfg)
    if result1 is None or result2 is None:
        raise RuntimeError(f"Landing segmentation failed: landing1={result1} landing2={result2}")

    t = df[C.SIM_TIME].to_numpy()
    print("\n--- landing-01 events ---")
    for name, idx, tt in events1:
        print(f"  {name:20s} idx={idx:6d} t={tt:8.2f}s")
    print("--- landing-02 events ---")
    for name, idx, tt in events2:
        print(f"  {name:20s} idx={idx:6d} t={tt:8.2f}s")

    # landing-02: approach maneuvering start (search the whole circuit region
    # between the end of the airborne gap start-of-descent and touchdown-02)
    # and final approach established (search from approach_maneuvering start
    # to touchdown-02).
    search_hi_for_am = result2["touchdown_confirmed_idx"]
    search_lo_for_am = result1["complete_idx"]  # right after landing-01 liftoff; covers the whole circuit
    am_start = detect_approach_maneuvering_start(df, cfg, feats29, search_lo_for_am, search_hi_for_am)
    if am_start is None:
        am_start = gap_end_idx
        am_reason = "not_detected_automatically_defaulted_to_end_of_airborne_gap_REVIEW_NEEDED"
    else:
        am_reason = "auto_detected"
    print(f"\nlanding-02 APPROACH_MANEUVERING start: idx={am_start} t={t[am_start]:.2f}s ({am_reason})")

    fa_idx = detect_final_approach_established(df, cfg, feats29, search_lo_for_am, result2["touchdown_confirmed_idx"])
    if fa_idx is None:
        fa_idx = am_start
        fa_reason = "not_detected_automatically_defaulted_to_approach_maneuvering_start_REVIEW_NEEDED"
    else:
        fa_reason = "auto_detected"
    print(f"landing-02 FINAL_APPROACH_ESTABLISHED: idx={fa_idx} t={t[fa_idx]:.2f}s ({fa_reason})")

    flare_idx = None
    for k in range(fa_idx, result2["touchdown_confirmed_idx"] + 1):
        if df[C.ALT_AGL_FT].iloc[k] <= runway_cfg.flare_agl_ft:
            flare_idx = k
            break
    if flare_idx is None:
        flare_idx = result2["contact_candidate_idx"]

    # ---- write episode_overrides.yaml (human-reviewable) ----
    overrides = {
        "circuito-completo": {
            "source_file": "circuito-completo.csv",
            "session_id": "circuito-completo",
            "airborne_gap_used_to_split": {
                "start_idx": int(gap_start_idx), "end_idx": int(gap_end_idx),
                "t_start_s": float(t[gap_start_idx]), "t_end_s": float(t[gap_end_idx]),
                "duration_s": float(gap_duration),
            },
            "landing_01": {
                "episode_id": "circuito-completo__landing-01",
                "runway_id": "02",
                "start_idx": int(result1["episode_start_idx"]), "start_t_s": float(t[result1["episode_start_idx"]]),
                "contact_candidate_idx": int(result1["contact_candidate_idx"]),
                "touchdown_confirmed_idx": int(result1["touchdown_confirmed_idx"]),
                "rollout_idx": int(result1["rollout_idx"]),
                "complete_idx": int(result1["complete_idx"]),
                "complete_reason": result1["complete_reason"],
                "end_t_s": float(t[result1["complete_idx"]]),
                "note": "Standardized ~3km/800ft start, same family as the 13 straight-landing files; touch-and-go (COMPLETE detected via liftoff, not low groundspeed).",
            },
            "landing_02": {
                "episode_id": "circuito-completo__landing-02",
                "runway_id": "29",
                "runway_id_note": "CONFIRMED via touchdown heading (~264-269 deg, matches RWY29 true heading ~270 deg) and touchdown lat/lon (~634m from apt.dat RWY29 threshold, on the 11/29 latitude band) -- see data/config/runway_29.yaml for full evidence. This differs from RWY02/20 used by every other episode. Flagged for human review despite strong geometric evidence.",
                "full_episode_start_idx": int(gap_end_idx),
                "approach_maneuvering_start_idx": int(am_start),
                "approach_maneuvering_detection": am_reason,
                "final_approach_established_idx": int(fa_idx),
                "final_approach_established_detection": fa_reason,
                "flare_idx": int(flare_idx),
                "contact_candidate_idx": int(result2["contact_candidate_idx"]),
                "touchdown_confirmed_idx": int(result2["touchdown_confirmed_idx"]),
                "rollout_idx": int(result2["rollout_idx"]),
                "complete_idx": int(result2["complete_idx"]),
                "complete_reason": result2["complete_reason"],
                "end_t_s": float(t[result2["complete_idx"]]),
                "note": "full_approach = [approach_maneuvering_start_idx, complete_idx]. baseline_core = [final_approach_established_idx, complete_idx] (minus any initialization_transient, N/A here since this is a real in-flight approach, not an artificial standardized start).",
            },
            "manual_review_needed": (am_reason != "auto_detected") or (fa_reason != "auto_detected"),
        }
    }
    with open(paths.CONFIG_DIR / "episode_overrides.yaml", "w") as f:
        yaml.safe_dump(overrides, f, sort_keys=False)
    print(f"\nWrote {paths.CONFIG_DIR / 'episode_overrides.yaml'}")

    # ---- diagnostic plots ----
    fig, axes = plt.subplots(8, 1, figsize=(14, 22), sharex=True)
    all_events = events1 + events2
    extra_events = [("APPROACH_MANEUVERING", am_start, t[am_start]),
                    ("FINAL_APPROACH_ESTABLISHED", fa_idx, t[fa_idx]),
                    ("FLARE", flare_idx, t[flare_idx])]
    all_events = all_events + extra_events

    def mark(ax):
        for name, idx, tt in all_events:
            ax.axvline(tt, color="gray", linestyle="--", linewidth=0.7, alpha=0.7)
        ax.axvline(t[gap_start_idx], color="orange", linestyle=":", linewidth=1.2)
        ax.axvline(t[gap_end_idx], color="orange", linestyle=":", linewidth=1.2)

    axes[0].plot(t, df[C.ALT_AGL_FT], color="tab:blue")
    axes[0].set_ylabel("AGL (ft)")
    axes[1].plot(t, df[C.gear_wow(0)], label="nose wow")
    axes[1].plot(t, df[C.gear_wow(1)], label="main1 wow")
    axes[1].plot(t, df[C.gear_wow(2)], label="main2 wow")
    axes[1].legend(fontsize=7)
    axes[1].set_ylabel("WOW")
    axes[2].plot(t, df[C.gear_compression(0)], label="nose")
    axes[2].plot(t, df[C.gear_compression(1)], label="main1")
    axes[2].plot(t, df[C.gear_compression(2)], label="main2")
    axes[2].legend(fontsize=7)
    axes[2].set_ylabel("compression-norm")
    axes[3].plot(t, df[C.GROUNDSPEED_KT], color="tab:green")
    axes[3].set_ylabel("Groundspeed (kt)")
    axes[4].plot(t, df[C.AIRSPEED_KT], color="tab:purple")
    axes[4].set_ylabel("Airspeed (kt)")
    axes[5].plot(t, df[C.FLAPS_CMD], color="tab:brown")
    axes[5].set_ylabel("Flaps [0..1]")
    axes[6].plot(t, df[C.THROTTLE_ENGINE0], color="tab:red")
    axes[6].set_ylabel("Throttle")
    axes[7].plot(t, df[C.ELEVATOR_CMD], color="tab:cyan")
    axes[7].set_ylabel("Elevator cmd")
    axes[7].set_xlabel("sim_time (s)")

    for ax in axes:
        mark(ax)

    fig.suptitle("circuito-completo.csv segmentation diagnostics\n"
                  "dashed=detected boundaries, dotted orange=airborne gap used to split the file")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_fig = paths.FIGURES_DIR / "circuito_completo_segmentation.png"
    fig.savefig(out_fig, dpi=130)
    print(f"Wrote {out_fig}")


if __name__ == "__main__":
    main()
