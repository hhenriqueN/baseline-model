#!/usr/bin/env python3
"""Stage 7: for every eligible episode, run the corrected row-level landing
state machine (pipeline/landing_fsm.py, spec section 5) -- CONTACT_CANDIDATE
-> TOUCHDOWN_CONFIRMED -> (BOUNCE)* -> ROLLOUT -> COMPLETE, with pre- and
post-confirmation contact loss looping back to FLARE -- plus the (unchanged,
already-correct) FLARE crossing and initialization-transient detection.

Root-cause note (spec section 4): the previous implementation recorded only a
single contact_candidate_s / touchdown_confirmed_s timestamp per episode and
painted every row between them as CONTACT_CANDIDATE, including any airborne
gaps in between (porpoise bounces / intermittent contact before the eventual
confirmed touchdown). That is what produced the 2.7s-38.6s "CONTACT_CANDIDATE"
anomalies. The landing FSM here evaluates the actual WOW time series row by
row, on the 10 Hz processed grid (the same grid the training datasets are
built from), so touchdown/rollout confirmation-duration thresholds translate
into an explicit, off-by-one-free number of consecutive samples, and genuine
bounce/intermittent-contact intervals are correctly attributed to
BOUNCE/FLARE instead of being folded into CONTACT_CANDIDATE.

Writes:
  data/manifests/episode_phase_boundaries.csv       (summary indices/timestamps, reporting)
  data/manifests/transient_diagnostics.csv          (unchanged: standardized-start episodes)
  data/manifests/landing_fsm_row_phase.csv          (per-row CONTACT_CANDIDATE/.../COMPLETE overlay, 10Hz grid)
  data/manifests/phase_transitions.csv              (spec section 6 transition table)
  data/manifests/bounce_events.csv                  (spec section 5 bounce/unconfirmed-contact records)
  data/manifests/phase_review_flags.csv             (non-fatal QA flags requiring human review)
  data/reports/figures/touchdown_diagnostics/<episode_id>.png (spec section 6 touchdown plots)
  data/reports/figures/transient_<episode_id>.png   (unchanged: initialization-transient diagnostic)

Fails (raises) if pipeline/state_machine_checks.py finds a hard violation
(ordering, unconfirmed nose-gear-only contact, unreported long candidate,
non-monotonic time, bad gear mapping) -- spec section 6's "stop and produce a
clear diagnostic report" requirement.
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
from pipeline.landing_fsm import run_landing_fsm
from pipeline.state_machine_checks import validate_episode
from pipeline.transients import detect_transient_end
from pipeline.runway import RunwayConfig

STANDARDIZED_START_EPISODES_NOTE = (
    "13 straight-landing files + circuito-completo__landing-01 all begin in the "
    "pilot-described standardized ~3km/800ft position with unconfigured flaps/speed."
)


def detect_flare_idx(df: pd.DataFrame, rcfg: RunwayConfig) -> tuple[int, bool]:
    """First index where AGL crosses at/below the configured flare threshold.
    Unchanged single-threshold logic (not part of the known bug -- FLARE is a
    genuine one-way crossing, never re-entered/looped, unlike the landing
    sequence states below it)."""
    agl = df[C.ALT_AGL_FT].to_numpy()
    idx = np.argmax(agl <= rcfg.flare_agl_ft) if np.any(agl <= rcfg.flare_agl_ft) else None
    if idx is None or agl[idx] > rcfg.flare_agl_ft:
        return len(df) - 1, False
    return int(idx), True


def main():
    paths.ensure_dirs()
    with open(paths.CONFIG_DIR / "preprocessing.yaml") as f:
        cfg = yaml.safe_load(f)
    with open(paths.CONFIG_DIR / "episode_overrides.yaml") as f:
        overrides = yaml.safe_load(f)
    runway02 = RunwayConfig.from_yaml(paths.CONFIG_DIR / "runway.yaml")
    runway29 = RunwayConfig.from_yaml(paths.CONFIG_DIR / "runway_29.yaml")

    episodes = pd.read_csv(paths.MANIFESTS_DIR / "episodes.csv")
    episodes = episodes[episodes.eligible_for_training].reset_index(drop=True)

    boundary_rows = []
    transient_rows = []
    row_phase_rows = []
    transition_rows = []
    bounce_rows = []
    review_flag_rows = []

    circuit_l2 = overrides["circuito-completo"]["landing_02"]

    diag_dir = paths.FIGURES_DIR / "touchdown_diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    for _, ep in episodes.iterrows():
        eid = ep["episode_id"]
        df_val = pd.read_csv(paths.VALIDATED_DIR / f"{eid}.csv")
        df10 = pd.read_csv(paths.PROCESSED_10HZ_DIR / f"{eid}.csv")
        n_val = len(df_val)
        rcfg = runway02 if ep["runway_id"] == 2 or str(ep["runway_id"]) == "02" else runway29

        flare_idx_val, flare_found = detect_flare_idx(df_val, rcfg)
        if not flare_found:
            review_flag_rows.append({"episode_id": eid, "flag": "flare_not_detected_via_agl_threshold_REVIEW_NEEDED"})
        flare_s = float(df_val[C.SIM_TIME].iloc[flare_idx_val])

        t10 = df10[C.SIM_TIME].to_numpy()
        start_idx_10 = int(np.searchsorted(t10, flare_s, side="left"))
        start_idx_10 = min(start_idx_10, len(df10) - 1)
        end_idx_10 = len(df10) - 1

        fsm = run_landing_fsm(df10, start_idx_10, end_idx_10, cfg)
        validate_episode(df10, fsm, cfg, eid)

        for flag in fsm["review_flags"]:
            review_flag_rows.append({"episode_id": eid, "flag": flag})
        for ev in fsm["events"]:
            transition_rows.append({"episode_id": eid, **ev})
        for b in fsm["bounce_records"]:
            bounce_rows.append({"episode_id": eid, **b})
        for i, label in enumerate(fsm["row_phase"]):
            if label is not None:
                idx = start_idx_10 + i
                row_phase_rows.append({
                    "episode_id": eid, "frame_10hz": int(df10["frame"].iloc[idx]),
                    "sim_time_s": float(t10[idx]), "row_phase": label,
                })

        def _t10(idx):
            return float(t10[idx]) if idx is not None else None

        row = {
            "episode_id": eid, "session_id": ep["session_id"],
            "t_episode_start_s": float(t10[0]), "t_episode_end_s": float(t10[-1]),
            "touchdown_confirmed": bool(fsm["touchdown_confirmed"]),
            "contact_candidate_s": _t10(fsm["contact_candidate_first_idx"]),
            "touchdown_confirmed_s": _t10(fsm["touchdown_confirmed_idx"]),
            "rollout_s": _t10(fsm["rollout_idx"]),
            "complete_s": _t10(fsm["complete_idx"]),
            "complete_reason": fsm["complete_reason"],
            "flare_s": flare_s,
            "is_standardized_start": bool(ep["has_known_initialization_transient"]) if eid != "circuito-completo__landing-02" else False,
            "approach_maneuvering_s": None,
            "final_approach_established_s": None,
            "transient_end_s": None,
            "transient_detected": None,
        }

        if eid == "circuito-completo__landing-02":
            raw_df_full = pd.read_csv(paths.RAW_DIR / "circuito-completo.csv")
            raw_df_full = raw_df_full.sort_values(C.SIM_TIME, kind="mergesort").reset_index(drop=True)
            row["approach_maneuvering_s"] = float(raw_df_full[C.SIM_TIME].iloc[circuit_l2["approach_maneuvering_start_idx"]])
            row["final_approach_established_s"] = float(raw_df_full[C.SIM_TIME].iloc[circuit_l2["final_approach_established_idx"]])
        elif row["is_standardized_start"]:
            touchdown_idx_for_transient = (
                np.searchsorted(df_val[C.SIM_TIME].to_numpy(), row["touchdown_confirmed_s"])
                if row["touchdown_confirmed_s"] is not None else n_val - 1
            )
            tr = detect_transient_end(df_val, cfg, rcfg, int(touchdown_idx_for_transient))
            tv = df_val[C.SIM_TIME].to_numpy()
            row["transient_end_s"] = float(tv[tr["transient_end_idx"]]) if tr["detected"] else None
            row["transient_detected"] = tr["detected"]
            dist_to_thr_m = -tr["along_track_m_at_end"] if tr["along_track_m_at_end"] is not None else None
            transient_rows.append({
                "episode_id": eid,
                "detected": tr["detected"], "reason": tr["reason"],
                "t_start_s": float(tv[0]),
                "t_transient_end_s": row["transient_end_s"],
                "duration_removed_s": (row["transient_end_s"] - float(tv[0])) if tr["detected"] else None,
                "n_samples_removed_from_baseline_core": int(
                    (df_val[C.SIM_TIME] < row["transient_end_s"]).sum()) if tr["detected"] else 0,
                "n_samples_total": n_val,
                "airspeed_kt_before": tr["airspeed_before"], "airspeed_kt_after": tr["airspeed_after"],
                "flaps_before": tr["flaps_before"], "flaps_after": tr["flaps_after"],
                "distance_to_threshold_m_at_end": dist_to_thr_m,
                "agl_ft_at_end": tr["agl_ft_at_end"],
            })

        boundary_rows.append(row)
        print(f"[phases] {eid}: confirmed={row['touchdown_confirmed']} "
              f"cand={row['contact_candidate_s']} touchdown={row['touchdown_confirmed_s']} "
              f"rollout={row['rollout_s']} complete={row['complete_s']} ({row['complete_reason']}) "
              f"flare={row['flare_s']:.1f}s transient_end={row['transient_end_s']} "
              f"n_review_flags={len(fsm['review_flags'])}")

        # ---- touchdown diagnostic plot (spec section 6) ----
        _plot_touchdown_diagnostics(df10, fsm, eid, diag_dir, flare_s)

    boundaries = pd.DataFrame(boundary_rows)
    boundaries.to_csv(paths.MANIFESTS_DIR / "episode_phase_boundaries.csv", index=False)

    transient_diag = pd.DataFrame(transient_rows)
    transient_diag.to_csv(paths.MANIFESTS_DIR / "transient_diagnostics.csv", index=False)

    pd.DataFrame(row_phase_rows).to_csv(paths.MANIFESTS_DIR / "landing_fsm_row_phase.csv", index=False)
    pd.DataFrame(transition_rows).to_csv(paths.MANIFESTS_DIR / "phase_transitions.csv", index=False)
    pd.DataFrame(bounce_rows).to_csv(paths.MANIFESTS_DIR / "bounce_events.csv", index=False)
    pd.DataFrame(review_flag_rows).to_csv(paths.MANIFESTS_DIR / "phase_review_flags.csv", index=False)

    n_not_detected = (~transient_diag.detected).sum() if len(transient_diag) else 0
    n_unconfirmed = (~boundaries.touchdown_confirmed).sum()
    print(f"\nWrote episode_phase_boundaries.csv ({len(boundaries)} rows), "
          f"transient_diagnostics.csv ({len(transient_diag)} rows, {n_not_detected} not auto-detected), "
          f"landing_fsm_row_phase.csv ({len(row_phase_rows)} rows), "
          f"phase_transitions.csv ({len(transition_rows)} rows), "
          f"bounce_events.csv ({len(bounce_rows)} rows), "
          f"phase_review_flags.csv ({len(review_flag_rows)} rows). "
          f"{n_unconfirmed} episode(s) with touchdown never auto-confirmed (manual review required).")

    for eid in ["voo-condicoes-normais", "circuito-completo__landing-01"]:
        df = pd.read_csv(paths.VALIDATED_DIR / f"{eid}.csv")
        t = df[C.SIM_TIME].to_numpy()
        b = boundaries[boundaries.episode_id == eid].iloc[0]
        fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
        axes[0].plot(t, df[C.AIRSPEED_KT]); axes[0].set_ylabel("Airspeed (kt)")
        axes[0].axhspan(55, 75, color="green", alpha=0.1)
        axes[1].plot(t, df[C.FLAPS_CMD]); axes[1].set_ylabel("Flaps [0..1]")
        axes[2].plot(t, df[C.ALT_AGL_FT]); axes[2].set_ylabel("AGL (ft)")
        for ax in axes:
            if pd.notna(b["transient_end_s"]):
                ax.axvline(b["transient_end_s"], color="red", linestyle="--", label="transient_end")
            if pd.notna(b["touchdown_confirmed_s"]):
                ax.axvline(b["touchdown_confirmed_s"], color="black", linestyle=":", label="touchdown")
        axes[0].legend(fontsize=7)
        axes[2].set_xlabel("sim_time (s)")
        fig.suptitle(f"Initialization transient diagnostic: {eid}")
        fig.tight_layout()
        out = paths.FIGURES_DIR / f"transient_{eid}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"Wrote {out}")


def _plot_touchdown_diagnostics(df10: pd.DataFrame, fsm: dict, eid: str, out_dir: Path, flare_s: float):
    t = df10[C.SIM_TIME].to_numpy()
    cand_idx = fsm["contact_candidate_first_idx"]
    lo_t = flare_s - 3.0
    hi_t = (t[fsm["complete_idx"]] + 3.0) if fsm["complete_idx"] is not None else (
        t[cand_idx] + 15.0 if cand_idx is not None else t[-1])
    mask = (t >= lo_t) & (t <= hi_t)
    if mask.sum() < 2:
        mask = np.ones(len(t), dtype=bool)
    sub = df10.loc[mask]
    ts = sub[C.SIM_TIME].to_numpy()

    phase_full = np.full(len(df10), "FLARE", dtype=object)
    base_offset = int(np.searchsorted(t, flare_s, side="left"))  # matches the main loop's start_idx_10
    for i, label in enumerate(fsm["row_phase"]):
        if label is not None:
            phase_full[base_offset + i] = label
    phase_codes = pd.Series(phase_full).astype("category")

    fig, axes = plt.subplots(9, 1, figsize=(12, 20), sharex=True)
    axes[0].plot(ts, sub[C.ALT_AGL_FT]); axes[0].set_ylabel("AGL (ft)")
    axes[1].plot(ts, sub[C.VERTICAL_SPEED_FPS]); axes[1].set_ylabel("VS (fps)")
    axes[2].plot(ts, sub[C.GROUNDSPEED_KT]); axes[2].set_ylabel("GS (kt)")
    axes[3].plot(ts, sub[C.gear_wow(0)], label="nose")
    axes[3].plot(ts, sub[C.gear_wow(1)], label="main_left")
    axes[3].plot(ts, sub[C.gear_wow(2)], label="main_right")
    axes[3].legend(fontsize=7); axes[3].set_ylabel("WOW")
    axes[4].plot(ts, sub[C.gear_compression(0)], label="nose")
    axes[4].plot(ts, sub[C.gear_compression(1)], label="main_left")
    axes[4].plot(ts, sub[C.gear_compression(2)], label="main_right")
    axes[4].legend(fontsize=7); axes[4].set_ylabel("compression-norm")
    axes[5].step(ts, phase_codes.cat.codes.to_numpy()[mask], where="post")
    axes[5].set_yticks(range(len(phase_codes.cat.categories)))
    axes[5].set_yticklabels(phase_codes.cat.categories, fontsize=6)
    axes[5].set_ylabel("phase")
    axes[6].plot(ts, sub[C.AILERON_CMD], label="aileron")
    axes[6].plot(ts, sub[C.RUDDER_CMD], label="rudder")
    axes[6].legend(fontsize=7); axes[6].set_ylabel("lat cmds")
    axes[7].plot(ts, sub[C.ELEVATOR_CMD], label="elevator")
    axes[7].legend(fontsize=7); axes[7].set_ylabel("elevator")
    axes[8].plot(ts, sub[C.THROTTLE_ENGINE0], label="throttle")
    axes[8].legend(fontsize=7); axes[8].set_ylabel("throttle")
    axes[8].set_xlabel("sim_time (s)")
    fig.suptitle(f"Touchdown diagnostics: {eid}")
    fig.tight_layout()
    out = out_dir / f"{eid}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
