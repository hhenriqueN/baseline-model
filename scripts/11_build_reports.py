#!/usr/bin/env python3
"""Stage 11: final reports -- dataset_summary.csv, phase distribution,
full_episodes vs baseline_core vs recovery_ablation comparison plot, fold
summary, and figures referenced by cleaning_report.md.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import paths, columns as C


def main():
    episodes = pd.read_csv(paths.MANIFESTS_DIR / "episodes.csv")
    boundaries = pd.read_csv(paths.MANIFESTS_DIR / "episode_phase_boundaries.csv").set_index("episode_id")
    transient = pd.read_csv(paths.MANIFESTS_DIR / "transient_diagnostics.csv").set_index("episode_id")
    feat_summary = pd.read_csv(paths.MANIFESTS_DIR / "feature_build_summary.csv").set_index("episode_id")
    resample_diag = pd.read_csv(paths.MANIFESTS_DIR / "resampling_diagnostics.csv").set_index("episode_id")

    rows = []
    for _, ep in episodes.iterrows():
        eid = ep["episode_id"]
        row = {
            "episode_id": eid, "session_id": ep["session_id"], "source_file": ep["source_file"],
            "scenario_id": ep["scenario_id"], "eligible_for_training": ep["eligible_for_training"],
            "runway_id": ep["runway_id"], "n_samples_raw": ep["n_samples_raw"],
            "duration_s_raw": round(ep["duration_s_raw"], 1),
        }
        if eid in resample_diag.index:
            row["n_samples_10hz"] = int(resample_diag.loc[eid, "n_resampled_samples"])
            row["effective_fs_raw_hz"] = round(resample_diag.loc[eid, "effective_fs_raw_hz"], 2)
        if eid in feat_summary.index:
            fs = feat_summary.loc[eid]
            row["n_unnormalized"] = fs["n_unnormalized"]
            row["n_baseline_core"] = fs["n_baseline_core"]
            row["n_full_approach"] = fs["n_full_approach"]
            row["n_recovery_ablation"] = fs["n_recovery_ablation"]
        if eid in boundaries.index:
            b = boundaries.loc[eid]
            row["touchdown_s"] = round(b["touchdown_confirmed_s"], 2)
            row["complete_s"] = round(b["complete_s"], 2)
            row["complete_reason"] = b["complete_reason"]
        if eid in transient.index:
            tr = transient.loc[eid]
            row["transient_detected"] = tr["detected"]
            row["transient_duration_removed_s"] = round(tr["duration_removed_s"], 2) if pd.notna(tr["duration_removed_s"]) else None
            row["airspeed_kt_before_transient"] = round(tr["airspeed_kt_before"], 1)
            row["airspeed_kt_after_transient"] = round(tr["airspeed_kt_after"], 1) if pd.notna(tr["airspeed_kt_after"]) else None
            row["flaps_before_transient"] = round(tr["flaps_before"], 2)
            row["flaps_after_transient"] = round(tr["flaps_after"], 2) if pd.notna(tr["flaps_after"]) else None
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(paths.REPORTS_DIR / "dataset_summary.csv", index=False)
    print(f"Wrote {paths.REPORTS_DIR / 'dataset_summary.csv'} ({len(summary)} episodes)")

    # ---- phase distribution across all eligible episodes (baseline_core view uses unnormalized to see all phases) ----
    all_phase_counts = {}
    for eid in episodes.loc[episodes.eligible_for_training, "episode_id"]:
        df = pd.read_csv(paths.FEATURES_UNNORM_DIR / f"{eid}.csv", usecols=["phase"])
        vc = df["phase"].value_counts()
        for k, v in vc.items():
            all_phase_counts[k] = all_phase_counts.get(k, 0) + v
    fig, ax = plt.subplots(figsize=(9, 5))
    phases = sorted(all_phase_counts, key=lambda k: -all_phase_counts[k])
    ax.bar(phases, [all_phase_counts[p] for p in phases])
    ax.set_ylabel("Samples (10 Hz)")
    ax.set_title("Phase distribution across all eligible episodes")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(paths.FIGURES_DIR / "phase_distribution.png", dpi=130)
    plt.close(fig)
    print(f"Wrote {paths.FIGURES_DIR / 'phase_distribution.png'}")

    # ---- full_episodes (unnormalized) vs baseline_core vs recovery_ablation comparison ----
    eid = "voo-condicoes-normais"
    unnorm = pd.read_csv(paths.FEATURES_UNNORM_DIR / f"{eid}.csv")
    bc = pd.read_csv(paths.FEATURES_BASELINE_CORE_DIR / f"{eid}.csv")
    ra = pd.read_csv(paths.FEATURES_RECOVERY_ABLATION_DIR / f"{eid}.csv")
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes[0].plot(unnorm[C.SIM_TIME], unnorm["lon__airspeed_kt"], color="lightgray", linewidth=4, label="full_episodes (unnormalized)")
    axes[0].plot(ra[C.SIM_TIME], ra["lon__airspeed_kt"], color="tab:orange", linestyle="--", label="recovery_ablation")
    axes[0].plot(bc[C.SIM_TIME], bc["lon__airspeed_kt"], color="tab:blue", label="baseline_core")
    axes[0].set_ylabel("Airspeed (kt)")
    axes[0].legend(fontsize=8)
    axes[1].plot(unnorm[C.SIM_TIME], unnorm["lon__flaps_state_norm"], color="lightgray", linewidth=4)
    axes[1].plot(ra[C.SIM_TIME], ra["lon__flaps_state_norm"], color="tab:orange", linestyle="--")
    axes[1].plot(bc[C.SIM_TIME], bc["lon__flaps_state_norm"], color="tab:blue")
    axes[1].set_ylabel("Flaps [0..1]")
    axes[1].set_xlabel("sim_time (s)")
    fig.suptitle(f"full_episodes vs baseline_core vs recovery_ablation: {eid}\n"
                 f"(baseline_core excludes the initialization_transient at episode start)")
    fig.tight_layout()
    fig.savefig(paths.FIGURES_DIR / f"variant_comparison_{eid}.png", dpi=130)
    plt.close(fig)
    print(f"Wrote {paths.FIGURES_DIR / f'variant_comparison_{eid}.png'}")

    # ---- variant comparison for landing-02 (full_approach vs baseline_core) ----
    eid2 = "circuito-completo__landing-02"
    unnorm2 = pd.read_csv(paths.FEATURES_UNNORM_DIR / f"{eid2}.csv")
    bc2 = pd.read_csv(paths.FEATURES_BASELINE_CORE_DIR / f"{eid2}.csv")
    fa2 = pd.read_csv(paths.FEATURES_FULL_APPROACH_DIR / f"{eid2}.csv")
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes[0].plot(unnorm2[C.SIM_TIME], unnorm2["lon__agl_ft"], color="lightgray", linewidth=4, label="full episode (incl. circuit)")
    axes[0].plot(fa2[C.SIM_TIME], fa2["lon__agl_ft"], color="tab:orange", linestyle="--", label="full_approach")
    axes[0].plot(bc2[C.SIM_TIME], bc2["lon__agl_ft"], color="tab:blue", label="baseline_core (final approach onward)")
    axes[0].set_ylabel("AGL (ft)")
    axes[0].legend(fontsize=8)
    axes[1].plot(unnorm2[C.SIM_TIME], unnorm2["phase"].astype("category").cat.codes, color="black")
    axes[1].set_ylabel("phase (categorical code)")
    axes[1].set_xlabel("sim_time (s)")
    fig.suptitle(f"full_approach vs baseline_core: {eid2}")
    fig.tight_layout()
    fig.savefig(paths.FIGURES_DIR / f"variant_comparison_{eid2}.png", dpi=130)
    plt.close(fig)
    print(f"Wrote {paths.FIGURES_DIR / f'variant_comparison_{eid2}.png'}")

    # ---- fold/session summary ----
    fold_summary = pd.read_csv(paths.SPLITS_DIR / "cv_folds_summary.csv")
    print("\nFold summary:")
    print(fold_summary.to_string(index=False))


if __name__ == "__main__":
    main()
