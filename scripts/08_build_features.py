#!/usr/bin/env python3
"""Stage 8: build unnormalized feature tables (lateral 11-in/2-out,
longitudinal 12-in/2-out) + phase/mask columns for every episode, and derive
the baseline_core / full_approach / recovery_ablation variants by row
filtering.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import paths, columns as C
from pipeline.runway import RunwayConfig
from pipeline.features import (
    build_episode_features, LATERAL_FEATURE_ORDER, LATERAL_LABEL_ORDER,
    LONGITUDINAL_FEATURE_ORDER, LONGITUDINAL_LABEL_ORDER, FEATURE_SCHEMA_VERSION,
)


def main():
    paths.ensure_dirs()
    with open(paths.CONFIG_DIR / "preprocessing.yaml") as f:
        cfg = yaml.safe_load(f)
    runway02 = RunwayConfig.from_yaml(paths.CONFIG_DIR / "runway.yaml")
    runway29 = RunwayConfig.from_yaml(paths.CONFIG_DIR / "runway_29.yaml")

    episodes = pd.read_csv(paths.MANIFESTS_DIR / "episodes.csv")
    boundaries = pd.read_csv(paths.MANIFESTS_DIR / "episode_phase_boundaries.csv").set_index("episode_id")
    row_phase = pd.read_csv(paths.MANIFESTS_DIR / "landing_fsm_row_phase.csv")

    schema = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "lateral_feature_order": LATERAL_FEATURE_ORDER,
        "lateral_label_order": LATERAL_LABEL_ORDER,
        "longitudinal_feature_order": LONGITUDINAL_FEATURE_ORDER,
        "longitudinal_label_order": LONGITUDINAL_LABEL_ORDER,
        "column_prefixes": {"lat__": "lateral feature", "laty__": "lateral label",
                              "lon__": "longitudinal feature", "lony__": "longitudinal label"},
    }
    with open(paths.FEATURES_DIR / "schema.json", "w") as f:
        json.dump(schema, f, indent=2)

    summary_rows = []
    for _, ep in episodes.iterrows():
        eid = ep["episode_id"]
        is_eligible = bool(ep["eligible_for_training"])
        src_dir = paths.PROCESSED_10HZ_DIR if is_eligible else paths.PROCESSED_EXCLUDED_DIR
        df10 = pd.read_csv(src_dir / f"{eid}.csv")
        rcfg = runway29 if str(ep["runway_id"]) == "29" else runway02

        if eid in boundaries.index:
            b = boundaries.loc[eid].to_dict()
        else:
            # voo-inicio-teste: not segmented (no meaningful landing structure
            # expected); build a degenerate boundary set spanning the whole
            # file so features can still be computed for parser-testing only.
            b = {
                "flare_s": df10[C.SIM_TIME].iloc[-1] + 1, "contact_candidate_s": df10[C.SIM_TIME].iloc[-1] + 1,
                "touchdown_confirmed_s": df10[C.SIM_TIME].iloc[-1] + 1, "rollout_s": df10[C.SIM_TIME].iloc[-1] + 1,
                "complete_s": df10[C.SIM_TIME].iloc[-1] + 1, "transient_end_s": None,
                "approach_maneuvering_s": None, "final_approach_established_s": None,
            }

        is_standardized_start = bool(b["is_standardized_start"]) if "is_standardized_start" in b else False
        has_am_fa = pd.notna(b.get("approach_maneuvering_s"))

        overlay = None
        ep_overlay = row_phase[row_phase.episode_id == eid]
        if len(ep_overlay):
            overlay_by_frame = dict(zip(ep_overlay.frame_10hz, ep_overlay.row_phase))
            overlay = df10["frame"].map(overlay_by_frame)

        feat = build_episode_features(df10, rcfg, cfg, b, is_standardized_start, has_am_fa, overlay)
        feat.insert(0, "eligible_for_training", is_eligible)

        out_unnorm = paths.FEATURES_UNNORM_DIR / f"{eid}.csv"
        feat.to_csv(out_unnorm, index=False)

        n_total = len(feat)
        if is_eligible:
            bc = feat[feat.baseline_core == 1]
            fa = feat[feat.full_approach == 1]
            ra = feat[feat.recovery_ablation == 1]
            bc.to_csv(paths.FEATURES_BASELINE_CORE_DIR / f"{eid}.csv", index=False)
            fa.to_csv(paths.FEATURES_FULL_APPROACH_DIR / f"{eid}.csv", index=False)
            ra.to_csv(paths.FEATURES_RECOVERY_ABLATION_DIR / f"{eid}.csv", index=False)
            n_bc, n_fa, n_ra = len(bc), len(fa), len(ra)
        else:
            n_bc = n_fa = n_ra = 0

        summary_rows.append({
            "episode_id": eid, "eligible_for_training": is_eligible,
            "n_unnormalized": n_total, "n_baseline_core": n_bc,
            "n_full_approach": n_fa, "n_recovery_ablation": n_ra,
        })
        print(f"[features] {eid}: unnorm={n_total} baseline_core={n_bc} full_approach={n_fa} recovery_ablation={n_ra}")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(paths.MANIFESTS_DIR / "feature_build_summary.csv", index=False)
    print(f"\nWrote schema to {paths.FEATURES_DIR / 'schema.json'} and per-episode feature tables.")


if __name__ == "__main__":
    main()
