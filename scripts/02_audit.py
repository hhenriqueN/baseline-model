#!/usr/bin/env python3
"""Stage 2: full audit of the raw data (per-file timing QA + column catalog
+ duplicate detection + control-range checks).

Writes:
  data/manifests/columns_catalog.csv
  data/manifests/duplicate_columns.csv
  data/manifests/file_timing_audit.csv
  data/manifests/control_range_audit.csv
  data/reports/qa_summary.md
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import paths, audit
from pipeline.io_utils import read_flight_csv


def main():
    paths.ensure_dirs()
    manifest = pd.read_csv(paths.MANIFESTS_DIR / "raw_files.csv")

    timing_rows = []
    frames = []
    for _, row in manifest.iterrows():
        df, fmt = read_flight_csv(row["raw_path"])
        # sort defensively by sim_time before timing audit (spec step 7)
        df = df.sort_values("sim_time", kind="mergesort").reset_index(drop=True)
        timing_rows.append(audit.audit_file_timing(df, row["source_file"]))
        df.insert(0, "source_file", row["source_file"])
        frames.append(df)
        print(f"[audit] {row['source_file']}: rows={len(df)} monotonic_after_sort_check_done")

    all_df = pd.concat(frames, ignore_index=True)
    timing_df = pd.DataFrame(timing_rows)
    timing_df.to_csv(paths.MANIFESTS_DIR / "file_timing_audit.csv", index=False)

    eligible_files = set(manifest.loc[manifest.eligible_for_training, "source_file"])
    baseline_mask = all_df["source_file"].isin(eligible_files).to_numpy()

    catalog = audit.build_columns_catalog(all_df, baseline_mask)

    dup_unit = audit.verify_unit_duplicates(all_df)
    dup_exact = audit.find_exact_duplicate_columns(all_df)
    dup_all = pd.concat([dup_unit, dup_exact], ignore_index=True, sort=False)

    # reflect confirmed duplicates back into the catalog
    for _, r in dup_all.iterrows():
        if r.get("confirmed_duplicate"):
            drop_col = r["drop_column"]
            if drop_col in catalog["column_name"].values:
                idx = catalog.index[catalog["column_name"] == drop_col][0]
                catalog.loc[idx, "duplicate_of"] = r["keep_column"]
                catalog.loc[idx, "conversion_relation"] = r["formula"]
                if catalog.loc[idx, "role"] not in ("label",):
                    catalog.loc[idx, "role"] = "excluded"
                    catalog.loc[idx, "exclusion_reason"] = "duplicate_unit_or_exact"
                    catalog.loc[idx, "selected_for_baseline"] = False

    catalog.to_csv(paths.MANIFESTS_DIR / "columns_catalog.csv", index=False)
    dup_all.to_csv(paths.MANIFESTS_DIR / "duplicate_columns.csv", index=False)

    ctrl = audit.check_control_ranges(all_df)
    ctrl.to_csv(paths.MANIFESTS_DIR / "control_range_audit.csv", index=False)

    # ---- exclusions.csv: every column NOT selected for baseline, with reason
    excl = catalog.loc[~catalog.selected_for_baseline, ["column_name", "role", "exclusion_reason", "duplicate_of"]]
    excl.to_csv(paths.MANIFESTS_DIR / "exclusions.csv", index=False)

    # ---- qa_summary.md
    lines = []
    lines.append("# QA Summary — Raw Data Audit\n")
    lines.append(f"Generated from {len(manifest)} raw files, {len(all_df)} total rows, {all_df.shape[1]-1} data columns.\n")
    lines.append("## Per-file timing audit\n")
    lines.append(timing_df.to_markdown(index=False))
    lines.append("\n\n## Control range violations\n")
    viol = ctrl[(ctrl.n_below_range > 0) | (ctrl.n_above_range > 0)]
    if len(viol):
        lines.append(viol.to_markdown(index=False))
    else:
        lines.append("No control-range violations detected across the curated control columns.")
    lines.append("\n\n## Confirmed duplicate columns\n")
    confirmed = dup_all[dup_all.confirmed_duplicate == True]  # noqa: E712
    if len(confirmed):
        lines.append(confirmed.to_markdown(index=False))
    else:
        lines.append("None confirmed.")
    lines.append("\n\n## Candidate duplicate pairs NOT confirmed (kept, flagged)\n")
    not_confirmed = dup_unit[dup_unit.confirmed_duplicate == False]  # noqa: E712
    if len(not_confirmed):
        lines.append(not_confirmed.to_markdown(index=False))
    else:
        lines.append("All curated candidates were confirmed.")
    lines.append("\n\n## Constant / near-constant columns (excluded)\n")
    const_cols = catalog[catalog.is_constant | catalog.is_near_constant]
    lines.append(f"{len(const_cols)} columns out of {len(catalog)} are constant or near-constant across the baseline-eligible files.")
    lines.append("\n\n## Role distribution\n")
    lines.append(catalog.role.value_counts().to_frame("count").to_markdown())

    with open(paths.REPORTS_DIR / "qa_summary.md", "w") as f:
        f.write("\n".join(str(x) for x in lines))

    print("\nWrote:")
    print(" -", paths.MANIFESTS_DIR / "columns_catalog.csv")
    print(" -", paths.MANIFESTS_DIR / "duplicate_columns.csv")
    print(" -", paths.MANIFESTS_DIR / "file_timing_audit.csv")
    print(" -", paths.MANIFESTS_DIR / "control_range_audit.csv")
    print(" -", paths.MANIFESTS_DIR / "exclusions.csv")
    print(" -", paths.REPORTS_DIR / "qa_summary.md")

    # sanity prints for the required source columns (section 8)
    from pipeline import columns as C
    missing_required = [c for c in C.REQUIRED_SOURCE_COLUMNS if c not in all_df.columns]
    if missing_required:
        raise RuntimeError(f"Missing required source columns in raw headers: {missing_required}")
    print("\nAll required section-8 source columns are present in the raw headers.")


if __name__ == "__main__":
    main()
