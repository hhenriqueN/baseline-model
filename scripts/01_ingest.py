#!/usr/bin/env python3
"""Stage 1: inventory raw CSVs, copy byte-for-byte into data/raw/, verify SHA-256.

Never modifies, moves or deletes the original CSVs sitting in the project
root. Produces data/manifests/raw_files.csv.
"""
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import paths
from pipeline.io_utils import sha256_of_file, detect_format, is_pilot_familiarization_file

SOURCE_DIR = paths.ROOT


def main():
    paths.ensure_dirs()
    source_csvs = sorted(
        p for p in SOURCE_DIR.glob("*.csv")
    )
    fallback_to_raw_dir = False
    if not source_csvs:
        # The project may have been relocated after the original loose
        # source CSVs (that used to sit next to pipeline/scripts/data at
        # SOURCE_DIR) were copied into data/raw/. data/raw/ is immutable and
        # already holds the verified byte-for-byte copies, so treat it as
        # the canonical source in that case rather than failing -- this
        # never re-derives or alters data/raw/ contents, only re-derives the
        # manifest's path bookkeeping.
        source_csvs = sorted(p for p in paths.RAW_DIR.glob("*.csv"))
        fallback_to_raw_dir = True
        if not source_csvs:
            raise RuntimeError(f"No source CSVs found in {SOURCE_DIR} or {paths.RAW_DIR}")
        print(f"[ingest] no loose source CSVs at {SOURCE_DIR}; treating already-ingested "
              f"{paths.RAW_DIR} as canonical (raw files are immutable, never regenerated).")

    rows = []
    for src in source_csvs:
        dst = paths.RAW_DIR / src.name
        sha_before = sha256_of_file(src)

        if fallback_to_raw_dir:
            copy_action = "source_missing_raw_dir_already_canonical"
        elif dst.exists() and sha256_of_file(dst) == sha_before:
            copy_action = "already_present_verified"
        else:
            shutil.copy2(src, dst)
            copy_action = "copied"

        sha_after = sha256_of_file(dst)
        integrity_ok = (sha_before == sha_after)
        if not integrity_ok:
            raise RuntimeError(f"SHA-256 mismatch copying {src} -> {dst}")

        fmt = detect_format(src)
        is_test_flight = is_pilot_familiarization_file(src.name)

        # cheap row/col count without full parse (header + line count)
        with open(src, "r", encoding=fmt.encoding, errors="replace") as f:
            header = f.readline()
        n_cols = header.count(fmt.delimiter) + 1
        with open(src, "rb") as f:
            n_lines = sum(1 for _ in f)
        n_rows = n_lines - 1  # minus header

        rows.append({
            "source_file": src.name,
            "source_path": str(src),
            "raw_path": str(dst),
            "sha256_source": sha_before,
            "sha256_raw_copy": sha_after,
            "integrity_ok": integrity_ok,
            "copy_action": copy_action,
            "size_bytes": src.stat().st_size,
            "n_rows_incl_header_minus1": n_rows,
            "n_cols_header": n_cols,
            "delimiter": fmt.delimiter,
            "decimal": fmt.decimal,
            "encoding": fmt.encoding,
            "eligible_for_training": not is_test_flight,
            "exclusion_reason": "pilot_familiarization" if is_test_flight else "",
        })
        print(f"[ok] {src.name}: sha256 verified, delimiter={fmt.delimiter!r} decimal={fmt.decimal!r} "
              f"rows={n_rows} cols={n_cols} eligible={not is_test_flight}")

    manifest = pd.DataFrame(rows)
    out_path = paths.MANIFESTS_DIR / "raw_files.csv"
    manifest.to_csv(out_path, index=False)
    print(f"\nWrote manifest: {out_path} ({len(manifest)} files)")

    n_test = manifest["exclusion_reason"].eq("pilot_familiarization").sum()
    if n_test != 1:
        print(f"WARNING: expected exactly 1 pilot_familiarization file, found {n_test}")


if __name__ == "__main__":
    main()
