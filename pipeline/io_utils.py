"""Robust reading of the raw FlightGear property-tree CSV exports.

Discovered during manual audit (2026-09-17):
- Most files use ',' as field delimiter and '.' as decimal separator.
- Two files (voo-com-rajada.csv, voo-normal2.csv) use ';' as field delimiter
  and ',' as decimal separator (pt-BR / European locale export). This is
  detected automatically per-file, never hardcoded by filename.
- No BOM, plain ASCII/UTF-8 headers observed in all 15 files.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class CsvFormat:
    delimiter: str
    decimal: str
    encoding: str


def sha256_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def detect_encoding(path: Path) -> str:
    with open(path, "rb") as f:
        head = f.read(1 << 20)
    try:
        head.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        return "latin-1"


def detect_format(path: Path) -> CsvFormat:
    encoding = detect_encoding(path)
    with open(path, "r", encoding=encoding, errors="replace") as f:
        header = f.readline()
    n_comma = header.count(",")
    n_semi = header.count(";")
    if n_semi > n_comma:
        return CsvFormat(delimiter=";", decimal=",", encoding=encoding)
    return CsvFormat(delimiter=",", decimal=".", encoding=encoding)


def read_flight_csv(path: Path, usecols=None) -> tuple[pd.DataFrame, CsvFormat]:
    """Read one raw flight CSV, auto-detecting delimiter/decimal/encoding.

    Returns the dataframe with original property-path column names untouched,
    plus the detected CsvFormat for logging/auditing.
    """
    fmt = detect_format(path)
    df = pd.read_csv(
        path,
        sep=fmt.delimiter,
        decimal=fmt.decimal,
        encoding=fmt.encoding,
        usecols=usecols,
        low_memory=False,
    )
    return df, fmt


def normalize_test_filename(name: str) -> str:
    """Lower-case, strip extension, collapse '-'/'_' for tolerant matching."""
    stem = Path(name).stem.lower()
    return stem.replace("_", "").replace("-", "")


TEST_FLIGHT_CANONICAL = normalize_test_filename("voo-inicio-teste.csv")


def is_pilot_familiarization_file(name: str) -> bool:
    return normalize_test_filename(name) == TEST_FLIGHT_CANONICAL
