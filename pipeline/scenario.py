"""Scenario_id assignment: filename-based, cross-checked against wind/turbulence
telemetry (spec section 17: "confirme os cenarios pelos dados... nao confie
apenas no filename").
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline import columns as C
from pipeline.io_utils import normalize_test_filename

FILENAME_SCENARIO_MAP = {
    "circuitocompleto": "circuit",
    "turbulencia05": "turbulence_05",
    "turbulencia08": "turbulence_08",
    "voo2com16nos": "wind_16kt",
    "voo2comrajada": "gust",
    "voo3com16nos": "wind_16kt",
    "voocom16nos": "wind_16kt",
    "voocom8nos": "wind_8kt",
    "voocomrajadavindodadireita": "gust_direction_variant",
    "voocomrajada": "gust",
    "voocondicoesnormais": "normal",
    "vooiniciotest": "unknown",  # pilot familiarization; excluded anyway
    "voonormal2": "normal",
    "voonormal3": "normal",
    "voo2com8nos": "wind_8kt",
}


def scenario_from_filename(filename: str) -> str:
    key = normalize_test_filename(filename)
    return FILENAME_SCENARIO_MAP.get(key, "unknown")


def confirm_scenario_signals(df: pd.DataFrame) -> dict:
    """Compute the environment statistics used to cross-check the
    filename-derived scenario_id (does not itself decide the label)."""
    ws = df[C.WIND_SPEED_KT]
    tb = df[C.TURBULENCE_MAGNITUDE]
    wh = df[C.WIND_FROM_HEADING_DEG]
    return {
        "qa_wind_speed_kt_mean": float(ws.mean()),
        "qa_wind_speed_kt_std": float(ws.std()),
        "qa_wind_from_heading_std_deg": float(wh.std()),
        "qa_turbulence_magnitude_max": float(tb.max()),
    }
