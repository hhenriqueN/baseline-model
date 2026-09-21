"""Stage 9: session-grouped cross-validation splits + fixed diagnostic holdout
(spec section 17). NEVER split by row; always by session_id, and the two
circuito-completo landings always travel together (they share session_id by
construction, see episodes.csv).
"""
from __future__ import annotations

import hashlib

import pandas as pd


def select_fixed_diagnostic_holdout(episodes: pd.DataFrame, scenario: str = "wind_16kt") -> str:
    """Deterministic pick: sha256(session_id), ascending, first one.
    Reproducible without relying on any random seed."""
    candidates = sorted(episodes.loc[episodes.scenario_id == scenario, "session_id"].unique())
    ranked = sorted((hashlib.sha256(s.encode()).hexdigest(), s) for s in candidates)
    return ranked[0][1]


def build_loso_folds(session_ids: list[str]) -> list[dict]:
    """Leave-one-session-out: one fold per session, that session is
    validation, all others are train."""
    folds = []
    for i, val_session in enumerate(sorted(session_ids)):
        train_sessions = [s for s in sorted(session_ids) if s != val_session]
        folds.append({"fold_id": i, "val_session": val_session, "train_sessions": train_sessions})
    return folds
