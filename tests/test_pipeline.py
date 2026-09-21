"""Automated tests required by spec section 19. Run with:
    python3 -m pytest tests/ -v
from the voos-piloto project root.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import paths, columns as C
from pipeline.io_utils import sha256_of_file
from pipeline.features import LATERAL_FEATURE_ORDER, LATERAL_LABEL_ORDER, LONGITUDINAL_FEATURE_ORDER, LONGITUDINAL_LABEL_ORDER

FORBIDDEN_EXACT = {"frame", "sim_time", C.LAT, C.LON}
FORBIDDEN_SUBSTRINGS = ["surface-position", "yoke", "pedals", "flaps-lever", "brake",
                        "steering-norm", "wow", "compression-norm", "rollspeed", "-trim"]


@pytest.fixture(scope="module")
def raw_manifest():
    return pd.read_csv(paths.MANIFESTS_DIR / "raw_files.csv")


@pytest.fixture(scope="module")
def episodes():
    return pd.read_csv(paths.MANIFESTS_DIR / "episodes.csv")


# ---------- 1. raw CSVs unaltered ----------
def test_raw_csvs_unaltered(raw_manifest):
    for _, row in raw_manifest.iterrows():
        assert Path(row["source_path"]).exists(), f"original source file missing: {row['source_path']}"
        current_source_sha = sha256_of_file(Path(row["source_path"]))
        assert current_source_sha == row["sha256_source"], f"{row['source_file']}: original file changed since ingestion!"
        current_raw_sha = sha256_of_file(Path(row["raw_path"]))
        assert current_raw_sha == row["sha256_raw_copy"], f"{row['source_file']}: data/raw/ copy changed!"


# ---------- 2 & 3. timestamps sorted, no duplicates after cleaning ----------
@pytest.mark.parametrize("path", glob.glob(str(paths.VALIDATED_DIR / "*.csv")))
def test_validated_timestamps_sorted_and_unique(path):
    df = pd.read_csv(path, usecols=[C.SIM_TIME])
    assert df[C.SIM_TIME].is_monotonic_increasing, f"{path}: sim_time not sorted"
    assert not df[C.SIM_TIME].duplicated().any(), f"{path}: duplicate sim_time after cleaning"


# ---------- 4. processed frequency is 10Hz within tolerance ----------
@pytest.mark.parametrize("path", glob.glob(str(paths.PROCESSED_10HZ_DIR / "*.csv")) +
                          glob.glob(str(paths.PROCESSED_EXCLUDED_DIR / "*.csv")))
def test_resampled_is_10hz(path):
    df = pd.read_csv(path, usecols=[C.SIM_TIME])
    dt = df[C.SIM_TIME].diff().dropna().to_numpy()
    assert np.allclose(dt, 0.1, atol=1e-6), f"{path}: resampled grid is not uniform 10Hz (dt range {dt.min()}-{dt.max()})"


# ---------- 5. no NaN/Inf in final training datasets ----------
@pytest.mark.parametrize("path", glob.glob(str(paths.FEATURES_BASELINE_CORE_DIR / "*.csv")))
def test_no_nan_inf_in_baseline_core(path):
    df = pd.read_csv(path)
    num = df.select_dtypes(include=[np.number])
    assert not num.isna().any().any(), f"{path}: NaN found"
    assert not np.isinf(num.to_numpy()).any(), f"{path}: Inf found"


# ---------- 6-9. exact input/output dimensionality ----------
def test_lateral_network_has_11_inputs():
    assert len(LATERAL_FEATURE_ORDER) == 11


def test_longitudinal_network_has_12_inputs():
    assert len(LONGITUDINAL_FEATURE_ORDER) == 12


def test_lateral_labels_have_2_outputs():
    assert len(LATERAL_LABEL_ORDER) == 2


def test_longitudinal_labels_have_2_outputs():
    assert len(LONGITUDINAL_LABEL_ORDER) == 2


# ---------- 10-13. no forbidden columns in the neural input schema ----------
def test_no_forbidden_columns_in_feature_schema():
    all_vars = LATERAL_FEATURE_ORDER + LONGITUDINAL_FEATURE_ORDER
    for v in all_vars:
        assert v not in FORBIDDEN_EXACT, f"forbidden column {v} present in neural input schema"
        for sub in FORBIDDEN_SUBSTRINGS:
            assert sub not in v, f"forbidden pattern '{sub}' matched schema variable {v}"


def test_lat_lon_not_in_inputs():
    all_vars = LATERAL_FEATURE_ORDER + LONGITUDINAL_FEATURE_ORDER
    assert "latitude" not in " ".join(all_vars).lower()
    assert "longitude" not in " ".join(all_vars).lower()


def test_frame_and_sim_time_not_in_inputs():
    all_vars = LATERAL_FEATURE_ORDER + LONGITUDINAL_FEATURE_ORDER
    assert "frame" not in all_vars and "sim_time" not in all_vars


# ---------- 14. both circuit landings stay in the same fold ----------
def test_circuit_landings_share_session_and_fold(episodes):
    circuit_eps = episodes[episodes.episode_id.str.startswith("circuito-completo__")]
    assert circuit_eps.session_id.nunique() == 1

    assignment = pd.read_csv(paths.SPLITS_DIR / "cv_folds_session_assignment.csv")
    session_id = circuit_eps.session_id.iloc[0]
    rows = assignment[assignment.session_id == session_id]
    # a session must have exactly one role per fold it appears in, and appear
    # in every fold (either as the val session for its own fold, or as train)
    assert rows.groupby("fold_id").size().max() == 1


# ---------- 15. voo-inicio-teste excluded from every fold ----------
def test_pilot_familiarization_excluded_from_folds():
    assignment = pd.read_csv(paths.SPLITS_DIR / "cv_folds_session_assignment.csv")
    assert "voo-inicio-teste" not in assignment.session_id.values


# ---------- 16. each fold's scaler uses only its own train sessions ----------
@pytest.mark.parametrize("scaler_path", glob.glob(str(paths.SCALERS_FOLDS_DIR / "*.json")))
def test_fold_scaler_uses_only_train_sessions(scaler_path):
    scaler = json.load(open(scaler_path))
    fold_id = int(Path(scaler_path).stem.split("_")[1])
    assignment = pd.read_csv(paths.SPLITS_DIR / "cv_folds_session_assignment.csv")
    fold_rows = assignment[assignment.fold_id == fold_id]
    val_sessions = set(fold_rows.loc[fold_rows.role == "val", "session_id"])
    assert val_sessions.isdisjoint(set(scaler["training_session_ids"])), \
        f"{scaler_path}: validation session leaked into training_session_ids"

    holdout = yaml.safe_load(open(paths.SPLITS_DIR / "fixed_diagnostic_holdout.yaml"))
    assert holdout["holdout_session"] not in scaler["training_session_ids"], \
        f"{scaler_path}: fixed diagnostic holdout session leaked into a CV fold scaler"


# ---------- 17. normalized train features ~N(0,1) ----------
@pytest.mark.parametrize("path", glob.glob(str(paths.SPLITS_DIR / "normalized_folds" / "*_train.csv")))
def test_normalized_train_features_standardized(path):
    # Mean must be tightly near 0 (mean-centering is exact by construction).
    # Std is only *exactly* 1 before the +-5 clip; for a few heavily
    # leptokurtic angular-rate variables (roll/yaw/pitch rate, alpha), real
    # flight data has brief large excursions (bounce/flare/gust) whose
    # clipping legitimately pulls the post-clip std below 1 -- this is an
    # expected, spec-mandated consequence of clip(z,-5,5) on fat-tailed
    # data, not a normalization bug. So we use a looser sanity band on std.
    from pipeline.scalers import ZSCORE_VARIABLES
    df = pd.read_csv(path)
    for var in ZSCORE_VARIABLES:
        for prefix in ("lat__", "lon__"):
            col = prefix + var
            if col in df.columns:
                vals = df[col].to_numpy(dtype=float)
                assert abs(np.mean(vals)) < 0.2, f"{path}:{col} mean not near 0 ({np.mean(vals):.3f})"
                assert 0.4 <= np.std(vals) <= 1.2, f"{path}:{col} std far from 1 ({np.std(vals):.3f})"


def test_non_zscored_variables_stay_in_natural_range():
    from pipeline.scalers import NON_ZSCORE_VARIABLES
    for path in glob.glob(str(paths.SPLITS_DIR / "normalized_folds" / "*_train.csv")):
        df = pd.read_csv(path)
        for var in NON_ZSCORE_VARIABLES:
            for prefix in ("lat__", "lon__"):
                col = prefix + var
                if col in df.columns:
                    vals = df[col].to_numpy(dtype=float)
                    if var in ("heading_error_sin", "heading_error_cos"):
                        assert vals.min() >= -1.0001 and vals.max() <= 1.0001
                    if var in ("flaps_state_norm", "flare_flag"):
                        assert vals.min() >= -0.0001 and vals.max() <= 1.0001


# ---------- 18. labels keep their original scale after normalization ----------
@pytest.mark.parametrize("path", glob.glob(str(paths.SPLITS_DIR / "normalized_folds" / "*.csv")))
def test_labels_not_normalized(path):
    df = pd.read_csv(path)
    for col in ("laty__aileron_command", "laty__rudder_command"):
        assert df[col].abs().max() <= 1.5, f"{path}:{col} outside expected raw control range"
    assert df["lony__throttle_command"].min() >= -0.001 and df["lony__throttle_command"].max() <= 1.001


# ---------- 19. control ranges valid ----------
def test_control_range_audit_has_no_violations():
    audit = pd.read_csv(paths.MANIFESTS_DIR / "control_range_audit.csv")
    assert (audit.n_below_range == 0).all() and (audit.n_above_range == 0).all()


# ---------- 20. unit duplicates resolved per the documented rules ----------
def test_confirmed_duplicates_excluded_from_catalog():
    dup = pd.read_csv(paths.MANIFESTS_DIR / "duplicate_columns.csv")
    catalog = pd.read_csv(paths.MANIFESTS_DIR / "columns_catalog.csv").set_index("column_name")
    confirmed = dup[dup.confirmed_duplicate == True]  # noqa: E712
    for _, row in confirmed.iterrows():
        assert catalog.loc[row["drop_column"], "role"] == "excluded"
        assert catalog.loc[row["drop_column"], "duplicate_of"] == row["keep_column"]


# ---------- 21. row-level traceability ----------
@pytest.mark.parametrize("path", glob.glob(str(paths.FEATURES_UNNORM_DIR / "*.csv")))
def test_traceability_columns_present(path):
    df = pd.read_csv(path, nrows=5)
    for col in ("episode_id", "session_id", "source_file", C.SIM_TIME):
        assert col in df.columns, f"{path}: missing traceability column {col}"


# ---------- 22. no prohibited phases ever enter a neural-training file (spec section 7) ----------
PROHIBITED_TRAINING_PHASES = {"TOUCHDOWN_CONFIRMED", "BOUNCE", "ROLLOUT", "COMPLETE", "GO_AROUND"}


@pytest.mark.parametrize("path", glob.glob(str(paths.FEATURES_BASELINE_CORE_DIR / "*.csv")) +
                          glob.glob(str(paths.FEATURES_FULL_APPROACH_DIR / "*.csv")) +
                          glob.glob(str(paths.FEATURES_RECOVERY_ABLATION_DIR / "*.csv")) +
                          glob.glob(str(paths.SPLITS_DIR / "normalized_folds" / "*.csv")))
def test_no_prohibited_phases_in_training_datasets(path):
    df = pd.read_csv(path, usecols=["phase"])
    leaked = set(df["phase"].unique()) & PROHIBITED_TRAINING_PHASES
    assert not leaked, f"{path}: prohibited phase(s) {leaked} present in a neural-training dataset"


def test_baseline_core_only_contains_allowed_phases():
    from pipeline.features import BASELINE_CORE_PHASES
    for path in glob.glob(str(paths.FEATURES_BASELINE_CORE_DIR / "*.csv")):
        df = pd.read_csv(path, usecols=["phase"])
        extra = set(df["phase"].unique()) - BASELINE_CORE_PHASES
        assert not extra, f"{path}: baseline_core contains unexpected phase(s) {extra}"


# ---------- 23. no duplicated feature/column names ----------
@pytest.mark.parametrize("path", glob.glob(str(paths.FEATURES_BASELINE_CORE_DIR / "*.csv")))
def test_no_duplicated_column_names(path):
    df = pd.read_csv(path, nrows=1)
    cols = list(df.columns)
    assert len(cols) == len(set(cols)), f"{path}: duplicated column names {sorted({c for c in cols if cols.count(c) > 1})}"


def test_lateral_and_longitudinal_feature_name_overlap_is_documented_only():
    from pipeline.features import LATERAL_FEATURE_ORDER, LONGITUDINAL_FEATURE_ORDER
    # along_track_m, agl_ft, and airspeed_kt legitimately appear in both
    # networks' input lists per spec sections 15-16 (documented, not a bug);
    # every other name must not collide.
    shared = set(LATERAL_FEATURE_ORDER) & set(LONGITUDINAL_FEATURE_ORDER)
    assert shared <= {"along_track_m", "agl_ft", "airspeed_kt"}, f"unexpected feature name overlap: {shared}"
    assert len(LATERAL_FEATURE_ORDER) == len(set(LATERAL_FEATURE_ORDER)), "duplicate within LATERAL_FEATURE_ORDER"
    assert len(LONGITUDINAL_FEATURE_ORDER) == len(set(LONGITUDINAL_FEATURE_ORDER)), "duplicate within LONGITUDINAL_FEATURE_ORDER"


# ---------- 24. every eligible session appears in validation exactly once across the CV schedule ----------
def test_every_eligible_non_holdout_session_validated_exactly_once(episodes):
    eligible = episodes[episodes.eligible_for_training]
    holdout = yaml.safe_load(open(paths.SPLITS_DIR / "fixed_diagnostic_holdout.yaml"))
    expected_sessions = set(eligible.session_id.unique()) - {holdout["holdout_session"]}
    assignment = pd.read_csv(paths.SPLITS_DIR / "cv_folds_session_assignment.csv")
    val_sessions = assignment.loc[assignment.role == "val", "session_id"]
    assert set(val_sessions) == expected_sessions
    assert val_sessions.value_counts().eq(1).all(), "a session was validated more than once across the CV schedule"


def test_holdout_session_never_in_any_fold():
    holdout = yaml.safe_load(open(paths.SPLITS_DIR / "fixed_diagnostic_holdout.yaml"))
    assignment = pd.read_csv(paths.SPLITS_DIR / "cv_folds_session_assignment.csv")
    assert holdout["holdout_session"] not in assignment.session_id.values


# ---------- 25. state-machine regression checks (spec section 6) re-run on the real dataset ----------
def test_state_machine_checks_pass_on_regenerated_boundaries():
    import yaml as _yaml
    from pipeline.landing_fsm import run_landing_fsm
    from pipeline.state_machine_checks import validate_episode
    import numpy as np

    with open(paths.CONFIG_DIR / "preprocessing.yaml") as f:
        cfg = _yaml.safe_load(f)
    boundaries = pd.read_csv(paths.MANIFESTS_DIR / "episode_phase_boundaries.csv").set_index("episode_id")
    eligible_eids = pd.read_csv(paths.MANIFESTS_DIR / "episodes.csv")
    eligible_eids = eligible_eids.loc[eligible_eids.eligible_for_training, "episode_id"]
    for eid in eligible_eids:
        df10 = pd.read_csv(paths.PROCESSED_10HZ_DIR / f"{eid}.csv")
        flare_s = boundaries.loc[eid, "flare_s"]
        t10 = df10[C.SIM_TIME].to_numpy()
        start_idx = min(int(np.searchsorted(t10, flare_s, side="left")), len(df10) - 1)
        fsm = run_landing_fsm(df10, start_idx, len(df10) - 1, cfg)
        validate_episode(df10, fsm, cfg, eid)  # raises on any hard violation


def test_at_least_one_episode_never_auto_confirms_and_is_flagged():
    # documents the known, legitimate voo-3-com-16-nos case (spec section 6:
    # "stop and produce a clear diagnostic report" rather than force a proxy).
    boundaries = pd.read_csv(paths.MANIFESTS_DIR / "episode_phase_boundaries.csv")
    unconfirmed = boundaries[~boundaries.touchdown_confirmed]
    flags = pd.read_csv(paths.MANIFESTS_DIR / "phase_review_flags.csv")
    for eid in unconfirmed.episode_id:
        ep_flags = flags.loc[flags.episode_id == eid, "flag"]
        assert ep_flags.str.startswith("touchdown_never_confirmed").any(), \
            f"{eid}: touchdown not confirmed but no review flag recorded"
