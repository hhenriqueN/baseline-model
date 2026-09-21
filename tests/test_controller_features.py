import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import paths, columns as C
from pipeline.runway import RunwayConfig, compute_runway_features
from pipeline.features import LATERAL_FEATURE_ORDER, LONGITUDINAL_FEATURE_ORDER
from controller.features_online import OnlineFeatureBuilder, to_normalized_vectors

RUNWAY_CFG = RunwayConfig.from_yaml(paths.CONFIG_DIR / "runway.yaml")

FAKE_TELEMETRY = {
    "lat_deg": RUNWAY_CFG.threshold_lat_deg - 0.02, "lon_deg": RUNWAY_CFG.threshold_lon_deg + 0.0003,
    "heading_deg": RUNWAY_CFG.true_heading_deg + 2.0,
    "alt_msl_ft": RUNWAY_CFG.threshold_elevation_ft + 500, "alt_agl_ft": 480,
    "airspeed_kt": 68, "roll_deg": -1.5, "pitch_deg": 2.0, "alpha_deg": 4.0,
    "sideslip_rad": 0.01, "roll_rate_degps": 0.5, "pitch_rate_degps": -0.2, "yaw_rate_degps": 0.1,
    "vertical_speed_fps": -12.0, "vbody_fps": 1.2, "flaps_pos": 0.99,
}


def test_online_features_match_offline_pipeline_directly():
    """The online feature builder must reproduce pipeline.runway.compute_runway_features
    EXACTLY for a single-row equivalent input (same function, no drift)."""
    builder = OnlineFeatureBuilder(RUNWAY_CFG, glideslope_floor_m=50.0)
    raw = builder.build_raw_features(FAKE_TELEMETRY)

    df = pd.DataFrame([{
        C.LAT: FAKE_TELEMETRY["lat_deg"], C.LON: FAKE_TELEMETRY["lon_deg"],
        C.HEADING_DEG: FAKE_TELEMETRY["heading_deg"],
        C.ALT_MSL_FT: FAKE_TELEMETRY["alt_msl_ft"], C.ALT_AGL_FT: FAKE_TELEMETRY["alt_agl_ft"],
        C.AIRSPEED_KT: FAKE_TELEMETRY["airspeed_kt"],
    }])
    rfeat = compute_runway_features(df, RUNWAY_CFG, lat_col=C.LAT, lon_col=C.LON, heading_col=C.HEADING_DEG,
                                      alt_msl_col=C.ALT_MSL_FT, airspeed_col=C.AIRSPEED_KT, agl_col=C.ALT_AGL_FT)
    assert raw["along_track_m"] == pytest.approx(float(rfeat["along_track_m"].iloc[0]))
    assert raw["cross_track_error_m"] == pytest.approx(float(rfeat["cross_track_error_m"].iloc[0]))
    assert raw["heading_error_sin"] == pytest.approx(float(rfeat["heading_error_sin"].iloc[0]))
    assert raw["glideslope_error_deg"] == pytest.approx(float(rfeat["glideslope_error_deg"].iloc[0]))


def test_online_features_produce_all_required_keys():
    builder = OnlineFeatureBuilder(RUNWAY_CFG, glideslope_floor_m=50.0)
    raw = builder.build_raw_features(FAKE_TELEMETRY)
    for f in LATERAL_FEATURE_ORDER:
        assert f in raw, f"missing lateral feature {f}"
    for f in LONGITUDINAL_FEATURE_ORDER:
        assert f in raw, f"missing longitudinal feature {f}"


def test_glideslope_floor_hold_is_causal_and_sticky():
    builder = OnlineFeatureBuilder(RUNWAY_CFG, glideslope_floor_m=50.0)
    far_telem = dict(FAKE_TELEMETRY, lat_deg=RUNWAY_CFG.threshold_lat_deg - 0.005)  # ~555m out, outside floor
    raw_far = builder.build_raw_features(far_telem)
    last_valid = raw_far["glideslope_error_deg"]

    near_telem = dict(FAKE_TELEMETRY, lat_deg=RUNWAY_CFG.threshold_lat_deg - 0.00005)  # ~5.5m out, inside floor
    raw_near = builder.build_raw_features(near_telem)
    assert raw_near["glideslope_error_deg"] == pytest.approx(last_valid), \
        "glideslope error must hold the last valid value once inside the floor distance"


def test_normalized_vectors_have_correct_dimensions_and_are_finite():
    import json
    scaler = json.load(open(paths.SCALERS_FINAL_DIR / "scaler_final_all_valid.json"))
    builder = OnlineFeatureBuilder(RUNWAY_CFG, glideslope_floor_m=50.0)
    raw = builder.build_raw_features(FAKE_TELEMETRY)
    lat_vec, lon_vec = to_normalized_vectors(raw, scaler)
    assert len(lat_vec) == 11
    assert len(lon_vec) == 12
    assert all(v == v and abs(v) != float("inf") for v in lat_vec + lon_vec)  # no NaN/Inf
    assert all(-5.0001 <= v <= 5.0001 for v in lat_vec + lon_vec if True), "clip policy should bound z-scored values"
