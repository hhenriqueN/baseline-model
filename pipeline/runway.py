"""Runway-relative geometry: local ENU projection, along/cross track, glideslope
and airspeed error features (spec section 13).

Sign conventions (documented here, tested in tests/test_runway.py):
  - along_track_m: 0 at the threshold, negative before it (still approaching),
    positive beyond it (over the runway surface), measured along true_heading_deg.
  - cross_track_error_m: 0 on the extended runway centerline, POSITIVE to the
    RIGHT of the runway direction (aircraft heading toward true_heading_deg),
    NEGATIVE to the left. (Matches common "right = positive" XTK convention.)
  - heading_error_deg = aircraft_heading - runway_true_heading, wrapped to
    [-180, 180]; heading_error_sin/cos are sin/cos of that wrapped error.
  - height_above_threshold_ft = altitude_msl_ft - threshold_elevation_ft.
  - glideslope_error_deg = actual glide angle (aircraft-to-threshold) minus the
    configured nominal glideslope_deg. Positive = aircraft is ABOVE the
    nominal 3-degree profile for its distance to the threshold.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import yaml

WGS84_A = 6378137.0  # equatorial radius, meters


@dataclass
class RunwayConfig:
    threshold_lat_deg: float
    threshold_lon_deg: float
    true_heading_deg: float
    threshold_elevation_ft: float
    runway_length_m: float
    runway_width_m: float
    glideslope_deg: float
    target_airspeed_kt: float
    flare_agl_ft: float

    @classmethod
    def from_yaml(cls, path) -> "RunwayConfig":
        with open(path) as f:
            cfg = yaml.safe_load(f)
        return cls(
            threshold_lat_deg=cfg["threshold_latitude_deg"],
            threshold_lon_deg=cfg["threshold_longitude_deg"],
            true_heading_deg=cfg["true_heading_deg"],
            threshold_elevation_ft=cfg["threshold_elevation_ft"],
            runway_length_m=cfg["runway_length_m"],
            runway_width_m=cfg["runway_width_m"],
            glideslope_deg=cfg["glideslope_deg"],
            target_airspeed_kt=cfg["target_airspeed_kt"],
            flare_agl_ft=cfg["flare_agl_ft"],
        )


def _meters_per_degree(lat0_deg: float) -> tuple[float, float]:
    """Equirectangular (local tangent-plane) approximation, adequate for the
    few-km scales involved here. Returns (m_per_deg_lat, m_per_deg_lon)."""
    lat0 = math.radians(lat0_deg)
    m_per_deg_lat = (math.pi / 180.0) * WGS84_A * (1 - 0.00669438 * math.sin(lat0) ** 2) ** 1.5 \
        / (1 - 0.00669438) ** 0.5  # good enough approximation; dominant term below is what matters
    # Simpler and sufficiently accurate for <10 km spans:
    m_per_deg_lat = 111132.92 - 559.82 * math.cos(2 * lat0) + 1.175 * math.cos(4 * lat0)
    m_per_deg_lon = 111412.84 * math.cos(lat0) - 93.5 * math.cos(3 * lat0)
    return m_per_deg_lat, m_per_deg_lon


def latlon_to_enu(lat_deg: np.ndarray, lon_deg: np.ndarray, cfg: RunwayConfig) -> tuple[np.ndarray, np.ndarray]:
    m_per_lat, m_per_lon = _meters_per_degree(cfg.threshold_lat_deg)
    north_m = (np.asarray(lat_deg) - cfg.threshold_lat_deg) * m_per_lat
    east_m = (np.asarray(lon_deg) - cfg.threshold_lon_deg) * m_per_lon
    return east_m, north_m


def along_cross_track(lat_deg, lon_deg, cfg: RunwayConfig) -> tuple[np.ndarray, np.ndarray]:
    east_m, north_m = latlon_to_enu(lat_deg, lon_deg, cfg)
    theta = math.radians(cfg.true_heading_deg)
    along = east_m * math.sin(theta) + north_m * math.cos(theta)
    cross = east_m * math.cos(theta) - north_m * math.sin(theta)
    return along, cross


def point_at_along_cross(along_m: float, cross_m: float, cfg: RunwayConfig) -> tuple[float, float]:
    """Inverse of along_cross_track: given a desired (along_track_m,
    cross_track_error_m) relative to the runway threshold/heading, return
    (lat_deg, lon_deg). Used by the real-time controller to compute an exact
    spawn position from the SAME runway geometry used to build training
    features, rather than hardcoding an approximate lat/lon separately
    (spec: initialization must come from the project runway config, not a
    duplicated hardcoded position).

    Derivation: along/cross are related to local east/north (meters) by the
    2x2 rotation-reflection matrix in along_cross_track; inverting it gives
    east = along*sin(theta) + cross*cos(theta), north = along*cos(theta) - cross*sin(theta).
    """
    theta = math.radians(cfg.true_heading_deg)
    east_m = along_m * math.sin(theta) + cross_m * math.cos(theta)
    north_m = along_m * math.cos(theta) - cross_m * math.sin(theta)
    m_per_lat, m_per_lon = _meters_per_degree(cfg.threshold_lat_deg)
    lat_deg = cfg.threshold_lat_deg + north_m / m_per_lat
    lon_deg = cfg.threshold_lon_deg + east_m / m_per_lon
    return lat_deg, lon_deg


def heading_error(heading_deg: np.ndarray, cfg: RunwayConfig) -> np.ndarray:
    err = (np.asarray(heading_deg) - cfg.true_heading_deg + 180.0) % 360.0 - 180.0
    return err


def compute_runway_features(df, cfg: RunwayConfig, lat_col, lon_col, heading_col, alt_msl_col, airspeed_col, agl_col):
    import pandas as pd
    along_m, cross_m = along_cross_track(df[lat_col].to_numpy(), df[lon_col].to_numpy(), cfg)
    hdg_err_deg = heading_error(df[heading_col].to_numpy(), cfg)
    hdg_err_rad = np.radians(hdg_err_deg)
    height_above_threshold_ft = df[alt_msl_col].to_numpy() - cfg.threshold_elevation_ft

    dist_to_threshold_m = -along_m  # positive while still approaching
    height_m = height_above_threshold_ft * 0.3048
    actual_glide_angle_deg = np.degrees(np.arctan2(height_m, dist_to_threshold_m))
    glideslope_error_deg = actual_glide_angle_deg - cfg.glideslope_deg

    airspeed_error_kt = df[airspeed_col].to_numpy() - cfg.target_airspeed_kt
    flare_flag = (df[agl_col].to_numpy() <= cfg.flare_agl_ft).astype(int)

    out = pd.DataFrame({
        "along_track_m": along_m,
        "cross_track_error_m": cross_m,
        "heading_error_deg": hdg_err_deg,
        "heading_error_sin": np.sin(hdg_err_rad),
        "heading_error_cos": np.cos(hdg_err_rad),
        "height_above_threshold_ft": height_above_threshold_ft,
        "glideslope_error_deg": glideslope_error_deg,
        "airspeed_error_kt": airspeed_error_kt,
        "flare_flag": flare_flag,
    }, index=df.index)
    return out
