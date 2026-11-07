"""GPS-derived distance and speed for FitRec records."""

from __future__ import annotations

import numpy as np

EARTH_RADIUS_KM = 6371.0088


def derive_distance_speed(
    latitude: list[float] | np.ndarray,
    longitude: list[float] | np.ndarray,
    timestamp: list[int] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-step distance in km and speed in m/s.

    This reproduces FitRec's ``3600 * haversine_km / dt`` formula and performs
    the mandatory km/h to m/s conversion before returning.
    """
    lat = np.asarray(latitude, dtype=float)
    lon = np.asarray(longitude, dtype=float)
    time_s = np.asarray(timestamp, dtype=float)
    if not (len(lat) == len(lon) == len(time_s)):
        raise AssertionError("GPS and timestamp channels must have equal lengths")
    if len(time_s) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    dlat = np.diff(lat_rad)
    dlon = np.diff(lon_rad)
    hav = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat_rad[:-1]) * np.cos(lat_rad[1:]) * np.sin(dlon / 2.0) ** 2
    )
    step_km = np.zeros(len(time_s), dtype=float)
    step_km[1:] = 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(hav, 0.0, 1.0)))

    dt_s = np.diff(time_s)
    if np.any(dt_s <= 0):
        raise AssertionError("timestamp must be strictly increasing")
    speed_kmh = np.zeros(len(time_s), dtype=float)
    speed_kmh[1:] = 3600.0 * step_km[1:] / dt_s
    if len(speed_kmh) > 1:
        speed_kmh[0] = speed_kmh[1]
    return step_km, speed_kmh / 3.6


def smooth_speed(
    speed_ms: np.ndarray, timestamp: list[int] | np.ndarray, window_s: float
) -> np.ndarray:
    """Time-scale-aware centred rolling median for noisy GPS speed."""
    speed = np.asarray(speed_ms, dtype=float)
    time_s = np.asarray(timestamp, dtype=float)
    if len(speed) < 2:
        return speed.copy()
    native_dt_s = float(np.median(np.diff(time_s)))
    window_points = max(1, int(round(window_s / native_dt_s)))
    if window_points % 2 == 0:
        window_points += 1
    import pandas as pd

    return (
        pd.Series(speed)
        .rolling(window_points, center=True, min_periods=1)
        .median()
        .to_numpy(dtype=float)
    )

