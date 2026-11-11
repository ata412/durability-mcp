"""Physiological scalar baselines calibrated inside each training fold."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _daniels_fraction(duration_min: np.ndarray) -> np.ndarray:
    return (
        0.8
        + 0.1894393 * np.exp(-0.012778 * duration_min)
        + 0.2989558 * np.exp(-0.1932605 * duration_min)
    )


def _daniels_velocity_from_vo2(vo2: np.ndarray) -> np.ndarray:
    # Positive root of 0.000104*v^2 + 0.182258*v - (vo2 + 4.60) = 0.
    a = 0.000104
    b = 0.182258
    discriminant = np.maximum(0.0, b * b + 4.0 * a * (vo2 + 4.60))
    return (-b + np.sqrt(discriminant)) / (2.0 * a)


def formula_scores(
    frame: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    """Create leakage-safe formula outputs used as one-dimensional baselines."""
    early_speed = (
        frame["early_external_load_mean"].to_numpy(dtype=float)
        * frame["cs_ms_asof"].to_numpy(dtype=float)
    )
    full_duration_s = (
        frame["early_cutoff_elapsed_s"].to_numpy(dtype=float)
        / float(config["features"]["early_window_frac"])
    )
    early_duration_s = frame["early_duration_min"].to_numpy(dtype=float) * 60.0
    exponent = float(config["models"]["tier0"]["riegel_exponent"])
    duration_ratio = full_duration_s / early_duration_s
    riegel_full_speed = early_speed * np.power(
        duration_ratio, 1.0 - exponent
    )
    riegel_retention = riegel_full_speed / early_speed

    early_speed_m_min = early_speed * 60.0
    early_vo2 = (
        -4.60
        + 0.182258 * early_speed_m_min
        + 0.000104 * np.square(early_speed_m_min)
    )
    early_vdot = early_vo2 / _daniels_fraction(early_duration_s / 60.0)
    target_vo2 = early_vdot * _daniels_fraction(full_duration_s / 60.0)
    daniels_full_speed = _daniels_velocity_from_vo2(target_vo2) / 60.0
    daniels_retention = daniels_full_speed / early_speed

    cs_full_speed = (
        frame["cs_ms_asof"].to_numpy(dtype=float)
        + frame["dprime_m_asof"].to_numpy(dtype=float) / full_duration_s
    )
    cs_retention = cs_full_speed / early_speed
    decoupling_extrapolation = (
        1.0
        + frame["early_decoupling_slope_per_h"].to_numpy(dtype=float)
        * (full_duration_s / 3_600.0)
        / frame["early_decoupling_mean"].to_numpy(dtype=float)
    )
    result = pd.DataFrame(
        {
            "tier0_riegel": riegel_retention,
            "tier0_daniels": daniels_retention,
            "tier0_cs_dprime": cs_retention,
            "tier0_decoupling_extrapolation": decoupling_extrapolation,
        },
        index=frame.index,
    )
    if "early_gap_speed_slope_ms_per_h" in frame:
        result["control_early_speed_trend"] = frame[
            "early_gap_speed_slope_ms_per_h"
        ].to_numpy(dtype=float)
    return result.replace([np.inf, -np.inf], np.nan)

