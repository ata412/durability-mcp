"""Grade-adjusted speed using the Minetti running-cost polynomial."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter


@dataclass(frozen=True)
class GapMetadata:
    gap_available: bool
    clipped_gradient_fraction: float
    clipped_multiplier_fraction: float


def minetti_cost(gradient: np.ndarray | float) -> np.ndarray:
    """Energy cost of running in J/kg/m for decimal gradient."""
    value = np.asarray(gradient, dtype=float)
    return (
        155.4 * value**5
        - 30.4 * value**4
        - 43.3 * value**3
        + 46.3 * value**2
        + 19.5 * value
        + 3.6
    )


def _odd_window(window_s: float, grid_dt_s: float, length: int) -> int:
    points = max(1, int(round(window_s / grid_dt_s)))
    if points % 2 == 0:
        points += 1
    maximum = length if length % 2 == 1 else length - 1
    return max(1, min(points, maximum))


def add_grade_adjusted_speed(
    frame: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, GapMetadata]:
    """Add smoothed altitude, gradient and grade-adjusted speed columns."""
    result = frame.copy()
    speed = result["smoothed_speed_ms"].to_numpy(dtype=float)
    altitude = result["altitude"].to_numpy(dtype=float)
    if len(result) < 2:
        raise AssertionError("Activity needs at least two points for GAP")
    if not np.isfinite(altitude).all():
        result["smoothed_altitude_m"] = np.nan
        result["gradient"] = np.nan
        result["gap_multiplier"] = 1.0
        result["gap_speed_ms"] = speed
        result["gap_available"] = False
        return result, GapMetadata(False, 0.0, 0.0)

    elapsed = result["elapsed_s"].to_numpy(dtype=float)
    grid_dt_s = float(np.median(np.diff(elapsed)))
    altitude_series = pd.Series(altitude).interpolate(
        limit_direction="both"
    ).to_numpy(dtype=float)
    median_points = _odd_window(
        config["gap"]["altitude_median_filter_s"], grid_dt_s, len(result)
    )
    smoothed_altitude = median_filter(
        altitude_series, size=median_points, mode="nearest"
    )
    savgol_points = _odd_window(
        config["gap"]["savgol_window_s"], grid_dt_s, len(result)
    )
    polyorder = int(config["gap"]["savgol_polyorder"])
    if savgol_points > polyorder:
        smoothed_altitude = savgol_filter(
            smoothed_altitude,
            window_length=savgol_points,
            polyorder=polyorder,
            mode="interp",
        )

    distance_m = result["cumulative_distance_m"].to_numpy(dtype=float)
    delta_distance = np.diff(distance_m)
    delta_altitude = np.diff(smoothed_altitude)
    step_gradient = np.zeros(len(result), dtype=float)
    usable = delta_distance >= config["gap"]["min_horizontal_step_m"]
    trailing_gradient = step_gradient[1:]
    trailing_gradient[usable] = delta_altitude[usable] / delta_distance[usable]
    if len(step_gradient) > 1:
        step_gradient[0] = step_gradient[1]
    if np.ptp(altitude_series) == 0.0:
        step_gradient.fill(0.0)

    gradient_limit = float(config["gap"]["gradient_clip"])
    gradient_clipped = np.abs(step_gradient) > gradient_limit
    clipped_gradient = np.clip(step_gradient, -gradient_limit, gradient_limit)
    raw_multiplier = minetti_cost(clipped_gradient) / float(minetti_cost(0.0))
    multiplier_low, multiplier_high = config["gap"]["multiplier_clip"]
    multiplier_clipped = (raw_multiplier < multiplier_low) | (
        raw_multiplier > multiplier_high
    )
    multiplier = np.clip(raw_multiplier, multiplier_low, multiplier_high)
    gap_speed = speed * multiplier
    # Preserve exact equality on genuinely flat terrain.
    gap_speed[clipped_gradient == 0.0] = speed[clipped_gradient == 0.0]

    result["smoothed_altitude_m"] = smoothed_altitude
    result["gradient"] = clipped_gradient
    result["gap_multiplier"] = multiplier
    result["gap_speed_ms"] = gap_speed
    result["gap_available"] = True
    metadata = GapMetadata(
        True,
        float(np.mean(gradient_clipped)),
        float(np.mean(multiplier_clipped)),
    )
    return result, metadata
