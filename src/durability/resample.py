"""Uniform-grid resampling without manufacturing 1 Hz precision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ResampleMetadata:
    native_dt_s: float
    native_n_points: int
    upsample_factor: float
    coarse_sampling: bool
    onset_sampling_eligible: bool
    gap_split: bool
    retained_duration_s: float
    grid_dt_s: float
    distance_speed_relative_error: float


def _longest_contiguous_block(
    frame: pd.DataFrame, max_gap_s: float
) -> tuple[pd.DataFrame, bool]:
    timestamps = pd.to_datetime(frame["timestamp"], utc=True)
    gaps_s = timestamps.diff().dt.total_seconds().to_numpy()
    boundaries = np.flatnonzero(gaps_s > max_gap_s)
    if boundaries.size == 0:
        return frame.reset_index(drop=True).copy(), False
    starts = np.r_[0, boundaries]
    stops = np.r_[boundaries, len(frame)]
    durations = np.asarray(
        [
            (timestamps.iloc[stop - 1] - timestamps.iloc[start]).total_seconds()
            for start, stop in zip(starts, stops, strict=True)
        ]
    )
    winner = int(np.argmax(durations))
    return frame.iloc[starts[winner] : stops[winner]].reset_index(drop=True).copy(), True


def _interpolate_with_gap_limit(
    source_time_s: np.ndarray,
    values: np.ndarray,
    target_time_s: np.ndarray,
    max_gap_s: float,
) -> np.ndarray:
    finite = np.isfinite(values)
    if not finite.any():
        return np.full(target_time_s.shape, np.nan, dtype=float)
    known_time = source_time_s[finite]
    known_values = values[finite]
    result = np.interp(target_time_s, known_time, known_values)
    if len(known_time) > 1:
        for left, right in zip(known_time[:-1], known_time[1:], strict=True):
            if right - left > max_gap_s:
                result[(target_time_s > left) & (target_time_s < right)] = np.nan
    result[target_time_s < known_time[0]] = np.nan
    result[target_time_s > known_time[-1]] = np.nan
    return result


def resample_activity(
    frame: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, ResampleMetadata]:
    """Keep the longest valid block and linearly interpolate onto the configured grid."""
    required = {
        "timestamp",
        "heart_rate",
        "altitude",
        "latitude",
        "longitude",
        "derived_distance_km",
        "derived_speed_ms",
        "smoothed_speed_ms",
    }
    missing = required - set(frame.columns)
    if missing:
        raise AssertionError(f"Missing Phase-1 stream columns: {sorted(missing)}")
    if len(frame) < 2:
        raise AssertionError("Activity needs at least two points for resampling")

    ordered = frame.sort_values("timestamp").reset_index(drop=True)
    timestamps = pd.to_datetime(ordered["timestamp"], utc=True)
    native_gaps_s = timestamps.diff().dt.total_seconds().iloc[1:].to_numpy()
    if np.any(native_gaps_s <= 0):
        raise AssertionError("timestamp must be strictly increasing")
    native_dt_s = float(np.median(native_gaps_s))
    grid_dt_s = float(config["resample"]["grid_dt_s"])
    upsample_factor = grid_dt_s / native_dt_s
    coarse = native_dt_s > config["resample"]["coarse_sampling_threshold_s"]

    block, gap_split = _longest_contiguous_block(
        ordered, config["resample"]["max_interp_gap_s"]
    )
    block_timestamp = pd.to_datetime(block["timestamp"], utc=True)
    source_time_s = (
        block_timestamp - block_timestamp.iloc[0]
    ).dt.total_seconds().to_numpy(dtype=float)
    retained_duration_s = float(source_time_s[-1])
    if retained_duration_s <= 0:
        raise AssertionError("Retained block has non-positive duration")
    target_time_s = np.arange(
        0.0, retained_duration_s + grid_dt_s * 0.5, grid_dt_s
    )
    target_time_s = target_time_s[target_time_s <= retained_duration_s]
    if len(target_time_s) < 2:
        raise AssertionError("Retained block is shorter than one grid interval")

    speed = block["derived_speed_ms"].to_numpy(dtype=float)
    step_km = block["derived_distance_km"].to_numpy(dtype=float).copy()
    step_km[0] = 0.0
    total_distance_m = float(np.nansum(step_km) * 1_000.0)
    dt_s = np.diff(source_time_s)
    time_weighted_speed = float(
        np.nansum(speed[1:] * dt_s) / retained_duration_s
    )
    distance_speed = total_distance_m / retained_duration_s
    denominator = max(abs(distance_speed), np.finfo(float).eps)
    relative_error = abs(time_weighted_speed - distance_speed) / denominator
    if relative_error > config["data"]["distance_speed_agreement_tol"]:
        raise AssertionError(
            "Cumulative distance and time-weighted speed disagree by "
            f"{relative_error:.3%}"
        )

    speed_low, speed_high = config["data"]["speed_sanity_ms"]
    median_speed = float(np.nanmedian(speed))
    if not speed_low <= median_speed <= speed_high:
        raise AssertionError(
            f"Median running speed {median_speed:.3f} m/s outside "
            f"[{speed_low}, {speed_high}]"
        )

    hr = block["heart_rate"].to_numpy(dtype=float, copy=True)
    hr_low, hr_high = config["athlete_params"]["hr_valid_range"]
    hr[(hr < hr_low) | (hr > hr_high)] = np.nan
    max_gap_s = float(config["resample"]["max_interp_gap_s"])
    continuous = {
        "heart_rate": hr,
        "altitude": block["altitude"].to_numpy(dtype=float),
        "latitude": block["latitude"].to_numpy(dtype=float),
        "longitude": block["longitude"].to_numpy(dtype=float),
        "derived_speed_ms": speed,
        "smoothed_speed_ms": block["smoothed_speed_ms"].to_numpy(dtype=float),
    }
    output: dict[str, Any] = {
        "timestamp": block_timestamp.iloc[0]
        + pd.to_timedelta(target_time_s, unit="s"),
        "elapsed_s": target_time_s,
    }
    for column, values in continuous.items():
        output[column] = _interpolate_with_gap_limit(
            source_time_s, values, target_time_s, max_gap_s
        )

    cumulative_distance_m = np.cumsum(np.nan_to_num(step_km) * 1_000.0)
    output["cumulative_distance_m"] = _interpolate_with_gap_limit(
        source_time_s, cumulative_distance_m, target_time_s, max_gap_s
    )
    result = pd.DataFrame(output)
    onset_eligible = bool(
        not coarse
        and upsample_factor <= config["resample"]["max_upsample_factor"]
        and retained_duration_s >= config["indices"]["onset_rolling_s"]
    )
    metadata = ResampleMetadata(
        native_dt_s=native_dt_s,
        native_n_points=len(ordered),
        upsample_factor=upsample_factor,
        coarse_sampling=coarse,
        onset_sampling_eligible=onset_eligible,
        gap_split=gap_split,
        retained_duration_s=retained_duration_s,
        grid_dt_s=grid_dt_s,
        distance_speed_relative_error=relative_error,
    )
    return result, metadata
