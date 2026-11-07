"""Leakage-aware athlete HRmax, Critical Speed and D-prime estimates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from durability.config import repo_path

HistoryLoader = Callable[
    [str, pd.Timestamp], tuple[pd.DataFrame, dict[str, pd.DataFrame]]
]


@dataclass(frozen=True)
class CSFit:
    cs_ms: float
    dprime_m: float
    r_squared: float
    n_durations: int


@dataclass(frozen=True)
class AthleteParams:
    athlete_id: str
    as_of: pd.Timestamp
    hrmax_bpm: float
    cs_ms: float
    dprime_m: float
    cs_r_squared: float
    n_preceding_activities: int
    eligible: bool
    rejection_reason: str


def fit_cs_dprime(
    best_speeds_by_duration_s: dict[float, float],
    config: dict[str, Any],
) -> CSFit:
    """Fit distance = CS*time + D-prime with ordinary least squares."""
    valid = sorted(
        (float(duration), float(speed))
        for duration, speed in best_speeds_by_duration_s.items()
        if np.isfinite(duration) and np.isfinite(speed) and duration > 0 and speed > 0
    )
    minimum = int(config["athlete_params"]["cs_min_durations"])
    if len(valid) < minimum:
        raise ValueError(f"CS fit needs at least {minimum} usable durations")
    duration_s = np.asarray([item[0] for item in valid], dtype=float)
    distance_m = np.asarray([item[0] * item[1] for item in valid], dtype=float)
    design = np.column_stack([duration_s, np.ones(len(duration_s))])
    coefficients, *_ = np.linalg.lstsq(design, distance_m, rcond=None)
    cs_ms, dprime_m = coefficients
    predicted = design @ coefficients
    residual = float(np.sum((distance_m - predicted) ** 2))
    total = float(np.sum((distance_m - distance_m.mean()) ** 2))
    r_squared = 1.0 if total == 0 and residual == 0 else 1.0 - residual / total
    return CSFit(float(cs_ms), float(dprime_m), r_squared, len(valid))


def _fastest_rolling_mean(
    speed: np.ndarray, elapsed_s: np.ndarray, duration_s: float
) -> float:
    if len(speed) < 2 or elapsed_s[-1] - elapsed_s[0] < duration_s:
        return np.nan
    dt_s = float(np.median(np.diff(elapsed_s)))
    window_points = max(2, int(np.ceil(duration_s / dt_s)))
    return float(
        pd.Series(speed)
        .rolling(window_points, min_periods=window_points)
        .mean()
        .max()
    )


def _sustained_hrmax(
    streams: dict[str, pd.DataFrame], sustain_s: float, valid_range: list[float]
) -> float:
    candidates: list[float] = []
    low, high = valid_range
    for stream in streams.values():
        column = "heart_rate" if "heart_rate" in stream else "heart_rate_clean"
        if column not in stream or len(stream) < 2:
            continue
        hr = stream[column].to_numpy(dtype=float, copy=True)
        hr[(hr < low) | (hr > high)] = np.nan
        elapsed = stream["elapsed_s"].to_numpy(dtype=float)
        dt_s = float(np.median(np.diff(elapsed)))
        points = max(2, int(np.ceil(sustain_s / dt_s)) + 1)
        sustained = (
            pd.Series(hr)
            .rolling(points, min_periods=points)
            .min()
            .max()
        )
        if np.isfinite(sustained):
            candidates.append(float(sustained))
    return max(candidates) if candidates else np.nan


class AthleteParameterEstimator:
    """Estimate parameters from a history loader guarded by a temporal assertion."""

    def __init__(
        self, history_loader: HistoryLoader, config: dict[str, Any]
    ) -> None:
        self._history_loader = history_loader
        self._config = config

    def estimate_params(
        self, athlete_id: str, as_of: pd.Timestamp
    ) -> AthleteParams:
        """Estimate parameters using only activities strictly before ``as_of``."""
        target_time = pd.Timestamp(as_of)
        if target_time.tzinfo is None:
            target_time = target_time.tz_localize("UTC")
        else:
            target_time = target_time.tz_convert("UTC")
        activities, streams = self._history_loader(str(athlete_id), target_time)
        if not activities.empty:
            activity_times = pd.to_datetime(activities["start_time"], utc=True)
            assert (activity_times < target_time).all(), (
                "Every input activity must be strictly before as_of"
            )
            assert (
                activities["athlete_id"].astype(str) == str(athlete_id)
            ).all(), "History loader returned a different athlete"

        n_preceding = len(activities)
        minimum = int(
            self._config["athlete_params"]["min_preceding_activities"]
        )
        if n_preceding < minimum:
            return self._ineligible(
                str(athlete_id),
                target_time,
                n_preceding,
                f"fewer_than_{minimum}_preceding_activities",
            )

        hrmax = _sustained_hrmax(
            streams,
            self._config["athlete_params"]["hrmax_sustain_s"],
            self._config["athlete_params"]["hr_valid_range"],
        )
        if not np.isfinite(hrmax):
            return self._ineligible(
                str(athlete_id), target_time, n_preceding, "hrmax_unavailable"
            )

        best: dict[float, float] = {}
        for duration_min in self._config["athlete_params"]["cs_durations_min"]:
            duration_s = float(duration_min) * 60.0
            efforts = []
            for stream in streams.values():
                if "gap_speed_ms" not in stream:
                    continue
                if "gap_available" in stream and not bool(
                    stream["gap_available"].any()
                ):
                    continue
                value = _fastest_rolling_mean(
                    stream["gap_speed_ms"].to_numpy(dtype=float),
                    stream["elapsed_s"].to_numpy(dtype=float),
                    duration_s,
                )
                if np.isfinite(value):
                    efforts.append(value)
            if efforts:
                best[duration_s] = max(efforts)
        try:
            fit = fit_cs_dprime(best, self._config)
        except ValueError:
            return AthleteParams(
                str(athlete_id),
                target_time,
                hrmax,
                np.nan,
                np.nan,
                np.nan,
                n_preceding,
                False,
                "insufficient_cs_durations",
            )
        if fit.r_squared < self._config["athlete_params"]["cs_min_r2"]:
            return AthleteParams(
                str(athlete_id),
                target_time,
                hrmax,
                fit.cs_ms,
                fit.dprime_m,
                fit.r_squared,
                n_preceding,
                False,
                "cs_r_squared_below_threshold",
            )
        return AthleteParams(
            str(athlete_id),
            target_time,
            hrmax,
            fit.cs_ms,
            fit.dprime_m,
            fit.r_squared,
            n_preceding,
            True,
            "",
        )

    @staticmethod
    def _ineligible(
        athlete_id: str,
        as_of: pd.Timestamp,
        n_preceding: int,
        reason: str,
    ) -> AthleteParams:
        return AthleteParams(
            athlete_id,
            as_of,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            n_preceding,
            False,
            reason,
        )


class ParquetHistoryLoader:
    """Load strictly preceding Phase-2 streams for one athlete."""

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        path = repo_path(config, config["data"]["preprocessed_activities"])
        self._activities = pd.read_parquet(path)
        self._root = repo_path(
            config, config["data"]["preprocessed_streams_dir"]
        )

    def __call__(
        self, athlete_id: str, as_of: pd.Timestamp
    ) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
        target_time = pd.Timestamp(as_of)
        if target_time.tzinfo is None:
            target_time = target_time.tz_localize("UTC")
        else:
            target_time = target_time.tz_convert("UTC")
        start_time = pd.to_datetime(self._activities["start_time"], utc=True)
        mask = (
            (self._activities["athlete_id"].astype(str) == str(athlete_id))
            & (start_time < target_time)
            & self._activities["phase2_accepted"].fillna(False)
        )
        history = self._activities.loc[mask].copy()
        streams: dict[str, pd.DataFrame] = {}
        for row in history.itertuples(index=False):
            path = (
                self._root
                / f"athlete={row.athlete_id}"
                / f"activity={row.activity_id}.parquet"
            )
            streams[str(row.activity_id)] = pd.read_parquet(path)
        return history, streams


_DEFAULT_ESTIMATOR: AthleteParameterEstimator | None = None


def configure_estimator(estimator: AthleteParameterEstimator) -> None:
    """Configure the module-level estimator used by the required public API."""
    global _DEFAULT_ESTIMATOR
    _DEFAULT_ESTIMATOR = estimator


def estimate_params(athlete_id: str, as_of: pd.Timestamp) -> AthleteParams:
    """Required leakage-aware public API."""
    if _DEFAULT_ESTIMATOR is None:
        raise RuntimeError("Call configure_estimator before estimate_params")
    return _DEFAULT_ESTIMATOR.estimate_params(athlete_id, as_of)
