"""Durability indices and leakage-safe batch parameter construction."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from durability.athlete_params import (
    AthleteParams,
    _fastest_rolling_mean,
    _sustained_hrmax,
    fit_cs_dprime,
)
from durability.config import repo_path
from durability.preprocess import reload_preprocessed_stream

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ActivityIndices:
    magnitude: float
    onset: float
    onset_elapsed_s: float
    drift: float
    cs_shift: float
    retention: float
    early_decoupling_mean: float
    late_decoupling_mean: float
    retained_duration_s: float
    n_retained_samples: int
    magnitude_available: bool
    onset_available: bool
    drift_available: bool
    cs_shift_available: bool
    retention_available: bool


def _finite_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else np.nan


def _fit_cs_for_segment(
    speed: np.ndarray,
    elapsed_s: np.ndarray,
    config: dict[str, Any],
) -> float:
    best: dict[float, float] = {}
    segment_time = elapsed_s - elapsed_s[0]
    for duration_min in config["athlete_params"]["cs_durations_min"]:
        duration_s = float(duration_min) * 60.0
        value = _fastest_rolling_mean(speed, segment_time, duration_s)
        if np.isfinite(value):
            best[duration_s] = value
    minimum = int(config["indices"]["cs_shift_min_durations"])
    if len(best) < minimum:
        return np.nan
    duration = np.asarray(sorted(best), dtype=float)
    distance = np.asarray([best[item] * item for item in duration], dtype=float)
    design = np.column_stack([duration, np.ones(len(duration))])
    coefficients, *_ = np.linalg.lstsq(design, distance, rcond=None)
    return float(coefficients[0])


def compute_activity_indices(
    stream: pd.DataFrame,
    params: AthleteParams,
    config: dict[str, Any],
    *,
    onset_sampling_eligible: bool = True,
) -> ActivityIndices:
    """Compute Phase-3 indices on one target activity."""
    if not params.eligible:
        raise ValueError("Athlete parameters must be eligible")
    required = {"elapsed_s", "heart_rate", "gap_speed_ms"}
    missing = required - set(stream.columns)
    if missing:
        raise AssertionError(f"Missing index input columns: {sorted(missing)}")

    elapsed = stream["elapsed_s"].to_numpy(dtype=float)
    hr = stream["heart_rate"].to_numpy(dtype=float)
    gap_speed = stream["gap_speed_ms"].to_numpy(dtype=float)
    if len(stream) < 2 or elapsed[-1] <= elapsed[0]:
        raise AssertionError("Activity is too short for indices")
    duration_s = float(elapsed[-1] - elapsed[0])
    fraction = (elapsed - elapsed[0]) / duration_s
    head = float(config["indices"]["trim_head_frac"])
    tail = float(config["indices"]["trim_tail_frac"])
    retained_mask = (fraction >= head) & (fraction <= 1.0 - tail)
    retained_indices = np.flatnonzero(retained_mask)
    if retained_indices.size < 2:
        raise AssertionError("Retained span has fewer than two samples")
    retained_start = float(elapsed[retained_indices[0]])
    retained_end = float(elapsed[retained_indices[-1]])
    retained_duration = retained_end - retained_start

    valid_external = np.isfinite(gap_speed) & (gap_speed > 0)
    decoupling = np.full(len(stream), np.nan, dtype=float)
    decoupling[valid_external] = (
        hr[valid_external] / params.hrmax_bpm
    ) / (gap_speed[valid_external] / params.cs_ms)

    early_start, early_end = config["indices"]["early_window"]
    early_mask = (fraction >= early_start) & (fraction < early_end)
    early_mean = _finite_mean(decoupling[early_mask])
    late_start_s = retained_end - (
        config["indices"]["late_window_frac"] * retained_duration
    )
    late_mask = retained_mask & (elapsed >= late_start_s)
    late_mean = _finite_mean(decoupling[late_mask])
    magnitude = (
        late_mean / early_mean
        if np.isfinite(early_mean) and early_mean > 0 and np.isfinite(late_mean)
        else np.nan
    )

    onset = np.nan
    onset_elapsed_s = np.nan
    if (
        onset_sampling_eligible
        and np.isfinite(early_mean)
        and early_mean > 0
    ):
        retained_decoupling = decoupling[retained_mask] / early_mean
        retained_time = elapsed[retained_mask]
        grid_dt_s = float(np.median(np.diff(retained_time)))
        rolling_points = max(
            1,
            int(round(config["indices"]["onset_rolling_s"] / grid_dt_s)),
        )
        rolling = (
            pd.Series(retained_decoupling)
            .rolling(rolling_points, center=True, min_periods=1)
            .mean()
            .to_numpy(dtype=float)
        )
        above = (
            np.isfinite(rolling)
            & (rolling > config["indices"]["onset_threshold"])
        )
        persistent = np.logical_and.accumulate(above[::-1])[::-1]
        candidates = np.flatnonzero(persistent)
        if candidates.size:
            onset_elapsed_s = float(retained_time[candidates[0]])
            onset = (onset_elapsed_s - retained_start) / retained_duration

    drift = np.nan
    retained_speed = gap_speed[retained_mask]
    finite_retained_speed = retained_speed[np.isfinite(retained_speed)]
    speed_median = (
        float(np.median(finite_retained_speed))
        if finite_retained_speed.size
        else np.nan
    )
    if np.isfinite(speed_median) and speed_median > 0:
        tolerance = config["indices"]["drift_speed_tolerance"]
        steady = (
            retained_mask
            & np.isfinite(hr)
            & np.isfinite(gap_speed)
            & (np.abs(gap_speed - speed_median) <= tolerance * speed_median)
        )
        if steady.sum() >= 2:
            time_h = (elapsed[steady] - elapsed[steady][0]) / 3_600.0
            drift = float(np.polyfit(time_h, hr[steady], 1)[0])

    cs_shift = np.nan
    gap_available = (
        bool(stream["gap_available"].all())
        if "gap_available" in stream
        else True
    )
    if gap_available:
        midpoint = retained_start + retained_duration / 2.0
        early_half = retained_mask & (elapsed <= midpoint)
        late_half = retained_mask & (elapsed >= midpoint)
        if early_half.sum() >= 2 and late_half.sum() >= 2:
            cs_early = _fit_cs_for_segment(
                gap_speed[early_half], elapsed[early_half], config
            )
            cs_late = _fit_cs_for_segment(
                gap_speed[late_half], elapsed[late_half], config
            )
            if np.isfinite(cs_early) and np.isfinite(cs_late):
                cs_shift = cs_late - cs_early

    retained_time = elapsed[retained_mask]
    quarter_duration = retained_duration / 4.0
    first_quarter = retained_mask & (
        elapsed <= retained_start + quarter_duration
    )
    final_quarter = retained_mask & (
        elapsed >= retained_end - quarter_duration
    )
    first_speed = _finite_mean(gap_speed[first_quarter])
    final_speed = _finite_mean(gap_speed[final_quarter])
    retention = (
        final_speed / first_speed
        if np.isfinite(first_speed) and first_speed > 0 and np.isfinite(final_speed)
        else np.nan
    )
    return ActivityIndices(
        magnitude=float(magnitude),
        onset=float(onset),
        onset_elapsed_s=float(onset_elapsed_s),
        drift=float(drift),
        cs_shift=float(cs_shift),
        retention=float(retention),
        early_decoupling_mean=early_mean,
        late_decoupling_mean=late_mean,
        retained_duration_s=retained_duration,
        n_retained_samples=int(retained_mask.sum()),
        magnitude_available=bool(np.isfinite(magnitude)),
        onset_available=bool(np.isfinite(onset)),
        drift_available=bool(np.isfinite(drift)),
        cs_shift_available=bool(np.isfinite(cs_shift)),
        retention_available=bool(np.isfinite(retention)),
    )


def build_parameter_evidence(
    activities: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    """Summarise each activity once for efficient historical parameter fits."""
    rows: list[dict[str, Any]] = []
    durations_min = config["athlete_params"]["cs_durations_min"]
    progress_every = int(config["data"]["indices_progress_every"])
    for number, row in enumerate(activities.itertuples(index=False), start=1):
        stream = reload_preprocessed_stream(
            config, row.athlete_id, row.activity_id
        )
        evidence: dict[str, Any] = {
            "activity_id": str(row.activity_id),
            "athlete_id": str(row.athlete_id),
            "start_time": row.start_time,
            "sustained_hrmax_bpm": _sustained_hrmax(
                {str(row.activity_id): stream},
                config["athlete_params"]["hrmax_sustain_s"],
                config["athlete_params"]["hr_valid_range"],
            ),
        }
        elapsed = stream["elapsed_s"].to_numpy(dtype=float)
        speed = stream["gap_speed_ms"].to_numpy(dtype=float)
        gap_available = bool(stream["gap_available"].all())
        for duration_min in durations_min:
            column = f"best_{duration_min}min_ms"
            evidence[column] = (
                _fastest_rolling_mean(
                    speed, elapsed, float(duration_min) * 60.0
                )
                if gap_available
                else np.nan
            )
        rows.append(evidence)
        if number % progress_every == 0:
            LOGGER.info(
                "Parameter evidence progress: %s/%s",
                number,
                len(activities),
            )
    result = pd.DataFrame(rows)
    path = repo_path(config, config["data"]["parameter_evidence"])
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(path, index=False)
    return result


def build_historical_params(
    evidence: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    """Build as-of parameter rows, updating state only after equal-time targets."""
    duration_columns = {
        float(duration_min) * 60.0: f"best_{duration_min}min_ms"
        for duration_min in config["athlete_params"]["cs_durations_min"]
    }
    output: list[dict[str, Any]] = []
    minimum_activities = int(
        config["athlete_params"]["min_preceding_activities"]
    )
    for athlete_id, athlete in evidence.groupby("athlete_id", sort=False):
        ordered = athlete.copy()
        ordered["start_time"] = pd.to_datetime(
            ordered["start_time"], utc=True
        )
        ordered = ordered.sort_values(["start_time", "activity_id"])
        n_preceding = 0
        historical_hrmax = np.nan
        historical_best = {
            duration_s: np.nan for duration_s in duration_columns
        }
        latest_history_time: pd.Timestamp | None = None
        for start_time, targets in ordered.groupby("start_time", sort=True):
            target_time = pd.Timestamp(start_time)
            if latest_history_time is not None:
                assert latest_history_time < target_time, (
                    "Historical parameter state must be strictly before target"
                )
            for target in targets.itertuples(index=False):
                eligible = False
                reason = ""
                cs_ms = dprime_m = r_squared = np.nan
                available_best = {
                    duration: speed
                    for duration, speed in historical_best.items()
                    if np.isfinite(speed)
                }
                if n_preceding < minimum_activities:
                    reason = (
                        f"fewer_than_{minimum_activities}_preceding_activities"
                    )
                elif not np.isfinite(historical_hrmax):
                    reason = "hrmax_unavailable"
                else:
                    try:
                        fit = fit_cs_dprime(available_best, config)
                        cs_ms = fit.cs_ms
                        dprime_m = fit.dprime_m
                        r_squared = fit.r_squared
                        if r_squared < config["athlete_params"]["cs_min_r2"]:
                            reason = "cs_r_squared_below_threshold"
                        else:
                            eligible = True
                    except ValueError:
                        reason = "insufficient_cs_durations"
                output.append(
                    {
                        "activity_id": str(target.activity_id),
                        "athlete_id": str(athlete_id),
                        "as_of": target_time,
                        "hrmax_bpm": historical_hrmax,
                        "cs_ms": cs_ms,
                        "dprime_m": dprime_m,
                        "cs_r_squared": r_squared,
                        "n_preceding_activities": n_preceding,
                        "params_eligible": eligible,
                        "params_rejection_reason": reason,
                    }
                )
            # Equal-time activities become history only after every target at
            # this timestamp has been estimated.
            for target in targets.itertuples(index=False):
                if np.isfinite(target.sustained_hrmax_bpm):
                    historical_hrmax = float(
                        np.nanmax(
                            [historical_hrmax, target.sustained_hrmax_bpm]
                        )
                    )
                for duration_s, column in duration_columns.items():
                    value = getattr(target, column)
                    if np.isfinite(value):
                        historical_best[duration_s] = float(
                            np.nanmax([historical_best[duration_s], value])
                        )
                n_preceding += 1
            latest_history_time = target_time
    return pd.DataFrame(output)


def materialize_phase3(
    config: dict[str, Any], *, limit: int | None = None
) -> pd.DataFrame:
    """Build leakage-safe parameters and durability indices for the corpus."""
    activity_path = repo_path(
        config, config["data"]["preprocessed_activities"]
    )
    phase2 = pd.read_parquet(activity_path)
    phase2 = phase2[phase2["phase2_accepted"].fillna(False)].copy()
    if limit is not None:
        phase2 = phase2.head(limit)
    evidence = build_parameter_evidence(phase2, config)
    historical = build_historical_params(evidence, config)
    targets = phase2.merge(
        historical,
        on=["activity_id", "athlete_id"],
        how="left",
        validate="one_to_one",
    )
    rows: list[dict[str, Any]] = []
    progress_every = int(config["data"]["indices_progress_every"])
    for number, target in enumerate(targets.itertuples(index=False), start=1):
        base = {
            "activity_id": str(target.activity_id),
            "athlete_id": str(target.athlete_id),
            "start_time": target.start_time,
            "sport": target.sport,
            "duration_s": target.duration_s,
            "phase2_retained_duration_s": target.retained_duration_s,
            "gap_split": target.gap_split,
            "gap_available": target.gap_available,
            "onset_sampling_eligible": target.onset_sampling_eligible,
            "hr_coverage": target.hr_coverage,
            "clipped_gradient_fraction": target.clipped_gradient_fraction,
            "hrmax_bpm": target.hrmax_bpm,
            "cs_ms": target.cs_ms,
            "dprime_m": target.dprime_m,
            "cs_r_squared": target.cs_r_squared,
            "n_preceding_activities": target.n_preceding_activities,
            "params_eligible": target.params_eligible,
            "params_rejection_reason": target.params_rejection_reason,
        }
        if not bool(target.params_eligible):
            rows.append(base | {"indices_computed": False})
            continue
        params = AthleteParams(
            athlete_id=str(target.athlete_id),
            as_of=pd.Timestamp(target.as_of),
            hrmax_bpm=float(target.hrmax_bpm),
            cs_ms=float(target.cs_ms),
            dprime_m=float(target.dprime_m),
            cs_r_squared=float(target.cs_r_squared),
            n_preceding_activities=int(target.n_preceding_activities),
            eligible=True,
            rejection_reason="",
        )
        try:
            stream = reload_preprocessed_stream(
                config, target.athlete_id, target.activity_id
            )
            indices = compute_activity_indices(
                stream,
                params,
                config,
                onset_sampling_eligible=bool(
                    target.onset_sampling_eligible
                ),
            )
            base.update(asdict(indices))
            base["indices_computed"] = True
            base["indices_rejection_reason"] = ""
        except (AssertionError, KeyError, TypeError, ValueError) as exc:
            base["indices_computed"] = False
            base["indices_rejection_reason"] = str(exc)
        rows.append(base)
        if number % progress_every == 0:
            LOGGER.info("Index progress: %s/%s", number, len(targets))
    result = pd.DataFrame(rows)
    output_path = repo_path(config, config["data"]["indices_output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    return result


def write_phase3_summary(
    indices: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    """Write compact MEASURED Phase-3 availability statistics."""
    computed = indices[indices["indices_computed"].fillna(False)]
    long_mask = (
        computed["phase2_retained_duration_s"]
        >= config["qc"]["min_duration_min"] * 60
    )
    metrics = {
        "input_activities": len(indices),
        "parameter_eligible_activities": int(
            indices["params_eligible"].fillna(False).sum()
        ),
        "indices_computed_activities": len(computed),
        "magnitude_available": int(
            computed["magnitude_available"].fillna(False).sum()
        ),
        "onset_available": int(
            computed["onset_available"].fillna(False).sum()
        ),
        "drift_available": int(
            computed["drift_available"].fillna(False).sum()
        ),
        "cs_shift_available": int(
            computed["cs_shift_available"].fillna(False).sum()
        ),
        "retention_available": int(
            computed["retention_available"].fillna(False).sum()
        ),
        "long_indices_computed": int(long_mask.sum()),
        "long_magnitude_available": int(
            (long_mask & computed["magnitude"].notna()).sum()
        ),
        "long_onset_available": int(
            (long_mask & computed["onset"].notna()).sum()
        ),
        "long_drift_available": int(
            (long_mask & computed["drift"].notna()).sum()
        ),
        "long_cs_shift_available": int(
            (long_mask & computed["cs_shift"].notna()).sum()
        ),
        "long_retention_available": int(
            (long_mask & computed["retention"].notna()).sum()
        ),
    }
    summary = pd.DataFrame(
        {
            "metric": list(metrics),
            "value": list(metrics.values()),
            "provenance": "MEASURED",
        }
    )
    path = repo_path(config, config["data"]["phase3_summary"])
    path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(path, index=False)
    return summary
