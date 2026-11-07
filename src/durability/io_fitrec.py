"""Streaming FitRec loader and Phase-1 materialisation."""

from __future__ import annotations

import ast
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from durability.config import repo_path
from durability.derive import derive_distance_speed, smooth_speed

LOGGER = logging.getLogger(__name__)

# Verified against the raw corpus by ``scripts/00_verify_schema.py``. The
# verifier fails on missing configured fields instead of silently adapting to
# an unexpected release.
FIELD_MAP = {
    "activity_id": "id",
    "athlete_id": "userId",
    "sport": "sport",
    "timestamp": "timestamp",
    "heart_rate": "heart_rate",
    "altitude": "altitude",
    "latitude": "latitude",
    "longitude": "longitude",
    "gender": "gender",
    "url": "url",
}


def iter_fitrec_records(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield one safe, literal-decoded record at a time."""
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = ast.literal_eval(line)
            except (SyntaxError, ValueError) as exc:
                raise ValueError(f"Invalid record at line {line_number}") from exc
            if not isinstance(record, dict):
                raise TypeError(f"Record at line {line_number} is not a dict")
            yield record


def is_running(record: dict[str, Any], config: dict[str, Any]) -> bool:
    """Return whether the exact observed sport label is configured as running."""
    return record.get(FIELD_MAP["sport"]) in set(config["data"]["sports_keep"])


def validate_record_structure(record: dict[str, Any], config: dict[str, Any]) -> int:
    """Assert the measured raw schema invariants and return point count."""
    missing = set(config["data"]["sequential_keys"]) - record.keys()
    if missing:
        raise AssertionError(f"Missing sequential fields: {sorted(missing)}")
    lengths = {key: len(record[key]) for key in config["data"]["sequential_keys"]}
    if len(set(lengths.values())) != 1:
        raise AssertionError(f"Sequential channel lengths differ: {lengths}")
    n_points = next(iter(lengths.values()))
    if n_points < 2:
        raise AssertionError("Activity needs at least two points")
    timestamp = np.asarray(record[FIELD_MAP["timestamp"]], dtype=float)
    if np.any(np.diff(timestamp) <= 0):
        raise AssertionError("timestamp must be strictly increasing")
    altitude = np.asarray(record[FIELD_MAP["altitude"]], dtype=float)
    finite_altitude = altitude[np.isfinite(altitude)]
    altitude_limit = config["data"]["altitude_range_max_m"]
    if finite_altitude.size and np.ptp(finite_altitude) >= altitude_limit:
        raise AssertionError(
            f"Altitude range must be under {altitude_limit} m"
        )
    return n_points


def make_stream_frame(
    record: dict[str, Any], config: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, float | bool]]:
    """Build a raw-grid stream with derived SI-unit channels and QC metadata."""
    n_points = validate_record_structure(record, config)
    timestamp = np.asarray(record[FIELD_MAP["timestamp"]], dtype=np.int64)
    distance_km, speed_ms = derive_distance_speed(
        record[FIELD_MAP["latitude"]],
        record[FIELD_MAP["longitude"]],
        timestamp,
    )
    smoothed = smooth_speed(
        speed_ms, timestamp, config["data"]["speed_smooth_window_s"]
    )
    native_dt_s = float(np.median(np.diff(timestamp)))
    speed_low, speed_high = config["data"]["speed_sanity_ms"]
    median_speed_ms = float(np.nanmedian(speed_ms))
    speed_sanity_pass = bool(speed_low <= median_speed_ms <= speed_high)

    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamp, unit="s", utc=True),
            "elapsed_s": timestamp - timestamp[0],
            "heart_rate": np.asarray(record[FIELD_MAP["heart_rate"]], dtype=float),
            "altitude": np.asarray(record[FIELD_MAP["altitude"]], dtype=float),
            "latitude": np.asarray(record[FIELD_MAP["latitude"]], dtype=float),
            "longitude": np.asarray(record[FIELD_MAP["longitude"]], dtype=float),
            "derived_distance_km": distance_km,
            "derived_speed_ms": speed_ms,
            "smoothed_speed_ms": smoothed,
        }
    )
    assert len(frame) == n_points
    return frame, {
        "native_dt_s": native_dt_s,
        "native_n_points": n_points,
        "upsample_factor": config["resample"]["grid_dt_s"] / native_dt_s,
        "coarse_sampling": native_dt_s
        > config["resample"]["coarse_sampling_threshold_s"],
        "speed_sanity_pass": speed_sanity_pass,
        "median_speed_ms": median_speed_ms,
    }


def assert_running_speed(
    median_speed_ms: float, config: dict[str, Any]
) -> None:
    """Reject a running activity whose median speed signals bad GPS or units."""
    low, high = config["data"]["speed_sanity_ms"]
    assert low <= median_speed_ms <= high, (
        f"Median running speed {median_speed_ms:.3f} m/s outside [{low}, {high}]"
    )


def _stream_path(
    streams_dir: Path, athlete_id: str | int, activity_id: str | int
) -> Path:
    return streams_dir / f"athlete={athlete_id}" / f"activity={activity_id}.parquet"


@dataclass
class SchemaStats:
    total_records: int = 0
    running_records: int = 0
    running_athletes: set[str] = field(default_factory=set)
    sport_counts: Counter[str] = field(default_factory=Counter)
    running_durations_s: list[float] = field(default_factory=list)
    hr_out_of_range: int = 0
    speed_out_of_range: int = 0
    speed_rejected_activities: int = 0
    malformed_records: int = 0


def materialize_phase1(
    config: dict[str, Any],
    *,
    limit: int | None = None,
) -> tuple[pd.DataFrame, SchemaStats]:
    """Stream raw FitRec, materialising running inventory and stream Parquets."""
    raw_path = repo_path(config, config["data"]["raw_json"])
    interim_dir = repo_path(config, config["data"]["interim_dir"])
    streams_dir = interim_dir / "streams"
    streams_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    stats = SchemaStats()
    hr_low, hr_high = config["athlete_params"]["hr_valid_range"]
    speed_low, speed_high = config["athlete_params"]["speed_valid_range"]

    for record in iter_fitrec_records(raw_path):
        if limit is not None and stats.total_records >= limit:
            break
        stats.total_records += 1
        sport = str(record.get(FIELD_MAP["sport"], "<missing>"))
        stats.sport_counts[sport] += 1
        if not is_running(record, config):
            continue
        stats.running_records += 1
        athlete_id = str(record.get(FIELD_MAP["athlete_id"]))
        activity_id = str(record.get(FIELD_MAP["activity_id"]))
        stats.running_athletes.add(athlete_id)
        raw_timestamp = record.get(FIELD_MAP["timestamp"], [])
        raw_duration_s = (
            float(raw_timestamp[-1] - raw_timestamp[0])
            if len(raw_timestamp) >= 2
            else np.nan
        )
        if np.isfinite(raw_duration_s) and raw_duration_s >= 0:
            stats.running_durations_s.append(raw_duration_s)

        try:
            frame, metadata = make_stream_frame(record, config)
        except (AssertionError, TypeError, ValueError) as exc:
            stats.malformed_records += 1
            LOGGER.debug("Rejected activity %s: %s", activity_id, exc)
            rows.append(
                {
                    "activity_id": activity_id,
                    "athlete_id": athlete_id,
                    "sport": sport,
                    "duration_s": raw_duration_s,
                    "n_points": len(raw_timestamp),
                    "accepted": False,
                    "rejection_reason": str(exc),
                    "stream_materialized": False,
                }
            )
            continue

        duration_s = float(frame["elapsed_s"].iloc[-1])
        hr = frame["heart_rate"].to_numpy()
        speed = frame["derived_speed_ms"].to_numpy()
        stats.hr_out_of_range += int(
            np.count_nonzero(np.isfinite(hr) & ((hr < hr_low) | (hr > hr_high)))
        )
        stats.speed_out_of_range += int(
            np.count_nonzero(
                np.isfinite(speed) & ((speed < speed_low) | (speed > speed_high))
            )
        )
        accepted = bool(metadata["speed_sanity_pass"])
        if not accepted:
            stats.speed_rejected_activities += 1

        stream_path = _stream_path(streams_dir, athlete_id, activity_id)
        stream_path.parent.mkdir(parents=True, exist_ok=True)
        if not stream_path.exists():
            frame.to_parquet(stream_path, index=False)
        altitude = frame["altitude"].to_numpy()
        ascent = np.diff(altitude)
        rows.append(
            {
                "activity_id": activity_id,
                "athlete_id": athlete_id,
                "sport": sport,
                "start_time": frame["timestamp"].iloc[0],
                "duration_s": duration_s,
                "n_points": len(frame),
                "distance_m": float(frame["derived_distance_km"].sum() * 1000.0),
                "has_hr": bool(np.isfinite(hr).any()),
                "has_altitude": bool(np.isfinite(altitude).any()),
                "mean_heart_rate": float(np.nanmean(hr)),
                "max_heart_rate": float(np.nanmax(hr)),
                "mean_speed_ms": float(np.nanmean(speed)),
                "median_speed_ms": metadata["median_speed_ms"],
                "total_ascent_m": float(np.nansum(ascent[ascent > 0])),
                "native_dt_s": metadata["native_dt_s"],
                "native_n_points": metadata["native_n_points"],
                "upsample_factor": metadata["upsample_factor"],
                "coarse_sampling": metadata["coarse_sampling"],
                "accepted": accepted,
                "rejection_reason": "" if accepted else "median_speed_sanity",
                "stream_materialized": True,
                "gender": record.get(FIELD_MAP["gender"]),
                "url": record.get(FIELD_MAP["url"]),
            }
        )

    activities = pd.DataFrame(rows)
    interim_dir.mkdir(parents=True, exist_ok=True)
    activities.to_parquet(interim_dir / "activities.parquet", index=False)
    return activities, stats


def reload_stream(
    config: dict[str, Any], athlete_id: str | int, activity_id: str | int
) -> pd.DataFrame:
    """Reload one materialised activity stream."""
    streams_dir = repo_path(config, config["data"]["interim_dir"]) / "streams"
    return pd.read_parquet(_stream_path(streams_dir, athlete_id, activity_id))
