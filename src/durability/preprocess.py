"""Phase-2 corpus preprocessing and materialisation."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from durability.config import repo_path
from durability.gap import add_grade_adjusted_speed
from durability.io_fitrec import reload_stream
from durability.resample import resample_activity

LOGGER = logging.getLogger(__name__)


@dataclass
class PreprocessStats:
    input_activities: int = 0
    written_activities: int = 0
    rejected_activities: int = 0
    gap_split_activities: int = 0
    onset_sampling_ineligible: int = 0
    gap_unavailable_activities: int = 0


def _output_path(
    root: Path, athlete_id: str | int, activity_id: str | int
) -> Path:
    return root / f"athlete={athlete_id}" / f"activity={activity_id}.parquet"


def preprocess_activity(
    frame: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Resample and grade-adjust one Phase-1 stream."""
    resampled, resample_metadata = resample_activity(frame, config)
    processed, gap_metadata = add_grade_adjusted_speed(resampled, config)
    metadata = asdict(resample_metadata) | asdict(gap_metadata)
    metadata["hr_coverage"] = float(processed["heart_rate"].notna().mean())
    metadata["n_resampled_points"] = len(processed)
    return processed, metadata


def materialize_phase2(
    config: dict[str, Any], *, limit: int | None = None
) -> tuple[pd.DataFrame, PreprocessStats]:
    """Materialise resampled, GAP-adjusted streams for Phase-1 accepted runs."""
    activities_path = (
        repo_path(config, config["data"]["interim_dir"]) / "activities.parquet"
    )
    activities = pd.read_parquet(activities_path)
    candidates = activities[activities["accepted"].fillna(False)].copy()
    if limit is not None:
        candidates = candidates.head(limit)
    output_root = repo_path(config, config["data"]["preprocessed_streams_dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    stats = PreprocessStats(input_activities=len(candidates))
    progress_every = int(config["data"]["preprocess_progress_every"])

    for number, row in enumerate(candidates.itertuples(index=False), start=1):
        base = row._asdict()
        try:
            raw_stream = reload_stream(
                config, row.athlete_id, row.activity_id
            )
            processed, metadata = preprocess_activity(raw_stream, config)
        except (AssertionError, KeyError, TypeError, ValueError) as exc:
            stats.rejected_activities += 1
            base.update(
                {
                    "phase2_accepted": False,
                    "phase2_rejection_reason": str(exc),
                }
            )
            rows.append(base)
            continue

        output_path = _output_path(
            output_root, row.athlete_id, row.activity_id
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        processed.to_parquet(output_path, index=False)
        stats.written_activities += 1
        stats.gap_split_activities += int(metadata["gap_split"])
        stats.onset_sampling_ineligible += int(
            not metadata["onset_sampling_eligible"]
        )
        stats.gap_unavailable_activities += int(
            not metadata["gap_available"]
        )
        base.update(metadata)
        base.update(
            {
                "phase2_accepted": True,
                "phase2_rejection_reason": "",
            }
        )
        rows.append(base)
        if number % progress_every == 0:
            LOGGER.info(
                "Phase 2 progress: %s/%s activities",
                number,
                len(candidates),
            )

    result = pd.DataFrame(rows)
    result_path = repo_path(config, config["data"]["preprocessed_activities"])
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(result_path, index=False)
    return result, stats


def reload_preprocessed_stream(
    config: dict[str, Any], athlete_id: str | int, activity_id: str | int
) -> pd.DataFrame:
    root = repo_path(config, config["data"]["preprocessed_streams_dir"])
    return pd.read_parquet(_output_path(root, athlete_id, activity_id))


def write_phase2_summary(
    activities: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    """Write compact MEASURED Phase-2 corpus statistics."""
    accepted = activities[activities["phase2_accepted"].fillna(False)]
    long_mask = accepted["retained_duration_s"] >= (
        config["qc"]["min_duration_min"] * 60
    )
    onset_mask = accepted["onset_sampling_eligible"] == True  # noqa: E712
    metrics = {
        "input_activities": len(activities),
        "accepted_activities": len(accepted),
        "rejected_activities": len(activities) - len(accepted),
        "accepted_athletes": accepted["athlete_id"].nunique(),
        "gap_split_activities": int((accepted["gap_split"] == True).sum()),  # noqa: E712
        "onset_sampling_eligible": int(onset_mask.sum()),
        "onset_sampling_ineligible": int((~onset_mask).sum()),
        "gap_available_activities": int(
            (accepted["gap_available"] == True).sum()  # noqa: E712
        ),
        "long_retained_activities": int(long_mask.sum()),
        "long_onset_eligible_activities": int((long_mask & onset_mask).sum()),
        "median_native_dt_s": float(accepted["native_dt_s"].median()),
        "median_retained_duration_min": float(
            accepted["retained_duration_s"].median() / 60.0
        ),
        "max_distance_speed_relative_error": float(
            accepted["distance_speed_relative_error"].max()
        ),
    }
    summary = pd.DataFrame(
        {
            "metric": list(metrics),
            "value": list(metrics.values()),
            "provenance": "MEASURED",
        }
    )
    path = repo_path(config, config["data"]["phase2_summary"])
    path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(path, index=False)
    return summary
