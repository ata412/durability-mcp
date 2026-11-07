"""Phase-4 activity eligibility and auditable QC waterfall."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from durability.config import repo_path
from durability.preprocess import reload_preprocessed_stream

LOGGER = logging.getLogger(__name__)


def build_waterfall(
    frame: pd.DataFrame,
    conditions: list[tuple[str, pd.Series]],
) -> tuple[pd.Series, pd.DataFrame]:
    """Apply ordered criteria and return a waterfall whose counts reconcile."""
    remaining = pd.Series(True, index=frame.index, dtype=bool)
    rows: list[dict[str, Any]] = []
    for criterion, condition in conditions:
        before = int(remaining.sum())
        valid = condition.reindex(frame.index).fillna(False).astype(bool)
        remaining &= valid
        after = int(remaining.sum())
        rows.append(
            {
                "criterion": criterion,
                "before": before,
                "removed": before - after,
                "remaining": after,
                "provenance": "MEASURED",
            }
        )
    waterfall = pd.DataFrame(rows)
    if not waterfall.empty:
        assert (
            waterfall["before"] - waterfall["removed"]
            == waterfall["remaining"]
        ).all()
        assert int(waterfall["removed"].sum() + remaining.sum()) == len(frame)
    return remaining, waterfall


def _gap_speed_cv(stream: pd.DataFrame) -> float:
    speed = stream["gap_speed_ms"].to_numpy(dtype=float)
    finite = speed[np.isfinite(speed)]
    if finite.size < 2:
        return np.nan
    mean = float(finite.mean())
    return float(finite.std(ddof=0) / mean) if mean > 0 else np.nan


def materialize_qc(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute GAP-speed CV, apply all ordered QC criteria, and write outputs."""
    indices_path = repo_path(config, config["data"]["indices_output"])
    frame = pd.read_parquet(indices_path)
    phase1_path = (
        repo_path(config, config["data"]["interim_dir"]) / "activities.parquet"
    )
    phase1 = pd.read_parquet(
        phase1_path, columns=["activity_id", "athlete_id", "has_hr"]
    )
    frame = frame.merge(
        phase1,
        on=["activity_id", "athlete_id"],
        how="left",
        validate="one_to_one",
    )
    min_duration_s = config["qc"]["min_duration_min"] * 60
    max_duration_s = config["qc"]["max_duration_min"] * 60
    running = frame["sport"].isin(config["data"]["sports_keep"])
    has_hr = frame["has_hr"].fillna(False).astype(bool)
    hr_coverage = frame["hr_coverage"] >= config["qc"]["min_hr_coverage"]
    duration = frame["duration_s"].between(
        min_duration_s, max_duration_s, inclusive="both"
    )
    gap_block = (frame["gap_split"] != True) | (  # noqa: E712
        frame["phase2_retained_duration_s"] >= min_duration_s
    )

    before_cv = running & has_hr & hr_coverage & duration & gap_block
    frame["gap_speed_cv"] = np.nan
    candidate_indices = frame.index[before_cv]
    progress_every = int(config["data"]["qc_progress_every"])
    for number, index in enumerate(candidate_indices, start=1):
        row = frame.loc[index]
        stream = reload_preprocessed_stream(
            config, row["athlete_id"], row["activity_id"]
        )
        frame.at[index, "gap_speed_cv"] = _gap_speed_cv(stream)
        if number % progress_every == 0:
            LOGGER.info(
                "QC speed-CV progress: %s/%s",
                number,
                len(candidate_indices),
            )
    speed_variability = (
        frame["gap_speed_cv"] < config["qc"]["max_gap_speed_cv"]
    )
    gradient_quality = (
        frame["clipped_gradient_fraction"]
        < config["qc"]["max_clipped_gradient_frac"]
    )
    parameters = (
        frame["params_eligible"].fillna(False)
        & (
            frame["n_preceding_activities"]
            >= config["athlete_params"]["min_preceding_activities"]
        )
    )
    indices_available = (
        frame["indices_computed"].fillna(False)
        & frame["magnitude"].notna()
        & frame["drift"].notna()
    )
    conditions = [
        ("sport_is_running", running),
        ("has_heart_rate", has_hr),
        ("heart_rate_coverage", hr_coverage),
        ("duration_range", duration),
        ("longest_gap_block", gap_block),
        ("gap_speed_variability", speed_variability),
        ("gradient_clipping", gradient_quality),
        ("preceding_athlete_parameters", parameters),
        ("durability_indices_available", indices_available),
    ]
    eligible, waterfall = build_waterfall(frame, conditions)
    frame["qc_eligible"] = eligible
    output_path = repo_path(config, config["data"]["qc_activities"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    waterfall_path = repo_path(config, config["outputs"]["qc_waterfall"])
    waterfall_path.parent.mkdir(parents=True, exist_ok=True)
    waterfall.to_csv(waterfall_path, index=False)
    return frame, waterfall

