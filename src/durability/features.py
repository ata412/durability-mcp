"""Leakage-safe history and early-window feature construction."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from durability.config import repo_path
from durability.preprocess import reload_preprocessed_stream

LOGGER = logging.getLogger(__name__)
IDENTIFIER_COLUMNS = ("activity_id", "athlete_id", "start_time")
TARGET_COLUMNS = ("magnitude", "onset", "retention", "drift")


def _utc_timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _finite_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else np.nan


def _linear_slope(
    x: np.ndarray, y: np.ndarray, minimum_observations: int
) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < minimum_observations:
        return np.nan
    x_valid = x[finite]
    if np.ptp(x_valid) <= 0:
        return np.nan
    return float(np.polyfit(x_valid, y[finite], 1)[0])


def _constant_value(frame: pd.DataFrame, column: str) -> float:
    if column not in frame:
        raise AssertionError(f"Early window missing {column}")
    values = frame[column].dropna().unique()
    if len(values) != 1:
        raise AssertionError(f"{column} must be constant in early window")
    return float(values[0])


def prepare_early_window(
    full_stream: pd.DataFrame,
    target_metadata: dict[str, float],
    config: dict[str, Any],
) -> pd.DataFrame:
    """Expose only the configured early fraction of a target stream."""
    if "elapsed_s" not in full_stream:
        raise AssertionError("Target stream missing elapsed_s")
    elapsed = full_stream["elapsed_s"].to_numpy(dtype=float)
    if len(elapsed) < 2 or not np.isfinite(elapsed).all():
        raise AssertionError("Target elapsed_s must contain finite samples")
    start = float(elapsed.min())
    duration_s = float(elapsed.max() - start)
    if duration_s <= 0:
        raise AssertionError("Target duration must be positive")
    fraction = float(config["features"]["early_window_frac"])
    cutoff = start + fraction * duration_s
    early = full_stream.loc[full_stream["elapsed_s"] <= cutoff].copy()
    early["target_duration_s"] = duration_s
    early["early_cutoff_elapsed_s"] = cutoff
    for column in ("hrmax_bpm", "cs_ms", "dprime_m"):
        if column not in target_metadata:
            raise AssertionError(f"Target metadata missing {column}")
        early[column] = float(target_metadata[column])
    return early.reset_index(drop=True)


def feature_outputs_equal(
    left: dict[str, Any], right: dict[str, Any]
) -> bool:
    """Compare feature dictionaries with NaN- and timestamp-aware semantics."""
    if set(left) != set(right):
        return False
    for key in left:
        left_value = left[key]
        right_value = right[key]
        if pd.isna(left_value) and pd.isna(right_value):
            continue
        if isinstance(left_value, (int, float, np.number, bool)) and isinstance(
            right_value, (int, float, np.number, bool)
        ):
            if not np.isclose(
                float(left_value),
                float(right_value),
                equal_nan=True,
            ):
                return False
        elif left_value != right_value:
            return False
    return True


class FeatureBuilder:
    """Build features from strict prior history plus an already-sliced target."""

    def __init__(
        self, history_activities: pd.DataFrame, config: dict[str, Any]
    ) -> None:
        history = history_activities.copy()
        required = {
            "activity_id",
            "athlete_id",
            "start_time",
            "duration_s",
            "distance_m",
        }
        missing = required - set(history.columns)
        if missing:
            raise AssertionError(
                f"History table missing columns: {sorted(missing)}"
            )
        history["athlete_id"] = history["athlete_id"].astype(str)
        history["activity_id"] = history["activity_id"].astype(str)
        history["start_time"] = pd.to_datetime(
            history["start_time"], utc=True
        )
        self._history = {
            athlete_id: athlete.sort_values(
                ["start_time", "activity_id"]
            ).reset_index(drop=True)
            for athlete_id, athlete in history.groupby(
                "athlete_id", sort=False
            )
        }
        self._config = config

    def _prior_history(
        self, athlete_id: str, target_time: pd.Timestamp
    ) -> pd.DataFrame:
        athlete = self._history.get(str(athlete_id))
        if athlete is None:
            return pd.DataFrame()
        history = athlete[athlete["start_time"] < target_time].copy()
        if not history.empty:
            assert (history["start_time"] < target_time).all(), (
                "Every feature-history activity must be strictly before target"
            )
            assert (history["athlete_id"] == str(athlete_id)).all()
        return history

    def _history_window(
        self,
        history: pd.DataFrame,
        target_time: pd.Timestamp,
        days: int,
    ) -> pd.DataFrame:
        if history.empty:
            return history
        lower = target_time - pd.Timedelta(days=int(days))
        return history[history["start_time"] >= lower]

    def _intensity_distribution(
        self,
        history: pd.DataFrame,
        hrmax_bpm: float,
    ) -> tuple[float, float, float]:
        if history.empty or "hr_histogram" not in history:
            return np.nan, np.nan, np.nan
        histograms = [
            np.asarray(value, dtype=float)
            for value in history["hr_histogram"]
            if value is not None
        ]
        if not histograms:
            return np.nan, np.nan, np.nan
        expected = (
            int(self._config["athlete_params"]["hr_valid_range"][1])
            - int(self._config["athlete_params"]["hr_valid_range"][0])
            + 1
        )
        if any(len(histogram) != expected for histogram in histograms):
            raise AssertionError("HR histogram has unexpected bin count")
        counts = np.sum(np.stack(histograms), axis=0)
        total = float(counts.sum())
        if total <= 0 or not np.isfinite(hrmax_bpm):
            return np.nan, np.nan, np.nan
        low_hr, high_hr = self._config["athlete_params"]["hr_valid_range"]
        centers = np.arange(int(low_hr), int(high_hr) + 1)
        zone_low, zone_high = self._config["features"][
            "hr_zone_fractions"
        ]
        low_cutoff = float(zone_low) * hrmax_bpm
        high_cutoff = float(zone_high) * hrmax_bpm
        low_fraction = float(counts[centers < low_cutoff].sum() / total)
        middle_fraction = float(
            counts[
                (centers >= low_cutoff) & (centers < high_cutoff)
            ].sum()
            / total
        )
        high_fraction = float(counts[centers >= high_cutoff].sum() / total)
        return low_fraction, middle_fraction, high_fraction

    def _parameter_trend(
        self,
        history: pd.DataFrame,
        target_time: pd.Timestamp,
        column: str,
    ) -> float:
        window_days = int(
            self._config["features"]["parameter_trend_days"]
        )
        recent = self._history_window(history, target_time, window_days)
        if recent.empty or column not in recent:
            return np.nan
        time_days = (
            recent["start_time"] - target_time
        ).dt.total_seconds().to_numpy(dtype=float) / 86_400.0
        values = recent[column].to_numpy(dtype=float)
        slope_per_day = _linear_slope(
            time_days,
            values,
            int(self._config["features"]["min_slope_observations"]),
        )
        return (
            float(slope_per_day * window_days)
            if np.isfinite(slope_per_day)
            else np.nan
        )

    def _history_features(
        self,
        history: pd.DataFrame,
        target_time: pd.Timestamp,
        hrmax_bpm: float,
    ) -> dict[str, Any]:
        features: dict[str, Any] = {
            "history_activity_count": int(len(history)),
            "max_history_start_time": (
                history["start_time"].max() if not history.empty else pd.NaT
            ),
        }
        for days in self._config["features"]["history_windows_days"]:
            recent = self._history_window(history, target_time, int(days))
            features[f"history_volume_km_{days}d"] = float(
                recent["distance_m"].sum() / 1_000.0
            )
            features[f"history_duration_h_{days}d"] = float(
                recent["duration_s"].sum() / 3_600.0
            )

        acute_days = int(self._config["features"]["acute_window_days"])
        chronic_days = int(self._config["features"]["chronic_window_days"])
        acute_duration = self._history_window(
            history, target_time, acute_days
        )["duration_s"].sum()
        chronic_duration = self._history_window(
            history, target_time, chronic_days
        )["duration_s"].sum()
        weekly_chronic = float(
            chronic_duration * acute_days / chronic_days
        )
        features["acute_chronic_duration_ratio"] = (
            float(acute_duration / weekly_chronic)
            if weekly_chronic > 0
            else np.nan
        )

        long_days = int(
            self._config["features"]["long_run_lookback_days"]
        )
        recent_long_window = self._history_window(
            history, target_time, long_days
        )
        long_threshold_s = (
            float(self._config["qc"]["min_duration_min"]) * 60.0
        )
        recent_long = recent_long_window[
            recent_long_window["duration_s"] >= long_threshold_s
        ]
        features["long_run_count_28d"] = int(len(recent_long))
        features["max_long_run_duration_min_28d"] = (
            float(recent_long["duration_s"].max() / 60.0)
            if not recent_long.empty
            else 0.0
        )
        all_long = history[history["duration_s"] >= long_threshold_s]
        features["days_since_last_long_run"] = (
            float(
                (
                    target_time - all_long["start_time"].max()
                ).total_seconds()
                / 86_400.0
            )
            if not all_long.empty
            else np.nan
        )

        intensity_history = self._history_window(
            history,
            target_time,
            int(self._config["features"]["intensity_history_days"]),
        )
        low, middle, high = self._intensity_distribution(
            intensity_history, hrmax_bpm
        )
        features["history_hr_low_fraction_90d"] = low
        features["history_hr_middle_fraction_90d"] = middle
        features["history_hr_high_fraction_90d"] = high

        if "gate_pass" in history:
            gate_history = history[history["gate_pass"].fillna(False)]
        else:
            gate_history = history.iloc[0:0]
        features["prior_gate_pass_count"] = int(len(gate_history))
        for column, output in (
            ("magnitude", "prior_gate_magnitude_mean"),
            ("onset", "prior_gate_onset_mean"),
            ("drift", "prior_gate_drift_mean_bpm_h"),
        ):
            features[output] = (
                float(gate_history[column].mean())
                if column in gate_history and gate_history[column].notna().any()
                else np.nan
            )
        features["cs_trend_ms_per_90d"] = self._parameter_trend(
            history, target_time, "cs_ms"
        )
        features["dprime_trend_m_per_90d"] = self._parameter_trend(
            history, target_time, "dprime_m"
        )
        return features

    def _early_features(self, early: pd.DataFrame) -> dict[str, Any]:
        required = {
            "elapsed_s",
            "heart_rate",
            "gap_speed_ms",
            "target_duration_s",
            "early_cutoff_elapsed_s",
            "hrmax_bpm",
            "cs_ms",
            "dprime_m",
        }
        missing = required - set(early.columns)
        if missing:
            raise AssertionError(
                f"Early window missing columns: {sorted(missing)}"
            )
        if early.empty:
            raise AssertionError("Early window is empty")
        target_duration_s = _constant_value(early, "target_duration_s")
        cutoff = _constant_value(early, "early_cutoff_elapsed_s")
        elapsed = early["elapsed_s"].to_numpy(dtype=float)
        assert float(np.nanmax(elapsed)) <= cutoff, (
            "Early feature builder received samples after cutoff"
        )
        expected_cutoff = (
            float(np.nanmin(elapsed))
            + float(self._config["features"]["early_window_frac"])
            * target_duration_s
        )
        assert np.isclose(cutoff, expected_cutoff), (
            "Early cutoff does not match configured target fraction"
        )

        hrmax = _constant_value(early, "hrmax_bpm")
        cs_ms = _constant_value(early, "cs_ms")
        dprime_m = _constant_value(early, "dprime_m")
        heart_rate = early["heart_rate"].to_numpy(dtype=float)
        gap_speed = early["gap_speed_ms"].to_numpy(dtype=float)
        internal = heart_rate / hrmax
        external = gap_speed / cs_ms
        decoupling = np.full(len(early), np.nan, dtype=float)
        valid_external = np.isfinite(external) & (external > 0)
        decoupling[valid_external] = (
            internal[valid_external] / external[valid_external]
        )
        elapsed_h = (elapsed - elapsed[0]) / 3_600.0
        minimum = int(
            self._config["features"]["min_slope_observations"]
        )

        finite_speed = gap_speed[np.isfinite(gap_speed)]
        mean_speed = _finite_mean(gap_speed)
        speed_cv = (
            float(finite_speed.std(ddof=0) / mean_speed)
            if finite_speed.size >= minimum and mean_speed > 0
            else np.nan
        )
        altitude_column = (
            "smoothed_altitude_m"
            if "smoothed_altitude_m" in early
            else "altitude"
        )
        if altitude_column in early:
            altitude = early[altitude_column].to_numpy(dtype=float)
            altitude_change = np.diff(altitude)
            valid_change = np.isfinite(altitude[:-1]) & np.isfinite(
                altitude[1:]
            )
            ascent = float(
                np.clip(altitude_change[valid_change], 0.0, None).sum()
            )
        else:
            ascent = np.nan

        temperature_column = next(
            (
                column
                for column in ("temperature_c", "temperature")
                if column in early
            ),
            None,
        )
        humidity_column = next(
            (
                column
                for column in ("humidity_pct", "humidity")
                if column in early
            ),
            None,
        )
        temperature = (
            _finite_mean(early[temperature_column].to_numpy(dtype=float))
            if temperature_column
            else np.nan
        )
        humidity = (
            _finite_mean(early[humidity_column].to_numpy(dtype=float))
            if humidity_column
            else np.nan
        )
        return {
            "hrmax_bpm_asof": hrmax,
            "cs_ms_asof": cs_ms,
            "dprime_m_asof": dprime_m,
            "early_duration_min": float(
                (elapsed[-1] - elapsed[0]) / 60.0
            ),
            "early_n_samples": int(len(early)),
            "early_decoupling_mean": _finite_mean(decoupling),
            "early_decoupling_slope_per_h": _linear_slope(
                elapsed_h, decoupling, minimum
            ),
            "early_internal_load_mean": _finite_mean(internal),
            "early_external_load_mean": _finite_mean(external),
            "early_hr_drift_bpm_h": _linear_slope(
                elapsed_h, heart_rate, minimum
            ),
            "early_cumulative_ascent_m": ascent,
            "early_gap_speed_cv": speed_cv,
            "early_temperature_c": temperature,
            "early_humidity_pct": humidity,
            "early_temperature_available": bool(
                np.isfinite(temperature)
            ),
            "early_humidity_available": bool(np.isfinite(humidity)),
        }

    def build_features(
        self,
        athlete_id: str,
        t0: pd.Timestamp,
        early_window_df: pd.DataFrame,
    ) -> dict[str, Any]:
        """Build one row without receiving any target samples after 40%."""
        target_time = _utc_timestamp(t0)
        history = self._prior_history(str(athlete_id), target_time)
        hrmax = _constant_value(early_window_df, "hrmax_bpm")
        return self._history_features(
            history, target_time, hrmax
        ) | self._early_features(early_window_df)


def materialize_feature_history_evidence(
    activities: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    """Store HR histograms needed for target-as-of intensity zones."""
    low, high = [
        int(value)
        for value in config["athlete_params"]["hr_valid_range"]
    ]
    edges = np.arange(low - 0.5, high + 1.5, 1.0)
    rows: list[dict[str, Any]] = []
    progress_every = int(config["data"]["feature_progress_every"])
    for number, activity in enumerate(
        activities.itertuples(index=False), start=1
    ):
        stream = reload_preprocessed_stream(
            config, activity.athlete_id, activity.activity_id
        )
        hr = stream["heart_rate"].to_numpy(dtype=float)
        finite = hr[np.isfinite(hr) & (hr >= low) & (hr <= high)]
        histogram, _ = np.histogram(finite, bins=edges)
        rows.append(
            {
                "activity_id": str(activity.activity_id),
                "athlete_id": str(activity.athlete_id),
                "hr_histogram": histogram.astype(np.int32).tolist(),
                "hr_observed_samples": int(histogram.sum()),
            }
        )
        if number % progress_every == 0:
            LOGGER.info(
                "Feature-history evidence progress: %s/%s",
                number,
                len(activities),
            )
    evidence = pd.DataFrame(rows)
    path = repo_path(config, config["data"]["feature_history_evidence"])
    path.parent.mkdir(parents=True, exist_ok=True)
    evidence.to_parquet(path, index=False)
    return evidence


def build_history_table(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join Phase-2 training metadata, HR evidence, and prior labels."""
    phase2 = pd.read_parquet(
        repo_path(config, config["data"]["preprocessed_activities"])
    )
    phase2 = phase2[phase2["phase2_accepted"].fillna(False)].copy()
    phase2["activity_id"] = phase2["activity_id"].astype(str)
    phase2["athlete_id"] = phase2["athlete_id"].astype(str)
    evidence = materialize_feature_history_evidence(phase2, config)
    gated = pd.read_parquet(
        repo_path(config, config["data"]["gated_indices"])
    )
    gated["activity_id"] = gated["activity_id"].astype(str)
    gated["athlete_id"] = gated["athlete_id"].astype(str)
    history = phase2[
        [
            "activity_id",
            "athlete_id",
            "start_time",
            "duration_s",
            "distance_m",
        ]
    ].merge(
        evidence,
        on=["activity_id", "athlete_id"],
        how="left",
        validate="one_to_one",
    )
    history = history.merge(
        gated[
            [
                "activity_id",
                "athlete_id",
                "gate_pass",
                "magnitude",
                "onset",
                "drift",
                "cs_ms",
                "dprime_m",
            ]
        ],
        on=["activity_id", "athlete_id"],
        how="left",
        validate="one_to_one",
    )
    return history, gated


def _feature_manifest(
    features: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    target_names = {f"target_{column}" for column in TARGET_COLUMNS}
    rows: list[dict[str, str]] = []
    for column in features.columns:
        if column in IDENTIFIER_COLUMNS:
            role = "identifier"
        elif column in target_names:
            role = "target"
        elif column in {
            "max_history_start_time",
            "early_cutoff_elapsed_s",
            "gate_pass",
        }:
            role = "audit"
        else:
            role = "feature"
        rows.append(
            {
                "column": column,
                "role": role,
                "provenance": "MEASURED",
            }
        )
    manifest = pd.DataFrame(rows)
    path = repo_path(config, config["outputs"]["feature_manifest"])
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(path, index=False)
    return manifest


def _write_phase6_summary(
    features: pd.DataFrame,
    manifest: pd.DataFrame,
    audit: dict[str, int],
    config: dict[str, Any],
) -> pd.DataFrame:
    feature_columns = manifest[manifest["role"] == "feature"]["column"]
    metrics: dict[str, Any] = {
        "feature_rows": len(features),
        "athletes": features["athlete_id"].nunique(),
        "feature_columns": len(feature_columns),
        "finite_magnitude_targets": features[
            "target_magnitude"
        ].notna().sum(),
        "finite_onset_targets": features["target_onset"].notna().sum(),
        "finite_retention_targets": features[
            "target_retention"
        ].notna().sum(),
        "finite_drift_targets": features["target_drift"].notna().sum(),
        "temporal_history_violations": int(
            (
                features["max_history_start_time"]
                >= features["start_time"]
            ).fillna(False).sum()
        ),
        "preceding_count_mismatches": audit[
            "preceding_count_mismatches"
        ],
        "late_window_audits": audit["audited"],
        "late_window_feature_mismatches": audit["feature_mismatches"],
        "planted_leaks_detected": audit["planted_leaks_detected"],
        "temperature_available_rows": features[
            "early_temperature_available"
        ].sum(),
        "humidity_available_rows": features[
            "early_humidity_available"
        ].sum(),
        "median_history_activities": features[
            "history_activity_count"
        ].median(),
        "median_early_samples": features["early_n_samples"].median(),
    }
    summary = pd.DataFrame(
        {
            "metric": list(metrics),
            "value": list(metrics.values()),
            "provenance": "MEASURED",
        }
    )
    path = repo_path(config, config["data"]["phase6_summary"])
    path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(path, index=False)
    return summary


def materialize_phase6(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the gate-passing prospective-prediction feature table."""
    history, gated = build_history_table(config)
    targets = gated[gated["gate_pass"].fillna(False)].copy()
    targets = targets.sort_values(
        ["athlete_id", "start_time", "activity_id"]
    ).reset_index(drop=True)
    builder = FeatureBuilder(history, config)
    rng = np.random.default_rng(int(config["seed"]))
    audit_n = min(
        int(config["features"]["leakage_audit_activities"]), len(targets)
    )
    audit_indices = set(
        rng.choice(len(targets), size=audit_n, replace=False).tolist()
    )
    audit = {
        "audited": 0,
        "feature_mismatches": 0,
        "planted_leaks_detected": 0,
        "preceding_count_mismatches": 0,
    }
    rows: list[dict[str, Any]] = []
    progress_every = int(config["data"]["feature_progress_every"])
    for number, target in enumerate(targets.itertuples(index=False), start=1):
        stream = reload_preprocessed_stream(
            config, target.athlete_id, target.activity_id
        )
        metadata = {
            "hrmax_bpm": target.hrmax_bpm,
            "cs_ms": target.cs_ms,
            "dprime_m": target.dprime_m,
        }
        early = prepare_early_window(stream, metadata, config)
        feature_values = builder.build_features(
            str(target.athlete_id),
            pd.Timestamp(target.start_time),
            early,
        )
        audit["preceding_count_mismatches"] += int(
            feature_values["history_activity_count"]
            != int(target.n_preceding_activities)
        )
        row = {
            "activity_id": str(target.activity_id),
            "athlete_id": str(target.athlete_id),
            "start_time": target.start_time,
            "early_cutoff_elapsed_s": _constant_value(
                early, "early_cutoff_elapsed_s"
            ),
            "gate_pass": bool(target.gate_pass),
            **feature_values,
            **{
                f"target_{column}": getattr(target, column)
                for column in TARGET_COLUMNS
            },
        }
        rows.append(row)

        target_index = number - 1
        if target_index in audit_indices:
            modified = stream.copy()
            cutoff = _constant_value(early, "early_cutoff_elapsed_s")
            late = modified["elapsed_s"] > cutoff
            modified.loc[late, "heart_rate"] = (
                modified.loc[late, "heart_rate"]
                + float(
                    config["features"]["leakage_probe_hr_offset_bpm"]
                )
            )
            modified_early = prepare_early_window(
                modified, metadata, config
            )
            modified_features = builder.build_features(
                str(target.athlete_id),
                pd.Timestamp(target.start_time),
                modified_early,
            )
            audit["audited"] += 1
            audit["feature_mismatches"] += int(
                not feature_outputs_equal(
                    feature_values, modified_features
                )
            )
            planted_original = {
                "full_activity_mean_hr": float(
                    stream["heart_rate"].mean()
                )
            }
            planted_modified = {
                "full_activity_mean_hr": float(
                    modified["heart_rate"].mean()
                )
            }
            audit["planted_leaks_detected"] += int(
                not feature_outputs_equal(
                    planted_original, planted_modified
                )
            )
        if number % progress_every == 0:
            LOGGER.info("Phase 6 progress: %s/%s", number, len(targets))

    features = pd.DataFrame(rows)
    temporal_violations = (
        features["max_history_start_time"] >= features["start_time"]
    ).fillna(False)
    assert not temporal_violations.any(), (
        "Feature table contains history at or after target start"
    )
    assert audit["preceding_count_mismatches"] == 0, (
        "Feature history count disagrees with leakage-safe Phase-3 history"
    )
    assert audit["feature_mismatches"] == 0, (
        "Real feature changed when only the late target window changed"
    )
    assert audit["planted_leaks_detected"] == audit["audited"], (
        "Leakage probe failed to detect planted full-activity mean HR"
    )
    output_path = repo_path(config, config["data"]["features_output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)
    manifest = _feature_manifest(features, config)
    summary = _write_phase6_summary(features, manifest, audit, config)
    return features, manifest, summary
