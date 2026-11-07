"""E25: single-athlete COROS FIT case study.

The production pipeline (GPS-derived speed, 10 s grid, GAP, leakage-guarded
as-of athlete parameters, Phase-3 indices, Phase-4 QC, conjunctive gate) is
reused unchanged on one athlete's native 1 Hz FIT export. Two sampling arms are
compared on the same runs:

* ``native_1hz``  - every native record, resampled to ``resample.grid_dt_s``.
* ``fitrec_500``  - the record thinned to FitRec's fixed 500-point budget first,
  so the stream looks like an Endomondo workout before the same resampling.

n = 1. Nothing here is a cross-athlete result; every number is MEASURED on this
athlete only. Reports stay aggregate: no coordinates and no maps.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from durability.athlete_params import AthleteParameterEstimator
from durability.config import repo_path
from durability.derive import derive_distance_speed, smooth_speed
from durability.gap import add_grade_adjusted_speed
from durability.gate import gate_pass
from durability.indices import compute_activity_indices
from durability.qc import _gap_speed_cv, build_waterfall
from durability.reliability import match_similar_sessions
from durability.resample import resample_activity

def kth_largest(*args, **kwargs):
    raise NotImplementedError('HRmax analysis is not part of the local tool')


sustained_peak = kth_largest

LOGGER = logging.getLogger(__name__)
INDEX_COLUMNS = ["magnitude", "drift", "onset", "retention", "cs_shift"]


# --------------------------------------------------------------------------- parsing
def parse_fit(path: str | Path, config: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    """Return session metadata and the per-record table of one FIT file."""
    import fitdecode

    settings = config["coros_case_study"]
    to_deg = float(settings["semicircle_to_deg"])
    meta: dict[str, Any] = {"file": Path(path).name, "sport": None, "sub_sport": None}
    rows: list[dict[str, Any]] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with fitdecode.FitReader(str(path)) as reader:
            for frame in reader:
                if not isinstance(frame, fitdecode.FitDataMessage):
                    continue
                if frame.name == "session":
                    meta["sport"] = _as_str(frame.get_value("sport", fallback=None))
                    meta["sub_sport"] = _as_str(frame.get_value("sub_sport", fallback=None))
                    meta["device_distance_m"] = frame.get_value("total_distance", fallback=None)
                elif frame.name == "record":
                    values = {field.name: field.value for field in frame.fields}
                    lat, lon = values.get("position_lat"), values.get("position_long")
                    altitude = values.get("enhanced_altitude")
                    if altitude is None:
                        altitude = values.get("altitude")
                    rows.append(
                        {
                            "timestamp": values.get("timestamp"),
                            "heart_rate": values.get("heart_rate"),
                            "altitude": altitude,
                            "latitude": None if lat is None else lat * to_deg,
                            "longitude": None if lon is None else lon * to_deg,
                        }
                    )
    records = pd.DataFrame(rows, columns=["timestamp", "heart_rate", "altitude", "latitude", "longitude"])
    records = records.dropna(subset=["timestamp"])
    records["timestamp"] = pd.to_datetime(records["timestamp"], utc=True)
    records = records.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    for column in ["heart_rate", "altitude", "latitude", "longitude"]:
        records[column] = pd.to_numeric(records[column], errors="coerce")
    if not records.empty:
        meta["start_time"] = records["timestamp"].iloc[0]
        meta["duration_s"] = float((records["timestamp"].iloc[-1] - records["timestamp"].iloc[0]).total_seconds())
        meta["hr_coverage"] = float(records["heart_rate"].notna().mean())
        meta["gps_coverage"] = float(records["latitude"].notna().mean())
    return meta, records


def _as_str(value: Any) -> str | None:
    return None if value is None else str(value)


def is_outdoor_run(meta: dict[str, Any], config: dict[str, Any]) -> bool:
    settings = config["coros_case_study"]
    return meta.get("sport") == settings["fit_running_sport"] and meta.get("sub_sport") not in set(
        settings["exclude_sub_sports"]
    )


# --------------------------------------------------------------------------- sampling arms
def thin_to_budget(records: pd.DataFrame, n_points: int) -> pd.DataFrame:
    """Keep ``n_points`` records evenly spread by index, as in FitRec's fixed budget."""
    if len(records) <= n_points:
        return records.reset_index(drop=True).copy()
    keep = np.unique(np.round(np.linspace(0, len(records) - 1, n_points)).astype(int))
    return records.iloc[keep].reset_index(drop=True).copy()


def build_phase1_stream(records: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """GPS-derived speed exactly as the FitRec path: haversine km/h, then m/s, then smooth."""
    gps = records.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
    if len(gps) < 2:
        raise AssertionError("Activity has fewer than two GPS fixes")
    time_s = np.asarray([value.timestamp() for value in gps["timestamp"]], dtype=float)
    step_km, speed_ms = derive_distance_speed(gps["latitude"].to_numpy(), gps["longitude"].to_numpy(), time_s)
    smoothed = smooth_speed(speed_ms, time_s, config["data"]["speed_smooth_window_s"])
    return pd.DataFrame(
        {
            "timestamp": gps["timestamp"],
            "elapsed_s": time_s - time_s[0],
            "heart_rate": gps["heart_rate"].to_numpy(dtype=float),
            "altitude": gps["altitude"].to_numpy(dtype=float),
            "latitude": gps["latitude"].to_numpy(dtype=float),
            "longitude": gps["longitude"].to_numpy(dtype=float),
            "derived_distance_km": step_km,
            "derived_speed_ms": speed_ms,
            "smoothed_speed_ms": smoothed,
        }
    )


def process_arm(records: pd.DataFrame, arm: str, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run one sampling arm through Phase 1-2 and return the 10 s GAP stream."""
    if arm == "fitrec_500":
        records = thin_to_budget(records, int(config["coros_case_study"]["fitrec_points"]))
    elif arm != "native_1hz":
        raise ValueError(f"Unknown sampling arm {arm}")
    phase1 = build_phase1_stream(records, config)
    resampled, resample_meta = resample_activity(phase1, config)
    processed, gap_meta = add_grade_adjusted_speed(resampled, config)
    metadata = {
        "native_dt_s": resample_meta.native_dt_s,
        "upsample_factor": resample_meta.upsample_factor,
        "onset_sampling_eligible": resample_meta.onset_sampling_eligible,
        "gap_split": resample_meta.gap_split,
        "retained_duration_s_phase2": resample_meta.retained_duration_s,
        "gap_available": gap_meta.gap_available,
        "clipped_gradient_fraction": gap_meta.clipped_gradient_fraction,
        "gps_distance_m": float(np.nansum(phase1["derived_distance_km"].to_numpy()[1:]) * 1000.0),
        "hr_coverage_10s": float(processed["heart_rate"].notna().mean()),
    }
    return processed, metadata


# --------------------------------------------------------------------------- athlete parameters
class _InMemoryHistory:
    """History loader over this athlete's processed runs; strictly-before is asserted upstream."""

    def __init__(self, sessions: pd.DataFrame, streams: dict[str, pd.DataFrame]) -> None:
        self._sessions = sessions
        self._streams = streams

    def __call__(self, athlete_id: str, as_of: pd.Timestamp) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
        prior = self._sessions[self._sessions["start_time"] < as_of]
        return prior[["activity_id", "athlete_id", "start_time"]].copy(), {
            key: self._streams[key] for key in prior["activity_id"]
        }


def asof_hrmax_e24(peaks: pd.Series, start_times: pd.Series, as_of: pd.Timestamp, k: int) -> float:
    """E24 estimator: k-th largest prior 60 s sustained peak."""
    prior = peaks[start_times < as_of]
    return kth_largest(prior.to_numpy(dtype=float), k)


# --------------------------------------------------------------------------- experiment
def build_arm_table(
    metas: list[dict[str, Any]], records: dict[str, pd.DataFrame], arm: str, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    settings = config["coros_case_study"]
    athlete_id = str(settings["athlete_id"])
    grid_dt_s = float(config["resample"]["grid_dt_s"])
    rows: list[dict[str, Any]] = []
    streams: dict[str, pd.DataFrame] = {}
    for meta in metas:
        row = {**meta, "athlete_id": athlete_id, "activity_id": meta["file"].removesuffix(".fit"), "arm": arm}
        try:
            stream, extra = process_arm(records[meta["file"]], arm, config)
            row.update(extra, phase2_accepted=True, phase2_rejection_reason="")
            streams[row["activity_id"]] = stream
            row["peak_hr_60s"] = sustained_peak(
                stream["heart_rate"].to_numpy(),
                grid_dt_s,
                config["hrmax_correction"]["sustain_s"],
                config["athlete_params"]["hr_valid_range"],
            )
            row["gap_speed_cv"] = _gap_speed_cv(stream)
            row["mean_gap_speed_ms"] = float(np.nanmean(stream["gap_speed_ms"]))
        except (AssertionError, ValueError) as exc:
            row.update(phase2_accepted=False, phase2_rejection_reason=str(exc)[:120])
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("start_time").reset_index(drop=True)
    history = table[table["phase2_accepted"]].reset_index(drop=True)
    estimator = AthleteParameterEstimator(_InMemoryHistory(history, streams), config)
    k = int(config["hrmax_correction"]["min_supporting_activities"])

    out: list[dict[str, Any]] = []
    for row in table.to_dict("records"):
        result = dict(row)
        if row["phase2_accepted"]:
            params = estimator.estimate_params(athlete_id, row["start_time"])
            result.update(
                n_preceding_activities=params.n_preceding_activities,
                params_eligible=params.eligible,
                params_rejection_reason=params.rejection_reason,
                hrmax_v1_bpm=params.hrmax_bpm,
                hrmax_e24_bpm=asof_hrmax_e24(history["peak_hr_60s"], history["start_time"], row["start_time"], k),
                cs_ms=params.cs_ms,
                cs_r_squared=params.cs_r_squared,
            )
            if params.eligible:
                stream = streams[row["activity_id"]]
                indices = compute_activity_indices(
                    stream, params, config, onset_sampling_eligible=row["onset_sampling_eligible"]
                )
                result.update({column: getattr(indices, column) for column in INDEX_COLUMNS})
                result["indices_computed"] = True
                hrmax_e24 = result["hrmax_e24_bpm"]
                if np.isfinite(hrmax_e24):
                    e24_params = type(params)(**{**params.__dict__, "hrmax_bpm": float(hrmax_e24)})
                    e24 = compute_activity_indices(
                        stream, e24_params, config, onset_sampling_eligible=row["onset_sampling_eligible"]
                    )
                    result["magnitude_e24"] = e24.magnitude
                    result["drift_e24"] = e24.drift
        out.append(result)
    frame = pd.DataFrame(out)
    for column in [*INDEX_COLUMNS, "magnitude_e24", "drift_e24"]:
        if column not in frame:
            frame[column] = np.nan
    frame["indices_computed"] = frame.get("indices_computed", False)
    frame["indices_computed"] = frame["indices_computed"].fillna(False).astype(bool)
    return apply_qc(frame, config)


def qc_conditions(frame: pd.DataFrame, config: dict[str, Any]) -> list[tuple[str, pd.Series]]:
    """The Phase-4 criteria, in the production order, on this athlete's runs."""
    qc = config["qc"]
    min_s, max_s = qc["min_duration_min"] * 60, qc["max_duration_min"] * 60
    accepted = frame["phase2_accepted"].fillna(False).astype(bool)
    return [
        ("outdoor_run_parsed", accepted),
        ("heart_rate_coverage", frame["hr_coverage"] >= qc["min_hr_coverage"]),
        ("duration_range", frame["duration_s"].between(min_s, max_s, inclusive="both")),
        (
            "longest_gap_block",
            (frame["gap_split"] != True) | (frame["retained_duration_s_phase2"] >= min_s),  # noqa: E712
        ),
        ("gap_speed_variability", frame["gap_speed_cv"] < qc["max_gap_speed_cv"]),
        ("gradient_clipping", frame["clipped_gradient_fraction"] < qc["max_clipped_gradient_frac"]),
        (
            "preceding_athlete_parameters",
            frame["params_eligible"].fillna(False).astype(bool)
            & (frame["n_preceding_activities"] >= config["athlete_params"]["min_preceding_activities"]),
        ),
        (
            "durability_indices_available",
            frame["indices_computed"] & frame["magnitude"].notna() & frame["drift"].notna(),
        ),
    ]


def apply_qc(frame: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible, waterfall = build_waterfall(frame, qc_conditions(frame, config))
    result = frame.copy()
    result["qc_eligible"] = eligible
    result["gate_pass"] = eligible & gate_pass(result["drift"], result["magnitude"], config).fillna(False)
    return result, waterfall


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    if len(a) == 0:
        return np.nan
    observed = float(np.mean(a == b))
    expected = float(np.mean(a) * np.mean(b) + np.mean(~a) * np.mean(~b))
    return np.nan if expected == 1.0 else (observed - expected) / (1.0 - expected)


def sampling_sensitivity(sessions: pd.DataFrame) -> pd.DataFrame:
    """Paired native-vs-500-point label comparison on runs QC-eligible in both arms."""
    native = sessions[sessions["arm"] == "native_1hz"].set_index("activity_id")
    thin = sessions[sessions["arm"] == "fitrec_500"].set_index("activity_id")
    both = native.index[native["qc_eligible"]].intersection(thin.index[thin["qc_eligible"]])
    rows: list[dict[str, Any]] = []
    for column in ["magnitude", "drift", "onset", "retention"]:
        a, b = native.loc[both, column].astype(float), thin.loc[both, column].astype(float)
        ok = a.notna() & b.notna()
        diff = (b - a)[ok]
        rows.append(
            {
                "index": column,
                "n_runs": int(ok.sum()),
                "native_median": float(a[ok].median()) if ok.any() else np.nan,
                "fitrec500_median": float(b[ok].median()) if ok.any() else np.nan,
                "median_diff": float(diff.median()) if ok.any() else np.nan,
                "mean_abs_diff": float(diff.abs().mean()) if ok.any() else np.nan,
                "spearman": float(a[ok].corr(b[ok], method="spearman")) if ok.sum() > 2 else np.nan,
                "provenance": "MEASURED",
            }
        )
    gate_a, gate_b = native.loc[both, "gate_pass"].to_numpy(), thin.loc[both, "gate_pass"].to_numpy()
    rows.append(
        {
            "index": "gate_pass",
            "n_runs": len(both),
            "native_median": float(np.mean(gate_a)) if len(both) else np.nan,
            "fitrec500_median": float(np.mean(gate_b)) if len(both) else np.nan,
            "median_diff": np.nan,
            "mean_abs_diff": float(np.mean(gate_a != gate_b)) if len(both) else np.nan,
            "spearman": cohen_kappa(gate_a, gate_b),
            "provenance": "MEASURED",
        }
    )
    return pd.DataFrame(rows)


def repeatability(sessions: pd.DataFrame, config: dict[str, Any], rng: np.random.Generator) -> pd.DataFrame:
    """Within-person SD in a repeat-matched cohort, against FitRec's SEM (same matching rule)."""
    settings = config["coros_case_study"]
    reference = pd.read_csv(repo_path(config, settings["fitrec_reference_table"])).set_index("index")
    rows: list[dict[str, Any]] = []
    eligible = sessions[(sessions["arm"] == "native_1hz") & sessions["qc_eligible"]]
    for column in ["magnitude", "drift", "onset", "retention"]:
        matched = match_similar_sessions(
            eligible,
            value_col=column,
            duration_tolerance=float(config["reliability"]["match_duration_tol"]),
            speed_tolerance=float(config["reliability"]["match_speed_tol"]),
            min_sessions=int(settings["min_sessions_matched"]),
        )
        values = matched[column].to_numpy(dtype=float)
        n = len(values)
        sd = float(np.std(values, ddof=1)) if n > 1 else np.nan
        if n > 1:
            draws = [np.std(rng.choice(values, n, replace=True), ddof=1) for _ in range(int(settings["bootstrap_n"]))]
            low, high = np.percentile(draws, [2.5, 97.5])
        else:
            low = high = np.nan
        rows.append(
            {
                "index": column,
                "n_matched_runs": n,
                "within_person_sd": sd,
                "sd_ci_low": float(low),
                "sd_ci_high": float(high),
                "all_eligible_sd": float(eligible[column].std(ddof=1)) if eligible[column].notna().sum() > 1 else np.nan,
                "fitrec_sem": float(reference.loc[column, "sem"]) if column in reference.index else np.nan,
                "fitrec_icc": float(reference.loc[column, "icc_1_1"]) if column in reference.index else np.nan,
                "provenance": "MEASURED",
            }
        )
    return pd.DataFrame(rows)


def run_experiment(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    settings = config["coros_case_study"]
    rng = np.random.default_rng(int(config["seed"]))
    raw_dir = repo_path(config, settings["raw_dir"])
    files = sorted(raw_dir.glob("*.fit"))
    if not files:
        raise FileNotFoundError(f"No FIT files under {raw_dir}")
    metas, records, sport_counts = [], {}, {}
    for path in files:
        meta, table = parse_fit(path, config)
        key = f"{meta.get('sport')}/{meta.get('sub_sport')}"
        sport_counts[key] = sport_counts.get(key, 0) + 1
        if is_outdoor_run(meta, config) and not table.empty:
            metas.append(meta)
            records[meta["file"]] = table
    LOGGER.info("E25 parsed %s FIT files; %s outdoor runs", len(files), len(metas))

    built = [build_arm_table(metas, records, arm, config) for arm in settings["arms"]]
    waterfall = pd.concat(
        [w.assign(arm=name) for (_, w), name in zip(built, settings["arms"], strict=True)], ignore_index=True
    )
    sessions = pd.concat([frame for frame, _ in built], ignore_index=True)
    sessions = sessions.drop(columns=[c for c in ["latitude", "longitude"] if c in sessions])  # privacy

    native = sessions[sessions["arm"] == "native_1hz"]
    eligible = native[native["qc_eligible"]]
    gps_vs_device = (native["gps_distance_m"] / native["device_distance_m"].astype(float)).replace([np.inf, -np.inf], np.nan)
    summary_items = {
        "fit_files": len(files),
        "outdoor_runs": len(metas),
        "sport_counts": "; ".join(f"{k}={v}" for k, v in sorted(sport_counts.items())),
        "span_days": float((native["start_time"].max() - native["start_time"].min()).days),
        "native_median_dt_s": float(native["native_dt_s"].median()),
        "qc_eligible_native": int(eligible.shape[0]),
        "qc_eligible_fitrec500": int(sessions[(sessions["arm"] == "fitrec_500") & sessions["qc_eligible"]].shape[0]),
        "gate_pass_native": int(eligible["gate_pass"].sum()),
        "gate_pass_rate_native": float(eligible["gate_pass"].mean()) if len(eligible) else np.nan,
        "median_magnitude_native": float(eligible["magnitude"].median()) if len(eligible) else np.nan,
        "median_drift_native": float(eligible["drift"].median()) if len(eligible) else np.nan,
        "gps_over_device_distance_median": float(gps_vs_device.median()),
        "gps_device_within_tol_frac": float(
            (gps_vs_device.sub(1.0).abs() <= settings["device_distance_check_tol"]).mean()
        ),
        "hrmax_v1_last_bpm": float(native["hrmax_v1_bpm"].dropna().iloc[-1]) if native["hrmax_v1_bpm"].notna().any() else np.nan,
        "hrmax_e24_last_bpm": float(native["hrmax_e24_bpm"].dropna().iloc[-1]) if native["hrmax_e24_bpm"].notna().any() else np.nan,
        "max_peak_hr_60s_bpm": float(native["peak_hr_60s"].max()),
        "gate_pass_e24_native": int(
            (eligible["drift_e24"].gt(config["gate"]["drift_bpm_per_h"]) & eligible["magnitude_e24"].gt(config["gate"]["magnitude"])).sum()
        ),
    }
    summary = pd.DataFrame({"metric": list(summary_items), "value": list(summary_items.values())})
    summary["provenance"] = "MEASURED"
    tables = {
        "sessions": sessions,
        "summary": summary,
        "sampling": sampling_sensitivity(sessions),
        "repeatability": repeatability(sessions, config, rng),
        "waterfall": waterfall,
    }
    _save(tables, config)
    return tables


def _save(tables: dict[str, pd.DataFrame], config: dict[str, Any]) -> None:
    outputs = config["coros_case_study"]["outputs"]
    for name, table in tables.items():
        path = repo_path(config, outputs[name])
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".parquet":
            table.to_parquet(path, index=False)
        else:
            table.to_csv(path, index=False)
