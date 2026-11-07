"""Read-only durability summaries from local FIT files. Output never contains GPS coordinates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from durability.athlete_params import AthleteParams
from durability.coros_case_study import is_outdoor_run, parse_fit, process_arm
from durability.indices import compute_activity_indices
from durability.pace_matched import pace_matched_indices
from durability.qc import _gap_speed_cv

REASONS = {
    "not_outdoor_run": "Not an outdoor run with GPS, so speed cannot be derived.",
    "unusable_stream": "Speed or GPS is outside the supported range; trail or hiking runs often fall here.",
    "heart_rate_coverage": "Too little heart-rate data.",
    "duration_range": "Duration is outside the supported 45 to 240 minutes.",
    "longest_gap_block": "Pauses leave no continuous block of at least 45 minutes.",
    "gap_speed_variability": "Pace is too variable for the standard indices.",
    "gradient_clipping": "Too many altitude samples have implausible gradients.",
    "indices_unavailable": "The indices could not be computed from this run.",
}
CAVEAT = (
    "Research measure, not a diagnosis. Heat, hydration, illness, terrain and sensor error can also change these values. "
    "One run is weak evidence: the gate agreed only slightly between consecutive FitRec sessions. Prefer the trend over several runs."
)


def _number(value: Any, digits: int = 3) -> float | None:
    value = float(value) if value is not None else np.nan
    return round(value, digits) if np.isfinite(value) else None


def _standard_indices(stream: pd.DataFrame, native_dt_s: float, config: dict[str, Any], athlete: str, start: pd.Timestamp) -> Any:
    # HRmax and CS cancel in magnitude and onset, and drift uses neither, so unit values are used here.
    params = AthleteParams(athlete, start, 1.0, 1.0, 0.0, 1.0, 0, True, "")
    # Onset eligibility follows the rule's stated intent: only coarse sampling is excluded.
    onset_ok = native_dt_s <= float(config["resample"]["coarse_sampling_threshold_s"])
    return compute_activity_indices(stream, params, config, onset_sampling_eligible=onset_ok)


def analyse_records(meta: dict[str, Any], records: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """Summarise one parsed run: standard indices if QC passes, experimental drift if only pace fails QC."""
    qc, gate = config["qc"], config["gate"]
    start = meta.get("start_time")
    result: dict[str, Any] = {
        "file": meta["file"],
        "date": start.date().isoformat() if start is not None else None,
        "duration_min": _number(meta.get("duration_s", np.nan) / 60.0, 1) if meta.get("duration_s") is not None else None,
        "status": "not_evaluable",
        "reason": None,
        "reason_text": None,
        "drift_bpm_h": None,
        "magnitude": None,
        "onset_fraction": None,
        "experimental_pace_matched_drift_bpm_h": None,
        "provenance": "MEASURED",
    }

    def stop(code: str) -> dict[str, Any]:
        result.update(reason=code, reason_text=REASONS[code])
        return result

    if not is_outdoor_run(meta, config) or records.empty:
        return stop("not_outdoor_run")
    try:
        stream, info = process_arm(records, "native_1hz", config)
    except (AssertionError, ValueError) as exc:
        result["reason_detail"] = str(exc) or None
        return stop("unusable_stream")
    if float(records["heart_rate"].notna().mean()) < qc["min_hr_coverage"]:
        return stop("heart_rate_coverage")
    minimum, maximum = qc["min_duration_min"] * 60, qc["max_duration_min"] * 60
    if not minimum <= float(meta["duration_s"]) <= maximum:
        return stop("duration_range")
    if info["gap_split"] and info["retained_duration_s_phase2"] < minimum:
        return stop("longest_gap_block")

    cv = _gap_speed_cv(stream)
    result["gap_speed_cv"] = _number(cv)
    experimental = pace_matched_indices(stream["elapsed_s"], stream["heart_rate"], stream["gap_speed_ms"], config)
    result["experimental_pace_matched_drift_bpm_h"] = _number(experimental["pm_drift"], 2)
    if not cv < qc["max_gap_speed_cv"]:
        return stop("gap_speed_variability")
    if not info["clipped_gradient_fraction"] < qc["max_clipped_gradient_frac"]:
        return stop("gradient_clipping")
    try:
        values = _standard_indices(stream, float(info["native_dt_s"]), config, "local", start)
    except (AssertionError, ValueError):
        return stop("indices_unavailable")
    if not (np.isfinite(values.magnitude) and np.isfinite(values.drift)):
        return stop("indices_unavailable")
    passes = values.drift > gate["drift_bpm_per_h"] and values.magnitude > gate["magnitude"]
    result.update(
        status="signal" if passes else "no_signal",
        drift_bpm_h=_number(values.drift, 2),
        magnitude=_number(values.magnitude),
        onset_fraction=_number(values.onset),
    )
    return result


class FitFolder:
    """A single folder of FIT files; names outside it are refused and results are cached by file time."""

    def __init__(self, folder: str | Path, config: dict[str, Any]) -> None:
        self.folder = Path(folder).expanduser().resolve()
        if not self.folder.is_dir():
            raise ValueError(f"FIT folder not found: {self.folder}")
        self.config = config
        self._cache: dict[tuple[str, int], dict[str, Any]] = {}

    def _path(self, name: str) -> Path:
        path = (self.folder / name).resolve()
        if Path(name).name != name or path.parent != self.folder or path.suffix.lower() != ".fit" or not path.is_file():
            raise ValueError("Unknown FIT file name; use a name returned by list_durability_runs")
        return path

    def run(self, name: str) -> dict[str, Any]:
        path = self._path(name)
        key = (name, path.stat().st_mtime_ns)
        if key not in self._cache:
            meta, records = parse_fit(path, self.config)
            self._cache[key] = analyse_records(meta, records, self.config)
        return dict(self._cache[key])

    def runs(self, start_date: str | None = None, end_date: str | None = None) -> list[dict[str, Any]]:
        rows = [self.run(p.name) for p in sorted(self.folder.iterdir()) if p.suffix.lower() == ".fit"]
        rows = [r for r in rows if r["reason"] != "not_outdoor_run" and r["date"] is not None]
        if start_date:
            rows = [r for r in rows if r["date"] >= start_date]
        if end_date:
            rows = [r for r in rows if r["date"] <= end_date]
        return sorted(rows, key=lambda r: (r["date"], r["file"]))


def compare_recent(values: list[float], config: dict[str, Any]) -> dict[str, Any]:
    """Compare the latest runs with this runner's own earlier runs, in time order."""
    settings = config["fit_tool"]
    recent_n, baseline_n = int(settings["trend_recent_runs"]), int(settings["trend_min_baseline_runs"])
    data = np.asarray(values, dtype=float)
    summary: dict[str, Any] = {"runs": len(data), "mean": _number(data.mean()) if len(data) else None, "sd": _number(data.std(ddof=1)) if len(data) > 1 else None}
    if len(data) < recent_n + baseline_n:
        return {**summary, "recent_vs_own_baseline": "insufficient_data", "runs_needed": recent_n + baseline_n}
    recent, baseline = data[-recent_n:], data[:-recent_n]
    difference = float(recent.mean() - baseline.mean())
    error = float(np.sqrt(recent.var(ddof=1) / len(recent) + baseline.var(ddof=1) / len(baseline)))
    limit = float(settings["trend_change_standard_errors"]) * error
    label = "no_clear_change" if abs(difference) <= limit else ("higher_than_own_baseline" if difference > 0 else "lower_than_own_baseline")
    return {**summary, "recent_mean": _number(recent.mean()), "baseline_mean": _number(baseline.mean()), "difference": _number(difference), "recent_vs_own_baseline": label}


def trend(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """Status counts plus standard and experimental trends. Experimental drift is never merged into the gate."""
    counts = pd.Series([r["status"] for r in rows], dtype=str).value_counts().to_dict()
    reasons = pd.Series([r["reason"] for r in rows if r["reason"]], dtype=str).value_counts().to_dict()
    pick = lambda key: [r[key] for r in rows if r[key] is not None]
    evaluated = counts.get("signal", 0) + counts.get("no_signal", 0)
    return {
        "runs": len(rows),
        "status_counts": {k: int(counts.get(k, 0)) for k in ("signal", "no_signal", "not_evaluable")},
        "signal_share_of_evaluated": _number(counts.get("signal", 0) / evaluated) if evaluated else None,
        "not_evaluable_reasons": {k: int(v) for k, v in reasons.items()},
        "standard_drift_bpm_h": compare_recent(pick("drift_bpm_h"), config),
        "standard_magnitude": compare_recent(pick("magnitude"), config),
        "experimental_pace_matched_drift_bpm_h": compare_recent(pick("experimental_pace_matched_drift_bpm_h"), config),
        "experimental_note": "Experimental pace-matched drift did not meet all validation criteria and is not part of the gate.",
        "provenance": "MEASURED",
        "caveat": CAVEAT,
    }


def expected_next(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """Pre-run expectation for the next evaluable run: own mean shrunk toward the FitRec population mean.

    This is the baseline that no FitRec model beat before a run starts; it is not a trained model.
    """
    settings = config["fit_tool"]["expectation"]
    strength = float(settings["shrinkage_strength"])
    out: dict[str, Any] = {}
    for key in ("drift_bpm_h", "magnitude"):
        own = np.asarray([r[key] for r in rows if r[key] is not None], dtype=float)
        population = float(settings["population_mean"][key])
        if len(own) < int(settings["min_own_runs"]):
            value, source = population, "population_mean_only"
        else:
            value, source = (len(own) * own.mean() + strength * population) / (len(own) + strength), "shrunk_personal_average"
        error = float(settings["typical_error_mae_athlete"][key])
        out[key] = {"expected": _number(value), "typical_error": _number(error), "own_runs_used": int(len(own)), "source": source}
    return {
        **out,
        "method": f"Own mean shrunk toward the FitRec population mean with strength {strength:g}.",
        "evidence": "Before a run starts, no model showed a supported gain over this baseline (FitRec, MEASURED). Typical error is equal-athlete MAE on FitRec next runs (PILOT MEASURED).",
        "provenance": "EXPECTATION FROM OWN HISTORY, NOT A FORECAST MODEL",
        "caveat": CAVEAT,
    }
