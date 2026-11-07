"""Repeated-session matching and ICC reliability analysis."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import norm

from durability.config import repo_path
from durability.preprocess import reload_preprocessed_stream

LOGGER = logging.getLogger(__name__)
INDEX_COLUMNS = ("magnitude", "onset", "drift", "cs_shift", "retention")


def spearman_brown(icc_1: float, k: int) -> float:
    """Reliability after averaging ``k`` interchangeable sessions."""
    if k < 1:
        raise ValueError("k must be at least 1")
    denominator = 1.0 + (k - 1) * icc_1
    if denominator == 0:
        return np.nan
    return float(k * icc_1 / denominator)


def _group_sufficient_statistics(
    frame: pd.DataFrame, group_col: str, value_col: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clean = frame[[group_col, value_col]].dropna()
    grouped = clean.groupby(group_col, sort=False)[value_col]
    count = grouped.size().to_numpy(dtype=float)
    total = grouped.sum().to_numpy(dtype=float)
    squared_total = grouped.apply(
        lambda values: float(np.square(values.to_numpy(dtype=float)).sum())
    ).to_numpy(dtype=float)
    return count, total, squared_total


def _icc_from_sufficient_statistics(
    count: np.ndarray, total: np.ndarray, squared_total: np.ndarray
) -> float:
    n_groups = len(count)
    n_observations = float(count.sum())
    if n_groups < 2 or n_observations <= n_groups:
        return np.nan
    means = total / count
    grand_mean = float(total.sum() / n_observations)
    ss_between = float(np.sum(count * np.square(means - grand_mean)))
    ss_within = float(np.sum(squared_total - np.square(total) / count))
    ms_between = ss_between / (n_groups - 1)
    ms_within = ss_within / (n_observations - n_groups)
    effective_sessions = (
        n_observations - float(np.square(count).sum() / n_observations)
    ) / (n_groups - 1)
    denominator = ms_between + (effective_sessions - 1.0) * ms_within
    if denominator == 0:
        return np.nan
    return float((ms_between - ms_within) / denominator)


def icc_one_way_random(
    frame: pd.DataFrame, group_col: str, value_col: str
) -> float:
    """Compute unbalanced ICC(1,1) by the one-way mean-squares formula."""
    count, total, squared_total = _group_sufficient_statistics(
        frame, group_col, value_col
    )
    return _icc_from_sufficient_statistics(count, total, squared_total)


def bootstrap_icc_by_athlete(
    frame: pd.DataFrame,
    group_col: str,
    value_col: str,
    *,
    n_bootstrap: int,
    confidence_level: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Percentile ICC interval from athlete-cluster bootstrap samples."""
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least 1")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie between 0 and 1")
    count, total, squared_total = _group_sufficient_statistics(
        frame, group_col, value_col
    )
    n_groups = len(count)
    if n_groups < 2:
        return np.nan, np.nan
    estimates = np.empty(n_bootstrap, dtype=float)
    for bootstrap_index in range(n_bootstrap):
        sampled = rng.integers(0, n_groups, size=n_groups)
        estimates[bootstrap_index] = _icc_from_sufficient_statistics(
            count[sampled], total[sampled], squared_total[sampled]
        )
    finite = estimates[np.isfinite(estimates)]
    if not finite.size:
        return np.nan, np.nan
    alpha = 1.0 - confidence_level
    low, high = np.quantile(finite, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(low), float(high)


def match_similar_sessions(
    frame: pd.DataFrame,
    *,
    value_col: str,
    duration_tolerance: float,
    speed_tolerance: float,
    min_sessions: int,
) -> pd.DataFrame:
    """Select one largest pairwise-compatible session cohort per athlete.

    The selected duration and speed ranges each have ``max / min - 1`` no
    greater than their configured tolerance. Thus every selected pair meets
    both similarity rules. Ties prefer the tightest normalised ranges and
    deterministic lower bounds.
    """
    required = {
        "activity_id",
        "athlete_id",
        "duration_s",
        "mean_gap_speed_ms",
        value_col,
    }
    missing = required - set(frame.columns)
    if missing:
        raise AssertionError(f"Missing matching columns: {sorted(missing)}")
    eligible = frame[
        frame[value_col].notna()
        & frame["duration_s"].gt(0)
        & frame["mean_gap_speed_ms"].gt(0)
    ].copy()
    selected: list[pd.DataFrame] = []
    for athlete_id, athlete in eligible.groupby("athlete_id", sort=True):
        athlete = athlete.sort_values("activity_id").reset_index(drop=True)
        durations = athlete["duration_s"].to_numpy(dtype=float)
        speeds = athlete["mean_gap_speed_ms"].to_numpy(dtype=float)
        best_mask: np.ndarray | None = None
        best_score: tuple[float, float, float, float] | None = None
        for duration_min in np.unique(durations):
            duration_mask = (durations >= duration_min) & (
                durations <= duration_min * (1.0 + duration_tolerance)
            )
            for speed_min in np.unique(speeds[duration_mask]):
                mask = (
                    duration_mask
                    & (speeds >= speed_min)
                    & (speeds <= speed_min * (1.0 + speed_tolerance))
                )
                count = int(mask.sum())
                duration_span = (
                    durations[mask].max() / durations[mask].min() - 1.0
                )
                speed_span = speeds[mask].max() / speeds[mask].min() - 1.0
                score = (
                    -float(count),
                    float(duration_span + speed_span),
                    float(duration_min),
                    float(speed_min),
                )
                if best_score is None or score < best_score:
                    best_score = score
                    best_mask = mask
        assert best_mask is not None and best_score is not None
        if int(best_mask.sum()) < min_sessions:
            continue
        cohort = athlete.loc[best_mask].copy()
        cohort["match_anchor_activity_id"] = str(
            cohort.iloc[0]["activity_id"]
        )
        cohort["matched_session_count"] = len(cohort)
        cohort["index"] = value_col
        selected.append(cohort)
    if not selected:
        return pd.DataFrame(
            columns=list(frame.columns)
            + [
                "match_anchor_activity_id",
                "matched_session_count",
                "index",
            ]
        )
    return pd.concat(selected, ignore_index=True)


def materialize_session_characteristics(
    frame: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    """Attach the MEASURED full-retained-stream mean GAP speed."""
    result = frame.copy()
    means: list[float] = []
    progress_every = int(config["data"]["qc_progress_every"])
    for number, row in enumerate(result.itertuples(index=False), start=1):
        stream = reload_preprocessed_stream(
            config, row.athlete_id, row.activity_id
        )
        speed = stream["gap_speed_ms"].to_numpy(dtype=float)
        finite = speed[np.isfinite(speed)]
        means.append(float(finite.mean()) if finite.size else np.nan)
        if number % progress_every == 0:
            LOGGER.info(
                "Reliability session-character progress: %s/%s",
                number,
                len(result),
            )
    result["mean_gap_speed_ms"] = means
    result["duration_s"] = result["phase2_retained_duration_s"]
    path = repo_path(config, config["data"]["reliability_characteristics"])
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(path, index=False)
    return result


def _matching_summary(matched: pd.DataFrame) -> dict[str, Any]:
    counts = matched.groupby("athlete_id").size()
    return {
        "n_athletes": int(len(counts)),
        "n_sessions": int(len(matched)),
        "median_sessions_per_athlete": (
            float(counts.median()) if len(counts) else np.nan
        ),
        "min_sessions_per_athlete": (
            int(counts.min()) if len(counts) else 0
        ),
        "max_sessions_per_athlete": (
            int(counts.max()) if len(counts) else 0
        ),
    }


def reliability_for_index(
    frame: pd.DataFrame,
    index: str,
    config: dict[str, Any],
    rng: np.random.Generator,
    *,
    with_bootstrap: bool = True,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Match comparable sessions and compute all reliability statistics."""
    settings = config["reliability"]
    matched = match_similar_sessions(
        frame,
        value_col=index,
        duration_tolerance=float(settings["match_duration_tol"]),
        speed_tolerance=float(settings["match_speed_tol"]),
        min_sessions=int(settings["min_sessions_per_athlete"]),
    )
    summary = _matching_summary(matched)
    icc = icc_one_way_random(matched, "athlete_id", index)
    if with_bootstrap:
        ci_low, ci_high = bootstrap_icc_by_athlete(
            matched,
            "athlete_id",
            index,
            n_bootstrap=int(settings["bootstrap_n"]),
            confidence_level=float(settings["confidence_level"]),
            rng=rng,
        )
    else:
        ci_low = ci_high = np.nan
    raw_sd = (
        float(matched[index].std(ddof=1)) if len(matched) > 1 else np.nan
    )
    sem = (
        raw_sd * np.sqrt(max(0.0, 1.0 - icc))
        if np.isfinite(raw_sd) and np.isfinite(icc)
        else np.nan
    )
    z_score = norm.ppf(
        0.5 + float(settings["confidence_level"]) / 2.0
    )
    mdc = float(z_score * np.sqrt(2.0) * sem) if np.isfinite(sem) else np.nan
    k_values = [int(k) for k in settings["spearman_brown_k"]]
    icc_by_k = {
        f"icc_k{k}": spearman_brown(icc, k)
        if np.isfinite(icc)
        else np.nan
        for k in k_values
    }
    target = float(settings["target_icc"])
    reaching = [
        k for k in k_values if np.isfinite(icc_by_k[f"icc_k{k}"])
        and icc_by_k[f"icc_k{k}"] >= target
    ]
    result = {
        "index": index,
        **summary,
        "icc_1_1": icc,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "confidence_level": float(settings["confidence_level"]),
        "raw_sd": raw_sd,
        "sem": sem,
        "mdc": mdc,
        "target_icc": target,
        "smallest_configured_k_reaching_target": (
            min(reaching) if reaching else np.nan
        ),
        **icc_by_k,
        "provenance": "MEASURED",
    }
    return result, matched


def _write_matching_summary(
    matched: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    if matched.empty:
        summary = pd.DataFrame()
    else:
        summary = (
            matched.groupby(["index", "athlete_id"], as_index=False)
            .agg(
                n_sessions=("activity_id", "size"),
                anchor_activity_id=(
                    "match_anchor_activity_id",
                    "first",
                ),
                duration_min_mean=("duration_s", lambda x: x.mean() / 60.0),
                mean_gap_speed_ms=("mean_gap_speed_ms", "mean"),
            )
        )
        summary["provenance"] = "MEASURED"
    path = repo_path(config, config["outputs"]["reliability_matching"])
    path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(path, index=False)
    return summary


def _plot_reliability(
    reliability: pd.DataFrame, config: dict[str, Any]
) -> Path:
    matplotlib_cache = repo_path(config, "outputs/.matplotlib")
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k_values = [int(k) for k in config["reliability"]["spearman_brown_k"]]
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    markers = ("o", "s", "^", "D", "P")
    for marker, (_, row) in zip(markers, reliability.iterrows()):
        values = [row[f"icc_k{k}"] for k in k_values]
        ci_low = [spearman_brown(row["ci_low"], k) for k in k_values]
        ci_high = [spearman_brown(row["ci_high"], k) for k in k_values]
        (line,) = ax.plot(
            k_values, values, marker=marker, label=row["index"]
        )
        ax.fill_between(
            k_values,
            ci_low,
            ci_high,
            color=line.get_color(),
            alpha=0.10,
            linewidth=0,
        )
    ax.axhline(
        config["reliability"]["target_icc"],
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="Target ICC",
    )
    ax.set_xlabel("Number of sessions averaged (k)")
    ax.set_ylabel("Reliability, ICC")
    ax.set_xticks(k_values)
    ax.set_ylim(-0.05, 1.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    path = repo_path(config, config["outputs"]["reliability_figure"])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def update_gate_sensitivity_icc(
    characteristics: pd.DataFrame,
    config: dict[str, Any],
    indices: Iterable[str] = INDEX_COLUMNS,
) -> pd.DataFrame:
    """Fill Phase-4 gate sensitivity with downstream point ICCs."""
    path = repo_path(config, config["outputs"]["gate_sensitivity"])
    sweep = pd.read_csv(path)
    rng = np.random.default_rng(int(config["seed"]))
    for index in indices:
        sweep[f"icc_{index}"] = np.nan
        sweep[f"n_athletes_{index}"] = 0
    for row_index, row in sweep.iterrows():
        passed = characteristics[
            (characteristics["drift"] > row["drift_threshold_bpm_per_h"])
            & (characteristics["magnitude"] > row["magnitude_threshold"])
        ]
        for index in indices:
            result, _ = reliability_for_index(
                passed,
                index,
                config,
                rng,
                with_bootstrap=False,
            )
            sweep.at[row_index, f"icc_{index}"] = result["icc_1_1"]
            sweep.at[row_index, f"n_athletes_{index}"] = result[
                "n_athletes"
            ]
    sweep["icc_status"] = "MEASURED_PHASE5"
    sweep.to_csv(path, index=False)
    return sweep


def run_reliability(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run Phase 5 and materialise its tables, intermediate, and figure."""
    gated_path = repo_path(config, config["data"]["gated_indices"])
    gated = pd.read_parquet(gated_path)
    qc_eligible = gated[gated["qc_eligible"].fillna(False)].copy()
    characteristics = materialize_session_characteristics(
        qc_eligible, config
    )
    gate_passing = characteristics[
        characteristics["gate_pass"].fillna(False)
    ].copy()
    rng = np.random.default_rng(int(config["seed"]))
    rows: list[dict[str, Any]] = []
    matched_frames: list[pd.DataFrame] = []
    for index in INDEX_COLUMNS:
        result, matched = reliability_for_index(
            gate_passing, index, config, rng
        )
        rows.append(result)
        matched_frames.append(matched)
        LOGGER.info(
            "Reliability %s: athletes=%s sessions=%s ICC=%.6f",
            index,
            result["n_athletes"],
            result["n_sessions"],
            result["icc_1_1"],
        )
    reliability = pd.DataFrame(rows)
    output_path = repo_path(config, config["outputs"]["reliability"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reliability.to_csv(output_path, index=False)
    matched_all = pd.concat(matched_frames, ignore_index=True)
    session_path = repo_path(config, config["data"]["reliability_sessions"])
    session_path.parent.mkdir(parents=True, exist_ok=True)
    matched_all.to_parquet(session_path, index=False)
    _write_matching_summary(matched_all, config)
    _plot_reliability(reliability, config)
    sensitivity = update_gate_sensitivity_icc(
        characteristics, config
    )
    return reliability, matched_all, sensitivity
