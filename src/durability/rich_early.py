"""E28: rich first-40% features, wider history and Huber LightGBM. Protocol: docs/E28_PROTOCOL.md.

Every number is PILOT MEASURED EXPLORATORY on FitRec. Folds 3 and 4 were already opened once in an
unregistered lab, so this is a registered reproduction and not an independent confirmation.
"""
from __future__ import annotations

import logging
import warnings
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lightgbm import LGBMRegressor
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

from durability.config import repo_path
from durability.evaluate import assert_group_split_integrity, regression_metrics, shuffle_targets_within_athlete
from durability.level_change import markdown_table
from durability.models.tier1_gbdt import make_gbdt_model
from durability.preprocess import reload_preprocessed_stream
from durability.quality_v2 import tier_masks
from durability.two_stage_subgroups import bootstrap_weights, inference, weighted_bootstrap

LOGGER = logging.getLogger(__name__)
PROVENANCE = "PILOT MEASURED EXPLORATORY"
KEYS = ["activity_id", "athlete_id"]
FORBIDDEN_EXACT = {"abs_lat", "duration_s", "gap_speed_cv", "gate_pass", "drift", "magnitude", "retention", "onset",
                   "late_decoupling_mean", "gap_split", "clipped_gradient_fraction", "post_drift", "distance_m"}
DURATION_DEPENDENT = ("early_", "h_longest_ratio", "h_dur_z", "_kern", "_k_neff")
STREAM_COLUMNS = ["elapsed_s", "heart_rate", "gap_speed_ms", "gradient", "cumulative_distance_m"]


# --------------------------------------------------------------------------- session features
def _slope(x: np.ndarray, y: np.ndarray, minimum: int) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    return float(np.polyfit(x[ok], y[ok], 1)[0]) if ok.sum() >= minimum and np.ptp(x[ok]) > 0 else np.nan


def _mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return finite.mean() if finite.size else np.float64(np.nan)  # numpy scalar: a zero reference must not raise


def window_features(stream: pd.DataFrame, hrmax: float, cs: float, horizon_s: float, config: dict[str, Any]) -> dict[str, float]:
    """Features from samples at or before ``early_window_frac * horizon_s`` only.

    ``horizon_s`` is the run duration for the 40% window and a fixed value for the run-length-unknown window.
    """
    settings = config["rich_early"]
    elapsed = stream["elapsed_s"].to_numpy(float)
    fraction = (elapsed - elapsed[0]) / horizon_s
    window = fraction <= float(config["features"]["early_window_frac"])
    out: dict[str, float] = {}
    if window.sum() < int(settings["min_window_points"]):
        return out
    start, mid = (float(v) for v in config["indices"]["early_window"])
    third = float(settings["segment_fracs"][0])
    least = int(settings["min_slope_points"])
    hr = stream["heart_rate"].to_numpy(float)[window]
    gap = stream["gap_speed_ms"].to_numpy(float)[window]
    grad = stream["gradient"].to_numpy(float)[window]
    frac = fraction[window]
    hours = (elapsed[window] - elapsed[0]) / 3600.0
    minutes = hours * 60.0
    with np.errstate(divide="ignore", invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        decoupling = (hr / hrmax) / (gap / cs)
        decoupling[~(gap > 0)] = np.nan
        rel, ext = hr / hrmax, gap / cs
        core = frac >= start
        centre = np.nanmedian(gap[core])
        tolerance = float(config["indices"]["drift_speed_tolerance"])
        steady = core & np.isfinite(hr) & np.isfinite(gap) & (np.abs(gap - centre) <= tolerance * centre)
        a, b, c = (frac >= start) & (frac < mid), (frac >= mid) & (frac < third), frac >= third
        ref, seg_b, seg_c = _mean(decoupling[a]), _mean(decoupling[b]), _mean(decoupling[c])
        out.update(f_steady_frac=steady.sum() / max(core.sum(), 1), f_drift_same=_slope(hours[steady], hr[steady], least),
                   f_drift_core=_slope(hours[core], hr[core], least), f_dec_ref=ref, f_dec_b=seg_b, f_dec_c=seg_c,
                   f_mag_b=seg_b / ref, f_mag_c=seg_c / ref, f_dec_slope=_slope(hours[core], decoupling[core], least))
        ok = core & np.isfinite(hr) & np.isfinite(gap) & (gap > 0)
        if ok.sum() >= int(settings["min_regression_points"]) and np.ptp(gap[ok]) > 0:
            design = np.column_stack([np.ones(ok.sum()), gap[ok] - gap[ok].mean(), hours[ok] - hours[ok].mean()])
            beta, *_ = np.linalg.lstsq(design, hr[ok], rcond=None)
            out.update(f_hr_speed_gain=beta[1], f_hr_time_adj=beta[2], f_hr_res_sd=(hr[ok] - design @ beta).std())
        low, high = (float(v) for v in settings["hr_zone_fracs"])
        out.update(f_hr_mean=_mean(rel), f_hr_sd=np.nanstd(rel), f_hr_core=_mean(rel[core]), f_hr_q90=np.nanquantile(rel, 0.9),
                   f_hr_c=_mean(rel[c]), f_hr_a=_mean(rel[a]), f_ext_mean=_mean(ext), f_ext_core=_mean(ext[core]),
                   f_ext_sd=np.nanstd(ext), f_ext_slope=_slope(hours[core], ext[core], least),
                   f_ext_c_a=_mean(ext[c]) / _mean(ext[a]), f_speed=_mean(gap), f_grad_abs=_mean(np.abs(grad)),
                   f_grad_pos=_mean(np.clip(grad, 0, None)), f_grad_net=_mean(grad),
                   f_above85=_mean((rel > low).astype(float)), f_above90=_mean((rel > high).astype(float)))
        for point in settings["checkpoints_min"]:
            out[f"f_hr_{int(point)}m"] = rel[min(int(np.searchsorted(minutes, float(point))), len(rel) - 1)]
        warm = minutes <= float(settings["warmup_min"])
        out["f_warm_rise"] = out["f_hr_core"] - _mean(rel[warm]) if warm.any() else np.nan
    return out


def strict_future_drift(stream: pd.DataFrame, config: dict[str, Any]) -> float:
    """Same steady-speed slope as the drift label, on samples after the 40% cutoff only (label, never a feature)."""
    elapsed = stream["elapsed_s"].to_numpy(float)
    hr, gap = stream["heart_rate"].to_numpy(float), stream["gap_speed_ms"].to_numpy(float)
    fraction = (elapsed - elapsed[0]) / (elapsed[-1] - elapsed[0])
    mask = (fraction > float(config["features"]["early_window_frac"])) & (fraction <= 1.0 - float(config["indices"]["trim_tail_frac"]))
    centre = np.nanmedian(gap[mask])
    steady = mask & np.isfinite(hr) & np.isfinite(gap) & (np.abs(gap - centre) <= float(config["indices"]["drift_speed_tolerance"]) * centre)
    return _slope((elapsed[steady] - elapsed[0]) / 3600.0, hr[steady], int(config["rich_early"]["min_slope_points"]))


def _fixed_horizon(config: dict[str, Any]) -> float:
    return float(config["rich_early"]["fixed_window_min"]) * 60.0 / float(config["features"]["early_window_frac"])


def session_record(stream: pd.DataFrame, activity_id: str, hrmax: float, cs: float, config: dict[str, Any]) -> dict[str, Any]:
    elapsed = stream["elapsed_s"].to_numpy(float)
    record: dict[str, Any] = {"activity_id": activity_id, "distance_m": float(stream["cumulative_distance_m"].iloc[-1])}
    record.update(window_features(stream, hrmax, cs, float(elapsed[-1] - elapsed[0]), config))
    fixed = window_features(stream, hrmax, cs, _fixed_horizon(config), config)
    record.update({f"m18_{name}": value for name, value in fixed.items()})
    record["post_drift"] = strict_future_drift(stream, config)
    return record


def _load_record(row, config):
    stream = reload_preprocessed_stream(config, row.athlete_id, row.activity_id)[STREAM_COLUMNS]
    return session_record(stream, row.activity_id, row.hrmax_bpm, row.cs_ms, config)


def all_sessions(config: dict[str, Any]) -> pd.DataFrame:
    table = pd.read_parquet(repo_path(config, config["data"]["gated_indices"]))
    table = table[table.indices_computed].copy()
    for key in KEYS:
        table[key] = table[key].astype(str)
    table["start_time"] = pd.to_datetime(table.start_time, utc=True)
    table["end_time"] = table.start_time + pd.to_timedelta(table.duration_s, unit="s")
    table["gate_pass"] = table.gate_pass.fillna(False).astype(bool)
    table["qc_eligible"] = table.qc_eligible.fillna(False).astype(bool)
    return table.sort_values(["athlete_id", "start_time", "activity_id"]).reset_index(drop=True)


# --------------------------------------------------------------------------- history
def history_row(group: pd.DataFrame, position: int, config: dict[str, Any]) -> dict[str, Any]:
    """History for one target: only sessions that ended before the target started."""
    settings = config["rich_early"]
    rules = settings["history"]
    start, end = group.start_time.values, group.end_time.values
    prior = end < start[position]
    drift = np.clip(group.drift.to_numpy(float), *settings["label_clip"]["drift"])
    magnitude = np.clip(group.magnitude.to_numpy(float), *settings["label_clip"]["magnitude"])
    log_duration = np.log(group.duration_s.to_numpy(float))
    distance = group.distance_m.to_numpy(float)
    external, terrain = group.f_ext_mean.to_numpy(float), group.f_grad_abs.to_numpy(float)
    early_drift, early_magnitude = group.f_drift_same.to_numpy(float), group.f_mag_c.to_numpy(float)
    pools = {"all": (group.gap_speed_cv.to_numpy(float) < float(rules["steady_max_gap_speed_cv"]))
             & (group.duration_s.to_numpy(float) >= float(rules["steady_min_duration_s"])),
             "qc": group.qc_eligible.to_numpy(), "gate": group.gate_pass.to_numpy()}
    row: dict[str, Any] = {"activity_id": group.activity_id.iat[position], "h_n_all": int(prior.sum())}
    days = (start[position] - start[prior]) / np.timedelta64(1, "D") if prior.any() else np.array([])
    for width in rules["load_windows_days"]:
        recent = days <= float(width)
        row[f"h_n_{width}d"] = int(recent.sum())
        row[f"h_km_{width}d"] = distance[prior][recent].sum() / 1000.0
        row[f"h_h_{width}d"] = group.duration_s.to_numpy(float)[prior][recent].sum() / 3600.0
    row["h_days_last"] = days.min() if len(days) else np.nan
    row["h_longest_ratio"] = log_duration[position] - log_duration[prior].max() if prior.any() else np.nan
    row["h_dur_z"] = (log_duration[position] - log_duration[prior].mean()) / (log_duration[prior].std() + 0.1) if prior.sum() > 2 else np.nan
    weights_scale = rules["kernel_bandwidth"]
    for pool, mask in pools.items():
        chosen = prior & mask & np.isfinite(drift) & np.isfinite(magnitude)
        count = int(chosen.sum())
        row[f"h_{pool}_n"] = count
        for short, label in (("d", drift), ("m", magnitude)):
            values = label[chosen]
            if count:
                decay = 0.5 ** (np.arange(count)[::-1] / float(rules["ewm_halflife_sessions"]))
                row[f"h_{pool}_{short}_mean"] = values.mean()
                row[f"h_{pool}_{short}_med"] = np.median(values)
                row[f"h_{pool}_{short}_ew5"] = (decay * values).sum() / decay.sum()
                row[f"h_{pool}_{short}_last3"] = values[-int(rules["recent_sessions"]):].mean()
            if count > 2:
                row[f"h_{pool}_{short}_sd"] = values.std(ddof=1)
        if count >= int(rules["min_context_sessions"]):
            kernel = np.nan_to_num(np.exp(-0.5 * (((log_duration[chosen] - log_duration[position]) / float(weights_scale["log_duration"])) ** 2
                                                  + ((external[chosen] - external[position]) / float(weights_scale["external_load"])) ** 2
                                                  + ((terrain[chosen] - terrain[position]) / float(weights_scale["abs_gradient"])) ** 2)))
            row[f"h_{pool}_k_neff"] = kernel.sum() ** 2 / ((kernel ** 2).sum() + 1e-9)
            if kernel.sum() > 1e-6:
                row[f"h_{pool}_d_kern"] = (kernel * drift[chosen]).sum() / kernel.sum()
                row[f"h_{pool}_m_kern"] = (kernel * magnitude[chosen]).sum() / kernel.sum()
            gap_d = (drift[chosen] - early_drift[chosen])
            gap_m = (magnitude[chosen] - early_magnitude[chosen])
            gap_d, gap_m = gap_d[np.isfinite(gap_d)], gap_m[np.isfinite(gap_m)]
            if gap_d.size >= int(rules["min_context_sessions"]):
                row[f"h_{pool}_d_minus_early"] = np.median(gap_d)
            if gap_m.size >= int(rules["min_context_sessions"]):
                row[f"h_{pool}_m_minus_early"] = np.median(gap_m)
            if np.ptp(log_duration[chosen]) > float(rules["min_duration_slope_range"]):
                row[f"h_{pool}_d_durslope"] = np.polyfit(log_duration[chosen], drift[chosen], 1)[0]
                row[f"h_{pool}_m_durslope"] = np.polyfit(log_duration[chosen], magnitude[chosen], 1)[0]
    return row


def _athlete_history(group, config):
    group = group.reset_index(drop=True)
    return [history_row(group, int(i), config) for i in np.flatnonzero(group.gate_pass.to_numpy())]


# --------------------------------------------------------------------------- frame and feature sets
def assert_allowed_features(columns: list[str]) -> None:
    bad = [c for c in columns if c in FORBIDDEN_EXACT or c.startswith(("target_", "aud_", "sh_"))]
    assert not bad, f"Forbidden E28 inputs: {bad}"


def feature_sets(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, list[str]]:
    manifest = pd.read_csv(repo_path(config, config["outputs"]["feature_manifest"]))
    base = manifest.loc[manifest.role.eq("feature"), "column"].tolist()
    early = [c for c in frame if c.startswith("f_")]
    pooled = [c for c in frame if c.startswith(("h_qc_", "h_all_"))]
    load = [c for c in frame if c.startswith("h_") and not c.startswith(("h_all_", "h_qc_", "h_gate_"))]
    core = base + early + pooled + load
    fixed = [c for c in frame if c.startswith("m18_")] + [
        c for c in base + pooled + load if not any(token in c for token in DURATION_DEPENDENT)]
    for columns in (core, fixed):
        assert_allowed_features(columns)
        assert len(columns) == len(set(columns))
    assert not any(token in c for c in fixed for token in DURATION_DEPENDENT if not c.startswith("m18_"))
    return {"manifest": base, "core": core, "fixed": fixed}


def build_frame(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    outputs = config["rich_early"]["outputs"]
    sessions = all_sessions(config)
    jobs = int(config["evaluate"]["n_jobs"])
    records = Parallel(n_jobs=jobs, batch_size=64)(delayed(_load_record)(row, config) for row in
                                                   sessions[KEYS + ["hrmax_bpm", "cs_ms"]].itertuples())
    features = pd.DataFrame(records)
    sessions = sessions.merge(features, on="activity_id", validate="one_to_one")
    rows = Parallel(n_jobs=jobs)(delayed(_athlete_history)(group, config) for _, group in sessions.groupby("athlete_id"))
    history = pd.DataFrame([row for part in rows for row in part])
    for name, table in (("session_features", features), ("history", history)):
        path = repo_path(config, outputs[name])
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(path, index=False)
    cohort = pd.read_parquet(repo_path(config, config["quality_cohort_prediction"]["outputs"]["cohort"]))
    for key in KEYS:
        cohort[key] = cohort[key].astype(str)
    cohort["start_time"] = pd.to_datetime(cohort.start_time, utc=True)
    assert cohort.gate_pass.all() and cohort.groupby("athlete_id").outer_fold.nunique().eq(1).all()
    frame = (cohort.merge(features, on="activity_id", validate="one_to_one")
             .merge(history, on="activity_id", validate="one_to_one")
             .sort_values(["athlete_id", "start_time"]).reset_index(drop=True))
    assert len(frame) == len(cohort)
    return frame, sessions


# --------------------------------------------------------------------------- models
def huber_alpha(y: np.ndarray, quantile: float) -> float:
    return float(np.quantile(np.abs(y - np.median(y)), quantile))


def _huber(y, seed, config):
    settings = config["rich_early"]
    return LGBMRegressor(objective="huber", alpha=huber_alpha(y, float(settings["huber_quantile"])), verbosity=-1,
                         n_jobs=int(config["evaluate"]["n_jobs"]), random_state=seed, **settings["lightgbm"])


def _cross_fold(frame, target_values, fit):
    """Each fold is predicted by a model fitted on the other four folds."""
    out = np.full(len(frame), np.nan)
    for fold in sorted(frame.outer_fold.unique()):
        train = np.flatnonzero(frame.outer_fold.ne(fold).to_numpy())
        test = np.flatnonzero(frame.outer_fold.eq(fold).to_numpy())
        assert_group_split_integrity(frame, train, test, group_col="athlete_id")
        out[test] = fit(frame.iloc[train], target_values[train], frame.iloc[test], int(fold))
    return out


def fit_huber(columns, config):
    seed0, bags = int(config["seed"]), int(config["rich_early"]["bag_seeds"])

    def fit(a, y, b, _fold):
        return np.mean([_huber(y, seed0 + i, config).fit(a[columns], y).predict(b[columns]) for i in range(bags)], axis=0)
    return fit


def _clip(values, target, config):
    name = {"target_drift": "drift", "target_magnitude": "magnitude", "strict_future_drift": "drift"}[target]
    return np.clip(values, *config["rich_early"]["label_clip"][name])


def predict_candidates(frame, sets, target, config):
    observed = frame[target].to_numpy(float)
    clipped = _clip(observed, target, config)
    short = "d" if "drift" in target else "m"
    strength = float(config["rich_early"]["shrinkage_strength"])
    seed = int(config["seed"])

    def production(a, y, b, _fold):
        return make_gbdt_model("tier1_lightgbm", config, seed=seed).fit(a[sets["manifest"]].to_numpy(float), y).predict(b[sets["manifest"]].to_numpy(float))

    def shrunk_fit(_a, y, b, _fold):
        population = y.mean()
        count = b["h_gate_n"].to_numpy(float)
        mean = b[f"h_gate_{short}_mean"].fillna(population).to_numpy(float)
        return (count * mean + strength * population) / (count + strength)

    def duration(a, y, b, _fold):
        settings = config["rich_early"]
        model = LGBMRegressor(**{**settings["lightgbm"], "n_estimators": int(settings["duration_only_trees"]), "colsample_bytree": 1.0},
                              verbosity=-1, n_jobs=int(config["evaluate"]["n_jobs"]), random_state=seed)
        return model.fit(a[["early_duration_min"]], y).predict(b[["early_duration_min"]])

    results = {"population_mean": _cross_fold(frame, clipped, lambda _a, y, b, _f: np.full(len(b), y.mean())),
               "shrunk_history": _cross_fold(frame, clipped, shrunk_fit),
               "production_lgbm": _cross_fold(frame, observed, production),
               "duration_only": _cross_fold(frame, clipped, duration),
               "huber_lgbm": _cross_fold(frame, clipped, fit_huber(sets["core"], config))}
    if target != "strict_future_drift":
        results["huber_18min"] = _cross_fold(frame, clipped, fit_huber(sets["fixed"], config))
    return results


def within_metrics(observed, predicted, athletes) -> dict[str, float]:
    athletes = np.asarray(athletes)
    y = observed - pd.Series(observed).groupby(athletes).transform("mean").to_numpy()
    p = predicted - pd.Series(predicted).groupby(athletes).transform("mean").to_numpy()
    total = (y ** 2).sum()
    return {"r2_within": 1 - ((y - p) ** 2).sum() / total if total > 0 else np.nan,
            "spearman_within": spearmanr(y, p).statistic if np.unique(p).size > 1 else np.nan}


# --------------------------------------------------------------------------- audit and probe
def _stream_audit(sessions, config, rng):
    columns = ["heart_rate", "gap_speed_ms", "gradient"]
    changed, bite, tested = 0, 0, 0
    sample = sessions[sessions.gate_pass].sample(int(config["rich_early"]["audit_runs"]), random_state=int(config["seed"]))
    frac = float(config["features"]["early_window_frac"])
    for row in sample[KEYS + ["hrmax_bpm", "cs_ms"]].itertuples():
        stream = reload_preprocessed_stream(config, row.athlete_id, row.activity_id)[STREAM_COLUMNS].copy()
        for column in columns:
            stream[column] = stream[column].astype(float)
        elapsed = stream.elapsed_s.to_numpy(float)
        span = elapsed[-1] - elapsed[0]
        for horizon, prefix in ((span, "f_"), (_fixed_horizon(config), "m18_")):
            base = window_features(stream, row.hrmax_bpm, row.cs_ms, horizon, config)
            mutated = stream.copy()
            after = (elapsed - elapsed[0]) / horizon > frac
            for column in columns:
                mutated.loc[after, column] = rng.uniform(1, 200, after.sum())
            new = window_features(mutated, row.hrmax_bpm, row.cs_ms, horizon, config)
            changed += int(any(not np.isclose(base[k], new.get(k, np.nan), equal_nan=True) for k in base))
            if prefix == "f_":
                bite += int(not np.isclose(strict_future_drift(stream, config), strict_future_drift(mutated, config), equal_nan=True))
        tested += 1
    return {"check": "stream_after_cutoff_mutated", "runs": tested, "runs_with_changed_features": changed, "mutation_changed_full_run_label": bite}


def _history_audit(sessions, config, rng):
    changed, tested = 0, 0
    sample = sessions[sessions.gate_pass].sample(int(config["rich_early"]["audit_runs"]), random_state=int(config["seed"]) + 1)
    for row in sample[KEYS].itertuples():
        group = sessions[sessions.athlete_id == row.athlete_id].reset_index(drop=True)
        position = int(np.flatnonzero(group.activity_id == row.activity_id)[0])
        base = history_row(group, position, config)
        mutated = group.copy()
        future = mutated.end_time >= mutated.start_time.iat[position]
        for column in ("drift", "magnitude", "f_drift_same", "f_mag_c", "distance_m"):
            mutated.loc[future, column] = rng.normal(size=future.sum()) * 50
        later = future.to_numpy() & (np.arange(len(group)) != position)
        for column in ("f_ext_mean", "f_grad_abs"):
            mutated.loc[later, column] = rng.normal(size=later.sum())
        new = history_row(mutated, position, config)
        keys = [k for k in base if k != "activity_id"]
        changed += int(set(base) != set(new) or any(not np.isclose(float(base[k]), float(new[k]), equal_nan=True) for k in keys))
        tested += 1
    return {"check": "target_and_future_labels_mutated", "runs": tested, "runs_with_changed_features": changed, "mutation_changed_full_run_label": np.nan}


def probe_status(leakage: pd.DataFrame, limit: float) -> dict[str, bool]:
    """A target is accepted only if the pooled shuffled R² is within the limit on every scope."""
    return {target: bool((part.r_squared <= limit).all()) for target, part in leakage.groupby("target")}


def shuffled_probe(frame, sets, config):
    targets = list(config["rich_early"]["targets"])
    limit = float(config["evaluate"]["leakage_r2_max"])
    shuffled = shuffle_targets_within_athlete(frame, targets, seed=int(config["seed"]))
    holdout = frame.outer_fold.isin(config["rich_early"]["holdout_folds"]).to_numpy()
    rows = []
    for target in targets:
        observed = shuffled[target].to_numpy(float)
        estimate = _cross_fold(shuffled, _clip(observed, target, config), fit_huber(sets["core"], config))
        for scope, mask in (("all_folds", np.ones(len(frame), bool)), ("holdout_3_4", holdout)):
            metrics = regression_metrics(observed[mask], estimate[mask])
            rows.append({"target": target, "scope": scope, "r_squared": metrics["r_squared"], "mae": metrics["mae"],
                         **within_metrics(observed[mask], estimate[mask], frame.athlete_id.to_numpy()[mask]),
                         "limit": limit, "exceeds_limit": metrics["r_squared"] > limit,
                         "provenance": "MEASURED WITHIN-ATHLETE SHUFFLED TARGET"})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- inference
def summarise(predictions, config):
    settings = config["rich_early"]
    rows, pairs = [], []
    holdout_folds = settings["holdout_folds"]
    for scope, data in (("holdout_3_4", predictions[predictions.outer_fold.isin(holdout_folds)]), ("all_folds", predictions)):
        for tier in settings["report_tiers"]:
            part_tier = data if tier == "Q0" else data[data.in_q1]
            for (target, model), part in part_tier.groupby(["target", "model"]):
                error = (part.observed - part.predicted).abs().groupby(part.athlete_id).mean()
                rows.append({"scope": scope, "tier": tier, "target": target, "model": model,
                             **regression_metrics(part.observed, part.predicted), "mae_athlete": error.mean(), "n_athletes": len(error),
                             **within_metrics(part.observed.to_numpy(), part.predicted.to_numpy(), part.athlete_id.to_numpy()), "provenance": PROVENANCE})
    primary = predictions[predictions.outer_fold.isin(holdout_folds)]
    plan = [("H1", "target_drift", "Q0", "huber_lgbm", "production_lgbm"), ("H2", "target_magnitude", "Q0", "huber_lgbm", "production_lgbm"),
            ("H3", "target_drift", "Q0", "huber_lgbm", "shrunk_history"), ("H4", "target_magnitude", "Q0", "huber_lgbm", "shrunk_history")]
    for target in settings["targets"]:
        plan += [("context_q1", target, "Q1", "huber_lgbm", "production_lgbm"), ("context_q1", target, "Q1", "huber_lgbm", "shrunk_history"),
                 ("context_18min", target, "Q0", "huber_18min", "production_lgbm"), ("context_18min", target, "Q0", "huber_18min", "shrunk_history")]
    plan.append(("context_strict_future", "strict_future_drift", "Q0", "huber_lgbm", "production_lgbm"))
    for hypothesis, target, tier, challenger, reference in plan:
        data = primary[primary.target.eq(target)]
        data = data if tier == "Q0" else data[data.in_q1]
        errors = {m: (p.observed - p.predicted).abs().groupby(p.athlete_id).mean() for m, p in data.groupby("model")}
        athletes = sorted(data.athlete_id.unique())
        weights = bootstrap_weights(len(athletes), int(settings["bootstrap_n"]), int(config["seed"]))
        delta = errors[challenger] - errors[reference]
        draws = weighted_bootstrap(delta, athletes, weights)
        pairs.append({"hypothesis": hypothesis, "target": target, "tier": tier, "challenger": challenger, "reference": reference,
                      "delta_mae_athlete": float(delta.mean()), **inference(delta.mean(), draws, float(settings["confidence_level"])),
                      "n_athletes": len(athletes), "provenance": PROVENANCE})
    contrasts = pd.DataFrame(pairs)
    contrasts["q_bh"] = multipletests(contrasts.p_bootstrap, method="fdr_bh")[1]
    contrasts["meets_gain_criterion"] = contrasts.ci_high.lt(0) & contrasts.q_bh.le(float(settings["fdr_alpha"]))
    return pd.DataFrame(rows), contrasts


def apply_probe(contrasts, status):
    parent = {"target_drift": "target_drift", "target_magnitude": "target_magnitude", "strict_future_drift": "target_drift"}
    contrasts = contrasts.copy()
    contrasts["probe_accepted"] = contrasts.target.map(parent).map(status)
    contrasts["established_gain"] = contrasts.meets_gain_criterion & contrasts.probe_accepted
    return contrasts


# --------------------------------------------------------------------------- orchestration
def run_experiment(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    settings = config["rich_early"]
    outputs = settings["outputs"]
    frame, sessions = build_frame(config)
    sets = feature_sets(frame, config)
    LOGGER.info("E28 features: manifest=%s core=%s fixed=%s rows=%s", len(sets["manifest"]), len(sets["core"]), len(sets["fixed"]), len(frame))
    rng = np.random.default_rng(int(config["seed"]))
    audit = pd.DataFrame([_stream_audit(sessions, config, rng), _history_audit(sessions, config, rng)])
    audit["provenance"] = "MEASURED"
    LOGGER.info("E28 mutation audit:\n%s", audit.to_string(index=False))
    if audit.runs_with_changed_features.sum() > 0:
        raise AssertionError("E28 halted: a feature changed after future data was mutated")
    assert audit.mutation_changed_full_run_label.iloc[0] > 0, "Stream mutation had no effect, so the audit proves nothing"

    in_q1 = tier_masks(frame, config)["Q1"].to_numpy()
    long_rows = []
    frame["strict_future_drift"] = frame["post_drift"]
    usable = frame["strict_future_drift"].notna().to_numpy()
    for target in list(settings["targets"]) + ["strict_future_drift"]:
        subset = frame[usable].reset_index(drop=True) if target == "strict_future_drift" else frame
        flags = in_q1[usable] if target == "strict_future_drift" else in_q1
        results = predict_candidates(subset, sets, target, config)
        for model, values in results.items():
            assert np.isfinite(values).all(), (target, model)
            part = subset[KEYS + ["outer_fold"]].copy()
            part["target"], part["model"], part["in_q1"] = target, model, flags
            part["observed"], part["predicted"], part["provenance"] = subset[target].to_numpy(float), values, PROVENANCE
            long_rows.append(part)
        LOGGER.info("E28 predicted %s", target)
    predictions = pd.concat(long_rows, ignore_index=True)

    # reproduction checks
    holdout = predictions.outer_fold.isin(settings["holdout_folds"])
    benchmark = pd.read_parquet(repo_path(config, config["data"]["benchmark_predictions"]))
    benchmark["activity_id"] = benchmark.activity_id.astype(str)
    checks = []
    for target in settings["targets"]:
        name = target.removeprefix("target_")
        ours = predictions[holdout & predictions.target.eq(target)]
        huber = ours[ours.model.eq("huber_lgbm")]
        r2 = regression_metrics(huber.observed, huber.predicted)["r_squared"]
        production = ours[ours.model.eq("production_lgbm")].set_index("activity_id")
        stored = benchmark[(benchmark.model == "tier1_lightgbm") & (benchmark.target == name)].set_index("activity_id").predicted.reindex(production.index)
        stored_r2 = regression_metrics(production.observed, stored)["r_squared"]
        ours_r2 = regression_metrics(production.observed, production.predicted)["r_squared"]
        checks.append({"target": target, "huber_r2": r2, "lab_r2": settings["lab_reference_r2"][target], "production_r2": ours_r2, "stored_benchmark_r2": stored_r2})
        LOGGER.info("E28 reproduction %s", checks[-1])
        if abs(r2 - float(settings["lab_reference_r2"][target])) > float(settings["reproduction_tol"]) or abs(ours_r2 - stored_r2) > 1e-3:
            raise AssertionError(f"E28 reproduction check failed: {checks[-1]}")

    leakage = shuffled_probe(frame, sets, config)
    LOGGER.info("E28 shuffled probe:\n%s", leakage.to_string(index=False))
    status = probe_status(leakage, float(config["evaluate"]["leakage_r2_max"]))
    summary, contrasts = summarise(predictions, config)
    contrasts = apply_probe(contrasts, status)
    tables = {"predictions": predictions, "summary": summary, "contrasts": contrasts, "leakage": leakage, "audit": audit}
    for name, table in tables.items():
        path = repo_path(config, outputs[name])
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(path, index=False) if path.suffix == ".parquet" else table.to_csv(path, index=False)
    _figure(summary, config)
    _report(summary, contrasts, leakage, audit, pd.DataFrame(checks), status, sets, config)
    return tables


def _figure(summary, config):
    order = ["population_mean", "shrunk_history", "duration_only", "production_lgbm", "huber_18min", "huber_lgbm"]
    names = {"population_mean": "Population mean", "shrunk_history": "Shrunk history", "duration_only": "Duration only",
             "production_lgbm": "Production LightGBM", "huber_18min": "E28, first 18 min", "huber_lgbm": "E28, first 40%"}
    data = summary[(summary.scope == "holdout_3_4") & (summary.tier == "Q0")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, (target, title) in zip(axes, [("target_drift", "Drift"), ("target_magnitude", "Magnitude")], strict=True):
        part = data[data.target == target].set_index("model").reindex(order)
        colours = ["#2a78d6" if m == "huber_lgbm" else "#8c8c8c" for m in order]
        ax.barh(range(len(order)), part.r_squared.clip(lower=0), height=0.55, color=colours)
        for i, value in enumerate(part.r_squared):
            ax.text(max(value, 0) + 0.006, i, f"{value:.3f}", va="center", fontsize=9, color="#404040")
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_yticks(range(len(order)), [names[m] for m in order], fontsize=9)
        ax.set_xlabel("Held-out R² (negative values drawn at 0)", fontsize=9)
        ax.set_xlim(0, 0.45)
        ax.set_xticks(np.arange(0, 0.41, 0.1))
        ax.grid(axis="x", color="#e5e5e5", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", length=0)
    fig.suptitle("E28 on folds 3 and 4, PILOT MEASURED EXPLORATORY (not an independent confirmation)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    path = repo_path(config, config["rich_early"]["outputs"]["figure"])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _report(summary, contrasts, leakage, audit, checks, status, sets, config):
    def table(frame, digits=4):
        return markdown_table(frame.round(digits))
    columns = ["target", "model", "mae", "mae_athlete", "r_squared", "spearman", "r2_within", "spearman_within", "n", "n_athletes"]
    view = lambda scope, tier: table(summary[(summary.scope == scope) & (summary.tier == tier)].sort_values(["target", "mae"])[columns])
    verdict = "\n".join(f"- **{t}: {'ACCEPTED' if ok else 'NOT ACCEPTED'}** by the shuffled-probe rule." for t, ok in status.items())
    lines = ["# E28: Rich first-40% features, wider history and Huber LightGBM", "",
             "All results are **PILOT MEASURED EXPLORATORY** on FitRec. Protocol: docs/E28_PROTOCOL.md. "
             f"Seed {config['seed']} (CONFIGURED). Features: {len(sets['core'])} (first 40%), {len(sets['fixed'])} (first 18 minutes).", "",
             "**This is a registered reproduction, not an independent confirmation.** Folds 3 and 4 were already opened once in an unregistered lab, and Huber LightGBM alone was chosen after that look.", "",
             "## Verdict", "", verdict, "",
             "A gain counts as established only if its contrast meets the criterion and the target passes the probe.", "",
             "## Shuffled probe", "", table(leakage), "",
             "## Mutation audit", "", table(audit), "",
             "## Reproduction checks", "", table(checks), "",
             "## Primary scope: folds 3 and 4", "", view("holdout_3_4", "Q0"), "",
             "### Q1 (retrospective membership)", "", view("holdout_3_4", "Q1"), "",
             "## Contrasts (athlete-weighted MAE, BH across all contrasts)", "", table(contrasts.drop(columns=["provenance"])), "",
             "## Secondary scope: all five folds (descriptive)", "", view("all_folds", "Q0"), "",
             "![E28](../../figures/e28_rich_early.png)", "",
             "## Limits", "",
             "- The 40% window requires knowing the run length. `huber_18min` shows the result without it.",
             "- The cohort is selected on the outcome through the gate, so these are not deployment scores.",
             "- Drift and magnitude labels overlap the input between 12% and 40% of the run. `strict_future_drift` removes this overlap for drift. Its test labels are not clipped here.",
             "- Centred 30 s speed smoothing lets samples just before the cutoff see about 15 s ahead (production pipeline, not measured here).",
             "- A true confirmation needs new data or a new cohort.", ""]
    path = repo_path(config, config["rich_early"]["outputs"]["report"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
