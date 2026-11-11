"""E29 pure feature code (no LightGBM, no torch). Protocol: docs/E29_PROTOCOL.md."""
from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd


def state_row(group: pd.DataFrame, position: int, config: dict[str, Any]) -> dict[str, Any]:
    """How the target's first-40% values compare with this runner's runs that ended before it started."""
    settings = config["runner_state"]
    rules = config["rich_early"]["history"]
    start, end = group.start_time.values, group.end_time.values
    duration = group.duration_s.to_numpy(float)
    usable = (group.gap_speed_cv.to_numpy(float) < float(rules["steady_max_gap_speed_cv"])) & (duration >= float(rules["steady_min_duration_s"]))
    prior = np.flatnonzero((end < start[position]) & usable)
    row: dict[str, Any] = {"activity_id": group.activity_id.iat[position]}
    minimum = int(settings["state"]["min_prior"])
    if len(prior) < minimum:
        return row
    external, log_duration = group.f_ext_mean.to_numpy(float), np.log(duration)
    kernel = np.nan_to_num(np.exp(-0.5 * (((external[prior] - external[position]) / float(settings["state"]["bandwidth_external"])) ** 2
                                          + ((log_duration[prior] - log_duration[position]) / float(settings["state"]["bandwidth_log_duration"])) ** 2)))
    for column in settings["state_columns"]:
        values = group[column].to_numpy(float)
        now, past = values[position], values[prior]
        ok = np.isfinite(past)
        if ok.sum() >= minimum and np.isfinite(now):
            row[f"s_{column}_z"] = (now - past[ok].mean()) / (past[ok].std(ddof=1) + 1e-6)
            row[f"s_{column}_d"] = now - past[ok].mean()
            row[f"s_{column}_last"] = now - past[ok][-1]
            if kernel[ok].sum() > 1e-3:
                row[f"s_{column}_kd"] = now - (kernel[ok] * past[ok]).sum() / kernel[ok].sum()
    return row


def sequence_row(stream: pd.DataFrame, hrmax: float, cs: float, config: dict[str, Any]) -> tuple[np.ndarray, float, list[float]] | None:
    """Select existing samples from the first 40% (no interpolation). The aux list is a training label and reads the full run."""
    rules = config["runner_state"]["encoder"]
    elapsed = stream["elapsed_s"].to_numpy(float)
    duration = elapsed[-1] - elapsed[0]
    length = int(rules["sequence_length"])
    fraction = (elapsed - elapsed[0]) / duration
    window = np.flatnonzero(fraction <= float(config["features"]["early_window_frac"]))
    if duration < float(rules["min_duration_s"]) or len(window) < length:
        return None
    relative = stream["heart_rate"].to_numpy(float) / hrmax
    external = stream["gap_speed_ms"].to_numpy(float) / cs
    clip = float(rules["gradient_clip"])
    gradient = np.clip(stream["gradient"].to_numpy(float), -clip, clip) / clip
    pick = window[np.linspace(0, len(window) - 1, length).round().astype(int)]
    sequence = np.column_stack([relative[pick], external[pick], gradient[pick], fraction[pick]]).astype(np.float32)
    aux: list[float] = []
    half = float(rules["aux_halfwidth"])
    with np.errstate(invalid="ignore"):
        for centre in rules["aux_fracs"]:
            mask = (fraction > float(centre) - half) & (fraction <= float(centre) + half)
            aux += [float(np.nanmean(relative[mask])) if np.isfinite(relative[mask]).any() else np.nan,
                    float(np.nanmean(external[mask])) if np.isfinite(external[mask]).any() else np.nan]
    return np.nan_to_num(sequence, nan=0.0, posinf=0.0, neginf=0.0), float(np.log(duration)), aux


def residual_row(index: np.ndarray, position: int, start: np.ndarray, end: np.ndarray, labels: np.ndarray, predicted: np.ndarray,
                 activity_id: str, config: dict[str, Any]) -> dict[str, Any]:
    """The encoder's past errors on this runner; ``index`` lists this runner's rows in time order."""
    rules = config["runner_state"]["residual"]
    target = index[position]
    prior = index[:position]
    prior = prior[end[prior] < start[target]]
    row: dict[str, Any] = {"activity_id": activity_id, "e_n": len(prior)}
    for column, name in enumerate(("d", "m")):
        errors = labels[prior, column] - predicted[prior, column]
        errors = errors[np.isfinite(errors)]
        if errors.size:
            weight = 0.5 ** (np.arange(errors.size)[::-1] / float(rules["ewm_halflife_sessions"]))
            row[f"e_{name}_mean"] = errors.mean()
            row[f"e_{name}_ew"] = (weight * errors).sum() / weight.sum()
            row[f"e_{name}_shr"] = errors.sum() / (errors.size + float(rules["shrink"]))
        if errors.size > 2:
            row[f"e_{name}_sd"] = errors.std(ddof=1)
    return row


def network_keys(folds: int = 5) -> list[tuple[int, ...]]:
    """One network on everything, one excluding each fold, one excluding each pair of folds."""
    return [(), *[(fold,) for fold in range(folds)], *itertools.combinations(range(folds), 2)]


def stacking_key(test_fold: int, row_fold: int) -> tuple[int, ...]:
    """The network whose training excluded both the test fold and the row's own fold."""
    return (test_fold,) if row_fold == test_fold else tuple(sorted((test_fold, row_fold)))


def guard(train: pd.DataFrame, *others: pd.DataFrame, quantiles: tuple[float, float]) -> list[pd.DataFrame]:
    """Clip every input to the training range, so linear leaves cannot extrapolate."""
    low, high = train.quantile(quantiles[0]), train.quantile(quantiles[1])
    return [frame.clip(low, high, axis=1) for frame in (train, *others)]


def cap_runs(frame: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    """At most ``cap`` randomly chosen rows per athlete, the same rows on every call."""
    order = frame.sample(frac=1.0, random_state=seed)
    return order.groupby("athlete_id", sort=False).head(cap).sort_index()


def athlete_folds(athletes: Any, folds: int, seed: int) -> dict[Any, int]:
    """Assign each athlete to one fold at random."""
    order = np.random.default_rng(seed).permutation(sorted(set(athletes)))
    return {athlete: position % folds for position, athlete in enumerate(order)}


def nested_subset(pool: Any, fraction: float, seed: int) -> list[Any]:
    """Random athletes from ``pool``; a smaller fraction is always contained in a larger one."""
    order = np.random.default_rng(seed).permutation(sorted(set(pool)))
    return list(order[: int(round(fraction * len(order)))])
