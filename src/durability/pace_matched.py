"""Experimental pace-matched indices for runs rejected only for GAP-speed variability."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def pace_matched_indices(
    elapsed_s: np.ndarray,
    heart_rate: np.ndarray,
    gap_speed_ms: np.ndarray,
    config: dict[str, Any],
) -> dict[str, float]:
    """Compare heart rate only at similar grade-adjusted speed, away from speed changes."""
    settings = config["pace_matched"]
    windows = config["indices"]
    t, hr, gap = (np.asarray(v, dtype=float) for v in (elapsed_s, heart_rate, gap_speed_ms))
    out = {"pm_drift": np.nan, "pm_magnitude": np.nan, "used_fraction": np.nan}
    if len(t) < 2 or t[-1] <= t[0]:
        return out
    fraction = (t - t[0]) / (t[-1] - t[0])
    retained = (fraction >= windows["trim_head_frac"]) & (fraction <= 1 - windows["trim_tail_frac"])
    if retained.sum() < 2:
        return out
    start, end = t[retained][0], t[retained][-1]
    late = retained & (t >= end - windows["late_window_frac"] * (end - start))
    early = (fraction >= windows["early_window"][0]) & (fraction < windows["early_window"][1])
    previous = (
        pd.Series(gap)
        .rolling(int(settings["transition_window_points"]), min_periods=int(settings["transition_min_points"]))
        .mean()
        .shift(1)
        .to_numpy()
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        settled = np.abs(gap / np.where(previous > 0, previous, np.nan) - 1) <= float(settings["transition_tolerance"])
        bins = np.floor(np.log(np.where(gap > 0, gap, np.nan)) / np.log1p(float(settings["bin_log_width"])))
    usable = retained & np.isfinite(hr) & np.isfinite(gap) & (gap > 0) & settled
    out["used_fraction"] = float(usable.sum() / retained.sum())

    frame = pd.DataFrame({"bin": bins[usable], "hours": t[usable] / 3600.0, "hr": hr[usable]})
    frame = frame[frame.groupby("bin")["hours"].transform("size") >= 2]
    if len(frame) >= int(settings["min_drift_samples"]):
        centred_t = frame["hours"] - frame.groupby("bin")["hours"].transform("mean")
        centred_hr = frame["hr"] - frame.groupby("bin")["hr"].transform("mean")
        if np.sqrt((centred_t**2).mean()) >= float(settings["min_within_bin_time_sd_h"]):
            out["pm_drift"] = float((centred_t * centred_hr).sum() / (centred_t**2).sum())

    def by_bin(mask: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame({"bin": bins[mask], "hr": hr[mask]}).groupby("bin")["hr"].agg(["mean", "size"])

    joined = by_bin(usable & early).join(by_bin(usable & late), how="inner", lsuffix="_early", rsuffix="_late")
    if len(joined):
        weight = np.minimum(joined["size_early"], joined["size_late"])
        if weight.sum() >= int(settings["min_magnitude_matched_samples"]):
            out["pm_magnitude"] = float((weight * joined["mean_late"] / joined["mean_early"]).sum() / weight.sum())
    return out
