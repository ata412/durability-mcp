"""Feed FIT files through the exact GoldenCheetah test path used by E29-E31.

GoldenCheetah streams carry seconds, cumulative km, heart rate and altitude. A FIT
run is converted to the same four channels, with km from consecutive GPS fixes
(haversine, as rule 13), then handed to ``goldencheetah.process_athlete`` unchanged.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pandas as pd

from durability import goldencheetah
from durability.coros_case_study import is_outdoor_run, parse_fit

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Distance between consecutive fixes; 0 for the first sample and wherever a fix is missing."""
    la, lo = np.radians(lat), np.radians(lon)
    a = np.sin(np.diff(la) / 2) ** 2 + np.cos(la[:-1]) * np.cos(la[1:]) * np.sin(np.diff(lo) / 2) ** 2
    step = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))
    return np.r_[0.0, np.nan_to_num(step)]


def fit_to_csv(records: pd.DataFrame) -> bytes:
    """secs, km, hr, alt from FIT records with a GPS fix; the target values are never read here."""
    fixed = records.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
    secs = (fixed.timestamp - fixed.timestamp.iloc[0]).dt.total_seconds().to_numpy()
    km = np.cumsum(haversine_km(fixed.latitude.to_numpy(float), fixed.longitude.to_numpy(float)))
    frame = pd.DataFrame({"secs": secs, "km": km, "hr": fixed.heart_rate.to_numpy(float), "alt": fixed.altitude.to_numpy(float)})
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    return buffer.getvalue().encode()


def fit_runs(folder: str | Path, config: dict[str, Any]) -> dict[pd.Timestamp, bytes]:
    """Outdoor runs keyed by the first GPS fix time, like GoldenCheetah's run start."""
    runs: dict[pd.Timestamp, bytes] = {}
    for path in sorted(Path(folder).glob("*.fit")):
        meta, records = parse_fit(path, config)
        if not is_outdoor_run(meta, config) or records.empty or records.latitude.notna().sum() < 2:
            continue
        start = records.dropna(subset=["latitude"]).timestamp.iloc[0]
        runs[start] = fit_to_csv(records)
    return runs


def process_fit_folder(folder: str | Path, athlete: str, config: dict[str, Any]) -> dict[str, Any]:
    """Per-athlete gated rows, labels and sequences, exactly as for a GoldenCheetah test athlete."""
    runs = fit_runs(folder, config)
    with mock.patch.object(goldencheetah, "parse_zip", return_value=runs):
        return goldencheetah.process_athlete(Path(f"{athlete}.zip"), config)
