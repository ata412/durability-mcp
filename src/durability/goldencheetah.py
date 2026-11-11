"""GoldenCheetah OpenData (CC0, OSF 6hfpz) through the production Phase 1-4 code. Used by E29 as an external holdout."""
from __future__ import annotations

import concurrent.futures as futures
import io
import json
import logging
import random
import re
import socket
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from durability.athlete_params import AthleteParameterEstimator
from durability.config import repo_path
from durability.coros_case_study import _gap_speed_cv, _InMemoryHistory, apply_qc, thin_to_budget
from durability.coros_transfer import _hr_histogram
from durability.derive import smooth_speed
from durability.features import FeatureBuilder, prepare_early_window
from durability.gap import add_grade_adjusted_speed
from durability.indices import compute_activity_indices
from durability.resample import resample_activity
from durability.rich_early import history_row, session_record
from durability.runner_state_features import sequence_row, state_row

LOGGER = logging.getLogger(__name__)


def download_holdout(config: dict[str, Any]) -> pd.DataFrame:
    """Sample athletes outside the seen list, with a saved manifest so the sample cannot change."""
    rules = config["runner_state"]["goldencheetah"]
    raw = repo_path(config, rules["raw_dir"])
    raw.mkdir(parents=True, exist_ok=True)
    socket.setdefaulttimeout(float(rules["download_timeout_s"]))
    retries = int(rules["download_retries"])

    def fetch_json(address):
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(address) as response:
                    return json.load(response)
            except OSError:
                if attempt == retries - 1:
                    raise
                time.sleep(15 * 2 ** attempt)

    path = repo_path(config, rules["manifest"])
    if path.exists():
        manifest = pd.read_csv(path)
    else:
        seen = set(repo_path(config, rules["seen_list"]).read_text().split())
        listing, url = [], f"https://api.osf.io/v2/nodes/{rules['osf_node']}/files/osfstorage/?page%5Bsize%5D=100"
        while url:
            page = fetch_json(url)
            listing += [(x["attributes"]["name"], x["attributes"]["size"], x["links"]["download"]) for x in page["data"]
                        if x["attributes"]["kind"] == "file" and x["attributes"]["name"].endswith(".zip")]
            url = page["links"].get("next")
        fresh = sorted(x for x in listing if x[0] not in seen and (x[1] or 0) < float(rules["max_zip_bytes"]))
        random.Random(int(config["seed"])).shuffle(fresh)
        manifest = pd.DataFrame(fresh[:int(rules["n_athletes"])], columns=["name", "bytes", "url"])
        assert not seen & set(manifest.name)
        manifest.to_csv(path, index=False)
        LOGGER.info("E29 holdout: %s of %s athletes selected, %.1f GB", len(manifest), len(listing), manifest.bytes.sum() / 1e9)

    def fetch(row):
        target = raw / row.name_
        if target.exists() and target.stat().st_size == row.bytes:
            return True
        part = Path(str(target) + ".part")
        for attempt in range(retries):
            try:
                urllib.request.urlretrieve(row.url, part)
                if part.stat().st_size != row.bytes:
                    raise OSError("size mismatch")
                part.rename(target)
                return True
            except OSError:
                time.sleep(15 * 2 ** attempt)
        return False
    with futures.ThreadPoolExecutor(int(rules["download_workers"])) as pool:
        done = list(pool.map(fetch, manifest.rename(columns={"name": "name_"}).itertuples()))
    failed = done.count(False)
    LOGGER.info("E29 downloads: %s of %s failed", failed, len(done))
    assert failed <= float(rules["max_download_failure_frac"]) * len(done), f"{failed} downloads failed"
    return manifest[done].reset_index(drop=True)


def parse_zip(path: Path, config: dict[str, Any]) -> dict[pd.Timestamp, bytes]:
    """Match Run entries in the summary JSON to their CSV files; the zip is read, never extracted."""
    hours = float(config["runner_state"]["goldencheetah"]["time_match_hours"])
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        summary = [n for n in names if n.endswith(".json")]
        if not summary:
            return {}
        meta = json.loads(archive.read(summary[0]).decode("utf-8", "ignore"))
        stamps = {}
        for name in names:
            match = re.match(r"(\d{4})_(\d\d)_(\d\d)_(\d\d)_(\d\d)_(\d\d)\.csv$", name)
            if match:
                stamps[name] = pd.Timestamp(*map(int, match.groups()))
        runs = {}
        for ride in meta.get("RIDES", []):
            if str(ride.get("sport")) != "Run":
                continue
            start = pd.Timestamp(ride["date"].replace(" UTC", "").replace("/", "-"), tz="UTC")
            naive = start.tz_localize(None)
            hits = [n for n, c in stamps.items() if c.minute == start.minute and c.second == start.second and abs((c - naive).total_seconds()) <= hours * 3600]
            if len(hits) == 1:
                runs[start] = archive.read(hits[0])
    return runs


def stream_from_csv(raw: bytes, config: dict[str, Any]) -> pd.DataFrame:
    """Phase-1 stream from the distance channel, on the FitRec 500-point budget."""
    rules = config["runner_state"]["goldencheetah"]
    frame = pd.read_csv(io.BytesIO(raw))
    if not {"secs", "km", "hr", "alt"}.issubset(frame.columns):
        raise ValueError("missing channels")
    frame = frame[["secs", "km", "hr", "alt"]].apply(pd.to_numeric, errors="coerce").dropna(subset=["secs", "km"]).sort_values("secs").drop_duplicates("secs")
    low, high = rules["hr_valid_range"]
    frame["hr"] = frame.hr.where(frame.hr.between(low, high))
    if frame.alt.notna().mean() < float(rules["min_altitude_coverage"]) or frame.alt.std() < float(rules["min_altitude_sd_m"]):
        raise ValueError("altitude flat or missing")
    frame["timestamp"] = pd.to_datetime(frame.secs, unit="s", utc=True)
    frame = thin_to_budget(frame.reset_index(drop=True), int(config["coros_case_study"]["fitrec_points"])).reset_index(drop=True)
    seconds, km = frame.secs.to_numpy(float), frame.km.to_numpy(float)
    speed = np.r_[0.0, np.diff(km) * 1000.0 / np.maximum(np.diff(seconds), 1e-6)]
    lo, hi = config["data"]["speed_sanity_ms"]
    if not lo <= np.nanmedian(speed) <= hi:
        raise ValueError("speed sanity")
    return pd.DataFrame({"timestamp": frame.timestamp, "elapsed_s": seconds - seconds[0], "heart_rate": frame.hr.to_numpy(float), "altitude": frame.alt.to_numpy(float),
                         "latitude": np.nan, "longitude": np.nan, "derived_distance_km": np.r_[0.0, np.diff(km)], "derived_speed_ms": speed,
                         "smoothed_speed_ms": smooth_speed(speed, seconds, config["data"]["speed_smooth_window_s"])})


def process_athlete(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    athlete = Path(path).stem
    empty = {"athlete_id": athlete, "gated": [], "labels": [], "sequences": [], "counts": {"runs": 0, "accepted": 0, "computed": 0, "qc": 0, "gate": 0}}
    try:
        runs = parse_zip(path, config)
    except (zipfile.BadZipFile, json.JSONDecodeError, ValueError) as exc:
        return {**empty, "error": str(exc)[:80]}
    rows, streams = [], {}
    for start, raw in sorted(runs.items()):
        row = {"athlete_id": athlete, "activity_id": f"{athlete}_{int(start.timestamp())}", "start_time": start, "file": "", "duration_s": np.nan, "phase2_accepted": False}
        try:
            phase1 = stream_from_csv(raw, config)
            resampled, rm = resample_activity(phase1, config)
            stream, gm = add_grade_adjusted_speed(resampled, config)
            step = np.r_[0.0, np.diff(stream.elapsed_s.to_numpy(float))]
            stream["cumulative_distance_m"] = np.cumsum(step * np.nan_to_num(stream.smoothed_speed_ms.to_numpy(float)))
            row.update(duration_s=float(phase1.elapsed_s.iloc[-1]), native_dt_s=rm.native_dt_s, onset_sampling_eligible=rm.onset_sampling_eligible, gap_split=rm.gap_split,
                       retained_duration_s_phase2=rm.retained_duration_s, gap_available=gm.gap_available, clipped_gradient_fraction=gm.clipped_gradient_fraction,
                       gps_distance_m=float(np.nansum(phase1.derived_distance_km) * 1000.0), hr_coverage=float(stream.heart_rate.notna().mean()),
                       gap_speed_cv=_gap_speed_cv(stream), phase2_accepted=True)
            streams[row["activity_id"]] = stream
        except (AssertionError, ValueError, KeyError) as exc:
            row["reject"] = str(exc)[:60]
        rows.append(row)
    if not streams:
        return {**empty, "counts": {**empty["counts"], "runs": len(rows)}}
    table = pd.DataFrame(rows).sort_values("start_time").reset_index(drop=True)
    accepted = table[table.phase2_accepted].reset_index(drop=True)
    estimator = AthleteParameterEstimator(_InMemoryHistory(accepted, streams), config)
    out = []
    for row in table.to_dict("records"):
        if row["phase2_accepted"]:
            try:
                params = estimator.estimate_params(athlete, row["start_time"])
                row.update(n_preceding_activities=params.n_preceding_activities, params_eligible=params.eligible, hrmax_bpm=params.hrmax_bpm, cs_ms=params.cs_ms, dprime_m=params.dprime_m, cs_r_squared=params.cs_r_squared)
                if params.eligible:
                    indices = compute_activity_indices(streams[row["activity_id"]], params, config, onset_sampling_eligible=row["onset_sampling_eligible"])
                    row.update(magnitude=indices.magnitude, drift=indices.drift, retention=indices.retention, onset=indices.onset, cs_shift=indices.cs_shift, early_decoupling_mean=indices.early_decoupling_mean, late_decoupling_mean=indices.late_decoupling_mean, indices_computed=True)
            except (AssertionError, ValueError) as exc:
                row["param_error"] = str(exc)[:60]
        out.append(row)
    frame = pd.DataFrame(out)
    for column in ("magnitude", "drift", "retention", "onset", "params_eligible", "n_preceding_activities", "hrmax_bpm", "cs_ms", "dprime_m", "indices_computed", "cs_r_squared", "clipped_gradient_fraction", "gap_split", "retained_duration_s_phase2", "hr_coverage", "gap_speed_cv"):
        if column not in frame:
            frame[column] = np.nan
    frame["indices_computed"] = frame.indices_computed.fillna(False).astype(bool)
    frame, _ = apply_qc(frame, config)
    counts = {"runs": len(frame), "accepted": int(frame.phase2_accepted.sum()), "computed": int(frame.indices_computed.sum()), "qc": int(frame.qc_eligible.sum()), "gate": int(frame.gate_pass.sum())}
    done = frame[frame.indices_computed].copy()
    if done.empty:
        return {**empty, "counts": counts}
    columns = ["elapsed_s", "heart_rate", "gap_speed_ms", "gradient", "cumulative_distance_m"]
    records = [session_record(streams[r.activity_id][columns], r.activity_id, r.hrmax_bpm, r.cs_ms, config) for r in done.itertuples()]
    done = done.merge(pd.DataFrame(records), on="activity_id", suffixes=("", "_w")).sort_values("start_time").reset_index(drop=True)
    done["end_time"] = done.start_time + pd.to_timedelta(done.duration_s, unit="s")
    done["qc_eligible"] = done.qc_eligible.fillna(False).astype(bool)
    done["gate_pass"] = done.gate_pass.fillna(False).astype(bool)
    for column in config["runner_state"]["state_columns"]:
        if column not in done:
            done[column] = np.nan
    sequences = []
    for r in done.itertuples():
        made = sequence_row(streams[r.activity_id], r.hrmax_bpm, r.cs_ms, config)
        if made is not None:
            sequences.append((r.activity_id, made[0], made[1]))
    labels = done[["activity_id", "athlete_id", "start_time", "end_time", "drift", "magnitude", "gate_pass"]].to_dict("records")
    positions = np.flatnonzero(done.gate_pass.to_numpy())
    if not len(positions):
        return {**empty, "counts": counts, "labels": labels, "sequences": sequences}
    history = pd.DataFrame([history_row(done, int(i), config) for i in positions])
    state = pd.DataFrame([state_row(done, int(i), config) for i in positions])
    source = accepted[["activity_id", "athlete_id", "start_time", "duration_s"]].copy()
    source["distance_m"] = accepted.gps_distance_m
    source["hr_histogram"] = [_hr_histogram(streams[a], config) for a in accepted.activity_id]
    indexed = frame.set_index("activity_id")
    for column in ("gate_pass", "magnitude", "onset", "drift", "cs_ms", "dprime_m"):
        source[column] = source.activity_id.map(indexed[column])
    source["gate_pass"] = source.gate_pass.fillna(False).astype(bool)
    builder = FeatureBuilder(source, config)
    manifest = []
    for r in done[done.gate_pass].itertuples():
        early = prepare_early_window(streams[r.activity_id], {"hrmax_bpm": r.hrmax_bpm, "cs_ms": r.cs_ms, "dprime_m": r.dprime_m}, config)
        manifest.append({"activity_id": r.activity_id, **builder.build_features(athlete, r.start_time, early)})
    result = done[done.gate_pass].merge(history, on="activity_id").merge(state, on="activity_id", how="left").merge(pd.DataFrame(manifest), on="activity_id", how="left", suffixes=("", "_mf"))
    return {"athlete_id": athlete, "gated": result.to_dict("records"), "labels": labels, "sequences": sequences, "counts": counts}
