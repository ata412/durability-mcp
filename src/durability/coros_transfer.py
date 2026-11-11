"""E25-B: train on all FitRec gated runs, predict one COROS athlete (n = 1 demo).

The FitRec feature table (Phase 6) and its manifest define the 40 features. The
same ``FeatureBuilder`` (strict-prior history + first 40 percent of the target)
builds the COROS rows, so the only thing that changes is the data source.

This is an out-of-distribution transfer demo on one athlete, not an A1 test:
there is no athlete-split CV here, because the test athlete is simply never in
FitRec. Every number is MEASURED and n = 1.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd

from durability.athlete_params import AthleteParameterEstimator
from durability.config import repo_path
from durability.coros_case_study import _InMemoryHistory, is_outdoor_run, parse_fit, process_arm
from durability.features import FeatureBuilder, prepare_early_window
from durability.gate import gate_pass
from durability.indices import compute_activity_indices

LOGGER = logging.getLogger(__name__)
TARGETS = ["magnitude", "drift", "retention"]


def _hr_histogram(stream: pd.DataFrame, config: dict[str, Any]) -> list[int]:
    low, high = (int(v) for v in config["athlete_params"]["hr_valid_range"])
    hr = stream["heart_rate"].to_numpy(dtype=float)
    finite = hr[np.isfinite(hr) & (hr >= low) & (hr <= high)]
    histogram, _ = np.histogram(finite, bins=np.arange(low - 0.5, high + 1.5, 1.0))
    return histogram.astype(np.int32).tolist()


def build_coros_features(config: dict[str, Any], sessions: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the native-arm COROS streams and compute FitRec-identical feature rows."""
    settings = config["coros_case_study"]
    athlete_id = str(settings["athlete_id"])
    native = sessions[(sessions["arm"] == "native_1hz") & sessions["phase2_accepted"]].copy()
    native = native.sort_values("start_time").reset_index(drop=True)
    raw_dir = repo_path(config, settings["raw_dir"])

    streams: dict[str, pd.DataFrame] = {}
    for row in native.itertuples(index=False):
        meta, records = parse_fit(raw_dir / row.file, config)
        assert is_outdoor_run(meta, config)
        streams[row.activity_id], _ = process_arm(records, "native_1hz", config)

    history = native[["activity_id", "athlete_id", "start_time", "duration_s"]].copy()
    history["distance_m"] = native["gps_distance_m"].to_numpy()
    history["hr_histogram"] = [_hr_histogram(streams[a], config) for a in native["activity_id"]]
    history["gate_pass"] = native["gate_pass"].to_numpy()
    for column in ["magnitude", "onset", "drift", "cs_ms"]:
        history[column] = native[column].to_numpy()

    estimator = AthleteParameterEstimator(_InMemoryHistory(native, streams), config)
    dprime = {}
    for row in native.itertuples(index=False):
        params = estimator.estimate_params(athlete_id, row.start_time)
        dprime[row.activity_id] = params
    history["dprime_m"] = [dprime[a].dprime_m if dprime[a].eligible else np.nan for a in native["activity_id"]]
    history.loc[~history["activity_id"].map(lambda a: dprime[a].eligible), "cs_ms"] = np.nan
    builder = FeatureBuilder(history, config)

    rows: list[dict[str, Any]] = []
    for row in native.itertuples(index=False):
        params = dprime[row.activity_id]
        if not params.eligible:
            continue
        stream = streams[row.activity_id]
        early = prepare_early_window(
            stream,
            {"hrmax_bpm": params.hrmax_bpm, "cs_ms": params.cs_ms, "dprime_m": params.dprime_m},
            config,
        )
        features = builder.build_features(athlete_id, row.start_time, early)
        indices = compute_activity_indices(stream, params, config, onset_sampling_eligible=False)
        rows.append(
            {
                "activity_id": row.activity_id,
                "athlete_id": athlete_id,
                "start_time": row.start_time,
                "qc_eligible": bool(row.qc_eligible),
                "gate_pass": bool(row.qc_eligible and gate_pass(indices.drift, indices.magnitude, config)),
                "duration_min": row.duration_s / 60.0,
                **features,
                **{f"target_{t}": getattr(indices, t) for t in TARGETS},
            }
        )
    return pd.DataFrame(rows)


def run_transfer(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    from durability.models.exploratory import make_exploratory_model
    from durability.models.tier1_gbdt import make_gbdt_model

    settings = config["coros_case_study"]
    seed = int(config["seed"])
    sessions = pd.read_parquet(repo_path(config, settings["outputs"]["sessions"]))
    coros = build_coros_features(config, sessions)

    fitrec = pd.read_parquet(repo_path(config, config["data"]["features_output"]))
    manifest = pd.read_csv(repo_path(config, config["outputs"]["feature_manifest"]))
    feature_columns = manifest.loc[manifest["role"] == "feature", "column"].tolist()
    missing = set(feature_columns) - set(coros.columns)
    assert not missing, f"COROS rows miss FitRec features: {sorted(missing)}"
    assert not set(coros["athlete_id"]) & set(fitrec["athlete_id"].astype(str)), "test athlete leaked into training"

    def matrix(frame: pd.DataFrame) -> pd.DataFrame:
        return frame[feature_columns].astype(float)

    rows: list[pd.DataFrame] = []
    for target in TARGETS:
        y = fitrec[f"target_{target}"].to_numpy(dtype=float)
        finite = np.isfinite(y)
        x_train = matrix(fitrec.loc[finite])
        predictions = {"control_fitrec_mean": np.full(len(coros), float(y[finite].mean()))}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for name, factory in (
                ("tier1_lightgbm", lambda: make_gbdt_model("tier1_lightgbm", config, seed=seed)),
                ("elastic_net", lambda: make_exploratory_model("elastic_net", config, seed=seed)),
            ):
                model = factory()
                model.fit(x_train, y[finite])
                predictions[name] = model.predict(matrix(coros))
        # Athlete's own strict-prior mean of the same label (no FitRec at all): the
        # personal baseline that won in the FitRec audit.
        own = []
        for row in coros.itertuples(index=False):
            prior = coros[(coros["start_time"] < row.start_time) & coros["qc_eligible"]]
            own.append(float(prior[f"target_{target}"].mean()) if len(prior) else np.nan)
        predictions["own_prior_mean"] = np.asarray(own)
        for model_name, values in predictions.items():
            rows.append(
                pd.DataFrame(
                    {
                        "activity_id": coros["activity_id"],
                        "qc_eligible": coros["qc_eligible"],
                        "gate_pass": coros["gate_pass"],
                        "target": target,
                        "model": model_name,
                        "observed": coros[f"target_{target}"].to_numpy(dtype=float),
                        "predicted": values,
                        "fitrec_target_sd": float(np.std(y[finite], ddof=1)),
                        "provenance": "MEASURED",
                    }
                )
            )
    predictions = pd.concat(rows, ignore_index=True)
    predictions["abs_error"] = (predictions["predicted"] - predictions["observed"]).abs()

    summary_rows = []
    for (cohort_name, mask_col) in (("qc_eligible", "qc_eligible"), ("all_param_eligible", None)):
        subset = predictions if mask_col is None else predictions[predictions[mask_col]]
        for (target, model), group in subset.groupby(["target", "model"], sort=False):
            ok = group["observed"].notna() & group["predicted"].notna()
            g = group[ok]
            summary_rows.append(
                {
                    "cohort": cohort_name,
                    "target": target,
                    "model": model,
                    "n_runs": int(len(g)),
                    "mae": float(g["abs_error"].mean()) if len(g) else np.nan,
                    "spearman": float(g["observed"].corr(g["predicted"], method="spearman")) if len(g) > 2 else np.nan,
                    "predicted_sd": float(g["predicted"].std(ddof=1)) if len(g) > 1 else np.nan,
                    "observed_sd": float(g["observed"].std(ddof=1)) if len(g) > 1 else np.nan,
                    "provenance": "MEASURED",
                }
            )
    summary = pd.DataFrame(summary_rows)
    out = settings["transfer_outputs"]
    for name, table in (("predictions", predictions), ("summary", summary), ("features", coros)):
        path = repo_path(config, out[name])
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(path, index=False) if path.suffix == ".parquet" else table.to_csv(path, index=False)
    return {"predictions": predictions, "summary": summary, "features": coros}
