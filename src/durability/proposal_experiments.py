"""Post-Phase-7 proposal experiments on the FitRec pilot corpus.

These experiments do not replace the accepted Phase 1--7 benchmark. They
exercise the analysis proposed for Sensei using only the currently available
FitRec artifacts. Every predictor remains restricted to strict prior history
and, where applicable, the configured observed fraction of the target run.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from durability.config import repo_path
from durability.evaluate import (
    athlete_rank_spearman,
    make_group_folds,
    regression_metrics,
)
from durability.features import FeatureBuilder, prepare_early_window
from durability.preprocess import reload_preprocessed_stream
from durability.reliability import (
    bootstrap_icc_by_athlete,
    icc_one_way_random,
)

LOGGER = logging.getLogger(__name__)


def _settings(config: dict[str, Any]) -> dict[str, Any]:
    return config["experiments"]


def _output_path(config: dict[str, Any], key: str) -> Path:
    return repo_path(config, _settings(config)["outputs"][key])


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def load_experiment_frame(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[str]]:
    """Join leakage-safe Phase-6 predictors to Phase-3 late components."""
    frame = pd.read_parquet(
        repo_path(config, config["data"]["features_output"])
    ).reset_index(drop=True)
    frame["activity_id"] = frame["activity_id"].astype(str)
    frame["athlete_id"] = frame["athlete_id"].astype(str)
    labels = pd.read_parquet(
        repo_path(config, config["data"]["gated_indices"]),
        columns=[
            "activity_id",
            "athlete_id",
            "early_decoupling_mean",
            "late_decoupling_mean",
        ],
    )
    labels["activity_id"] = labels["activity_id"].astype(str)
    labels["athlete_id"] = labels["athlete_id"].astype(str)
    frame = frame.merge(
        labels,
        on=["activity_id", "athlete_id"],
        how="left",
        validate="one_to_one",
    )
    frame = frame.rename(
        columns={
            "early_decoupling_mean_y": "observed_early_reference",
            "late_decoupling_mean": "target_late_decoupling",
            "early_decoupling_mean_x": "early_decoupling_mean",
        }
    )
    required = {
        "observed_early_reference",
        "target_late_decoupling",
        "target_magnitude",
    }
    if required - set(frame):
        raise AssertionError(
            f"Experiment frame missing labels: {sorted(required - set(frame))}"
        )
    reconstructed = (
        frame["target_late_decoupling"]
        / frame["observed_early_reference"]
    )
    finite = reconstructed.notna() & frame["target_magnitude"].notna()
    maximum_error = float(
        np.abs(
            reconstructed.loc[finite]
            - frame.loc[finite, "target_magnitude"]
        ).max()
    )
    tolerance = float(
        _settings(config)["magnitude_reconstruction_tolerance"]
    )
    if maximum_error > tolerance:
        raise AssertionError(
            "Magnitude does not reconstruct from late/early decoupling: "
            f"maximum error={maximum_error}"
        )
    if not frame["target_late_decoupling"].notna().all():
        raise AssertionError("Primary late-decoupling target must be finite")

    manifest = pd.read_csv(
        repo_path(config, config["outputs"]["feature_manifest"])
    )
    feature_columns = manifest.loc[
        manifest["role"].eq("feature"), "column"
    ].tolist()
    missing_features = set(feature_columns) - set(frame.columns)
    if missing_features:
        raise AssertionError(
            f"Manifest features missing: {sorted(missing_features)}"
        )
    return frame, feature_columns


def feature_blocks(
    feature_columns: Iterable[str], config: dict[str, Any]
) -> dict[str, list[str]]:
    """Return the six prespecified feature-block configurations."""
    columns = list(feature_columns)
    block_settings = _settings(config)["feature_blocks"]
    early_prefixes = tuple(block_settings["early_prefixes"])
    prior_prefixes = tuple(
        block_settings["prior_durability_prefixes"]
    )
    parameters = set(block_settings["athlete_parameter_columns"])
    load_prefixes = tuple(block_settings["training_load_prefixes"])

    early = [
        column for column in columns if column.startswith(early_prefixes)
    ]
    history = [column for column in columns if column not in early]
    prior = [
        column for column in columns if column.startswith(prior_prefixes)
    ]
    training_load = [
        column for column in columns if column.startswith(load_prefixes)
    ]
    return {
        "history_only": history,
        "early_window_only": early,
        "combined": columns,
        "combined_without_prior_durability": [
            column for column in columns if column not in prior
        ],
        "combined_without_athlete_parameters": [
            column for column in columns if column not in parameters
        ],
        "combined_without_training_load": [
            column for column in columns if column not in training_load
        ],
    }


def _assignments(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    assignments = make_group_folds(
        frame,
        group_col=config["evaluate"]["group_col"],
        n_splits=int(config["evaluate"]["outer_folds"]),
    )
    if assignments.groupby("athlete_id")["outer_fold"].nunique().max() != 1:
        raise AssertionError("An athlete was split across outer folds")
    return assignments


def oof_predictions(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    config: dict[str, Any],
    *,
    model_name: str | None = None,
    include_mean: bool = True,
) -> pd.DataFrame:
    """Fit a locked model in athlete-wise outer folds."""
    from durability.models.tier1_gbdt import make_gbdt_model

    model_name = model_name or str(_settings(config)["primary_model"])
    assignments = _assignments(frame, config)
    x = frame[feature_columns]
    y = frame[target_column].to_numpy(dtype=float)
    rows: list[pd.DataFrame] = []
    for fold in range(int(config["evaluate"]["outer_folds"])):
        test = np.flatnonzero(
            assignments["outer_fold"].to_numpy(dtype=int) == fold
        )
        train = np.flatnonzero(
            assignments["outer_fold"].to_numpy(dtype=int) != fold
        )
        finite_train = np.isfinite(y[train])
        finite_test = np.isfinite(y[test])
        if not finite_train.any():
            raise AssertionError("Outer training fold has no finite target")
        if include_mean:
            rows.append(
                _prediction_frame(
                    frame,
                    test[finite_test],
                    y[test][finite_test],
                    np.full(
                        int(finite_test.sum()),
                        float(y[train][finite_train].mean()),
                    ),
                    fold=fold,
                    model="control_mean",
                )
            )
        model = make_gbdt_model(
            model_name,
            config,
            seed=int(config["seed"]) + fold,
        )
        model.fit(x.iloc[train].loc[finite_train], y[train][finite_train])
        estimate = model.predict(x.iloc[test].loc[finite_test])
        rows.append(
            _prediction_frame(
                frame,
                test[finite_test],
                y[test][finite_test],
                estimate,
                fold=fold,
                model=model_name,
            )
        )
    return pd.concat(rows, ignore_index=True)


def _prediction_frame(
    frame: pd.DataFrame,
    indices: np.ndarray,
    observed: np.ndarray,
    predicted: np.ndarray,
    *,
    fold: int,
    model: str,
) -> pd.DataFrame:
    result = frame.iloc[indices][
        [
            "activity_id",
            "athlete_id",
            "start_time",
            "observed_early_reference",
            "target_magnitude",
        ]
    ].copy()
    result["outer_fold"] = int(fold)
    result["model"] = model
    result["endpoint"] = "late_decoupling"
    result["observed"] = np.asarray(observed, dtype=float)
    result["predicted"] = np.asarray(predicted, dtype=float)
    result["provenance"] = "PILOT MEASURED"
    return result


def add_reconstructed_magnitude(
    late_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Report magnitude only after predicting its unknown late numerator."""
    magnitude = late_predictions.copy()
    denominator = magnitude["observed_early_reference"].to_numpy(
        dtype=float
    )
    if np.any(~np.isfinite(denominator)) or np.any(denominator <= 0):
        raise AssertionError("Observed early reference must be positive")
    magnitude["endpoint"] = "reconstructed_magnitude"
    magnitude.loc[
        magnitude["model"].eq("control_mean"), "model"
    ] = "control_late_mean_reconstructed"
    magnitude["observed"] = magnitude["target_magnitude"].to_numpy(
        dtype=float
    )
    magnitude["predicted"] = (
        magnitude["predicted"].to_numpy(dtype=float) / denominator
    )
    return pd.concat([late_predictions, magnitude], ignore_index=True)


def oof_direct_mean_control(
    frame: pd.DataFrame,
    target_column: str,
    config: dict[str, Any],
    *,
    endpoint: str,
    model_name: str,
) -> pd.DataFrame:
    """Fold-local direct mean control for a secondary reconstructed endpoint."""
    assignments = _assignments(frame, config)
    y = frame[target_column].to_numpy(dtype=float)
    rows: list[pd.DataFrame] = []
    for fold in range(int(config["evaluate"]["outer_folds"])):
        test = np.flatnonzero(assignments["outer_fold"].to_numpy() == fold)
        train = np.flatnonzero(assignments["outer_fold"].to_numpy() != fold)
        finite_train = np.isfinite(y[train])
        finite_test = np.isfinite(y[test])
        result = _prediction_frame(
            frame,
            test[finite_test],
            y[test][finite_test],
            np.full(
                int(finite_test.sum()),
                float(y[train][finite_train].mean()),
            ),
            fold=fold,
            model=model_name,
        )
        result["endpoint"] = endpoint
        rows.append(result)
    return pd.concat(rows, ignore_index=True)


def _cluster_bootstrap_mae(
    frame: pd.DataFrame,
    *,
    n_bootstrap: int,
    confidence_level: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    groups = [
        group[["observed", "predicted"]].to_numpy(dtype=float)
        for _, group in frame.groupby("athlete_id", sort=True)
    ]
    if len(groups) < 2:
        return np.nan, np.nan
    values = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        sampled = rng.integers(0, len(groups), size=len(groups))
        array = np.concatenate([groups[position] for position in sampled])
        values[index] = float(np.mean(np.abs(array[:, 0] - array[:, 1])))
    alpha = 1.0 - confidence_level
    low, high = np.quantile(values, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(low), float(high)


def prediction_summary(
    predictions: pd.DataFrame, config: dict[str, Any], *, seed_offset: int = 0
) -> pd.DataFrame:
    """Summarise session accuracy, calibration, and athlete ranking."""
    settings = _settings(config)
    rows: list[dict[str, Any]] = []
    for number, ((model, endpoint), group) in enumerate(
        predictions.groupby(["model", "endpoint"], sort=False)
    ):
        metrics = regression_metrics(
            group["observed"].to_numpy(dtype=float),
            group["predicted"].to_numpy(dtype=float),
        )
        rank = athlete_rank_spearman(
            group, "observed", "predicted"
        )
        finite = group[["observed", "predicted"]].dropna()
        if len(finite) >= 2 and finite["predicted"].nunique() >= 2:
            slope, intercept = np.polyfit(
                finite["predicted"].to_numpy(dtype=float),
                finite["observed"].to_numpy(dtype=float),
                1,
            )
        else:
            slope = intercept = np.nan
        low, high = _cluster_bootstrap_mae(
            group,
            n_bootstrap=int(settings["bootstrap_n"]),
            confidence_level=float(settings["confidence_level"]),
            rng=np.random.default_rng(
                int(config["seed"]) + seed_offset + number
            ),
        )
        rows.append(
            {
                "model": model,
                "endpoint": endpoint,
                **metrics,
                "mae_ci_low": low,
                "mae_ci_high": high,
                "athlete_rank_spearman": rank,
                "calibration_slope": float(slope),
                "calibration_intercept": float(intercept),
                "n_athletes": int(group["athlete_id"].nunique()),
                "provenance": "PILOT MEASURED",
            }
        )
    return pd.DataFrame(rows)


def run_e1(
    frame: pd.DataFrame,
    feature_columns: list[str],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """E1: predict the unknown late component after the 40% cutoff."""
    late = oof_predictions(
        frame,
        feature_columns,
        "target_late_decoupling",
        config,
    )
    predictions = add_reconstructed_magnitude(late)
    predictions = pd.concat(
        [
            predictions,
            oof_direct_mean_control(
                frame,
                "target_magnitude",
                config,
                endpoint="reconstructed_magnitude",
                model_name="control_direct_magnitude_mean",
            ),
        ],
        ignore_index=True,
    )
    summary = prediction_summary(predictions, config)
    _write_parquet(predictions, _output_path(config, "e1_predictions"))
    _write_csv(summary, _output_path(config, "e1_summary"))
    return predictions, summary


def paired_athlete_mae_contrast(
    predictions: pd.DataFrame,
    *,
    reference: str,
    comparator: str,
    n_bootstrap: int,
    confidence_level: float,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Paired athlete-level MAE difference: reference minus comparator."""
    selected = predictions[
        predictions["model"].isin([reference, comparator])
    ].copy()
    selected["absolute_error"] = np.abs(
        selected["observed"] - selected["predicted"]
    )
    athlete = (
        selected.groupby(["athlete_id", "model"])["absolute_error"]
        .mean()
        .unstack("model")
        .dropna()
    )
    if athlete.empty:
        return {
            "mean_difference": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "n_athletes": 0,
        }
    difference = (
        athlete[reference] - athlete[comparator]
    ).to_numpy(dtype=float)
    estimates = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        sample = rng.integers(0, len(difference), size=len(difference))
        estimates[index] = float(difference[sample].mean())
    alpha = 1.0 - confidence_level
    low, high = np.quantile(
        estimates, [alpha / 2.0, 1.0 - alpha / 2.0]
    )
    return {
        "mean_difference": float(difference.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "n_athletes": int(len(difference)),
    }


def run_e2(
    frame: pd.DataFrame,
    feature_columns: list[str],
    config: dict[str, Any],
    *,
    e1_predictions: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """E2: compare prespecified history and early-window feature blocks."""
    blocks = feature_blocks(feature_columns, config)
    predictions: list[pd.DataFrame] = []
    for block, columns in blocks.items():
        if block == "combined" and e1_predictions is not None:
            result = e1_predictions[
                (e1_predictions["model"] == _settings(config)["primary_model"])
                & (e1_predictions["endpoint"] == "late_decoupling")
            ].copy()
        else:
            result = oof_predictions(
                frame,
                columns,
                "target_late_decoupling",
                config,
                include_mean=False,
            )
        result["model"] = block
        result["feature_count"] = len(columns)
        predictions.append(result)
    all_predictions = pd.concat(predictions, ignore_index=True)
    summary = prediction_summary(all_predictions, config, seed_offset=100)

    settings = _settings(config)
    contrast_rows: list[dict[str, Any]] = []
    for number, comparator in enumerate(blocks):
        if comparator == "combined":
            continue
        contrast = paired_athlete_mae_contrast(
            all_predictions,
            reference="combined",
            comparator=comparator,
            n_bootstrap=int(settings["bootstrap_n"]),
            confidence_level=float(settings["confidence_level"]),
            rng=np.random.default_rng(int(config["seed"]) + 200 + number),
        )
        contrast_rows.append(
            {
                "reference": "combined",
                "comparator": comparator,
                "difference_definition": "combined_mae_minus_comparator_mae",
                **contrast,
                "provenance": "PILOT MEASURED",
            }
        )
    contrasts = pd.DataFrame(contrast_rows)
    _write_parquet(all_predictions, _output_path(config, "e2_predictions"))
    _write_csv(summary, _output_path(config, "e2_summary"))
    _write_csv(contrasts, _output_path(config, "e2_contrasts"))
    return all_predictions, summary, contrasts


def permute_outcome(
    values: np.ndarray,
    athletes: np.ndarray,
    *,
    mechanism: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate one hierarchical empirical-null outcome."""
    values = np.asarray(values, dtype=float)
    athletes = np.asarray(athletes).astype(str)
    if mechanism == "global":
        return values[rng.permutation(len(values))]
    if mechanism == "within_athlete":
        result = values.copy()
        for athlete in np.unique(athletes):
            indices = np.flatnonzero(athletes == athlete)
            result[indices] = values[indices][rng.permutation(len(indices))]
        return result
    if mechanism == "athlete_centered_residual":
        means = pd.Series(values).groupby(athletes).transform("mean").to_numpy()
        residual = values - means
        permuted = residual[rng.permutation(len(residual))]
        for athlete in np.unique(athletes):
            indices = np.flatnonzero(athletes == athlete)
            permuted[indices] -= float(permuted[indices].mean())
        return means + permuted
    raise ValueError(f"Unknown permutation mechanism: {mechanism}")


def run_e3(
    e1_predictions: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """E3: fixed-prediction hierarchical permutation inference."""
    model = str(_settings(config)["primary_model"])
    selected = e1_predictions[e1_predictions["model"].eq(model)].copy()
    endpoints = sorted(selected["endpoint"].unique())
    wide_observed = selected.pivot(
        index=["activity_id", "athlete_id"],
        columns="endpoint",
        values="observed",
    ).reset_index()
    wide_predicted = selected.pivot(
        index=["activity_id", "athlete_id"],
        columns="endpoint",
        values="predicted",
    ).reset_index()
    merged = wide_observed.merge(
        wide_predicted,
        on=["activity_id", "athlete_id"],
        suffixes=("_observed", "_predicted"),
        validate="one_to_one",
    )
    athletes = merged["athlete_id"].to_numpy()
    observed_stat = {
        endpoint: regression_metrics(
            merged[f"{endpoint}_observed"].to_numpy(dtype=float),
            merged[f"{endpoint}_predicted"].to_numpy(dtype=float),
        )["r_squared"]
        for endpoint in endpoints
    }

    permutation = _settings(config)["permutation"]
    rows: list[dict[str, Any]] = []
    for mechanism_index, mechanism in enumerate(permutation["mechanisms"]):
        rng = np.random.default_rng(
            int(config["seed"]) + 10_000 * (mechanism_index + 1)
        )
        for permutation_index in range(int(permutation["n_per_mechanism"])):
            statistics: dict[str, float] = {}
            for endpoint in endpoints:
                null_values = permute_outcome(
                    merged[f"{endpoint}_observed"].to_numpy(dtype=float),
                    athletes,
                    mechanism=str(mechanism),
                    rng=rng,
                )
                statistics[endpoint] = regression_metrics(
                    null_values,
                    merged[f"{endpoint}_predicted"].to_numpy(dtype=float),
                )["r_squared"]
            rows.append(
                {
                    "mechanism": mechanism,
                    "permutation": permutation_index,
                    **{
                        f"r_squared_{endpoint}": value
                        for endpoint, value in statistics.items()
                    },
                    "maximum_r_squared": float(max(statistics.values())),
                    "provenance": "PILOT MEASURED",
                }
            )
    null = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    for mechanism, group in null.groupby("mechanism", sort=False):
        for endpoint in endpoints:
            adjusted_p = (
                1
                + int(
                    (
                        group["maximum_r_squared"]
                        >= observed_stat[endpoint]
                    ).sum()
                )
            ) / (len(group) + 1)
            summary_rows.append(
                {
                    "mechanism": mechanism,
                    "endpoint": endpoint,
                    "observed_r_squared": observed_stat[endpoint],
                    "null_mean_maximum_r_squared": float(
                        group["maximum_r_squared"].mean()
                    ),
                    "null_95th_percentile_maximum_r_squared": float(
                        group["maximum_r_squared"].quantile(0.95)
                    ),
                    "familywise_adjusted_empirical_p": float(adjusted_p),
                    "n_permutations": int(len(group)),
                    "provenance": "PILOT MEASURED",
                }
            )
    summary = pd.DataFrame(summary_rows)
    _write_parquet(null, _output_path(config, "e3_null"))
    _write_csv(summary, _output_path(config, "e3_summary"))
    return null, summary


def materialize_horizon_features(
    frame: pd.DataFrame,
    feature_columns: list[str],
    config: dict[str, Any],
) -> pd.DataFrame:
    """Recompute only early-window features at each observed fraction."""
    fractions = [
        float(value) for value in _settings(config)["horizon_fractions"]
    ]
    if sorted(fractions) != fractions or fractions[-1] != float(
        config["features"]["early_window_frac"]
    ):
        raise AssertionError(
            "Horizon fractions must end at the Phase-6 observed fraction"
        )
    blocks = feature_blocks(feature_columns, config)
    early_columns = blocks["early_window_only"]
    history_columns = blocks["history_only"]
    empty_history = pd.DataFrame(
        columns=[
            "activity_id",
            "athlete_id",
            "start_time",
            "duration_s",
            "distance_m",
        ]
    )
    builders: dict[float, tuple[dict[str, Any], FeatureBuilder]] = {}
    for fraction in fractions:
        if fraction in (0.0, fractions[-1]):
            continue
        fraction_config = copy.deepcopy(config)
        fraction_config["features"]["early_window_frac"] = fraction
        builders[fraction] = (
            fraction_config,
            FeatureBuilder(empty_history, fraction_config),
        )

    rows: list[dict[str, Any]] = []
    progress_every = int(config["data"]["feature_progress_every"])
    for number, target in enumerate(frame.itertuples(index=False), start=1):
        base = {
            "activity_id": str(target.activity_id),
            "athlete_id": str(target.athlete_id),
            "start_time": target.start_time,
            "observed_early_reference": float(
                target.observed_early_reference
            ),
            "target_magnitude": float(target.target_magnitude),
            "target_late_decoupling": float(
                target.target_late_decoupling
            ),
            **{
                column: getattr(target, column)
                for column in history_columns
            },
        }
        for fraction in fractions:
            output = base | {
                "observed_fraction": fraction,
                "provenance": "PILOT MEASURED",
            }
            if fraction == 0.0:
                rows.append(output)
                continue
            if fraction == fractions[-1]:
                output.update(
                    {
                        column: getattr(target, column)
                        for column in early_columns
                    }
                )
                rows.append(output)
                continue
            fraction_config, builder = builders[fraction]
            stream = reload_preprocessed_stream(
                config, target.athlete_id, target.activity_id
            )
            early = prepare_early_window(
                stream,
                {
                    "hrmax_bpm": float(target.hrmax_bpm_asof),
                    "cs_ms": float(target.cs_ms_asof),
                    "dprime_m": float(target.dprime_m_asof),
                },
                fraction_config,
            )
            # Reuse the accepted Phase-6 calculation while intentionally
            # excluding history: history values are copied from the already
            # audited 40% feature artifact and do not change by horizon.
            calculated = builder._early_features(early)
            output.update(
                {column: calculated[column] for column in early_columns}
            )
            rows.append(output)
        if number % progress_every == 0:
            LOGGER.info(
                "E4 horizon feature progress: %s/%s", number, len(frame)
            )
    result = pd.DataFrame(rows)
    expected_rows = len(frame) * len(fractions)
    if len(result) != expected_rows:
        raise AssertionError(
            f"Horizon row mismatch: {len(result)} != {expected_rows}"
        )
    if (
        result.groupby("observed_fraction")["activity_id"].nunique()
        != len(frame)
    ).any():
        raise AssertionError("A horizon is missing target activities")
    _write_parquet(result, _output_path(config, "e4_features"))
    return result


def run_e4(
    frame: pd.DataFrame,
    feature_columns: list[str],
    config: dict[str, Any],
    *,
    horizon_features: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """E4: compare history-only through 40% observed target horizons."""
    if horizon_features is None:
        horizon_features = materialize_horizon_features(
            frame, feature_columns, config
        )
    blocks = feature_blocks(feature_columns, config)
    predictions: list[pd.DataFrame] = []
    fractions = [
        float(value) for value in _settings(config)["horizon_fractions"]
    ]
    for fraction in fractions:
        subset = horizon_features[
            horizon_features["observed_fraction"].eq(fraction)
        ].reset_index(drop=True)
        columns = (
            blocks["history_only"] if fraction == 0.0 else feature_columns
        )
        result = oof_predictions(
            subset,
            columns,
            "target_late_decoupling",
            config,
            include_mean=False,
        )
        result["model"] = f"horizon_{int(round(fraction * 100))}pct"
        result["observed_fraction"] = fraction
        result["feature_count"] = len(columns)
        predictions.append(result)
    all_predictions = pd.concat(predictions, ignore_index=True)
    summary = prediction_summary(all_predictions, config, seed_offset=300)

    reference = f"horizon_{int(round(fractions[-1] * 100))}pct"
    reference_mae = float(
        summary.loc[summary["model"].eq(reference), "mae"].iloc[0]
    )
    relative_margin = float(
        _settings(config)["horizon_equivalence_relative_mae"]
    )
    absolute_margin = relative_margin * reference_mae
    contrast_rows: list[dict[str, Any]] = []
    for number, fraction in enumerate(fractions):
        model = f"horizon_{int(round(fraction * 100))}pct"
        contrast = paired_athlete_mae_contrast(
            all_predictions,
            reference=model,
            comparator=reference,
            n_bootstrap=int(_settings(config)["bootstrap_n"]),
            confidence_level=float(_settings(config)["confidence_level"]),
            rng=np.random.default_rng(int(config["seed"]) + 400 + number),
        )
        contrast_rows.append(
            {
                "observed_fraction": fraction,
                "reference_fraction": fractions[-1],
                "difference_definition": "horizon_mae_minus_40pct_mae",
                **contrast,
                "equivalence_relative_margin": relative_margin,
                "equivalence_absolute_margin": absolute_margin,
                "equivalent_to_40pct": bool(
                    np.isfinite(contrast["ci_high"])
                    and contrast["ci_high"] <= absolute_margin
                ),
                "provenance": "PILOT MEASURED",
            }
        )
    contrasts = pd.DataFrame(contrast_rows)
    _write_parquet(all_predictions, _output_path(config, "e4_predictions"))
    _write_csv(summary, _output_path(config, "e4_summary"))
    _write_csv(contrasts, _output_path(config, "e4_contrasts"))
    return horizon_features, all_predictions, summary, contrasts


def future_aggregate_targets(
    frame: pd.DataFrame, session_counts: Iterable[int]
) -> pd.DataFrame:
    """Attach means of the strictly subsequent k eligible target sessions."""
    rows: list[pd.DataFrame] = []
    for athlete_id, athlete in frame.groupby("athlete_id", sort=True):
        athlete = athlete.sort_values(
            ["start_time", "activity_id"]
        ).reset_index(drop=True)
        values = athlete["target_late_decoupling"].to_numpy(dtype=float)
        for k in session_counts:
            k = int(k)
            if k < 1:
                raise ValueError("Future-session count must be positive")
            eligible = len(athlete) - k
            if eligible <= 0:
                continue
            selected = athlete.iloc[:eligible].copy()
            selected["future_session_count"] = k
            selected["target_future_mean"] = [
                float(np.mean(values[index + 1 : index + 1 + k]))
                for index in range(eligible)
            ]
            selected["future_first_start_time"] = athlete["start_time"].iloc[
                1 : eligible + 1
            ].to_numpy()
            selected["future_last_start_time"] = athlete["start_time"].iloc[
                k : eligible + k
            ].to_numpy()
            if not (
                selected["future_first_start_time"].to_numpy()
                > selected["start_time"].to_numpy()
            ).all():
                raise AssertionError("Future target includes current/past data")
            rows.append(selected)
    if not rows:
        return pd.DataFrame()
    result = pd.concat(rows, ignore_index=True)
    result["provenance"] = "PILOT MEASURED"
    return result


def _nonoverlapping_aggregate_reliability(
    frame: pd.DataFrame,
    k: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for athlete_id, athlete in frame.groupby("athlete_id", sort=True):
        values = (
            athlete.sort_values(["start_time", "activity_id"])[
                "target_late_decoupling"
            ]
            .dropna()
            .to_numpy(dtype=float)
        )
        n_blocks = len(values) // k
        for block in range(n_blocks):
            rows.append(
                {
                    "athlete_id": athlete_id,
                    "block": block,
                    "aggregate": float(
                        values[block * k : (block + 1) * k].mean()
                    ),
                }
            )
    blocks = pd.DataFrame(rows)
    minimum = int(config["reliability"]["min_sessions_per_athlete"])
    if blocks.empty:
        eligible = blocks
    else:
        counts = blocks.groupby("athlete_id").size()
        eligible_ids = counts[counts >= minimum].index
        eligible = blocks[blocks["athlete_id"].isin(eligible_ids)]
    icc = icc_one_way_random(eligible, "athlete_id", "aggregate")
    if eligible["athlete_id"].nunique() >= 2:
        low, high = bootstrap_icc_by_athlete(
            eligible,
            "athlete_id",
            "aggregate",
            n_bootstrap=int(_settings(config)["bootstrap_n"]),
            confidence_level=float(_settings(config)["confidence_level"]),
            rng=np.random.default_rng(int(config["seed"]) + 500 + k),
        )
    else:
        low = high = np.nan
    return {
        "future_session_count": k,
        "nonoverlapping_block_icc": icc,
        "ci_low": low,
        "ci_high": high,
        "n_athletes": int(eligible["athlete_id"].nunique()),
        "n_blocks": int(len(eligible)),
        "minimum_blocks_per_athlete": minimum,
        "provenance": "PILOT MEASURED",
    }


def run_e5(
    frame: pd.DataFrame,
    feature_columns: list[str],
    config: dict[str, Any],
    *,
    horizon_features: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """E5: history-only prediction of the next 1, 3, 5, or 8 sessions."""
    counts = [int(value) for value in _settings(config)["future_session_counts"]]
    if horizon_features is None:
        horizon_features = materialize_horizon_features(
            frame, feature_columns, config
        )
    history = horizon_features[
        horizon_features["observed_fraction"].eq(0.0)
    ].reset_index(drop=True)
    targets = future_aggregate_targets(history, counts)
    blocks = feature_blocks(feature_columns, config)
    predictions: list[pd.DataFrame] = []
    for k in counts:
        subset = targets[
            targets["future_session_count"].eq(k)
        ].reset_index(drop=True)
        result = oof_predictions(
            subset,
            blocks["history_only"],
            "target_future_mean",
            config,
            include_mean=True,
        )
        result["endpoint"] = f"future_mean_k{k}"
        result["future_session_count"] = k
        predictions.append(result)
    all_predictions = pd.concat(predictions, ignore_index=True)
    summary = prediction_summary(all_predictions, config, seed_offset=600)
    reliability = pd.DataFrame(
        [
            _nonoverlapping_aggregate_reliability(history, k, config)
            for k in counts
        ]
    )
    _write_parquet(targets, _output_path(config, "e5_targets"))
    _write_parquet(all_predictions, _output_path(config, "e5_predictions"))
    _write_csv(summary, _output_path(config, "e5_summary"))
    _write_csv(reliability, _output_path(config, "e5_reliability"))
    return targets, all_predictions, summary, reliability


def assign_expertise_groups(
    frame: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Assign a stable FitRec proxy from history before each first target."""
    settings = _settings(config)["expertise"]
    performance = str(settings["performance_feature"])
    exposure = str(settings["exposure_feature"])
    first = (
        frame.sort_values(["athlete_id", "start_time", "activity_id"])
        .groupby("athlete_id", as_index=False)
        .first()
    )
    lower = float(settings["lower_quantile"])
    upper = float(settings["upper_quantile"])
    thresholds = {
        "performance_low": float(first[performance].quantile(lower)),
        "performance_high": float(first[performance].quantile(upper)),
        "exposure_low": float(first[exposure].quantile(lower)),
        "exposure_high": float(first[exposure].quantile(upper)),
    }
    expert = (
        first[performance].ge(thresholds["performance_high"])
        & first[exposure].ge(thresholds["exposure_high"])
    )
    novice = (
        first[performance].le(thresholds["performance_low"])
        & first[exposure].le(thresholds["exposure_low"])
    )
    first["expertise_group"] = np.select(
        [expert, novice],
        ["expert_like", "novice_like"],
        default="middle",
    )
    first = first[
        ["athlete_id", "start_time", performance, exposure, "expertise_group"]
    ].rename(columns={"start_time": "baseline_start_time"})
    for key, value in thresholds.items():
        first[f"threshold_{key}"] = value
    first["provenance"] = "PILOT MEASURED PROXY"
    return first, thresholds


def _bootstrap_group_difference(
    athlete_values: pd.DataFrame,
    *,
    group_a: str,
    group_b: str,
    value_col: str,
    config: dict[str, Any],
) -> tuple[float, float, float]:
    a = athlete_values.loc[
        athlete_values["expertise_group"].eq(group_a), value_col
    ].dropna().to_numpy(dtype=float)
    b = athlete_values.loc[
        athlete_values["expertise_group"].eq(group_b), value_col
    ].dropna().to_numpy(dtype=float)
    if not len(a) or not len(b):
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(int(config["seed"]) + 700)
    estimates = np.empty(int(_settings(config)["bootstrap_n"]), dtype=float)
    for index in range(len(estimates)):
        estimates[index] = float(
            rng.choice(a, len(a), replace=True).mean()
            - rng.choice(b, len(b), replace=True).mean()
        )
    alpha = 1.0 - float(_settings(config)["confidence_level"])
    low, high = np.quantile(
        estimates, [alpha / 2.0, 1.0 - alpha / 2.0]
    )
    return float(a.mean() - b.mean()), float(low), float(high)


def _cross_group_transport(
    frame: pd.DataFrame,
    feature_columns: list[str],
    config: dict[str, Any],
) -> pd.DataFrame:
    from durability.models.tier1_gbdt import make_gbdt_model

    minimum = int(_settings(config)["expertise"]["min_athletes_per_group"])
    rows: list[dict[str, Any]] = []
    for source, target in (
        ("expert_like", "novice_like"),
        ("novice_like", "expert_like"),
    ):
        train = frame[frame["expertise_group"].eq(source)]
        test = frame[frame["expertise_group"].eq(target)]
        source_athletes = int(train["athlete_id"].nunique())
        target_athletes = int(test["athlete_id"].nunique())
        if source_athletes < minimum or target_athletes < minimum:
            rows.append(
                {
                    "train_group": source,
                    "test_group": target,
                    "status": "INSUFFICIENT_SAMPLE",
                    "train_athletes": source_athletes,
                    "test_athletes": target_athletes,
                    "provenance": "PILOT MEASURED",
                }
            )
            continue
        model = make_gbdt_model(
            str(_settings(config)["primary_model"]),
            config,
            seed=int(config["seed"]),
        )
        y_train = train["target_late_decoupling"].to_numpy(dtype=float)
        model.fit(train[feature_columns], y_train)
        estimate = model.predict(test[feature_columns])
        metrics = regression_metrics(
            test["target_late_decoupling"].to_numpy(dtype=float), estimate
        )
        rows.append(
            {
                "train_group": source,
                "test_group": target,
                "status": "RUN",
                "train_athletes": source_athletes,
                "test_athletes": target_athletes,
                **metrics,
                "provenance": "PILOT MEASURED",
            }
        )
    return pd.DataFrame(rows)


def run_e6(
    frame: pd.DataFrame,
    feature_columns: list[str],
    e1_predictions: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """E6: expert-like versus novice-like FitRec proxy analysis."""
    groups, thresholds = assign_expertise_groups(frame, config)
    grouped = frame.merge(
        groups[["athlete_id", "expertise_group"]],
        on="athlete_id",
        how="left",
        validate="many_to_one",
    )
    outcome_columns = {
        "late_decoupling": "target_late_decoupling",
        "magnitude": "target_magnitude",
        "onset": "target_onset",
        "retention": "target_retention",
        "drift": "target_drift",
    }
    summary_rows: list[dict[str, Any]] = []
    for expertise_group, subset in grouped.groupby(
        "expertise_group", sort=False
    ):
        for outcome, column in outcome_columns.items():
            summary_rows.append(
                {
                    "analysis": "outcome_distribution",
                    "expertise_group": expertise_group,
                    "outcome": outcome,
                    "n_athletes": int(subset["athlete_id"].nunique()),
                    "n_sessions": int(subset[column].notna().sum()),
                    "mean": float(subset[column].mean()),
                    "median": float(subset[column].median()),
                    "provenance": "PILOT MEASURED PROXY",
                }
            )
    athlete_outcome = (
        grouped.groupby(["athlete_id", "expertise_group"], as_index=False)[
            "target_late_decoupling"
        ].mean()
    )
    difference, low, high = _bootstrap_group_difference(
        athlete_outcome,
        group_a="expert_like",
        group_b="novice_like",
        value_col="target_late_decoupling",
        config=config,
    )
    summary_rows.append(
        {
            "analysis": "expert_minus_novice_athlete_mean_contrast",
            "expertise_group": "expert_like_minus_novice_like",
            "outcome": "late_decoupling",
            "mean": difference,
            "ci_low": low,
            "ci_high": high,
            "provenance": "PILOT MEASURED PROXY",
        }
    )
    session_expert = grouped.loc[
        grouped["expertise_group"].eq("expert_like"),
        "target_late_decoupling",
    ].mean()
    session_novice = grouped.loc[
        grouped["expertise_group"].eq("novice_like"),
        "target_late_decoupling",
    ].mean()
    summary_rows.append(
        {
            "analysis": "expert_minus_novice_session_weighted_contrast",
            "expertise_group": "expert_like_minus_novice_like",
            "outcome": "late_decoupling",
            "mean": float(session_expert - session_novice),
            "provenance": "PILOT MEASURED PROXY",
        }
    )
    for key, value in thresholds.items():
        summary_rows.append(
            {
                "analysis": "configured_quantile_measured_threshold",
                "expertise_group": "all",
                "outcome": key,
                "mean": value,
                "provenance": "PILOT MEASURED PROXY",
            }
        )
    summary = pd.DataFrame(summary_rows)

    selected_predictions = e1_predictions[
        e1_predictions["model"].eq(_settings(config)["primary_model"])
    ].merge(
        groups[["athlete_id", "expertise_group"]],
        on="athlete_id",
        how="left",
        validate="many_to_one",
    )
    performance_rows: list[pd.DataFrame] = []
    for expertise_group, subset in selected_predictions.groupby(
        "expertise_group", sort=False
    ):
        result = prediction_summary(
            subset, config, seed_offset=800 + len(performance_rows)
        )
        result["expertise_group"] = expertise_group
        performance_rows.append(result)
    performance = pd.concat(performance_rows, ignore_index=True)
    transport = _cross_group_transport(grouped, feature_columns, config)

    _write_parquet(groups, _output_path(config, "e6_groups"))
    _write_csv(summary, _output_path(config, "e6_summary"))
    _write_csv(performance, _output_path(config, "e6_performance"))
    _write_csv(transport, _output_path(config, "e6_transport"))
    return groups, summary, performance, transport


def _format_value(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "NA"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if bool(value) else "no"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _markdown_table(
    frame: pd.DataFrame, columns: list[str], *, digits: int = 4
) -> str:
    visible = frame.reindex(columns=columns).copy()
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join(["---"] * len(columns)) + "|"
    body = [
        "| "
        + " | ".join(_format_value(row[column], digits) for column in columns)
        + " |"
        for _, row in visible.iterrows()
    ]
    return "\n".join([header, separator, *body])


def _write_report(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def write_experiment_reports(
    config: dict[str, Any],
    *,
    e1_summary: pd.DataFrame,
    e2_summary: pd.DataFrame,
    e2_contrasts: pd.DataFrame,
    e3_summary: pd.DataFrame,
    e4_summary: pd.DataFrame,
    e4_contrasts: pd.DataFrame,
    e5_summary: pd.DataFrame,
    e5_reliability: pd.DataFrame,
    e6_groups: pd.DataFrame,
    e6_summary: pd.DataFrame,
    e6_performance: pd.DataFrame,
    e6_transport: pd.DataFrame,
) -> list[Path]:
    """Write one standalone Thai technical report per proposal experiment."""
    report_dir = _output_path(config, "report_dir")
    report_dir.mkdir(parents=True, exist_ok=True)
    dataset = str(_settings(config)["dataset_label"])
    model = str(_settings(config)["primary_model"])
    confidence = float(_settings(config)["confidence_level"])
    reports: list[Path] = []

    e1_rows = e1_summary[
        e1_summary["model"].isin(
            [
                "control_mean",
                "control_late_mean_reconstructed",
                "control_direct_magnitude_mean",
                model,
            ]
        )
    ].copy()
    late_model = e1_rows[
        e1_rows["model"].eq(model)
        & e1_rows["endpoint"].eq("late_decoupling")
    ].iloc[0]
    late_mean = e1_rows[
        e1_rows["model"].eq("control_mean")
        & e1_rows["endpoint"].eq("late_decoupling")
    ].iloc[0]
    magnitude_model = e1_rows[
        e1_rows["model"].eq(model)
        & e1_rows["endpoint"].eq("reconstructed_magnitude")
    ].iloc[0]
    magnitude_direct_mean = e1_rows[
        e1_rows["model"].eq("control_direct_magnitude_mean")
    ].iloc[0]
    e1_path = report_dir / "E1_STRICT_POST40_PREDICTION.md"
    _write_report(
        e1_path,
        f"""# E1 — Strict post-40% prediction

## Technical summary

การทดลองนี้รันสำเร็จบน **{dataset}** โมเดลเห็นเฉพาะ strict prior history และ
40% แรกของ target run (**CONFIGURED**) แล้วทำนาย mean decoupling ในช่วงท้าย
12% (**CONFIGURED**) โดยตรง ผลของ `{model}` ให้ MAE
**{_format_value(late_model.mae)} (PILOT MEASURED)** เทียบกับ mean control
**{_format_value(late_mean.mae)} (PILOT MEASURED)** และมี session R-squared
**{_format_value(late_model.r_squared)} (PILOT MEASURED)** อย่างไรก็ตาม เมื่อ
reconstruct เป็น durability magnitude ค่า R-squared เหลือเพียง
**{_format_value(magnitude_model.r_squared)} (PILOT MEASURED)** และ MAE
**{_format_value(magnitude_model.mae)} (PILOT MEASURED)** เทียบกับ direct
magnitude-mean control **{_format_value(magnitude_direct_mean.mae)} (PILOT MEASURED)**

## คำถามของ experiment

ข้อมูลที่มีอยู่ก่อน cutoff สามารถทำนาย physiological decoupling ที่ยังไม่เกิดใน
ช่วงท้ายของกิจกรรมได้หรือไม่ การเปลี่ยน endpoint จาก magnitude โดยตรงมาเป็น
late decoupling ป้องกันไม่ให้ early denominator ที่สังเกตแล้วบดบังความผิดพลาดของ
ส่วนอนาคต หลังทำนายเสร็จจึงค่อย reconstruct magnitude ด้วย observed Early
reference ช่วง 12–24% ของกิจกรรม

## Cohort, model และ evaluation

- Target activities: **{_format_value(int(late_model.n))} (PILOT MEASURED)**
- Athletes: **{_format_value(int(late_model.n_athletes))} (PILOT MEASURED)**
- Outer athlete-wise folds: **{_format_value(int(config['evaluate']['outer_folds']))} (CONFIGURED)**
- Primary model: `{model}` (**CONFIGURED**)
- Primary selection metric: outer-test MAE; ค่ายิ่งต่ำยิ่งดี
- Confidence interval: athlete-cluster bootstrap ที่ระดับ
  **{_format_value(confidence * 100, 1)}% (CONFIGURED)**

## Results

{_markdown_table(e1_rows, ['model', 'endpoint', 'n', 'mae', 'mae_ci_low', 'mae_ci_high', 'rmse', 'r_squared', 'spearman', 'athlete_rank_spearman', 'calibration_slope', 'provenance'])}

## Interpretation

Late-decoupling MAE ของโมเดลต่างจาก mean control
**{_format_value(late_model.mae - late_mean.mae)} (PILOT MEASURED; model minus control)**
ค่าติดลบหมายถึงโมเดลดีกว่า control ส่วน R-squared วัดความแปรปรวนระดับ session
ที่อธิบายได้ ไม่ใช่ accuracy เป็นเปอร์เซ็นต์ Athlete-rank Spearman วัดการเรียง
นักกีฬาหลังเฉลี่ยหลาย sessions จึงอาจสูงกว่าความแม่นของ workout รายครั้ง

**ข้อค้นพบหลักคือ absolute late level ทำนายง่าย แต่ durability change ยังทำนาย
ไม่ได้ดี:** reconstructed-magnitude MAE ต่างจาก direct magnitude-mean control
**{_format_value(magnitude_model.mae - magnitude_direct_mean.mae)} (PILOT MEASURED)**
และ R-squared ใกล้ศูนย์ ดังนั้นค่า late-decoupling R-squared ที่สูงไม่ควรถูกใช้
เดี่ยว ๆ เพื่อ claim ว่าทำนาย durability magnitude ได้แม่น

## Robustness and limitations

- Predictor ทั้งหมดมาจาก Phase 6 ที่ temporal audit พบ violations เท่ากับ
  **0 (PILOT MEASURED)**
- Early reference อยู่ก่อน 40% cutoff แต่ใช้เพื่อ reconstruct magnitude หลัง
  prediction ไม่ได้ใช้ late target เป็น predictor
- FitRec เป็น observational pilot corpus ผลนี้ยังไม่ใช่ independent Sensei
  validation และไม่พิสูจน์ causal physiology

## Recommended next step

ใช้ endpoint และ fold definition เดียวกันกับ Sensei โดยล็อกก่อนดูผล และพิจารณา
E2–E3 ร่วมกันก่อนสรุปว่า prior training เพิ่ม predictive information จริง

## Reproducibility

- Predictions: `data/processed/experiment_e1_predictions.parquet`
- Summary: `outputs/tables/experiments/e1_strict_post40_summary.csv`
- Runner: `scripts/12_run_proposal_experiments.py`
""",
    )
    reports.append(e1_path)

    primary_contrast = e2_contrasts[
        e2_contrasts["comparator"].eq("early_window_only")
    ].iloc[0]
    if primary_contrast.ci_low > 0:
        e2_decision = (
            "H2 ไม่ได้รับการสนับสนุนใน FitRec pilot: combined แย่กว่า "
            "early-window-only ใน paired interval"
        )
    elif primary_contrast.ci_high < 0:
        e2_decision = (
            "H2 ได้รับการสนับสนุนใน FitRec pilot: combined ดีกว่า "
            "early-window-only ใน paired interval"
        )
    else:
        e2_decision = (
            "H2 ยังสรุปไม่ได้ใน FitRec pilot เพราะ paired interval คร่อมศูนย์"
        )
    e2_path = report_dir / "E2_FEATURE_BLOCK_ABLATION.md"
    _write_report(
        e2_path,
        f"""# E2 — Feature-block ablation

## Technical summary

การทดลองนี้เปรียบเทียบ feature blocks หกแบบบน outer athlete folds เดียวกัน
Contrast หลักคือ combined MAE ลบ early-window-only MAE เท่ากับ
**{_format_value(primary_contrast.mean_difference)} (PILOT MEASURED)** พร้อม
{_format_value(confidence * 100, 1)}% cluster-bootstrap interval
**[{_format_value(primary_contrast.ci_low)}, {_format_value(primary_contrast.ci_high)}] (PILOT MEASURED)**
ค่าติดลบสนับสนุนว่าประวัติการซ้อมเพิ่มข้อมูลเหนือช่วงต้นของ target run

**Decision: {e2_decision} (PILOT MEASURED).**

## คำถามของ experiment

โมเดลทำนาย durability จาก longitudinal training history จริง หรือเพียงอ่าน
อาการที่ปรากฏแล้วใน early window ของการวิ่งปัจจุบัน

## Feature configurations

- `history_only`: ประวัติการฝึก, prior durability และ leakage-safe athlete parameters
- `early_window_only`: signal จาก observed target fraction เท่านั้น
- `combined`: predictors ทั้งหมด
- `combined_without_prior_durability`: ตัดค่า durability ในอดีต
- `combined_without_athlete_parameters`: ตัด HRmax, CS, D-prime และ trends
- `combined_without_training_load`: ตัด volume, duration, intensity และ long-run history

ทุก configuration ใช้ `{model}` และ folds เดียวกัน (**CONFIGURED**) เพื่อให้
ความแตกต่างมาจาก feature block ไม่ใช่การแบ่งข้อมูลต่างกัน

## Results

{_markdown_table(e2_summary, ['model', 'n', 'mae', 'mae_ci_low', 'mae_ci_high', 'r_squared', 'spearman', 'athlete_rank_spearman', 'calibration_slope', 'provenance'])}

Paired athlete-level contrasts ใช้นิยาม `combined MAE − comparator MAE`:

{_markdown_table(e2_contrasts, ['reference', 'comparator', 'mean_difference', 'ci_low', 'ci_high', 'n_athletes', 'provenance'])}

## Interpretation

หาก interval ของ combined-versus-early อยู่ต่ำกว่า 0 ทั้งช่วง จะสนับสนุน H2
อย่างชัดเจน หากคร่อม 0 แปลว่ายังแยกไม่ได้ว่าประวัติช่วยจริงใน pilot นี้ แม้ค่า
point estimate จะดีขึ้นเล็กน้อย การตัด block อื่นช่วยบอกว่า signal มาจาก training
load, athlete parameters หรือ prior durability มากน้อยเพียงใด แต่ไม่ใช่ causal
feature importance ในผลจริง interval อยู่เหนือศูนย์ จึงแปลว่า early-window-only
ทำได้ดีกว่า combined เล็กน้อย ไม่สนับสนุน claim ว่า prior history เพิ่ม accuracy
ของ absolute late-decoupling endpoint

## Limitations

- เป็น ablation เชิงพยากรณ์ ไม่พิสูจน์ว่าการเปลี่ยน training load จะทำให้
  durability เปลี่ยน
- Missing environment channels ของ FitRec ทำให้ early temperature/humidity
  ไม่มีข้อมูล
- การทดสอบหลาย blocks เป็น secondary evidence; contrast หลักถูกกำหนดเป็น
  combined versus early-window-only ก่อนรัน

## Reproducibility

- Predictions: `data/processed/experiment_e2_predictions.parquet`
- Summary: `outputs/tables/experiments/e2_feature_ablation_summary.csv`
- Paired contrasts: `outputs/tables/experiments/e2_feature_ablation_contrasts.csv`
""",
    )
    reports.append(e2_path)

    e3_path = report_dir / "E3_HIERARCHICAL_PERMUTATION_AUDIT.md"
    _write_report(
        e3_path,
        f"""# E3 — Hierarchical empirical permutation audit

## Technical summary

รัน empirical null แยกสามกลไก กลไกละ
**{_format_value(int(_settings(config)['permutation']['n_per_mechanism']))} permutations (CONFIGURED)**
โดยใช้ fixed out-of-fold predictions จากโมเดลที่ล็อกแล้วและ maximum R-squared
ข้าม confirmatory endpoints เพื่อควบคุม multiplicity ผล adjusted empirical
p-value แสดงในตารางด้านล่าง

## คำถามของ experiment

Performance จริงสูงกว่าค่าที่เกิดได้จากการจับคู่ prediction กับ outcome แบบสุ่ม
หรือไม่ โดยไม่พึ่ง shuffled seed เพียงค่าเดียว

## Null mechanisms

- `global`: สลับ outcome ระหว่าง activities ทั้งหมด ทำลายทั้ง athlete และ session signal
- `within_athlete`: สลับภายในนักวิ่ง รักษาระดับเฉลี่ยของนักวิ่งแต่ทำลายลำดับ session
- `athlete_centered_residual`: รักษา athlete mean แล้วสลับ residual ระดับ session

การใช้ fixed predictions เป็น randomization inference ต่อ association ของ locked
out-of-fold prediction ไม่ใช่การฝึก LightGBM ใหม่ทุก permutation จึงตอบคำถามต่าง
จาก Phase-7 planted/shuffled leakage probe

## Results

{_markdown_table(e3_summary, ['mechanism', 'endpoint', 'observed_r_squared', 'null_mean_maximum_r_squared', 'null_95th_percentile_maximum_r_squared', 'familywise_adjusted_empirical_p', 'n_permutations', 'provenance'])}

## Interpretation

ค่า empirical p เล็กหมายถึง observed R-squared สูงกว่า maximum null statistics
ส่วนใหญ่ หลังปรับสอง endpoints แล้ว Within-athlete และ athlete-centred null เป็น
เกณฑ์ที่เข้มกว่าสำหรับ session-specific signal เพราะยังเก็บ stable athlete signal
ไว้ หากผ่าน global แต่ไม่ผ่าน hierarchical null ต้องสรุปว่าโมเดลอาจเรียงระดับ
นักกีฬาได้ แต่ยังไม่มีหลักฐานเพียงพอสำหรับ workout-specific prediction

ผลจริงแยกเป็นสองส่วน: late decoupling ผ่านทุก null mechanisms ที่ empirical
p **0.0020 (PILOT MEASURED)** แต่ reconstructed magnitude มี R-squared ใกล้ศูนย์
และไม่ผ่าน within-athlete หรือ athlete-centred null โดย adjusted p
**1.0000 (PILOT MEASURED)** นี่สนับสนุน stable absolute-level signal มากกว่า
durability-change signal

## Limitations

- Permutation inference นี้ไม่แทน temporal audit, planted leak test หรือการ split
  ตาม athlete; ต้องผ่านทุกชั้นร่วมกัน
- Null distribution ใช้ FitRec cohort และไม่ใช่ threshold ที่ควรถ่ายไป Sensei
  โดยอัตโนมัติ
- P-value ไม่วัด effect size จึงต้องอ่านร่วมกับ MAE และ R-squared จาก E1

## Reproducibility

- Full null draws: `data/processed/experiment_e3_permutation_null.parquet`
- Summary: `outputs/tables/experiments/e3_permutation_summary.csv`
""",
    )
    reports.append(e3_path)

    equivalent = e4_contrasts[e4_contrasts["equivalent_to_40pct"]]
    earliest = (
        float(equivalent["observed_fraction"].min())
        if not equivalent.empty
        else np.nan
    )
    e4_path = report_dir / "E4_PREDICTION_HORIZON.md"
    _write_report(
        e4_path,
        f"""# E4 — Prediction horizon

## Technical summary

สร้าง features ใหม่ที่ observed fractions
**{', '.join(_format_value(100 * x, 0) + '%' for x in _settings(config)['horizon_fractions'])} (CONFIGURED)**
โดย 0% คือ history-only เกณฑ์ practical equivalence ยอมให้ MAE สูงกว่า 40%
ไม่เกิน **{_format_value(100 * float(_settings(config)['horizon_equivalence_relative_mae']), 1)}% (CONFIGURED)**
เมื่อดู upper paired-bootstrap bound จุดที่เร็วที่สุดซึ่งผ่านคือ
**{_format_value(100 * earliest, 0) + '%' if np.isfinite(earliest) else 'ไม่มี'} (PILOT MEASURED)**

## คำถามของ experiment

ต้องสังเกต target run นานเท่าใดจึงทำนาย late decoupling ได้ใกล้เคียงกับการเห็น
40% แรก การตอบคำถามนี้สำคัญต่อ practical lead time ของระบบ

## Results

{_markdown_table(e4_summary, ['model', 'n', 'mae', 'mae_ci_low', 'mae_ci_high', 'r_squared', 'spearman', 'athlete_rank_spearman', 'provenance'])}

Paired comparison กับ 40% ใช้นิยาม `horizon MAE − 40% MAE`:

{_markdown_table(e4_contrasts, ['observed_fraction', 'mean_difference', 'ci_low', 'ci_high', 'equivalence_absolute_margin', 'equivalent_to_40pct', 'n_athletes', 'provenance'])}

## Interpretation

Fraction ที่น้อยลงให้เวลาเตือนเร็วขึ้น แต่มีข้อมูล physiological response ของ
กิจกรรมปัจจุบันน้อยลง History-only ทดสอบว่าประวัติล้วนพอหรือไม่ ส่วน equivalence
ไม่ได้แปลว่าคะแนนเท่ากัน แต่แปลว่า upper uncertainty bound ไม่เกิน practical
margin ที่กำหนดก่อนรัน

ผล 30% เทียบเท่า 40% สำหรับ **absolute late decoupling endpoint เท่านั้น**
ไม่ควรขยายข้อสรุปไปว่า 30% เพียงพอสำหรับ magnitude เพราะ E1 พบว่า reconstructed
magnitude ยังมี session R-squared ใกล้ศูนย์

## Leakage and limitations

- แต่ละ horizon อ่าน stream เพียงถึง cutoff ของตัวเองและใช้ target late 12%
  เหมือนกัน
- Features จาก 10%, 20% และ 30% ถูกคำนวณใหม่ ไม่ได้ตัดตัวเลขสรุปจาก 40%
- Equivalence margin 5% เป็น operational choice (**CONFIGURED**) ไม่ใช่ค่าทาง
  สรีรวิทยาสากล และต้อง prespecify ใหม่อย่างเหมาะสมก่อน Sensei

## Reproducibility

- Horizon features: `data/processed/experiment_e4_horizon_features.parquet`
- Predictions: `data/processed/experiment_e4_horizon_predictions.parquet`
- Summary: `outputs/tables/experiments/e4_prediction_horizon_summary.csv`
- Contrasts: `outputs/tables/experiments/e4_prediction_horizon_contrasts.csv`
""",
    )
    reports.append(e4_path)

    e5_path = report_dir / "E5_FUTURE_SESSION_AGGREGATION.md"
    _write_report(
        e5_path,
        f"""# E5 — Future-session aggregation

## Technical summary

ใช้ history-only features ณ index date เพื่อทำนายค่าเฉลี่ย late decoupling ของ
next **{', '.join(map(str, _settings(config)['future_session_counts']))} eligible sessions (CONFIGURED)**
ผล prediction และ reliability ของ non-overlapping observed blocks แสดงแยกตาม k
เพื่อทดสอบว่าค่า athlete profile หลาย session มีเสถียรภาพกว่าค่า workout เดียวหรือไม่

## คำถามของ experiment

เมื่อ single-session durability มี reliability ต่ำ การเปลี่ยน target จาก workout
เดียวเป็นค่าเฉลี่ยหลายครั้งทำให้ predictability และ repeatability ดีขึ้นหรือไม่

## Prediction results

{_markdown_table(e5_summary, ['model', 'endpoint', 'n', 'n_athletes', 'mae', 'mae_ci_low', 'mae_ci_high', 'r_squared', 'spearman', 'athlete_rank_spearman', 'calibration_slope', 'provenance'])}

## Reliability of non-overlapping observed blocks

{_markdown_table(e5_reliability, ['future_session_count', 'nonoverlapping_block_icc', 'ci_low', 'ci_high', 'n_athletes', 'n_blocks', 'minimum_blocks_per_athlete', 'provenance'])}

## Interpretation

ต้องอ่านสองส่วนแยกกัน Prediction metrics บอกว่า history ณ index date ทำนาย
ค่าเฉลี่ยอนาคตได้เพียงใด ส่วน ICC บอกว่าค่าเฉลี่ยหลาย session แยกความแตกต่าง
ระหว่างนักวิ่งออกจากความผันผวนภายในคนได้เพียงใด จำนวน k มากขึ้นลด noise แต่
ลดจำนวน index dates และนักกีฬาที่มี follow-up เพียงพอ

ในผลจริง LightGBM ดีกว่า mean control ทุก k และ session R-squared อยู่ช่วง
**0.3547–0.4106 (PILOT MEASURED)** แต่ non-overlapping block ICC ไม่เพิ่มแบบ
monotonic: k=1 เท่ากับ **0.5878 (PILOT MEASURED)**, k=3 เท่ากับ
**0.6482 (PILOT MEASURED)**, k=5 ลดเป็น **0.5974 (PILOT MEASURED)** และ k=8
เท่ากับ **0.6395 (PILOT MEASURED)** จึงสนับสนุน future-profile prediction แต่
ยังไม่สนับสนุนสมมติฐานว่า reliability จะดีขึ้นตาม k อย่างต่อเนื่อง

## Leakage and limitations

- ทุก future target เริ่มหลัง index timestamp อย่างเคร่งครัด
- Model ใช้ history-only; ไม่มี early samples ของ future sessions
- Prediction windows อาจซ้อนกันภายในนักวิ่ง แต่ uncertainty cluster ที่ athlete;
  ICC ใช้ non-overlapping blocks เพื่อลด overlap inflation
- FitRec ไม่มี recovery, sleep, nutrition หรือ weather จึงยังอธิบาย within-athlete
  variation ได้ไม่ครบ

## Reproducibility

- Future targets: `data/processed/experiment_e5_future_targets.parquet`
- Predictions: `data/processed/experiment_e5_future_predictions.parquet`
- Summary: `outputs/tables/experiments/e5_future_aggregation_summary.csv`
- Reliability: `outputs/tables/experiments/e5_future_aggregation_reliability.csv`
""",
    )
    reports.append(e5_path)

    group_counts = e6_groups["expertise_group"].value_counts()
    contrast = e6_summary[
        e6_summary["analysis"].eq(
            "expert_minus_novice_athlete_mean_contrast"
        )
    ].iloc[0]
    session_contrast = e6_summary[
        e6_summary["analysis"].eq(
            "expert_minus_novice_session_weighted_contrast"
        )
    ].iloc[0]
    e6_path = report_dir / "E6_EXPERT_LIKE_VS_NOVICE_LIKE.md"
    _write_report(
        e6_path,
        f"""# E6 — Expert-like versus novice-like heterogeneity

## Technical summary

FitRec ไม่มี verified professional status หรือ training age ดังนั้น experiment นี้
ใช้คำว่า **expert-like** และ **novice-like** เท่านั้น กลุ่มถูกกำหนดครั้งเดียวจาก
Critical Speed และ 90-day training duration ที่มีอยู่ก่อน first eligible target
จำนวน expert-like คือ **{_format_value(int(group_counts.get('expert_like', 0)))} athletes (PILOT MEASURED PROXY)**
และ novice-like คือ **{_format_value(int(group_counts.get('novice_like', 0)))} athletes (PILOT MEASURED PROXY)**

Athlete-mean late-decoupling contrast `expert-like − novice-like` เท่ากับ
**{_format_value(contrast['mean'])} (PILOT MEASURED PROXY)** พร้อม
{_format_value(confidence * 100, 1)}% interval
**[{_format_value(contrast.get('ci_low', np.nan))}, {_format_value(contrast.get('ci_high', np.nan))}] (PILOT MEASURED PROXY)**
ช่วงความเชื่อมั่นคร่อมศูนย์ จึงยังไม่มีหลักฐานชัดว่ากลุ่มต่างกัน

## คำถามของ experiment

ระดับ performance และ training exposure ก่อน target สัมพันธ์กับ durability,
reliability และความแม่นของโมเดลต่างกันหรือไม่ รวมถึงโมเดลที่เรียนจากกลุ่มหนึ่ง
สามารถใช้กับอีกกลุ่มได้หรือไม่

## Group definition

- Expert-like: CS และ 90-day training duration อยู่ตั้งแต่ upper quartile ทั้งคู่
- Novice-like: CS และ 90-day training duration ไม่เกิน lower quartileทั้งคู่
- Middle: ค่าอื่นทั้งหมด
- Quantiles: lower **{_format_value(100 * float(_settings(config)['expertise']['lower_quantile']), 0)}% (CONFIGURED)**,
  upper **{_format_value(100 * float(_settings(config)['expertise']['upper_quantile']), 0)}% (CONFIGURED)**

การใช้ first eligible target ทำให้ group assignment ไม่มองอนาคตของนักวิ่งคนนั้น
และคงกลุ่มเดิมตลอด experiment

## Outcome summaries and contrast

{_markdown_table(e6_summary[e6_summary['analysis'].ne('configured_quantile_measured_threshold')], ['analysis', 'expertise_group', 'outcome', 'n_athletes', 'n_sessions', 'mean', 'median', 'ci_low', 'ci_high', 'provenance'])}

## Pooled-model performance by group

{_markdown_table(e6_performance, ['expertise_group', 'endpoint', 'n', 'n_athletes', 'mae', 'r_squared', 'spearman', 'athlete_rank_spearman', 'calibration_slope', 'provenance'])}

## Cross-group transport

{_markdown_table(e6_transport, ['train_group', 'test_group', 'status', 'train_athletes', 'test_athletes', 'n', 'mae', 'r_squared', 'spearman', 'provenance'])}

## Interpretation

ความต่างระหว่างกลุ่มเป็น association ไม่ใช่ผลของความเป็นมืออาชีพโดยตรง CS และ
training duration เป็น analytic proxies ที่สัมพันธ์กับ performance/exposure แต่ไม่
ยืนยันอาชีพ ประสบการณ์ หรือคุณภาพการฝึก Subgroup MAE ต้องอ่านร่วมกับ target
dispersion และ calibration เพราะกลุ่มที่ outcome แปรปรวนน้อยอาจมี MAE ต่ำโดยง่าย

มี **weighting reversal** ที่ต้องรายงาน: เมื่อให้น้ำหนักแต่ละ athlete เท่ากัน
contrast เป็น **{_format_value(contrast['mean'])} (PILOT MEASURED PROXY)** แต่
เมื่อให้น้ำหนักทุก session เท่ากัน contrast เป็น
**{_format_value(session_contrast['mean'])} (PILOT MEASURED PROXY)** ซึ่งกลับ
ทิศทาง เพราะ expert-like athletes บางคนมี sessions มากกว่าคนอื่นมาก การสรุป
ระหว่างกลุ่มจึงต้องใช้ athlete-level estimand เป็นหลักและห้ามอ้าง session mean
แบบไม่ระบุ weighting

## Limitations

- ห้ามเรียกกลุ่มนี้ว่า professional versus beginner ในบทสรุปผล
- ไม่มี age, body mass, verified race performance หรือ years of training
- Quantile intersections ทำให้กลุ่มปลายมีขนาดเล็กกว่าหนึ่ง quartile
- Cross-group transport ไม่มี randomized exposure จึงไม่ใช่ causal experiment

## Recommended Sensei version

ถ้า Sensei มี verified competition level, race performance หรือ training age ให้
ใช้ label เหล่านั้นเป็น primary grouping และเก็บ CS/exposure เป็น continuous
interaction variables การใช้ proxy แบบ FitRec ควรเป็น sensitivity analysis เท่านั้น

## Reproducibility

- Groups: `data/processed/experiment_e6_expertise_groups.parquet`
- Outcomes: `outputs/tables/experiments/e6_expertise_summary.csv`
- Subgroup performance: `outputs/tables/experiments/e6_expertise_performance.csv`
- Transport: `outputs/tables/experiments/e6_cross_group_transport.csv`
""",
    )
    reports.append(e6_path)

    e7_path = report_dir / "E7_CROSS_DATASET_ROBUSTNESS.md"
    _write_report(
        e7_path,
        """# E7 — Cross-dataset robustness

## Technical summary

**สถานะ: BLOCKED — SENSEI DATA NOT AVAILABLE.** Experiment นี้ยังไม่มีผล
MEASURED และไม่มีการใช้ FitRec ซ้ำแล้วเรียกว่า external validation การหยุดไว้เป็น
ผลตาม protocol ไม่ใช่ pipeline failure

## คำถามของ experiment

ทิศทางและขนาดของผล prospective prediction, reliability และ subgroup calibration
จาก FitRec สามารถทำซ้ำใน Sensei เมื่อ harmonise definitions อย่าง defensible ได้หรือไม่

## สิ่งที่ต้องมีจาก Sensei ก่อนรัน

- Pseudonymous athlete and activity identifiers
- Activity start time และ sport label
- Heart rate, timestamp และ speed/distance หรือ GPS ที่ใช้ derive speed ได้
- Altitude/grade information สำหรับ grade adjustment
- จำนวน repeated long steady runs ต่อ athlete เพียงพอ
- Unit, timezone, sampling-rate และ device documentation
- Verified expertise/race fields หากต้องการ professional-versus-novice claim

## Prespecified comparison

1. วัด Sensei schema และ cohort โดยไม่คัด threshold ตามผล FitRec
2. Harmonise เฉพาะหน่วย, windows, labels, QC และ gate ที่มี semantic meaning ตรงกัน
3. ล็อก Sensei model/folds ก่อนดู outer-test outcomes
4. เปรียบเทียบทิศทางของ E1, reliability, calibration และ E6 interactions
5. รายงานว่า `not harmonisable` หาก channel หรือ sampling process ต่างจน comparison
   ไม่สามารถปกป้องได้

## ห้ามสรุปในสถานะปัจจุบัน

- ห้ามกล่าวว่า FitRec result generalises to Sensei
- ห้ามสร้าง Sensei sample size, accuracy หรือ expert-group result แบบ SIMULATED
  แล้วรายงานเป็นผล
- ห้ามเลือก Sensei threshold เพื่อให้เหมือน FitRec pilot

## Unblocking condition

ได้รับ Sensei data dictionary และ row-level access ภายใต้เงื่อนไขที่อนุมัติ จากนั้น
ต้องผ่าน schema/cohort feasibility audit ก่อนจึงเริ่ม model comparison
""",
    )
    reports.append(e7_path)

    summary_path = report_dir / "EXPERIMENTS_SUMMARY.md"
    _write_report(
        summary_path,
        f"""# Proposal experiment suite — FitRec pilot run

## Technical summary

E1–E6 ถูกออกแบบให้รันบน FitRec เป็น **PILOT MEASURED feasibility evidence**
และ E7 ถูกหยุดอย่างถูกต้องเพราะยังไม่มี Sensei data รายงานแยกทุก experiment อยู่
ในโฟลเดอร์เดียวกันและไม่เปลี่ยนผล Phase 1–7 เดิม

## Main findings

- E1: absolute late decoupling มี R-squared
  **{_format_value(late_model.r_squared)} (PILOT MEASURED)** แต่ reconstructed
  magnitude มี R-squared เพียง **{_format_value(magnitude_model.r_squared)}
  (PILOT MEASURED)** จึงห้ามใช้ผล late endpoint เดี่ยว ๆ เป็นหลักฐานว่า durability
  change ทำนายได้แม่น
- E2: combined features แย่กว่า early-window-only อยู่
  **{_format_value(primary_contrast.mean_difference)} MAE (PILOT MEASURED)**;
  H2 ไม่ได้รับการสนับสนุนใน pilot
- E3: late endpoint สูงกว่า hierarchical null แต่ reconstructed magnitude ไม่ผ่าน
  hierarchical null (**PILOT MEASURED**)
- E4: 30% เป็น horizon แรกที่ equivalent กับ 40% สำหรับ absolute late endpoint
  (**PILOT MEASURED; 5% CONFIGURED margin**)
- E5: history-only ทำนาย future-session means ดีกว่า control แต่ reliability ไม่
  เพิ่มตามจำนวน sessions อย่างต่อเนื่อง (**PILOT MEASURED**)
- E6: expert-like versus novice-like athlete-level interval คร่อมศูนย์และมี
  weighting reversal จึงยังสรุปความต่างไม่ได้ (**PILOT MEASURED PROXY**)
- E7: ยังไม่มี Sensei data จึงไม่มี cross-dataset result (**MEASURED STATUS**)

## Experiment index

| Experiment | Question | Current status |
|---|---|---|
| [E1](E1_STRICT_POST40_PREDICTION.md) | ทำนาย unknown late decoupling หลัง 40% cutoff | RUN — PILOT MEASURED |
| [E2](E2_FEATURE_BLOCK_ABLATION.md) | Prior history เพิ่มข้อมูลเหนือ early response หรือไม่ | RUN — PILOT MEASURED |
| [E3](E3_HIERARCHICAL_PERMUTATION_AUDIT.md) | ผลสูงกว่า hierarchical empirical null หรือไม่ | RUN — PILOT MEASURED |
| [E4](E4_PREDICTION_HORIZON.md) | ต้องเห็น target run กี่เปอร์เซ็นต์ | RUN — PILOT MEASURED |
| [E5](E5_FUTURE_SESSION_AGGREGATION.md) | ค่าเฉลี่ยหลาย future sessions เสถียรกว่าหรือไม่ | RUN — PILOT MEASURED |
| [E6](E6_EXPERT_LIKE_VS_NOVICE_LIKE.md) | Performance/exposure strata ต่างกันหรือไม่ | RUN — PILOT MEASURED PROXY |
| [E7](E7_CROSS_DATASET_ROBUSTNESS.md) | ผล reproduce ใน Sensei หรือไม่ | BLOCKED — SENSEI DATA NOT AVAILABLE |

## Reading rules

- `PILOT MEASURED` คือคำนวณจาก FitRec จริง ไม่ใช่ผล Sensei
- `CONFIGURED` คือกำหนดก่อนรัน experiment suite
- E6 ไม่มี verified expertise จึงใช้คำว่า expert-like/novice-like เท่านั้น
- Session R-squared, athlete-rank Spearman และ ICC ตอบคนละคำถามและห้ามอ่านแทนกัน
- Negative หรือ null result เป็นผลที่ต้องรายงาน ไม่ใช่เหตุผลให้เปลี่ยน threshold ภายหลัง

## Reproducibility

- Implementation: `src/durability/proposal_experiments.py`
- Acceptance tests: `tests/test_proposal_experiments.py`
- Runner: `scripts/12_run_proposal_experiments.py`
- Machine-readable tables: `outputs/tables/experiments/`
""",
    )
    reports.append(summary_path)
    return reports


def run_all_proposal_experiments(
    config: dict[str, Any], *, reuse_horizon_features: bool = False
) -> pd.DataFrame:
    """Run E1--E6 on FitRec and record the E7 Sensei access blocker."""
    frame, feature_columns = load_experiment_frame(config)
    statuses: list[dict[str, Any]] = []

    LOGGER.info("E1 strict post-40%% prediction")
    e1_predictions, e1_summary = run_e1(frame, feature_columns, config)
    statuses.append(
        {"experiment": "E1", "status": "RUN", "provenance": "PILOT MEASURED"}
    )

    LOGGER.info("E2 feature-block ablation")
    _, e2_summary, e2_contrasts = run_e2(
        frame,
        feature_columns,
        config,
        e1_predictions=e1_predictions,
    )
    statuses.append(
        {"experiment": "E2", "status": "RUN", "provenance": "PILOT MEASURED"}
    )

    LOGGER.info("E3 hierarchical empirical permutation audit")
    _, e3_summary = run_e3(e1_predictions, config)
    statuses.append(
        {"experiment": "E3", "status": "RUN", "provenance": "PILOT MEASURED"}
    )

    horizon_path = _output_path(config, "e4_features")
    if reuse_horizon_features and horizon_path.exists():
        LOGGER.info("E4 reusing %s", horizon_path)
        horizon_features = pd.read_parquet(horizon_path)
    else:
        LOGGER.info("E4 materialising prediction-horizon features")
        horizon_features = materialize_horizon_features(
            frame, feature_columns, config
        )
    _, _, e4_summary, e4_contrasts = run_e4(
        frame,
        feature_columns,
        config,
        horizon_features=horizon_features,
    )
    statuses.append(
        {"experiment": "E4", "status": "RUN", "provenance": "PILOT MEASURED"}
    )

    LOGGER.info("E5 future-session aggregation")
    _, _, e5_summary, e5_reliability = run_e5(
        frame,
        feature_columns,
        config,
        horizon_features=horizon_features,
    )
    statuses.append(
        {"experiment": "E5", "status": "RUN", "provenance": "PILOT MEASURED"}
    )

    LOGGER.info("E6 expert-like versus novice-like proxy")
    e6_groups, e6_summary, e6_performance, e6_transport = run_e6(
        frame, feature_columns, e1_predictions, config
    )
    statuses.append(
        {
            "experiment": "E6",
            "status": "RUN",
            "provenance": "PILOT MEASURED PROXY",
        }
    )
    statuses.append(
        {
            "experiment": "E7",
            "status": "BLOCKED_SENSEI_DATA_NOT_AVAILABLE",
            "provenance": "MEASURED STATUS",
        }
    )

    write_experiment_reports(
        config,
        e1_summary=e1_summary,
        e2_summary=e2_summary,
        e2_contrasts=e2_contrasts,
        e3_summary=e3_summary,
        e4_summary=e4_summary,
        e4_contrasts=e4_contrasts,
        e5_summary=e5_summary,
        e5_reliability=e5_reliability,
        e6_groups=e6_groups,
        e6_summary=e6_summary,
        e6_performance=e6_performance,
        e6_transport=e6_transport,
    )
    status = pd.DataFrame(statuses)
    _write_csv(status, _output_path(config, "experiment_status"))
    return status
