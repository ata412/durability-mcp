"""Athlete-wise nested evaluation and mandatory leakage controls."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

from durability.config import repo_path

LOGGER = logging.getLogger(__name__)

def make_group_folds(
    frame: pd.DataFrame, *, group_col: str, n_splits: int
) -> pd.DataFrame:
    """Assign every athlete wholly to one deterministic outer fold."""
    if frame[group_col].nunique() < n_splits:
        raise ValueError("Fewer groups than requested folds")
    splitter = GroupKFold(n_splits=n_splits)
    retained_columns = [
        column
        for column in ("activity_id", group_col)
        if column in frame.columns
    ]
    assignments = frame[retained_columns].copy()
    assignments["outer_fold"] = np.full(len(frame), -1, dtype=int)
    placeholder = np.zeros(len(frame))
    for fold, (_, test_indices) in enumerate(
        splitter.split(placeholder, groups=frame[group_col])
    ):
        assignments.iloc[
            test_indices, assignments.columns.get_loc("outer_fold")
        ] = fold
    assert (assignments["outer_fold"] >= 0).all()
    assert (
        assignments.groupby(group_col)["outer_fold"]
        .nunique()
        .max()
        == 1
    )
    return assignments


def assert_group_split_integrity(
    frame: pd.DataFrame,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    *,
    group_col: str,
) -> None:
    """Assert athlete disjointness; this makes within-athlete leakage impossible."""
    train_groups = set(frame.iloc[train_indices][group_col].astype(str))
    validation_groups = set(
        frame.iloc[validation_indices][group_col].astype(str)
    )
    assert train_groups.isdisjoint(validation_groups), (
        "An athlete appears in both training and validation"
    )


def inner_group_splits(
    frame: pd.DataFrame,
    train_indices: np.ndarray,
    *,
    group_col: str,
    n_splits: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return absolute row indices for inner athlete-wise folds."""
    outer_train = frame.iloc[train_indices]
    splitter = GroupKFold(n_splits=n_splits)
    local_x = np.zeros(len(outer_train))
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for local_train, local_validation in splitter.split(
        local_x, groups=outer_train[group_col]
    ):
        absolute_train = train_indices[local_train]
        absolute_validation = train_indices[local_validation]
        assert_group_split_integrity(
            frame,
            absolute_train,
            absolute_validation,
            group_col=group_col,
        )
        splits.append((absolute_train, absolute_validation))
    return splits


def shuffle_targets_within_athlete(
    frame: pd.DataFrame,
    target_columns: Iterable[str],
    *,
    seed: int,
) -> pd.DataFrame:
    """Permute each target independently within athletes, preserving missingness."""
    shuffled = frame.copy()
    rng = np.random.default_rng(seed)
    for _, indices in frame.groupby("athlete_id", sort=True).groups.items():
        row_indices = np.asarray(list(indices))
        for column in target_columns:
            values = frame.loc[row_indices, column].to_numpy(copy=True)
            shuffled.loc[row_indices, column] = values[
                rng.permutation(len(values))
            ]
    return shuffled


def regression_metrics(
    observed: np.ndarray, predicted: np.ndarray
) -> dict[str, float]:
    """Compute session-level regression metrics on finite pairs."""
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    finite = np.isfinite(observed) & np.isfinite(predicted)
    if finite.sum() < 2:
        return {
            "mae": np.nan,
            "rmse": np.nan,
            "r_squared": np.nan,
            "spearman": np.nan,
            "n": int(finite.sum()),
        }
    truth = observed[finite]
    estimate = predicted[finite]
    correlation = (
        spearmanr(truth, estimate).statistic
        if np.unique(truth).size > 1 and np.unique(estimate).size > 1
        else np.nan
    )
    return {
        "mae": float(mean_absolute_error(truth, estimate)),
        "rmse": float(np.sqrt(mean_squared_error(truth, estimate))),
        "r_squared": float(r2_score(truth, estimate)),
        "spearman": float(correlation),
        "n": int(len(truth)),
    }


def athlete_rank_spearman(
    frame: pd.DataFrame,
    observed_col: str,
    predicted_col: str,
) -> float:
    """Spearman correlation after averaging labels within each athlete."""
    clean = frame[
        ["athlete_id", observed_col, predicted_col]
    ].dropna()
    athlete = clean.groupby("athlete_id")[
        [observed_col, predicted_col]
    ].mean()
    if len(athlete) < 2:
        return np.nan
    if (
        athlete[observed_col].nunique() < 2
        or athlete[predicted_col].nunique() < 2
    ):
        return np.nan
    return float(
        spearmanr(
            athlete[observed_col], athlete[predicted_col]
        ).statistic
    )


def enforce_leakage_probe(
    probe: pd.DataFrame, *, maximum_r_squared: float
) -> None:
    """Halt whenever a shuffled-target model retains suspicious performance."""
    failed = probe[
        probe["r_squared"].notna()
        & (probe["r_squared"] > maximum_r_squared)
    ]
    if not failed.empty:
        details = failed[
            ["model", "target", "r_squared"]
        ].to_dict(orient="records")
        raise RuntimeError(f"Leakage probe failed: {details}")


def calibration_table(
    predictions: pd.DataFrame, *, n_bins: int
) -> pd.DataFrame:
    """Quantile-bin observed against predicted values by model and target."""
    rows: list[dict[str, Any]] = []
    for (model, target), group in predictions.groupby(
        ["model", "target"], sort=False
    ):
        clean = group[["observed", "predicted"]].dropna().copy()
        if len(clean) < n_bins:
            continue
        clean["bin"] = pd.qcut(
            clean["predicted"],
            q=n_bins,
            labels=False,
            duplicates="drop",
        )
        for bin_index, values in clean.groupby("bin"):
            rows.append(
                {
                    "model": model,
                    "target": target,
                    "bin": int(bin_index),
                    "n": len(values),
                    "mean_predicted": float(values["predicted"].mean()),
                    "mean_observed": float(values["observed"].mean()),
                    "provenance": "MEASURED",
                }
            )
    return pd.DataFrame(rows)


def _prediction_rows(
    frame: pd.DataFrame,
    indices: np.ndarray,
    prediction: np.ndarray,
    *,
    model: str,
    tier: str,
    target: str,
    fold: int,
    observed_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    source = frame if observed_frame is None else observed_frame
    return pd.DataFrame(
        {
            "activity_id": frame.iloc[indices]["activity_id"].astype(str).to_numpy(),
            "athlete_id": frame.iloc[indices]["athlete_id"].astype(str).to_numpy(),
            "outer_fold": fold,
            "model": model,
            "tier": tier,
            "target": target,
            "observed": source.iloc[indices][f"target_{target}"].to_numpy(
                dtype=float
            ),
            "predicted": np.asarray(prediction, dtype=float),
            "provenance": "MEASURED",
        }
    )


def _fit_scalar_prediction(
    train_score: np.ndarray,
    train_target: np.ndarray,
    test_score: np.ndarray,
) -> np.ndarray:
    finite = np.isfinite(train_target)
    if finite.sum() < 2:
        return np.full(len(test_score), np.nan)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    train_x = imputer.fit_transform(
        np.asarray(train_score, dtype=float).reshape(-1, 1)
    )
    test_x = imputer.transform(
        np.asarray(test_score, dtype=float).reshape(-1, 1)
    )
    model = LinearRegression()
    model.fit(train_x[finite], train_target[finite])
    return model.predict(test_x)


def _fold_arrays(
    assignments: pd.DataFrame, fold: int
) -> tuple[np.ndarray, np.ndarray]:
    test = np.flatnonzero(assignments["outer_fold"].to_numpy() == fold)
    train = np.flatnonzero(assignments["outer_fold"].to_numpy() != fold)
    return train, test


def _tabular_predictions(
    frame: pd.DataFrame,
    assignments: pd.DataFrame,
    feature_columns: list[str],
    config: dict[str, Any],
    *,
    observed_frame: pd.DataFrame | None = None,
    include_inner: bool = True,
) -> tuple[list[pd.DataFrame], list[dict[str, Any]]]:
    """Run mean, formula, and Tier-1 models on shared athlete folds."""
    from durability.models.tier0_formulas import formula_scores
    from durability.models.tier1_gbdt import make_gbdt_model

    outcomes = frame if observed_frame is None else observed_frame
    predictions: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    targets = list(config["evaluate"]["targets"])
    formulas = formula_scores(frame, config)
    outer_folds = int(config["evaluate"]["outer_folds"])
    group_col = config["evaluate"]["group_col"]
    tabular = frame[feature_columns]

    for fold in range(outer_folds):
        train, test = _fold_arrays(assignments, fold)
        assert_group_split_integrity(
            frame, train, test, group_col=group_col
        )
        for target in targets:
            target_column = f"target_{target}"
            train_target = outcomes.iloc[train][target_column].to_numpy(
                dtype=float
            )
            test_target = outcomes.iloc[test][target_column].to_numpy(
                dtype=float
            )
            finite_train = np.isfinite(train_target)
            train_mean = (
                float(train_target[finite_train].mean())
                if finite_train.any()
                else np.nan
            )
            predictions.append(
                _prediction_rows(
                    frame,
                    test,
                    np.full(len(test), train_mean),
                    model="control_mean",
                    tier="control",
                    target=target,
                    fold=fold,
                    observed_frame=outcomes,
                )
            )
            for model_name in formulas:
                estimate = _fit_scalar_prediction(
                    formulas.iloc[train][model_name].to_numpy(dtype=float),
                    train_target,
                    formulas.iloc[test][model_name].to_numpy(dtype=float),
                )
                predictions.append(
                    _prediction_rows(
                        frame,
                        test,
                        estimate,
                        model=model_name,
                        tier=(
                            "control"
                            if model_name.startswith("control_")
                            else "tier0"
                        ),
                        target=target,
                        fold=fold,
                        observed_frame=outcomes,
                    )
                )

            for model_name in ("tier1_xgboost", "tier1_lightgbm"):
                if include_inner:
                    inner_splits = inner_group_splits(
                        frame,
                        train,
                        group_col=group_col,
                        n_splits=int(config["evaluate"]["inner_folds"]),
                    )
                    for inner_fold, (
                        inner_train,
                        inner_validation,
                    ) in enumerate(inner_splits):
                        inner_y = outcomes.iloc[inner_train][
                            target_column
                        ].to_numpy(dtype=float)
                        inner_valid_y = outcomes.iloc[inner_validation][
                            target_column
                        ].to_numpy(dtype=float)
                        valid_train = np.isfinite(inner_y)
                        model = make_gbdt_model(
                            model_name,
                            config,
                            seed=int(config["seed"]) + fold,
                        )
                        model.fit(
                            tabular.iloc[inner_train].loc[valid_train],
                            inner_y[valid_train],
                        )
                        inner_prediction = model.predict(
                            tabular.iloc[inner_validation]
                        )
                        inner_metrics = regression_metrics(
                            inner_valid_y, inner_prediction
                        )
                        fold_rows.append(
                            {
                                "model": model_name,
                                "tier": "tier1",
                                "target": target,
                                "outer_fold": fold,
                                "inner_fold": inner_fold,
                                "split": "inner_validation",
                                **inner_metrics,
                                "provenance": "MEASURED",
                            }
                        )
                model = make_gbdt_model(
                    model_name,
                    config,
                    seed=int(config["seed"]) + fold,
                )
                model.fit(
                    tabular.iloc[train].loc[finite_train],
                    train_target[finite_train],
                )
                estimate = model.predict(tabular.iloc[test])
                predictions.append(
                    _prediction_rows(
                        frame,
                        test,
                        estimate,
                        model=model_name,
                        tier="tier1",
                        target=target,
                        fold=fold,
                        observed_frame=outcomes,
                    )
                )
                metrics = regression_metrics(test_target, estimate)
                fold_rows.append(
                    {
                        "model": model_name,
                        "tier": "tier1",
                        "target": target,
                        "outer_fold": fold,
                        "inner_fold": np.nan,
                        "split": "outer_test",
                        **metrics,
                        "provenance": "MEASURED",
                    }
                )
    return predictions, fold_rows


def _deep_predictions(
    frame: pd.DataFrame,
    assignments: pd.DataFrame,
    sequences: np.ndarray,
    config: dict[str, Any],
    *,
    observed_frame: pd.DataFrame | None = None,
    include_inner: bool = True,
) -> tuple[list[pd.DataFrame], list[dict[str, Any]]]:
    """Run multi-task Tier-2 models on the same outer and inner folds."""
    from durability.models.tier2_deep import fit_deep_model

    outcomes = frame if observed_frame is None else observed_frame
    target_names = list(config["evaluate"]["targets"])
    target_columns = [f"target_{target}" for target in target_names]
    targets = outcomes[target_columns].to_numpy(dtype=float)
    predictions: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    group_col = config["evaluate"]["group_col"]
    for fold in range(int(config["evaluate"]["outer_folds"])):
        train, test = _fold_arrays(assignments, fold)
        assert_group_split_integrity(
            frame, train, test, group_col=group_col
        )
        for model_offset, model_name in enumerate(
            (
                "tier2_tcn",
                "tier2_lstm",
                "tier2_patchtst",
                "tier2_itransformer",
            )
        ):
            if include_inner:
                for inner_fold, (
                    inner_train,
                    inner_validation,
                ) in enumerate(
                    inner_group_splits(
                        frame,
                        train,
                        group_col=group_col,
                        n_splits=int(config["evaluate"]["inner_folds"]),
                    )
                ):
                    inner_prediction, _, _, _ = fit_deep_model(
                        model_name,
                        sequences[inner_train],
                        targets[inner_train],
                        sequences[inner_validation],
                        config,
                        seed=(
                            int(config["seed"])
                            + fold * 100
                            + inner_fold * 10
                            + model_offset
                        ),
                        epochs=int(
                            config["models"]["tier2"]["inner_epochs"]
                        ),
                    )
                    for target_index, target in enumerate(target_names):
                        metrics = regression_metrics(
                            targets[inner_validation, target_index],
                            inner_prediction[:, target_index],
                        )
                        fold_rows.append(
                            {
                                "model": model_name,
                                "tier": "tier2",
                                "target": target,
                                "outer_fold": fold,
                                "inner_fold": inner_fold,
                                "split": "inner_validation",
                                **metrics,
                                "provenance": "MEASURED",
                            }
                        )
            estimate, _, _, _ = fit_deep_model(
                model_name,
                sequences[train],
                targets[train],
                sequences[test],
                config,
                seed=int(config["seed"]) + fold * 100 + model_offset,
            )
            for target_index, target in enumerate(target_names):
                predictions.append(
                    _prediction_rows(
                        frame,
                        test,
                        estimate[:, target_index],
                        model=model_name,
                        tier="tier2",
                        target=target,
                        fold=fold,
                        observed_frame=outcomes,
                    )
                )
                metrics = regression_metrics(
                    targets[test, target_index],
                    estimate[:, target_index],
                )
                fold_rows.append(
                    {
                        "model": model_name,
                        "tier": "tier2",
                        "target": target,
                        "outer_fold": fold,
                        "inner_fold": np.nan,
                        "split": "outer_test",
                        **metrics,
                        "provenance": "MEASURED",
                    }
                )
    return predictions, fold_rows


def _foundation_predictions(
    frame: pd.DataFrame,
    assignments: pd.DataFrame,
    forecasts: pd.DataFrame,
    config: dict[str, Any],
    *,
    observed_frame: pd.DataFrame | None = None,
) -> list[pd.DataFrame]:
    """Evaluate zero-shot forecasts and fold-local few-shot calibration."""
    outcomes = frame if observed_frame is None else observed_frame
    outputs: list[pd.DataFrame] = []
    if forecasts.empty:
        return outputs
    for (base_model, target, history_cap), forecast in forecasts.groupby(
        ["model", "target", "history_cap"], sort=False
    ):
        aligned = frame[["activity_id"]].merge(
            forecast[["activity_id", "predicted"]],
            on="activity_id",
            how="left",
            validate="one_to_one",
        )
        raw_prediction = aligned["predicted"].to_numpy(dtype=float)
        model_name = f"{base_model}_k{int(history_cap)}"
        for fold in range(int(config["evaluate"]["outer_folds"])):
            train, test = _fold_arrays(assignments, fold)
            outputs.append(
                _prediction_rows(
                    frame,
                    test,
                    raw_prediction[test],
                    model=model_name,
                    tier="tier3",
                    target=target,
                    fold=fold,
                    observed_frame=outcomes,
                )
            )
            calibrated = _fit_scalar_prediction(
                raw_prediction[train],
                outcomes.iloc[train][f"target_{target}"].to_numpy(
                    dtype=float
                ),
                raw_prediction[test],
            )
            outputs.append(
                _prediction_rows(
                    frame,
                    test,
                    calibrated,
                    model=model_name.replace(
                        "_zeroshot_", "_fewshot_"
                    ),
                    tier="tier3",
                    target=target,
                    fold=fold,
                    observed_frame=outcomes,
                )
            )
    return outputs


def summarise_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate OOF metrics and compare every model against the mean control."""
    rows: list[dict[str, Any]] = []
    for (model, tier, target), group in predictions.groupby(
        ["model", "tier", "target"], sort=False
    ):
        metrics = regression_metrics(group["observed"], group["predicted"])
        rank = athlete_rank_spearman(group, "observed", "predicted")
        fold_mae = [
            regression_metrics(values["observed"], values["predicted"])["mae"]
            for _, values in group.groupby("outer_fold")
        ]
        rows.append(
            {
                "model": model,
                "tier": tier,
                "target": target,
                **metrics,
                "athlete_rank_spearman": rank,
                "fold_mae_sd": float(np.nanstd(fold_mae, ddof=1)),
                "provenance": "MEASURED",
            }
        )
    result = pd.DataFrame(rows)
    mean_mae = (
        result[result["model"] == "control_mean"]
        .set_index("target")["mae"]
        .to_dict()
    )
    result["mean_control_mae"] = result["target"].map(mean_mae)
    result["beats_mean_mae"] = result["mae"] < result["mean_control_mae"]
    return result


def _fold_metrics_from_predictions(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model, tier, target, fold), group in predictions.groupby(
        ["model", "tier", "target", "outer_fold"], sort=False
    ):
        rows.append(
            {
                "model": model,
                "tier": tier,
                "target": target,
                "outer_fold": fold,
                "inner_fold": np.nan,
                "split": "outer_test",
                **regression_metrics(
                    group["observed"], group["predicted"]
                ),
                "provenance": "MEASURED",
            }
        )
    return pd.DataFrame(rows)


def _compute_shap_importance(
    frame: pd.DataFrame,
    feature_columns: list[str],
    sequences: np.ndarray,
    benchmark: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit full-data explanation models for best Tier 1 and Tier 2 variants."""
    import joblib
    import shap

    from durability.models.tier1_gbdt import make_gbdt_model
    from durability.models.tier2_deep import fit_deep_model

    rows: list[dict[str, Any]] = []
    statuses: list[dict[str, Any]] = []
    target_names = list(config["evaluate"]["targets"])
    magnitude = benchmark[benchmark["target"] == "magnitude"]
    models_dir = repo_path(config, config["outputs"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    sample_n = min(
        int(config["evaluate"]["shap_sample_rows"]), len(frame)
    )
    sample = frame.sample(
        n=sample_n, random_state=int(config["seed"])
    )
    sample_indices = sample.index.to_numpy()

    tier1 = magnitude[magnitude["tier"] == "tier1"].sort_values("mae")
    if not tier1.empty:
        model_name = str(tier1.iloc[0]["model"])
        target_column = "target_magnitude"
        finite = frame[target_column].notna().to_numpy()
        try:
            model = make_gbdt_model(
                model_name, config, seed=int(config["seed"])
            )
            model.fit(
                frame.loc[finite, feature_columns],
                frame.loc[finite, target_column],
            )
            transformed = model.named_steps["imputer"].transform(
                sample[feature_columns]
            )
            explainer = shap.TreeExplainer(
                model.named_steps["regressor"]
            )
            values = np.asarray(explainer.shap_values(transformed))
            importance = np.mean(np.abs(values), axis=0)
            for column, value in zip(
                feature_columns, importance, strict=True
            ):
                rows.append(
                    {
                        "model": model_name,
                        "tier": "tier1",
                        "target": "magnitude",
                        "feature": column,
                        "mean_abs_shap": float(value),
                        "provenance": "MEASURED",
                    }
                )
            joblib.dump(model, models_dir / f"{model_name}_magnitude.joblib")
            statuses.append(
                {
                    "tier": "tier1",
                    "model": f"{model_name}_shap",
                    "status": "RUN",
                    "reason": "",
                    "provenance": "MEASURED",
                }
            )
        except Exception as exc:
            statuses.append(
                {
                    "tier": "tier1",
                    "model": f"{model_name}_shap",
                    "status": "SKIPPED",
                    "reason": f"{type(exc).__name__}:{exc}",
                    "provenance": "MEASURED",
                }
            )

    tier2 = magnitude[magnitude["tier"] == "tier2"].sort_values("mae")
    if not tier2.empty:
        model_name = str(tier2.iloc[0]["model"])
        target_matrix = frame[
            [f"target_{target}" for target in target_names]
        ].to_numpy(dtype=float)
        try:
            _, model, _, _ = fit_deep_model(
                model_name,
                sequences,
                target_matrix,
                sequences[sample_indices],
                config,
                seed=int(config["seed"]),
            )
            background_indices = sample_indices[
                : min(50, len(sample_indices))
            ]
            explainer = shap.GradientExplainer(
                model,
                __import__("torch").from_numpy(
                    np.asarray(
                        sequences[background_indices], dtype=np.float32
                    )
                ),
            )
            values = np.asarray(
                explainer.shap_values(
                    __import__("torch").from_numpy(
                        np.asarray(
                            sequences[
                                sample_indices[: min(100, sample_n)]
                            ],
                            dtype=np.float32,
                        )
                    )
                )
            )
            # Current SHAP returns [samples, time, channels, outputs].
            if values.ndim == 4:
                magnitude_values = values[..., 0]
                channel_importance = np.mean(
                    np.abs(magnitude_values), axis=(0, 1)
                )
            else:
                raise ValueError(
                    f"Unexpected deep SHAP shape {values.shape}"
                )
            channel_names = config["models"]["tier2"]["channels"]
            for channel, value in zip(
                channel_names, channel_importance, strict=True
            ):
                rows.append(
                    {
                        "model": model_name,
                        "tier": "tier2",
                        "target": "magnitude",
                        "feature": channel,
                        "mean_abs_shap": float(value),
                        "provenance": "MEASURED",
                    }
                )
            import torch

            torch.save(
                model.state_dict(),
                models_dir / f"{model_name}_multitask.pt",
            )
            statuses.append(
                {
                    "tier": "tier2",
                    "model": f"{model_name}_shap",
                    "status": "RUN",
                    "reason": "",
                    "provenance": "MEASURED",
                }
            )
        except Exception as exc:
            statuses.append(
                {
                    "tier": "tier2",
                    "model": f"{model_name}_shap",
                    "status": "SKIPPED",
                    "reason": f"{type(exc).__name__}:{exc}",
                    "provenance": "MEASURED",
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(statuses)


def run_benchmark(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run Phase 7 end to end and halt on a failed leakage probe."""
    from durability.models.tier2_deep import (
        materialize_sequence_features,
        sequence_array,
    )
    from durability.models.tier3_foundation import (
        run_foundation_forecasts,
    )

    frame = pd.read_parquet(
        repo_path(config, config["data"]["features_output"])
    ).reset_index(drop=True)
    frame["activity_id"] = frame["activity_id"].astype(str)
    frame["athlete_id"] = frame["athlete_id"].astype(str)
    manifest = pd.read_csv(
        repo_path(config, config["outputs"]["feature_manifest"])
    )
    feature_columns = manifest.loc[
        manifest["role"] == "feature", "column"
    ].tolist()
    sequence_table = materialize_sequence_features(frame, config)
    frame = frame.merge(
        sequence_table[
            [
                "activity_id",
                "athlete_id",
                "early_gap_speed_slope_ms_per_h",
            ]
        ],
        on=["activity_id", "athlete_id"],
        how="left",
        validate="one_to_one",
    )
    sequence_table = frame[["activity_id", "athlete_id"]].merge(
        sequence_table,
        on=["activity_id", "athlete_id"],
        how="left",
        validate="one_to_one",
    )
    sequences = sequence_array(sequence_table)

    assignments = make_group_folds(
        frame,
        group_col=config["evaluate"]["group_col"],
        n_splits=int(config["evaluate"]["outer_folds"]),
    )
    split_path = repo_path(config, config["data"]["cv_splits"])
    split_path.parent.mkdir(parents=True, exist_ok=True)
    assignments.to_parquet(split_path, index=False)

    prediction_frames, inner_rows = _tabular_predictions(
        frame, assignments, feature_columns, config
    )
    deep_predictions, deep_inner = _deep_predictions(
        frame, assignments, sequences, config
    )
    prediction_frames.extend(deep_predictions)
    inner_rows.extend(deep_inner)

    foundation_forecasts, model_status = run_foundation_forecasts(
        frame, config
    )
    prediction_frames.extend(
        _foundation_predictions(
            frame, assignments, foundation_forecasts, config
        )
    )
    predictions = pd.concat(prediction_frames, ignore_index=True)

    shuffled = shuffle_targets_within_athlete(
        frame,
        [f"target_{target}" for target in config["evaluate"]["targets"]],
        seed=int(config["seed"]),
    )
    probe_frames, _ = _tabular_predictions(
        frame,
        assignments,
        feature_columns,
        config,
        observed_frame=shuffled,
        include_inner=False,
    )
    deep_probe, _ = _deep_predictions(
        frame,
        assignments,
        sequences,
        config,
        observed_frame=shuffled,
        include_inner=False,
    )
    probe_frames.extend(deep_probe)
    probe_frames.extend(
        _foundation_predictions(
            frame,
            assignments,
            foundation_forecasts,
            config,
            observed_frame=shuffled,
        )
    )
    probe_predictions = pd.concat(probe_frames, ignore_index=True)
    leakage_rows = []
    for (model, target), group in probe_predictions.groupby(
        ["model", "target"], sort=False
    ):
        leakage_rows.append(
            {
                "model": model,
                "target": target,
                **regression_metrics(
                    group["observed"], group["predicted"]
                ),
                "provenance": "MEASURED_SHUFFLED_TARGET",
            }
        )
    leakage = pd.DataFrame(leakage_rows)
    leakage_path = repo_path(config, config["outputs"]["leakage_probe"])
    leakage_path.parent.mkdir(parents=True, exist_ok=True)
    leakage.to_csv(leakage_path, index=False)
    enforce_leakage_probe(
        leakage,
        maximum_r_squared=float(
            config["evaluate"]["leakage_r2_max"]
        ),
    )

    benchmark = summarise_predictions(predictions)
    benchmark_path = repo_path(config, config["outputs"]["benchmark"])
    benchmark.to_csv(benchmark_path, index=False)
    predictions_path = repo_path(
        config, config["data"]["benchmark_predictions"]
    )
    predictions.to_parquet(predictions_path, index=False)
    fold_metrics = pd.concat(
        [
            pd.DataFrame(inner_rows),
            _fold_metrics_from_predictions(predictions),
        ],
        ignore_index=True,
    )
    fold_metrics.to_csv(
        repo_path(config, config["outputs"]["benchmark_folds"]),
        index=False,
    )
    calibration = calibration_table(
        predictions,
        n_bins=int(config["evaluate"]["calibration_bins"]),
    )
    calibration.to_csv(
        repo_path(config, config["outputs"]["calibration"]), index=False
    )
    cold_start = benchmark[
        benchmark["model"].str.startswith("tier3_")
    ].copy()
    cold_start.to_csv(
        repo_path(config, config["outputs"]["cold_start"]), index=False
    )
    shap_importance, explanation_status = _compute_shap_importance(
        frame,
        feature_columns,
        sequences,
        benchmark,
        config,
    )
    shap_importance.to_csv(
        repo_path(config, config["outputs"]["shap_importance"]),
        index=False,
    )
    if not explanation_status.empty:
        model_status = pd.concat(
            [model_status, explanation_status], ignore_index=True
        )
    model_status.to_csv(
        repo_path(config, config["outputs"]["model_status"]), index=False
    )
    return benchmark, leakage, model_status


def refresh_optional_and_explanations(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Refresh Tier 3 and SHAP outputs without retraining completed lower tiers."""
    from durability.models.tier2_deep import sequence_array
    from durability.models.tier3_foundation import run_foundation_forecasts

    frame = pd.read_parquet(
        repo_path(config, config["data"]["features_output"])
    ).reset_index(drop=True)
    frame["activity_id"] = frame["activity_id"].astype(str)
    frame["athlete_id"] = frame["athlete_id"].astype(str)
    sequence_table = pd.read_parquet(
        repo_path(config, config["data"]["sequence_features"])
    )
    sequence_table["activity_id"] = sequence_table["activity_id"].astype(str)
    sequence_table["athlete_id"] = sequence_table["athlete_id"].astype(str)
    frame = frame.merge(
        sequence_table[
            [
                "activity_id",
                "athlete_id",
                "early_gap_speed_slope_ms_per_h",
            ]
        ],
        on=["activity_id", "athlete_id"],
        how="left",
        validate="one_to_one",
    )
    sequence_table = frame[["activity_id", "athlete_id"]].merge(
        sequence_table,
        on=["activity_id", "athlete_id"],
        how="left",
        validate="one_to_one",
    )
    sequences = sequence_array(sequence_table)
    assignments = pd.read_parquet(
        repo_path(config, config["data"]["cv_splits"])
    )
    existing_predictions = pd.read_parquet(
        repo_path(config, config["data"]["benchmark_predictions"])
    )
    lower_predictions = existing_predictions[
        existing_predictions["tier"] != "tier3"
    ]
    forecasts, foundation_status = run_foundation_forecasts(frame, config)
    foundation_frames = _foundation_predictions(
        frame, assignments, forecasts, config
    )
    predictions = pd.concat(
        [lower_predictions, *foundation_frames], ignore_index=True
    )
    predictions.to_parquet(
        repo_path(config, config["data"]["benchmark_predictions"]),
        index=False,
    )
    benchmark = summarise_predictions(predictions)
    benchmark.to_csv(
        repo_path(config, config["outputs"]["benchmark"]), index=False
    )
    existing_folds = pd.read_csv(
        repo_path(config, config["outputs"]["benchmark_folds"])
    )
    folds = pd.concat(
        [
            existing_folds[existing_folds["tier"] != "tier3"],
            _fold_metrics_from_predictions(
                pd.concat(foundation_frames, ignore_index=True)
            ),
        ],
        ignore_index=True,
    )
    folds.to_csv(
        repo_path(config, config["outputs"]["benchmark_folds"]),
        index=False,
    )
    calibration_table(
        predictions,
        n_bins=int(config["evaluate"]["calibration_bins"]),
    ).to_csv(
        repo_path(config, config["outputs"]["calibration"]), index=False
    )
    benchmark[benchmark["tier"] == "tier3"].to_csv(
        repo_path(config, config["outputs"]["cold_start"]), index=False
    )

    shuffled = shuffle_targets_within_athlete(
        frame,
        [f"target_{target}" for target in config["evaluate"]["targets"]],
        seed=int(config["seed"]),
    )
    foundation_probe = _foundation_predictions(
        frame,
        assignments,
        forecasts,
        config,
        observed_frame=shuffled,
    )
    foundation_probe_frame = pd.concat(
        foundation_probe, ignore_index=True
    )
    foundation_leakage = []
    for (model, target), group in foundation_probe_frame.groupby(
        ["model", "target"], sort=False
    ):
        foundation_leakage.append(
            {
                "model": model,
                "target": target,
                **regression_metrics(
                    group["observed"], group["predicted"]
                ),
                "provenance": "MEASURED_SHUFFLED_TARGET",
            }
        )
    existing_leakage = pd.read_csv(
        repo_path(config, config["outputs"]["leakage_probe"])
    )
    leakage = pd.concat(
        [
            existing_leakage[
                ~existing_leakage["model"].str.startswith("tier3_")
            ],
            pd.DataFrame(foundation_leakage),
        ],
        ignore_index=True,
    )
    leakage.to_csv(
        repo_path(config, config["outputs"]["leakage_probe"]), index=False
    )
    enforce_leakage_probe(
        leakage,
        maximum_r_squared=float(
            config["evaluate"]["leakage_r2_max"]
        ),
    )

    manifest = pd.read_csv(
        repo_path(config, config["outputs"]["feature_manifest"])
    )
    feature_columns = manifest.loc[
        manifest["role"] == "feature", "column"
    ].tolist()
    shap_importance, explanation_status = _compute_shap_importance(
        frame,
        feature_columns,
        sequences,
        benchmark,
        config,
    )
    shap_importance.to_csv(
        repo_path(config, config["outputs"]["shap_importance"]),
        index=False,
    )
    previous_status = pd.read_csv(
        repo_path(config, config["outputs"]["model_status"])
    )
    model_status = pd.concat(
        [
            foundation_status,
            previous_status[previous_status["tier"].isin(["tier1", "tier2"])],
            explanation_status,
        ],
        ignore_index=True,
    ).drop_duplicates(["tier", "model"], keep="last")
    model_status.to_csv(
        repo_path(config, config["outputs"]["model_status"]), index=False
    )
    return benchmark, leakage, model_status
