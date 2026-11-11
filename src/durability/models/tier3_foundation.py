"""Optional foundation-model forecasts of prior durability-label histories."""

from __future__ import annotations

import importlib.util
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from durability.config import repo_path

LOGGER = logging.getLogger(__name__)
_CHRONOS_PIPELINE: Any | None = None
_TIMESFM_MODEL: Any | None = None


def package_status() -> pd.DataFrame:
    """Report actual optional-package availability without substitution."""
    rows = []
    for package, module in (
        ("chronos", "chronos"),
        ("moment", "momentfm"),
        ("timesfm", "timesfm"),
    ):
        available = importlib.util.find_spec(module) is not None
        rows.append(
            {
                "tier": "tier3",
                "model": package,
                "status": "AVAILABLE" if available else "SKIPPED",
                "reason": (
                    ""
                    if available
                    else "package_not_installed_or_incompatible"
                ),
                "provenance": "MEASURED",
            }
        )
    return pd.DataFrame(rows)


def build_label_contexts(
    frame: pd.DataFrame,
    *,
    target_column: str,
    history_cap: int,
) -> list[np.ndarray]:
    """Build strict-prior label sequences aligned to the input row order."""
    work = frame[
        ["activity_id", "athlete_id", "start_time", target_column]
    ].copy()
    work["start_time"] = pd.to_datetime(work["start_time"], utc=True)
    contexts: dict[str, np.ndarray] = {}
    for _, athlete in work.groupby("athlete_id", sort=False):
        ordered = athlete.sort_values(["start_time", "activity_id"])
        history: list[float] = []
        for _, equal_time in ordered.groupby("start_time", sort=True):
            context = np.asarray(history[-history_cap:], dtype=np.float32)
            for row in equal_time.itertuples(index=False):
                contexts[str(row.activity_id)] = context.copy()
            for value in equal_time[target_column]:
                if np.isfinite(value):
                    history.append(float(value))
    return [
        contexts[str(activity_id)]
        for activity_id in frame["activity_id"].astype(str)
    ]


def _chronos_forecast(
    contexts: list[np.ndarray], config: dict[str, Any]
) -> np.ndarray:
    global _CHRONOS_PIPELINE

    import torch
    from chronos import ChronosBoltPipeline

    cache = repo_path(config, config["outputs"]["models_dir"]) / "huggingface"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache))
    if _CHRONOS_PIPELINE is None:
        _CHRONOS_PIPELINE = ChronosBoltPipeline.from_pretrained(
            config["models"]["tier3"]["chronos_model_id"],
            device_map="cpu",
            dtype=torch.float32,
            cache_dir=str(cache),
        )
    valid_indices = [
        index for index, context in enumerate(contexts) if len(context)
    ]
    predictions = np.full(len(contexts), np.nan, dtype=float)
    if not valid_indices:
        return predictions
    inputs = [
        torch.as_tensor(contexts[index], dtype=torch.float32)
        for index in valid_indices
    ]
    _, means = _CHRONOS_PIPELINE.predict_quantiles(
        inputs,
        prediction_length=1,
        quantile_levels=[0.5],
    )
    predictions[valid_indices] = means[:, 0].detach().cpu().numpy()
    return predictions


def _timesfm_forecast(
    contexts: list[np.ndarray], config: dict[str, Any]
) -> np.ndarray:
    global _TIMESFM_MODEL

    import timesfm

    cache = repo_path(config, config["outputs"]["models_dir"]) / "huggingface"
    cache.mkdir(parents=True, exist_ok=True)
    valid_indices = [
        index for index, context in enumerate(contexts) if len(context)
    ]
    predictions = np.full(len(contexts), np.nan, dtype=float)
    if not valid_indices:
        return predictions
    if _TIMESFM_MODEL is None:
        _TIMESFM_MODEL = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            config["models"]["tier3"]["timesfm_model_id"],
            cache_dir=str(cache),
            torch_compile=False,
        )
        maximum_context = max(
            int(value)
            for value in config["models"]["tier3"]["cold_start_history"]
        )
        _TIMESFM_MODEL.compile(
            timesfm.ForecastConfig(
                max_context=maximum_context,
                max_horizon=1,
                normalize_inputs=True,
                per_core_batch_size=int(
                    config["models"]["tier3"]["forecast_batch_size"]
                ),
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
            )
        )
    point, _ = _TIMESFM_MODEL.forecast(
        horizon=1,
        inputs=[contexts[index] for index in valid_indices],
    )
    predictions[valid_indices] = point[:, 0]
    return predictions


def run_foundation_forecasts(
    frame: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run installed zero-shot models; record failures and unsupported modes."""
    status = package_status()
    rows: list[pd.DataFrame] = []
    for model_name, forecast_function in (
        ("chronos", _chronos_forecast),
        ("timesfm", _timesfm_forecast),
    ):
        status_index = status.index[status["model"] == model_name][0]
        if status.at[status_index, "status"] != "AVAILABLE":
            continue
        try:
            for target in config["models"]["tier3"]["primary_targets"]:
                target_column = f"target_{target}"
                for history_cap in config["models"]["tier3"][
                    "cold_start_history"
                ]:
                    contexts = build_label_contexts(
                        frame,
                        target_column=target_column,
                        history_cap=int(history_cap),
                    )
                    prediction = forecast_function(contexts, config)
                    rows.append(
                        pd.DataFrame(
                            {
                                "activity_id": frame[
                                    "activity_id"
                                ].astype(str),
                                "athlete_id": frame[
                                    "athlete_id"
                                ].astype(str),
                                "model": f"tier3_{model_name}_zeroshot",
                                "target": target,
                                "history_cap": int(history_cap),
                                "observed": frame[target_column],
                                "predicted": prediction,
                                "context_sessions": [
                                    len(context) for context in contexts
                                ],
                                "provenance": "MEASURED",
                            }
                        )
                    )
            status.at[status_index, "status"] = "RUN"
        except Exception as exc:  # Optional tier must not block lower tiers.
            LOGGER.warning("%s skipped after runtime failure: %s", model_name, exc)
            status.at[status_index, "status"] = "SKIPPED"
            status.at[status_index, "reason"] = (
                f"runtime_failure:{type(exc).__name__}:{exc}"
            )
    for model_name in ("chronos", "moment", "timesfm"):
        for mode in ("few_shot_calibration", "fine_tuned"):
            rows_status = {
                "tier": "tier3",
                "model": f"{model_name}_{mode}",
                "status": (
                    "PENDING_CALIBRATION"
                    if mode == "few_shot_calibration"
                    and (
                        status.loc[
                            status["model"] == model_name, "status"
                        ].iloc[0]
                        == "RUN"
                    )
                    else "SKIPPED"
                ),
                "reason": (
                    ""
                    if mode == "few_shot_calibration"
                    and (
                        status.loc[
                            status["model"] == model_name, "status"
                        ].iloc[0]
                        == "RUN"
                    )
                    else (
                        "base_model_unavailable"
                        if mode == "few_shot_calibration"
                        else "package_has_no_validated_target_finetuning_path"
                    )
                ),
                "provenance": "MEASURED",
            }
            status = pd.concat(
                [status, pd.DataFrame([rows_status])], ignore_index=True
            )
    status.loc[
        status["status"] == "PENDING_CALIBRATION", "status"
    ] = "RUN"
    forecasts = (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame(
            columns=[
                "activity_id",
                "athlete_id",
                "model",
                "target",
                "history_cap",
                "observed",
                "predicted",
                "context_sessions",
                "provenance",
            ]
        )
    )
    return forecasts, status
