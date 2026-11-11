"""Tier-1 XGBoost and LightGBM replication models."""

from __future__ import annotations

from typing import Any

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def make_gbdt_model(
    model_name: str, config: dict[str, Any], *, seed: int
) -> Pipeline:
    """Create the preregistered 5,000-tree, 0.005-learning-rate model."""
    settings = config["models"]["tier1_smyth_replication"]
    common = {
        "n_estimators": int(settings["n_estimators"]),
        "learning_rate": float(settings["learning_rate"]),
        "max_depth": int(settings["max_depth"]),
        "random_state": int(seed),
        "n_jobs": int(config["evaluate"]["n_jobs"]),
    }
    if model_name == "tier1_xgboost":
        from xgboost import XGBRegressor

        regressor = XGBRegressor(
            **common,
            objective="reg:squarederror",
            verbosity=0,
        )
    elif model_name == "tier1_lightgbm":
        from lightgbm import LGBMRegressor

        regressor = LGBMRegressor(
            **common,
            num_leaves=int(settings["lightgbm_num_leaves"]),
            verbosity=-1,
        )
    else:
        raise ValueError(f"Unknown Tier-1 model: {model_name}")
    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    keep_empty_features=True,
                ),
            ),
            ("regressor", regressor),
        ]
    )

