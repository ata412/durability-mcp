"""Fixed exploratory tabular models outside the preregistered benchmark."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_exploratory_model(
    model_name: str, config: dict[str, Any], *, seed: int
) -> Pipeline:
    """Create one fixed exploratory model without outer-test tuning."""
    settings = config["models"]["exploratory"]
    n_jobs = int(config["evaluate"]["n_jobs"])
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)

    if model_name == "elastic_net":
        model_settings = settings[model_name]
        steps = [
            ("imputer", imputer),
            ("scaler", StandardScaler()),
            (
                "regressor",
                ElasticNet(
                    alpha=float(model_settings["alpha"]),
                    l1_ratio=float(model_settings["l1_ratio"]),
                    max_iter=int(model_settings["max_iter"]),
                    random_state=int(seed),
                ),
            ),
        ]
    elif model_name in {"random_forest", "extra_trees"}:
        model_settings = settings[model_name]
        model_class = (
            RandomForestRegressor
            if model_name == "random_forest"
            else ExtraTreesRegressor
        )
        steps = [
            ("imputer", imputer),
            (
                "regressor",
                model_class(
                    n_estimators=int(model_settings["n_estimators"]),
                    min_samples_leaf=int(
                        model_settings["min_samples_leaf"]
                    ),
                    max_features=model_settings["max_features"],
                    random_state=int(seed),
                    n_jobs=n_jobs,
                ),
            ),
        ]
    elif model_name == "hist_gradient_boosting":
        model_settings = settings[model_name]
        steps = [
            ("imputer", imputer),
            (
                "regressor",
                HistGradientBoostingRegressor(
                    max_iter=int(model_settings["max_iter"]),
                    learning_rate=float(model_settings["learning_rate"]),
                    max_leaf_nodes=int(model_settings["max_leaf_nodes"]),
                    l2_regularization=float(
                        model_settings["l2_regularization"]
                    ),
                    random_state=int(seed),
                ),
            ),
        ]
    elif model_name == "catboost":
        from catboost import CatBoostRegressor

        model_settings = settings[model_name]
        steps = [
            ("imputer", imputer),
            (
                "regressor",
                CatBoostRegressor(
                    iterations=int(model_settings["iterations"]),
                    learning_rate=float(model_settings["learning_rate"]),
                    depth=int(model_settings["depth"]),
                    l2_leaf_reg=float(model_settings["l2_leaf_reg"]),
                    loss_function="RMSE",
                    random_seed=int(seed),
                    thread_count=n_jobs,
                    verbose=False,
                    allow_writing_files=False,
                ),
            ),
        ]
    else:
        raise ValueError(f"Unknown exploratory model: {model_name}")
    return Pipeline(steps)
