"""Conjunctive pacing-intent gate and sensitivity analysis."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from durability.config import repo_path


def gate_pass(
    drift: float | np.ndarray | pd.Series,
    magnitude: float | np.ndarray | pd.Series,
    config: dict[str, Any],
) -> bool | np.ndarray | pd.Series:
    """Production gate: both metabolic-cost criteria must pass."""
    result = (drift > config["gate"]["drift_bpm_per_h"]) & (
        magnitude > config["gate"]["magnitude"]
    )
    return bool(result) if np.isscalar(result) else result


def diagnostic_disjunctive_gate(
    drift: float, magnitude: float, config: dict[str, Any]
) -> bool:
    """Test-only counterfactual documenting why OR is invalid."""
    return bool(
        (drift > config["gate"]["drift_bpm_per_h"])
        or (magnitude > config["gate"]["magnitude"])
    )


def apply_gate(
    frame: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    """Apply the conjunctive gate only to QC-eligible activities."""
    result = frame.copy()
    metabolic_rise = gate_pass(result["drift"], result["magnitude"], config)
    result["gate_pass"] = (
        result["qc_eligible"].fillna(False) & metabolic_rise.fillna(False)
    )
    output_path = repo_path(config, config["data"]["gated_indices"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    return result


def sensitivity_sweep(
    frame: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    """Report retention at every configured conjunctive threshold pair."""
    eligible = frame[frame["qc_eligible"].fillna(False)]
    rows: list[dict[str, Any]] = []
    for drift_threshold in config["gate"]["sweep"]["drift_bpm_per_h"]:
        for magnitude_threshold in config["gate"]["sweep"]["magnitude"]:
            passed = (eligible["drift"] > drift_threshold) & (
                eligible["magnitude"] > magnitude_threshold
            )
            n_pass = int(passed.sum())
            rows.append(
                {
                    "drift_threshold_bpm_per_h": drift_threshold,
                    "magnitude_threshold": magnitude_threshold,
                    "n_qc_eligible": len(eligible),
                    "n_gate_pass": n_pass,
                    "retention_rate": (
                        n_pass / len(eligible) if len(eligible) else np.nan
                    ),
                    "icc_status": "DEFERRED_TO_PHASE5",
                    "provenance": "MEASURED",
                }
            )
    result = pd.DataFrame(rows)
    path = repo_path(config, config["outputs"]["gate_sensitivity"])
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False)
    return result

