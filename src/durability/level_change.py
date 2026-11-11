"""E14: distinguish prediction of absolute physiological level from change."""
from __future__ import annotations

import logging
from itertools import product

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.stats.multitest import multipletests

from durability.config import repo_path
from durability.evaluate import (
    assert_group_split_integrity, regression_metrics, shuffle_targets_within_athlete,
)
from durability.models.tier1_gbdt import make_gbdt_model
from durability.proposal_experiments import load_experiment_frame
from durability.two_stage_subgroups import bootstrap_weights, inference, weighted_bootstrap

LOGGER = logging.getLogger(__name__)


def early_baselines(train_early, train_late, test_early):
    """Fit all simple controls strictly inside an outer training fold."""
    train_early = np.asarray(train_early, dtype=float)
    train_late = np.asarray(train_late, dtype=float)
    test_early = np.asarray(test_early, dtype=float)
    assert np.all(train_early > 0) and np.all(test_early > 0)
    mean_change = np.mean(train_late - train_early)
    mean_magnitude = np.mean(train_late / train_early)
    affine = LinearRegression().fit(train_early[:, None], train_late)
    predictions = {
        "early_persistence": test_early.copy(),
        "early_plus_mean_change": test_early + mean_change,
        "early_times_mean_magnitude": test_early * mean_magnitude,
        "early_affine_calibration": affine.predict(test_early[:, None]),
    }
    fitted = dict(mean_additive_change=mean_change, mean_magnitude=mean_magnitude,
                  affine_intercept=float(affine.intercept_),
                  affine_slope=float(affine.coef_[0]))
    return predictions, fitted


def endpoints(early, observed_late, predicted_late):
    """Preserve exact paired errors when changing the target representation."""
    early = np.asarray(early, dtype=float)
    observed_late = np.asarray(observed_late, dtype=float)
    predicted_late = np.asarray(predicted_late, dtype=float)
    assert np.all(early > 0)
    return {
        "late_decoupling": (observed_late, predicted_late),
        "additive_change": (observed_late-early, predicted_late-early),
        "magnitude": (observed_late/early, predicted_late/early),
    }


def audit_normalization(frame, settings):
    early = frame[settings["early_column"]].to_numpy(dtype=float)
    late = frame[settings["late_column"]].to_numpy(dtype=float)
    magnitude = frame[settings["magnitude_column"]].to_numpy(dtype=float)
    rows = []
    for cs_scale, hrmax_scale in product(settings["parameter_audit_scales"], repeat=2):
        scale = cs_scale / hrmax_scale
        error = float(np.max(abs((late*scale)/(early*scale)-magnitude)))
        assert error < settings["identity_tolerance"]
        rows.append(dict(cs_scale=cs_scale, hrmax_scale=hrmax_scale,
                         max_magnitude_error=error,
                         provenance="MEASURED ALGEBRA AUDIT"))
    return pd.DataFrame(rows)


def summarise(predictions, settings, seed):
    summary_rows, contrast_rows = [], []
    athletes = sorted(predictions.athlete_id.unique())
    weights = bootstrap_weights(len(athletes), settings["bootstrap_n"], seed)
    contrasts = [
        ("e1_frozen_lightgbm", "early_persistence"),
        ("e1_frozen_lightgbm", "early_plus_mean_change"),
        ("e1_frozen_lightgbm", "early_affine_calibration"),
        ("direct_matched_lightgbm", "early_plus_mean_change"),
        ("residual_change_lightgbm", "early_plus_mean_change"),
        ("residual_change_lightgbm", "early_persistence"),
        ("residual_change_lightgbm", "direct_matched_lightgbm"),
    ]
    for endpoint, data in predictions.groupby("endpoint"):
        errors = {}
        for model, part in data.groupby("model"):
            error = abs(part.observed - part.predicted)
            athlete_error = error.groupby(part.athlete_id).mean()
            errors[model] = athlete_error
            metrics = regression_metrics(part.observed, part.predicted)
            summary_rows.append(dict(endpoint=endpoint, model=model, **metrics,
                                     mae_athlete=athlete_error.mean(),
                                     n_athletes=part.athlete_id.nunique(),
                                     provenance="PILOT MEASURED EXPLORATORY"))
        for challenger, reference in contrasts:
            difference = errors[challenger] - errors[reference]
            assert difference.notna().all()
            draws = weighted_bootstrap(difference, athletes, weights)
            point = difference.mean()
            contrast_rows.append(dict(endpoint=endpoint, challenger=challenger,
                reference=reference, delta_mae_athlete=point,
                **inference(point, draws, settings["confidence_level"]),
                n_athletes=len(difference), provenance="PILOT MEASURED EXPLORATORY"))
    summary, contrasts = pd.DataFrame(summary_rows), pd.DataFrame(contrast_rows)
    # Late and additive errors are identical; do not count duplicated hypotheses twice.
    unique = contrasts[contrasts.endpoint.ne("late_decoupling")].copy()
    unique["q_bh"] = multipletests(unique.p_bootstrap, method="fdr_bh")[1]
    for row in unique.itertuples(index=False):
        mask = (contrasts.challenger.eq(row.challenger)
                & contrasts.reference.eq(row.reference)
                & (contrasts.endpoint.isin(["late_decoupling", "additive_change"])
                   if row.endpoint == "additive_change" else contrasts.endpoint.eq(row.endpoint)))
        contrasts.loc[mask, "q_bh"] = row.q_bh
    contrasts["supported_gain"] = contrasts.ci_high.lt(0) & contrasts.q_bh.le(settings["fdr_alpha"])
    return summary, contrasts


def markdown_table(table):
    table = table.round(6)
    return "\n".join([
        "| " + " | ".join(table.columns) + " |",
        "| " + " | ".join("---" for _ in table.columns) + " |",
        *["| " + " | ".join(str(x) for x in row) + " |"
          for row in table.itertuples(index=False, name=None)],
    ])


def run_level_change(config):
    settings = config["level_change_prediction"]
    frame, features = load_experiment_frame(config)
    frozen = pd.read_parquet(repo_path(config, config["experiments"]["outputs"]["e1_predictions"]))
    frozen = frozen[frozen.model.eq(config["experiments"]["primary_model"])
                    & frozen.endpoint.eq("late_decoupling")].copy()
    assert not frozen.activity_id.duplicated().any()
    frame = frame.merge(frozen[["activity_id", "athlete_id", "outer_fold", "observed", "predicted"]],
                        on=["activity_id", "athlete_id"], how="left", validate="one_to_one")
    assert len(frame) == len(frozen) and frame.outer_fold.notna().all()
    assert frame.groupby("athlete_id").outer_fold.nunique().eq(1).all()
    early = frame[settings["early_column"]].to_numpy(dtype=float)
    late = frame[settings["late_column"]].to_numpy(dtype=float)
    assert np.isfinite(early).all() and (early > 0).all() and np.isfinite(late).all()
    assert np.allclose(frame.observed, late)
    assert (pd.to_datetime(frame.max_history_start_time, utc=True)
            < pd.to_datetime(frame.start_time, utc=True)).all()
    assert config["indices"]["early_window"][1] <= config["features"]["early_window_frac"]
    assert not any(x.startswith("target_") for x in features)
    legal_features = list(dict.fromkeys(features + [settings["early_column"]]))
    design = frame[legal_features].copy()
    frame["target_change"] = late - early
    if settings["leakage_shuffle"] != "within_athlete":
        raise ValueError("Only within-athlete change shuffling is supported")
    shuffled = shuffle_targets_within_athlete(frame, ["target_change"], seed=config["seed"])
    pd.testing.assert_frame_equal(design, shuffled[legal_features])
    predictions, calibrations, leakage = [], [], []
    for fold in sorted(frame.outer_fold.unique()):
        train = np.flatnonzero(frame.outer_fold.ne(fold))
        test = np.flatnonzero(frame.outer_fold.eq(fold))
        assert_group_split_integrity(frame, train, test, group_col=config["evaluate"]["group_col"])
        controls, calibration = early_baselines(early[train], late[train], early[test])
        calibration.update(outer_fold=int(fold), n_train=len(train), n_test=len(test),
                           provenance="PILOT MEASURED")
        calibrations.append(calibration)
        controls["e1_frozen_lightgbm"] = frame.iloc[test].predicted.to_numpy()
        for name, target in [("direct_matched_lightgbm", late),
                             ("residual_change_lightgbm", late-early)]:
            LOGGER.info("E14 fold=%s fitting %s", fold, name)
            model = make_gbdt_model(settings["model"], config, seed=int(config["seed"])+int(fold))
            model.fit(design.iloc[train], target[train])
            estimate = model.predict(design.iloc[test])
            if name == "residual_change_lightgbm":
                estimate = early[test] + estimate
            controls[name] = estimate
        for name, estimate in controls.items():
            assert np.isfinite(estimate).all()
            for endpoint, (observed, predicted) in endpoints(early[test], late[test], estimate).items():
                part = frame.iloc[test][["activity_id", "athlete_id", "start_time", "outer_fold"]].copy()
                part["early_reference"] = early[test]
                part["model"], part["endpoint"] = name, endpoint
                part["observed"], part["predicted"] = observed, predicted
                part["provenance"] = "PILOT MEASURED EXPLORATORY"
                predictions.append(part)
        LOGGER.info("E14 fold=%s fitting shuffled change control", fold)
        model = make_gbdt_model(settings["model"], config, seed=int(config["seed"])+int(fold))
        model.fit(design.iloc[train], shuffled.iloc[train].target_change)
        leakage.append(pd.DataFrame(dict(athlete_id=frame.iloc[test].athlete_id,
            observed=shuffled.iloc[test].target_change,
            predicted=model.predict(design.iloc[test]))))
    predictions = pd.concat(predictions, ignore_index=True)
    summary, contrasts = summarise(predictions, settings, int(config["seed"]))
    leak = pd.concat(leakage, ignore_index=True)
    leakage_summary = pd.DataFrame([dict(model="residual_change_lightgbm", endpoint="additive_change",
        **regression_metrics(leak.observed, leak.predicted),
        provenance="MEASURED SHUFFLED TARGET", seed=config["seed"])])
    audit = audit_normalization(frame, settings)
    outputs = settings["outputs"]
    for name, table in [("predictions", predictions), ("summary", summary),
                        ("contrasts", contrasts), ("folds", pd.DataFrame(calibrations)),
                        ("leakage", leakage_summary), ("parameter_audit", audit)]:
        path = repo_path(config, outputs[name])
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".parquet":
            table.to_parquet(path, index=False)
        else:
            table.to_csv(path, index=False)
    assert leakage_summary.r_squared.iloc[0] <= config["evaluate"]["leakage_r2_max"], "E14 change leakage rule failed"
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.7))
    order = list(controls)
    for ax, endpoint in zip(axes, ["late_decoupling", "additive_change", "magnitude"]):
        table = summary[summary.endpoint.eq(endpoint)].set_index("model").loc[order]
        colors = ["#009E73" if name == "residual_change_lightgbm"
                  else "#0072B2" if "lightgbm" in name else "#8394A0" for name in order]
        ax.barh(np.arange(len(table)), table.r_squared, color=colors)
        lower = min(0, table.r_squared.min())
        upper = max(0, table.r_squared.max())
        span = max(upper-lower, 0.1)
        for index, value in enumerate(table.r_squared):
            ax.text(value + (0.025*span if value >= 0 else -0.025*span),
                    index, f"{value:.3f}", va="center",
                    ha="left" if value >= 0 else "right", fontsize=9)
        ax.set_xlim(lower-0.18*span, upper+0.18*span)
        ax.set_yticks(np.arange(len(table)), [x.replace("_", " ") for x in order])
        ax.axvline(0, color="grey", linestyle="--")
        ax.invert_yaxis()
        ax.set_title(endpoint.replace("_", " "))
        ax.set_xlabel("Session R² (higher is better)")
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle("E14: level vs change — FitRec PILOT MEASURED EXPLORATORY")
    fig.tight_layout()
    fig.savefig(repo_path(config, outputs["figure"]), dpi=180)
    plt.close(fig)
    primary = contrasts[(contrasts.endpoint.eq("additive_change"))
                        & contrasts.challenger.eq("residual_change_lightgbm")
                        & contrasts.reference.eq("early_plus_mean_change")].iloc[0]
    metrics = summary.set_index(["endpoint", "model"])
    control = metrics.loc[("additive_change", "early_plus_mean_change")]
    residual = metrics.loc[("additive_change", "residual_change_lightgbm")]
    relative_session_gain = 100*(control.mae-residual.mae)/control.mae
    relative_athlete_gain = 100*(control.mae_athlete-residual.mae_athlete)/control.mae_athlete
    control_late = metrics.loc[("late_decoupling", "early_plus_mean_change")]
    frozen_late = metrics.loc[("late_decoupling", "e1_frozen_lightgbm")]
    lines = ["# E14 — Physiological level versus within-session change", "",
        "All observed results are **PILOT MEASURED EXPLORATORY** on FitRec. Protocol: docs/E14_PROTOCOL.md.",
        f"Cohort: {len(frame)} sessions / {frame.athlete_id.nunique()} athletes. Seed: {config['seed']}.", "",
        "## Models and endpoints", "",
        "Persistence copies the observed early reference to the late segment. Additive and multiplicative controls learn one correction from training sessions. Affine calibration learns a slope/intercept from early to late using training data only.",
        "Frozen E1 preserves the original result. Matched direct and residual LightGBM both receive the same manifest features plus the exact early reference. Residual learns late minus early, then adds early back.",
        "Late and additive-change MAE are mathematically identical for any one method. R² changes with the outcome variance; high late-level R² does not establish change prediction. Magnitude divides errors by the observed early reference.", "",
        "## Results", "", markdown_table(summary), "",
        "## Paired equal-athlete MAE comparisons", "",
        "Negative delta favours challenger. Pointwise athlete-bootstrap CIs; BH applied across change and magnitude tests, excluding duplicated late-error hypotheses.",
        markdown_table(contrasts), "",
        "## Primary result", "",
        f"Residual model vs training-mean change: delta MAE {primary.delta_mae_athlete:.6f}, 95% CI [{primary.ci_low:.6f}, {primary.ci_high:.6f}], q={primary.q_bh:.6f}; supported gain={primary.supported_gain} (PILOT MEASURED).", "",
        "## What the results mean", "",
        f"The simple early-plus-mean-change control attains late R² {control_late.r_squared:.6f}, compared with frozen E1 {frozen_late.r_squared:.6f} (PILOT MEASURED). High absolute-level prediction therefore does not require sophisticated modelling.",
        f"Learning change reduces session MAE from {control.mae:.6f} to {residual.mae:.6f}, a {relative_session_gain:.2f}% reduction; equal-athlete reduction is {relative_athlete_gain:.2f}% (PILOT MEASURED). The primary paired contrast supports a modest incremental gain over the additive-mean control.",
        f"However, residual-model change R² is only {residual.r_squared:.6f} (PILOT MEASURED). Most additive-change variability remains unexplained. Superiority over matched direct LightGBM is not supported after BH adjustment, even though pointwise CIs favour residual prediction.",
        f"The within-athlete shuffled-change model still obtains R² {leakage_summary.r_squared.iloc[0]:.6f} (MEASURED SHUFFLED TARGET). It passes the configured halt rule, but retains athlete-level signal. The small gap from the real change score does not establish prediction of session-specific timing; a repeated hierarchical permutation comparison is needed for that claim.", "",
        "## Leakage and normalization audit", "", markdown_table(leakage_summary), "",
        "Shuffled change is scored as change, not absolute late level. All features stayed identical after target shuffling.",
        markdown_table(audit), "",
        "Fixed CS/HRmax contributes a shared multiplicative scale to early and late decoupling. This cancels in magnitude and normalized onset input. It does not cancel in absolute decoupling; it does not validate the physiological interpretation of any endpoint.", "",
        "## Limitations", "",
        "Only gate-passing sessions are included; selection truncates the outcome distribution. Models are locked configurations without new tuning. The exact early reference is added to BOTH matched models; their scores need not reproduce frozen E1. Time fractions assume final activity duration is known retrospectively. Within-athlete permutation retains athlete-level signal, so a passed probe is evidence against the tested leakage mechanism, not proof of all leakage absence. External and physiological validation remain necessary.", "",
        "![Level and change prediction](../../figures/e14_level_change_comparison.png)"]
    repo_path(config, outputs["report"]).write_text("\n".join(lines)+"\n", encoding="utf-8")
    return summary, contrasts
