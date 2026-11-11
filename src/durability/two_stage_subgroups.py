"""E13: paired subgroup evaluation of frozen E12 predictions."""
from __future__ import annotations

from itertools import combinations
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from durability.config import repo_path


def shrink(mean, count, population, strength):
    return (count * mean + strength * population) / (count + strength)


def training_cutoffs(training, column, quantiles):
    """Equal-athlete thresholds from training data only."""
    values = training.groupby("athlete_id")[column].mean().dropna()
    if values.empty:
        raise ValueError(f"No training evidence for {column}")
    return values.quantile(quantiles).to_numpy()


def pair_predictions(predictions, history, settings, minimum):
    chosen = predictions[
        predictions.minimum_prior_sessions.eq(minimum)
        & predictions.regime.eq("personalized")
    ]
    a = chosen[chosen.model.eq(settings["challenger"])].copy()
    b = chosen[chosen.model.eq(settings["reference"])].copy()
    keys = ["activity_id", "athlete_id"]
    pairs = a.merge(b[keys + ["observed", "predicted", "outer_fold"]],
                    on=keys, suffixes=("", "_direct"), how="outer",
                    validate="one_to_one", indicator=True)
    assert pairs._merge.eq("both").all(), "Models must cover identical sessions"
    assert np.allclose(pairs.observed, pairs.observed_direct)
    assert pairs.outer_fold.eq(pairs.outer_fold_direct).all()
    assert pairs.groupby("athlete_id").outer_fold.nunique().eq(1).all()
    pairs["error_two_stage"] = abs(pairs.observed - pairs.predicted)
    pairs["error_direct"] = abs(pairs.observed - pairs.predicted_direct)
    assert np.isfinite(pairs[["observed", "predicted", "predicted_direct"]]).all().all()
    pairs["delta_error"] = pairs.error_two_stage - pairs.error_direct
    columns = keys + ["target_history_count", "target_history_sd",
                      "history_duration_h_28d", "target_history_max_start_time"]
    pairs = pairs.drop(columns="_merge").merge(history[columns], on=keys,
                                               validate="one_to_one")
    assert (pd.to_datetime(pairs.target_history_max_start_time, utc=True)
            < pd.to_datetime(pairs.start_time, utc=True)).all()
    assert len(pairs) == len(a)
    return pairs


def assign_subgroups(pairs, history, settings):
    """Assign held-out rows with fold-specific training-only cutoffs."""
    result, thresholds = [], []
    labels = settings["quantile_labels"]
    minimum = int(pairs.minimum_prior_sessions.iloc[0])
    for fold, test in pairs.groupby("outer_fold"):
        train = history[history.outer_fold.ne(fold)].copy()
        assert not set(train.athlete_id) & set(test.athlete_id)
        population = train.target_drift.mean()
        strengths = test.selected_shrinkage_strength.dropna().unique()
        assert len(strengths) == 1
        train["baseline"] = shrink(train.target_history_mean,
                                   train.target_history_count,
                                   population, strengths[0])
        train = train[train.target_history_count.ge(minimum)]
        count = test.copy()
        count["factor"] = "prior_sessions"
        count["group"] = np.asarray(settings["history_labels"])[
            np.searchsorted(settings["history_boundaries"],
                            test.target_history_count, side="right")]
        result.append(count)
        for factor, column in settings["factors"].items():
            cuts = training_cutoffs(train, column, settings["quantiles"])
            part = test.copy()
            part["factor"] = factor
            part["group"] = np.asarray(labels)[np.searchsorted(
                cuts, part[column], side="right")]
            part.loc[part[column].isna(), "group"] = "missing"
            result.append(part)
            thresholds.append(dict(outer_fold=fold, factor=factor, column=column,
                                   lower_cutoff=cuts[0], upper_cutoff=cuts[1],
                                   n_training_athletes=train.athlete_id.nunique(),
                                   provenance="PILOT MEASURED"))
    return pd.concat(result, ignore_index=True), pd.DataFrame(thresholds)


def bootstrap_weights(n_athletes, n_bootstrap, seed):
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_athletes, np.full(n_athletes, 1 / n_athletes),
                           size=n_bootstrap)


def weighted_bootstrap(athlete_values, athletes, weights):
    values = athlete_values.reindex(athletes)
    mask = values.notna().to_numpy()
    numerator = weights[:, mask] @ values.dropna().to_numpy()
    denominator = weights[:, mask].sum(axis=1)
    return np.divide(numerator, denominator, out=np.full(len(weights), np.nan),
                     where=denominator > 0)


def inference(point, draws, confidence):
    draws = draws[np.isfinite(draws)]
    alpha = 1 - confidence
    low, high = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    p = (1 + np.count_nonzero(abs(draws - point) >= abs(point))) / (len(draws) + 1)
    return dict(ci_low=low, ci_high=high, p_bootstrap=p)


def summarise(memberships, settings, seed):
    athletes = sorted(memberships.athlete_id.unique())
    weights = bootstrap_weights(len(athletes), settings["bootstrap_n"], seed)
    rows, interactions, estimates = [], [], {}
    for (factor, group), part in memberships.groupby(["factor", "group"]):
        per_person = part.groupby("athlete_id")[["error_direct", "error_two_stage",
                                                  "delta_error"]].mean()
        point = per_person.delta_error.mean()
        draws = weighted_bootstrap(per_person.delta_error, athletes, weights)
        estimates[factor, group] = (point, draws)
        rows.append(dict(factor=factor, group=group, n_sessions=len(part),
                         n_athletes=len(per_person),
                         direct_mae_athlete=per_person.error_direct.mean(),
                         two_stage_mae_athlete=per_person.error_two_stage.mean(),
                         delta_mae_athlete=point,
                         direct_mae_session=part.error_direct.mean(),
                         two_stage_mae_session=part.error_two_stage.mean(),
                         delta_mae_session=part.delta_error.mean(),
                         **inference(point, draws, settings["confidence_level"]),
                         sufficient_size=len(per_person) >= settings["minimum_athletes"],
                         provenance="PILOT MEASURED EXPLORATORY"))
    for factor in memberships.factor.unique():
        groups = settings["history_labels"] if factor == "prior_sessions" else settings["quantile_labels"]
        for a, b in combinations(groups, 2):
            if (factor, a) not in estimates or (factor, b) not in estimates:
                continue
            pa, da = estimates[factor, a]
            pb, db = estimates[factor, b]
            interactions.append(dict(factor=factor, group_a=a, group_b=b,
                                     delta_difference=pa-pb,
                                     **inference(pa-pb, da-db, settings["confidence_level"]),
                                     provenance="PILOT MEASURED EXPLORATORY"))
    summary, contrasts = pd.DataFrame(rows), pd.DataFrame(interactions)
    for table in (summary, contrasts):
        table["q_bh"] = multipletests(table.p_bootstrap, method="fdr_bh")[1]
    summary["supported_gain"] = (summary.ci_high.lt(0)
        & summary.q_bh.le(settings["fdr_alpha"]) & summary.sufficient_size)
    return summary, contrasts


def run_subgroups(config):
    settings = config["two_stage_subgroups"]
    source = config["two_stage_prediction"]
    predictions = pd.read_parquet(repo_path(config, source["outputs"]["predictions"]))
    history = pd.read_parquet(repo_path(config, source["outputs"]["history"]))
    pairs = pair_predictions(predictions, history, settings,
                             source["primary_min_prior_sessions"])
    members, thresholds = assign_subgroups(pairs, history, settings)
    summary, contrasts = summarise(members, settings, config["seed"])
    outputs = settings["outputs"]
    for name, table in [("paired", pairs), ("memberships", members),
                        ("thresholds", thresholds), ("summary", summary),
                        ("interactions", contrasts)]:
        path = repo_path(config, outputs[name])
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".parquet":
            table.to_parquet(path, index=False)
        else:
            table.to_csv(path, index=False)
    fig, ax = plt.subplots(figsize=(10, 8))
    y = np.arange(len(summary))
    ax.errorbar(summary.delta_mae_athlete, y,
                xerr=np.array([summary.delta_mae_athlete-summary.ci_low,
                               summary.ci_high-summary.delta_mae_athlete]),
                fmt="o", color="#0072B2", capsize=3)
    ax.axvline(0, color="grey", linestyle="--")
    ax.set_yticks(y, summary.factor + " / " + summary.group)
    ax.invert_yaxis()
    ax.set_xlabel("Athlete-weighted MAE: Two-stage − Direct (bpm/h)\nNegative favours Two-stage; pointwise 95% athlete-bootstrap CI")
    ax.set_title("E13 subgroup analysis — FitRec PILOT MEASURED EXPLORATORY")
    fig.tight_layout()
    fig.savefig(repo_path(config, outputs["figure"]), dpi=200)
    plt.close(fig)
    def markdown(table):
        table = table.round(5)
        return "\n".join([
            "| " + " | ".join(table.columns) + " |",
            "| " + " | ".join("---" for _ in table.columns) + " |",
            *["| " + " | ".join(str(x) for x in row) + " |"
              for row in table.itertuples(index=False, name=None)],
        ])

    overall = pairs.groupby("athlete_id")[["error_direct", "error_two_stage", "delta_error"]].mean()
    best = summary.loc[summary.delta_mae_athlete.idxmin()]
    lines = ["# E13 — Two-stage benefit by subgroup", "",
             "All results: **PILOT MEASURED EXPLORATORY**, FitRec. Seed: " + str(config["seed"]), "",
             "Protocol: docs/E13_PROTOCOL.md. Reuses frozen E12 predictions; no new model fitting.",
             f"Paired cohort: {len(pairs)} sessions / {pairs.athlete_id.nunique()} athletes.",
             "Groups describe the athlete's state before each target; one athlete may occur in several groups.",
             "Continuous cutoffs are training-fold athlete-weighted tertiles. Baseline reconstructed with training-only population mean.",
             "Primary: equal athlete weights. Secondary: equal session weights. Negative delta favours Two-stage.",
             "Pointwise bootstrap CIs; centered-bootstrap p-values are approximate. BH applied separately to subgroup and interaction families.",
             "A significant subgroup and a nonsignificant subgroup do not establish heterogeneity: use the interaction contrasts.", "",
             "## Interpretation", "",
             f"Overall athlete-weighted delta = {overall.delta_error.mean():.6f} bpm/h (PILOT MEASURED); reconciles with E12.",
             f"Supported subgroup gains after BH FDR: {int(summary.supported_gain.sum())} (PILOT MEASURED).",
             f"Largest point-estimate reduction: {best.factor} / {best.group}, delta {best.delta_mae_athlete:.4f} bpm/h, "
             f"95% CI [{best.ci_low:.4f}, {best.ci_high:.4f}], BH q={best.q_bh:.4f} (PILOT MEASURED).",
             "No group can currently be recommended as a confirmed beneficiary. A larger history does not guarantee better two-stage performance.",
             "Athlete-weighted and session-weighted results can differ because prolific athletes contribute more to session-weighted estimates.",
             "Comparisons use the E12 minimum-three-history model throughout. The ≥8 subgroup is not the separately fitted E12 minimum-eight sensitivity model.",
             "Subgroup counts are not distinct athlete cohorts: a runner can move from one history band to another.",
             "A training tertile uses equal-athlete mean feature values, so held-out session counts need not divide into thirds.", "",
             "## Subgroups", "", markdown(summary), "",
             "## Between-group contrasts", "", markdown(contrasts), "",
             "These secondary results cannot isolate architecture from added history information or confirm external generalisation.",
             "![Subgroup forest plot](../../figures/e13_subgroup_forest.png)"]
    repo_path(config, outputs["report"]).write_text("\n".join(lines)+"\n", encoding="utf-8")
    logging.info("E13 complete: %s paired sessions, %s athletes", len(pairs), pairs.athlete_id.nunique())
    return summary, contrasts
