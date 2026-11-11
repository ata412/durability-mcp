"""E15-v2: quality dose-response, clean prior history, strongest baselines.

Protocol (locked before fitting): docs/E15V2_PROTOCOL.md. Every number is
PILOT MEASURED EXPLORATORY on FitRec.
"""
from __future__ import annotations

import logging
import warnings
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from durability.config import repo_path
from durability.evaluate import assert_group_split_integrity, regression_metrics, shuffle_targets_within_athlete
from durability.level_change import markdown_table
from durability.models.exploratory import make_exploratory_model
from durability.models.tier1_gbdt import make_gbdt_model
from durability.next_run_prediction import completed_history_mask
from durability.reliability import (
    _group_sufficient_statistics,
    _icc_from_sufficient_statistics,
    reliability_for_index,
)
from durability.two_stage_subgroups import bootstrap_weights, inference, weighted_bootstrap

LOGGER = logging.getLogger(__name__)
PROVENANCE = "PILOT MEASURED EXPLORATORY"
KEYS = ["activity_id", "athlete_id"]


# --------------------------------------------------------------------------- tiers
def tier_masks(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, pd.Series]:
    """Nested quality tiers from config; no duration criterion by design."""
    masks: dict[str, pd.Series] = {}
    for name, rule in config["quality_v2"]["tiers"].items():
        mask = frame["gap_speed_cv"].lt(rule["max_gap_speed_cv"])
        if rule["exclude_gap_split"]:
            mask &= frame["gap_split"].eq(False)
        mask &= frame["clipped_gradient_fraction"].lt(rule["max_clipped_gradient_fraction"])
        masks[name] = mask.fillna(False)
    names = list(masks)
    for lower, higher in zip(names[:-1], names[1:], strict=True):
        assert not (masks[higher] & ~masks[lower]).any(), f"{higher} is not nested in {lower}"
    return masks


# --------------------------------------------------------------------------- prior history
def prior_summaries(frame: pd.DataFrame, history_mask: pd.Series, outcomes: list[str], prefix: str) -> pd.DataFrame:
    """Strictly prior *completed* sessions (end < start), restricted to ``history_mask``."""
    ordered = frame.assign(_hist=history_mask.to_numpy()).copy()
    ordered["start_time"] = pd.to_datetime(ordered["start_time"], utc=True)
    ordered["end_time"] = ordered["start_time"] + pd.to_timedelta(ordered["duration_s"], unit="s")
    ordered = ordered.sort_values(["athlete_id", "start_time", "activity_id"])
    rows = []
    for athlete, group in ordered.groupby("athlete_id", sort=True):
        history: list[dict[str, Any]] = []
        for time, tied in group.groupby("start_time", sort=True):
            prior = [r for r in history if r["end_time"] < time]
            for current in tied.to_dict("records"):
                row = {"activity_id": current["activity_id"], "athlete_id": athlete, f"{prefix}count": len(prior),
                       f"{prefix}max_end_time": max((r["end_time"] for r in prior), default=pd.NaT)}
                for column in outcomes:
                    name = column.removeprefix("target_")
                    values = np.array([r[column] for r in prior], dtype=float)
                    finite = values[np.isfinite(values)]
                    row[f"{prefix}{name}_mean"] = finite.mean() if finite.size else np.nan
                    row[f"{prefix}{name}_sd"] = finite.std(ddof=1) if finite.size > 1 else np.nan
                    row[f"{prefix}{name}_last"] = values[-1] if values.size else np.nan
                rows.append(row)
            history.extend(r for r in tied.to_dict("records") if r["_hist"])
    result = pd.DataFrame(rows)
    return frame[KEYS].merge(result, on=KEYS, how="left", validate="one_to_one")


def shrunk(count: np.ndarray, mean: np.ndarray, population: float, strength: float) -> np.ndarray:
    mean = np.where(np.isfinite(mean), mean, population)
    return (count * mean + strength * population) / (count + strength)


# --------------------------------------------------------------------------- Part A: ICC per tier
def tier_reliability(config: dict[str, Any], rng: np.random.Generator) -> pd.DataFrame:
    settings = config["quality_v2"]
    characteristics = pd.read_parquet(repo_path(config, config["data"]["reliability_characteristics"]))
    gate = characteristics[characteristics["gate_pass"].fillna(False)].reset_index(drop=True)
    masks = tier_masks(gate, config)
    reference = pd.read_csv(repo_path(config, config["outputs"]["reliability"])).set_index("index")
    rows, matched_by = [], {}
    for index in settings["icc_indices"]:
        for tier, mask in masks.items():
            result, matched = reliability_for_index(gate.loc[mask], index, config, rng)
            matched_by[(index, tier)] = matched
            rows.append({"index": index, "tier": tier, "n_sessions_in_tier": int(mask.sum()),
                         "n_athletes": result["n_athletes"], "n_sessions": result["n_sessions"],
                         "icc": result["icc_1_1"], "ci_low": result["ci_low"], "ci_high": result["ci_high"],
                         "label_sd_tier": float(gate.loc[mask, index].std(ddof=1)),
                         "raw_sd_matched": result["raw_sd"], "sem": result["sem"], "provenance": PROVENANCE})
        q0 = next(r for r in rows if r["index"] == index and r["tier"] == "Q0")
        assert abs(q0["icc"] - float(reference.loc[index, "icc_1_1"])) <= settings["icc_reproduction_tol"], (
            f"Q0 {index} ICC {q0['icc']:.4f} does not reproduce production {reference.loc[index, 'icc_1_1']:.4f}")
    table = pd.DataFrame(rows)
    # H1: joint athlete bootstrap of ICC(Q_k) - ICC(Q0) on the matched cohorts.
    tiers = list(masks)
    diffs = []
    for index in settings["icc_indices"]:
        stats = {}
        for tier in tiers:
            matched = matched_by[(index, tier)]
            count, total, squared = _group_sufficient_statistics(matched, "athlete_id", index)
            athletes = matched.dropna(subset=[index]).groupby("athlete_id", sort=False).size().index
            stats[tier] = pd.DataFrame({"count": count, "total": total, "sq": squared}, index=athletes.astype(str))
        union = sorted(set().union(*[set(s.index) for s in stats.values()]))
        draws = {tier: np.empty(int(settings["bootstrap_n"])) for tier in tiers}
        for b in range(int(settings["bootstrap_n"])):
            sample = pd.Index(np.asarray(union)[rng.integers(0, len(union), len(union))])
            for tier in tiers:
                picked = stats[tier].reindex(sample).dropna()
                draws[tier][b] = _icc_from_sufficient_statistics(
                    picked["count"].to_numpy(), picked["total"].to_numpy(), picked["sq"].to_numpy())
        base = table[(table["index"] == index) & (table.tier == "Q0")].icc.iloc[0]
        for tier in tiers[1:]:
            point = table[(table["index"] == index) & (table.tier == tier)].icc.iloc[0] - base
            delta = draws[tier] - draws["Q0"]
            delta = delta[np.isfinite(delta)]
            low, high = np.quantile(delta, [0.025, 0.975])
            diffs.append({"index": index, "tier": tier, "icc_minus_q0": point, "ci_low": low, "ci_high": high,
                          "supported_increase": bool(low > 0), "provenance": PROVENANCE})
    return table, pd.DataFrame(diffs)


# --------------------------------------------------------------------------- Part B: prediction
def _fit_predict(name: str, config: dict[str, Any], x_train, y_train, x_test, weight=None, seed=0):
    model = (make_gbdt_model("tier1_lightgbm", config, seed=seed) if name.startswith("lgbm")
             else make_exploratory_model("elastic_net", config, seed=seed))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if weight is None:
            model.fit(x_train, y_train)
        else:
            model.fit(x_train, y_train, regressor__sample_weight=weight)
    return model.predict(x_test)


def run_prediction(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    settings = config["quality_v2"]
    target = settings["target"]
    strength = float(settings["shrinkage_strength"])
    frame = pd.read_parquet(repo_path(config, config["quality_cohort_prediction"]["outputs"]["cohort"]))
    for key in KEYS:
        frame[key] = frame[key].astype(str)
    frame["start_time"] = pd.to_datetime(frame["start_time"], utc=True)
    assert frame.gate_pass.all() and frame.groupby("athlete_id").outer_fold.nunique().eq(1).all()
    masks = tier_masks(frame, config)
    tiers = list(masks)
    rank = sum(masks[t].astype(int) for t in tiers[1:])  # 0..3 tiers passed above Q0

    manifest = pd.read_csv(repo_path(config, config["outputs"]["feature_manifest"]))
    manifest_features = manifest.loc[manifest.role.eq("feature"), "column"].tolist()
    prior_gate = [c for c in manifest_features if c.startswith("prior_gate_")]
    outcomes = list(settings["prior_outcomes"])

    allhist = prior_summaries(frame, pd.Series(True, index=frame.index), outcomes, "hall_")
    frame = frame.merge(allhist, on=KEYS, validate="one_to_one")
    activities = pd.read_parquet(repo_path(config, str(repo_path(config, config["data"]["interim_dir"]) / "activities.parquet")))
    eligible = frame["hall_count"].ge(1) & completed_history_mask(frame, activities)
    assert (frame.loc[eligible, "hall_max_end_time"] < frame.loc[eligible, "start_time"]).all()

    predictions, probes = [], []
    for tier in tiers:
        clean = prior_summaries(frame, masks[tier], outcomes, "hcl_")
        work = frame.merge(clean, on=KEYS, validate="one_to_one")
        ok = work["hcl_max_end_time"].isna() | (work["hcl_max_end_time"] < work["start_time"])
        assert ok.all(), "clean history must end before target start"
        all_cols = list(dict.fromkeys(manifest_features + [c for c in allhist if c.startswith("hall_") and c != "hall_max_end_time"]))
        clean_cols = [c for c in manifest_features if c not in prior_gate] + [
            c for c in clean if c.startswith("hcl_") and c != "hcl_max_end_time"]
        assert not any(c.startswith(("target_",)) for c in all_cols + clean_cols)
        tier_mask = masks[tier].to_numpy()
        shuffled = shuffle_targets_within_athlete(work, [target], seed=config["seed"])
        for fold in sorted(work.outer_fold.unique()):
            train = np.flatnonzero(work.outer_fold.ne(fold).to_numpy() & eligible.to_numpy())
            test = np.flatnonzero(work.outer_fold.eq(fold).to_numpy() & eligible.to_numpy() & tier_mask)
            assert_group_split_integrity(work, train, test, group_col="athlete_id")
            a, b = work.iloc[train], work.iloc[test]
            y = a[target].to_numpy(dtype=float)
            population = float(y.mean())
            seed = int(config["seed"]) + int(fold)
            strict_train = train[tier_mask[train]]
            LOGGER.info("E15v2 tier=%s fold=%s train=%s strict_train=%s test=%s", tier, fold, len(train), len(strict_train), len(test))
            est = {
                "population_mean": np.full(len(b), population),
                "shrunk_all": shrunk(b.hall_count.to_numpy(float), b.hall_magnitude_mean.to_numpy(float), population, strength),
                "shrunk_clean": shrunk(b.hcl_count.to_numpy(float), b.hcl_magnitude_mean.to_numpy(float), population, strength),
                "en_all": _fit_predict("en", config, a[all_cols], y, b[all_cols], seed=seed),
                "en_clean": _fit_predict("en", config, a[clean_cols], y, b[clean_cols], seed=seed),
                "en_clean_strict": _fit_predict("en", config, work.iloc[strict_train][clean_cols],
                                                work.iloc[strict_train][target].to_numpy(float), b[clean_cols], seed=seed),
                "en_clean_weighted": _fit_predict("en", config, a[clean_cols], y, b[clean_cols],
                                                  weight=1.0 + rank.iloc[train].to_numpy(float), seed=seed),
                "lgbm_all": _fit_predict("lgbm", config, a[all_cols], y, b[all_cols], seed=seed),
                "lgbm_clean": _fit_predict("lgbm", config, a[clean_cols], y, b[clean_cols], seed=seed),
            }
            probe = _fit_predict("en", config, a[clean_cols], shuffled.iloc[train][target].to_numpy(float), b[clean_cols], seed=seed)
            probes.append(pd.DataFrame({"tier": tier, "observed": shuffled.iloc[test][target].to_numpy(), "predicted": probe}))
            for model, values in est.items():
                assert np.isfinite(values).all(), model
                part = b[KEYS + ["outer_fold"]].copy()
                part["tier"], part["model"] = tier, model
                part["observed"], part["predicted"] = b[target].to_numpy(float), np.asarray(values)
                part["provenance"] = PROVENANCE
                predictions.append(part)
    predictions = pd.concat(predictions, ignore_index=True)
    probe = pd.concat(probes, ignore_index=True)
    leakage = pd.DataFrame([{"tier": t, **regression_metrics(p.observed, p.predicted), "provenance": "MEASURED WITHIN-ATHLETE SHUFFLED TARGET"}
                            for t, p in probe.groupby("tier")])
    leakage["halted"] = leakage.r_squared.gt(config["evaluate"]["leakage_r2_max"])
    assert not leakage.halted.any(), "E15v2 shuffled-target leakage rule failed"
    tier_counts = pd.DataFrame([{"tier": t, "n_sessions": int(m.sum()), "n_athletes": frame.loc[m, "athlete_id"].nunique(),
                                 "n_test_rows_eligible": int((m & eligible).sum()),
                                 "median_duration_min": float(frame.loc[m, "duration_s"].median() / 60),
                                 "target_sd": float(frame.loc[m, target].std(ddof=1)), "provenance": "MEASURED COHORT COUNTS"}
                                for t, m in masks.items()])
    return predictions, leakage, tier_counts


def contrasts_table(predictions: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    settings = config["quality_v2"]
    summary, rows = [], []
    for tier, data in predictions.groupby("tier", sort=True):
        athletes = sorted(data.athlete_id.unique())
        weights = bootstrap_weights(len(athletes), settings["bootstrap_n"], config["seed"])
        errors = {}
        for model, part in data.groupby("model"):
            error = (part.observed - part.predicted).abs().groupby(part.athlete_id).mean()
            errors[model] = error
            summary.append({"tier": tier, "model": model, **regression_metrics(part.observed, part.predicted),
                            "mae_athlete": error.mean(), "n_athletes": len(error), "provenance": PROVENANCE})
        pairs = []
        if tier == settings["primary_tier"]:
            pairs += [("en_clean", "shrunk_all", "H2"), ("en_clean_strict", "en_clean", "H5"),
                      ("en_clean_weighted", "en_clean", "H5")]
        if tier != "Q0":
            pairs += [("shrunk_clean", "shrunk_all", "H3"), ("en_clean", "en_all", "H4")]
        pairs += [("en_all", "shrunk_all", "context"), ("lgbm_clean", "shrunk_all", "context"),
                  ("lgbm_clean", "lgbm_all", "context")]
        for challenger, reference, hypothesis in pairs:
            delta = errors[challenger] - errors[reference]
            draws = weighted_bootstrap(delta, athletes, weights)
            rows.append({"tier": tier, "hypothesis": hypothesis, "challenger": challenger, "reference": reference,
                         "delta_mae_athlete": float(delta.mean()), **inference(delta.mean(), draws, settings["confidence_level"]),
                         "n_athletes": len(athletes), "provenance": PROVENANCE})
    contrasts = pd.DataFrame(rows)
    contrasts["q_bh"] = multipletests(contrasts.p_bootstrap, method="fdr_bh")[1]
    contrasts["supported_gain"] = contrasts.ci_high.lt(0) & contrasts.q_bh.le(settings["fdr_alpha"])
    return pd.DataFrame(summary), contrasts


# --------------------------------------------------------------------------- orchestration
def run_experiment(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    settings = config["quality_v2"]
    rng = np.random.default_rng(int(config["seed"]))
    icc, icc_diff = tier_reliability(config, rng)
    LOGGER.info("E15v2 Part A done:\n%s", icc.to_string(index=False))
    predictions, leakage, tiers = run_prediction(config)
    summary, contrasts = contrasts_table(predictions, config)
    tables = {"icc": pd.concat([icc, icc_diff], ignore_index=True), "predictions": predictions, "summary": summary,
              "contrasts": contrasts, "leakage": leakage, "tiers": tiers}
    out = settings["outputs"]
    for name, table in tables.items():
        path = repo_path(config, out[name])
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(path, index=False) if path.suffix == ".parquet" else table.to_csv(path, index=False)
    _figure(icc, summary, config)
    _report(icc, icc_diff, summary, contrasts, leakage, tiers, config)
    return tables


def _figure(icc, summary, config):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for index, colour in [("magnitude", "#0072B2"), ("drift", "#D55E00")]:
        part = icc[icc["index"] == index]
        axes[0].errorbar(part.tier, part.icc, yerr=[part.icc - part.ci_low, part.ci_high - part.icc],
                         marker="o", capsize=4, color=colour, label=index)
    axes[0].set_ylabel("ICC(1,1), matched sessions"); axes[0].set_title("A. Reliability by quality tier")
    axes[0].legend()
    for model, style in [("shrunk_all", "k--"), ("shrunk_clean", "k:"), ("en_all", "C0-"), ("en_clean", "C2-"), ("lgbm_clean", "C3-")]:
        part = summary[summary.model == model].sort_values("tier")
        axes[1].plot(part.tier, part.mae_athlete, style, marker="o", label=model)
    axes[1].set_ylabel("Athlete-weighted MAE, magnitude (lower is better)")
    axes[1].set_title("B. Prediction error by test tier"); axes[1].legend(fontsize=8)
    fig.suptitle("E15-v2 FitRec — PILOT MEASURED EXPLORATORY")
    fig.tight_layout(); fig.savefig(repo_path(config, config["quality_v2"]["outputs"]["figure"]), dpi=180); plt.close(fig)


def _report(icc, icc_diff, summary, contrasts, leakage, tiers, config):
    def r(t, digits=4):
        return markdown_table(t.round(digits))
    primary = contrasts[contrasts.hypothesis.isin(["H2", "H3", "H4", "H5"])]
    lines = ["# E15-v2 — Quality dose-response, clean history, strongest baselines", "",
             "All results: **PILOT MEASURED EXPLORATORY**, FitRec. Protocol locked before fitting: docs/E15V2_PROTOCOL.md. "
             f"Seed={config['seed']}.", "",
             "## Tiers", "", r(tiers, 3), "",
             "## Part A — reliability per tier", "", r(icc), "", "### H1: ICC(tier) − ICC(Q0), joint athlete bootstrap", "", r(icc_diff), "",
             "## Part B — next-run magnitude prediction", "", r(summary.sort_values(["tier", "mae_athlete"])), "",
             "### Hypothesis contrasts (negative delta favours challenger; BH across all Part-B contrasts)", "", r(primary), "",
             "### Context contrasts", "", r(contrasts[contrasts.hypothesis.eq("context")]), "",
             "## Leakage probe (en_clean, within-athlete shuffled target)", "", r(leakage), "",
             "## Limitations", "",
             "Test-session tier membership is retrospective (full-session GAP-speed CV). Only clean-history arms are online-deployable. "
             "Tiers restrict label range (see target_sd), which can lower ICC and R² mechanically. All sessions remain gate-passing.", "",
             "![E15-v2](../../figures/e15v2_quality_dose_response.png)"]
    repo_path(config, config["quality_v2"]["outputs"]["report"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
