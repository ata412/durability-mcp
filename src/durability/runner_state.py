"""E29: runner state + pretrained encoder + guarded linear-leaf Huber LightGBM. Protocol: docs/E29_PROTOCOL.md.

Every number is PILOT MEASURED EXPLORATORY. The GoldenCheetah holdout is the only independent evidence here.
"""
from __future__ import annotations

import logging
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lightgbm import LGBMRegressor
from statsmodels.stats.multitest import multipletests

from durability.config import repo_path
from durability.evaluate import assert_group_split_integrity, regression_metrics, shuffle_targets_within_athlete
from durability.goldencheetah import download_holdout, process_athlete
from durability.level_change import markdown_table
from durability.models.tier1_gbdt import make_gbdt_model
from durability.preprocess import reload_preprocessed_stream
from durability.quality_v2 import tier_masks
from durability.rich_early import STREAM_COLUMNS, build_frame, feature_sets, huber_alpha, within_metrics
from durability.runner_state_features import athlete_folds, cap_runs, guard, nested_subset, residual_row, sequence_row, stacking_key, state_row
from durability.two_stage_subgroups import bootstrap_weights, inference, weighted_bootstrap

LOGGER = logging.getLogger(__name__)
PROVENANCE = "PILOT MEASURED EXPLORATORY"
KEYS = ["activity_id", "athlete_id"]


# --------------------------------------------------------------------------- prepare stage (no torch)
def _sequence_job(row, config):
    stream = reload_preprocessed_stream(config, row.athlete_id, row.activity_id)
    made = sequence_row(stream, row.hrmax_bpm, row.cs_ms, config)
    return None if made is None else (row.activity_id, row.athlete_id, *made)


def _athlete_job(path, cache, config):
    """One athlete, saved to disk so a long run can resume. A failure drops the athlete and is counted."""
    target = cache / f"{path.stem}.pkl"
    if target.exists():
        return pd.read_pickle(target)
    try:
        result = process_athlete(path, config)
    except Exception as exc:  # noqa: BLE001 - one bad athlete must not end a run of thousands
        LOGGER.warning("E29 athlete %s failed: %s", path.stem, exc)
        result = {"athlete_id": path.stem, "gated": [], "labels": [], "sequences": [], "counts": {"runs": 0, "accepted": 0, "computed": 0, "qc": 0, "gate": 0}, "error": repr(exc)[:80]}
    partial = target.with_suffix(".part")
    pd.to_pickle(result, partial, compression=None)
    partial.rename(target)
    return result


def run_prepare_stage(config: dict[str, Any]) -> None:
    settings = config["runner_state"]
    work = repo_path(config, settings["outputs"]["work_dir"])
    work.mkdir(parents=True, exist_ok=True)
    if not (work / "fitrec_seq.npz").exists():
        _prepare_fitrec(config, settings, work)
    _prepare_holdout(config, settings, work)


def _prepare_fitrec(config, settings, work) -> None:
    frame, sessions = build_frame(config)
    sessions = sessions.copy()
    jobs = int(config["evaluate"]["n_jobs"])
    wanted = sessions[sessions.duration_s >= float(settings["encoder"]["min_duration_s"])]
    made = [m for m in Parallel(n_jobs=jobs, batch_size=64)(delayed(_sequence_job)(r, config) for r in wanted[KEYS + ["hrmax_bpm", "cs_ms"]].itertuples()) if m]
    ids = np.array([m[0] for m in made])
    table = pd.DataFrame({"activity_id": ids}).merge(sessions, on="activity_id", validate="one_to_one")
    folds = frame.drop_duplicates("athlete_id").set_index("athlete_id").outer_fold
    clip = config["rich_early"]["label_clip"]
    labels = np.column_stack([np.clip(table.drift, *clip["drift"]), np.clip(table.magnitude, *clip["magnitude"]),
                              np.clip(table.retention, *settings["encoder"]["retention_clip"]), np.array([m[4] for m in made], dtype=float)]).astype(np.float32)
    np.savez(work / "fitrec_seq.npz", ids=ids, athlete=table.athlete_id.to_numpy(), fold=table.athlete_id.map(folds).to_numpy(float),
             x=np.stack([m[2] for m in made]), ld=np.array([m[3] for m in made], np.float32), y=labels)
    table[["activity_id", "athlete_id", "start_time", "end_time", "drift", "magnitude", "gate_pass"]].to_parquet(work / "fitrec_labels.parquet", index=False)
    LOGGER.info("E29 FitRec sequences: %s", len(ids))


def _prepare_holdout(config, settings, work) -> None:
    rules = settings["goldencheetah"]
    manifest = download_holdout(config)
    raw = repo_path(config, rules["raw_dir"])
    cache = work / "gc_cache"
    cache.mkdir(exist_ok=True)
    results = Parallel(n_jobs=int(rules["process_workers"]), batch_size=1)(delayed(_athlete_job)(raw / name, cache, config) for name in manifest.name)
    gated = pd.DataFrame([row for r in results for row in r["gated"]])
    labels_gc = pd.DataFrame([row for r in results for row in r["labels"]])
    seq = [item for r in results for item in r["sequences"]]
    counts = pd.DataFrame([r["counts"] for r in results]).sum().to_dict()
    per_athlete = gated.groupby("athlete_id").size() if len(gated) else pd.Series(dtype=int)
    cohort = {**counts, "athletes_sampled": len(pd.read_csv(repo_path(config, rules["manifest"]))), "athletes_downloaded": len(results), "athletes_with_gated_runs": len(per_athlete), "zip_errors": sum("error" in r for r in results),
              "median_gated_runs_per_athlete": per_athlete.median(), "largest_athlete_gated_runs": per_athlete.max()}
    LOGGER.info("E29 holdout cohort: %s", cohort)
    assert len(gated), "E29 holdout has no gated runs"
    path = repo_path(config, settings["outputs"]["cohort"])
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{**cohort, "provenance": "MEASURED"}]).to_csv(path, index=False)
    gated.to_pickle(work / "gc_gated.pkl")
    labels_gc.to_parquet(work / "gc_labels.parquet", index=False)
    np.savez(work / "gc_seq.npz", ids=np.array([s[0] for s in seq]), x=np.stack([s[1] for s in seq]), ld=np.array([s[2] for s in seq], np.float32))


# --------------------------------------------------------------------------- evaluate stage (no torch)
def _state_table(sessions, config):
    def one(group):
        group = group.reset_index(drop=True)
        return [state_row(group, int(i), config) for i in np.flatnonzero(group.gate_pass.to_numpy())]
    parts = Parallel(n_jobs=int(config["evaluate"]["n_jobs"]))(delayed(one)(g) for _, g in sessions.groupby("athlete_id"))
    return pd.DataFrame([row for part in parts for row in part])


def _residuals(labels, ids, predicted, athletes, config):
    """Encoder errors for gated rows of ``athletes``, with a network that never saw them."""
    clip = config["rich_early"]["label_clip"]
    order = labels.set_index("activity_id").reindex(ids)
    start, end = order.start_time.values, order.end_time.values
    truth = np.column_stack([np.clip(order.drift.to_numpy(float), *clip["drift"]), np.clip(order.magnitude.to_numpy(float), *clip["magnitude"])])
    gate, person = order.gate_pass.to_numpy(bool), order.athlete_id.to_numpy()
    rows = []
    for athlete in athletes:
        index = np.flatnonzero(person == athlete)
        index = index[np.argsort(start[index], kind="stable")]
        rows += [residual_row(index, p, start, end, truth, predicted, ids[i], config) for p, i in enumerate(index) if gate[i]]
    return pd.DataFrame(rows)


def _fit_predict(train_x, train_y, test_x, config):
    rules = config["runner_state"]["model"]
    alpha = huber_alpha(train_y, float(rules["huber_quantile"]))
    return np.mean([LGBMRegressor(objective="huber", alpha=alpha, verbosity=-1, n_jobs=int(config["evaluate"]["n_jobs"]), random_state=int(config["seed"]) + i, **rules["lightgbm"]).fit(train_x, train_y).predict(test_x)
                    for i in range(int(rules["bag_seeds"]))], axis=0)


def _e28_bag(train_x, train_y, test_x, config):
    rules = config["rich_early"]
    alpha = huber_alpha(train_y, float(rules["huber_quantile"]))
    return np.mean([LGBMRegressor(objective="huber", alpha=alpha, verbosity=-1, n_jobs=int(config["evaluate"]["n_jobs"]), random_state=int(config["seed"]) + i, **rules["lightgbm"]).fit(train_x, train_y).predict(test_x)
                    for i in range(int(rules["bag_seeds"]))], axis=0)


class _Stack:
    """Cross-fitted encoder outputs and errors for FitRec rows."""

    def __init__(self, frame, config):
        work = repo_path(config, config["runner_state"]["outputs"]["work_dir"])
        store = np.load(work / "fitrec_out.npz", allow_pickle=True)
        self.keys = [tuple(int(v) for v in k.split("_")) if k else () for k in store["keys"]]
        ids = store["ids"]
        labels = pd.read_parquet(work / "fitrec_labels.parquet")
        fold = frame.drop_duplicates("athlete_id").set_index("athlete_id").outer_fold
        people = labels.set_index("activity_id").reindex(ids).athlete_id
        self.frame = frame
        self.fold = frame.outer_fold.to_numpy()
        position = pd.Series(np.arange(len(ids)), index=ids)
        where = position.reindex(frame.activity_id).to_numpy(float)
        assert np.isfinite(where).all(), "every gated FitRec run needs a sequence"
        self.where = where.astype(int)
        self.out = {key: store["out"][i] for i, key in enumerate(self.keys)}
        self.resid = {}
        for key in self.keys:
            if key:
                athletes = people[people.map(fold).isin(key)].unique()
                table = _residuals(labels, ids, self.out[key], athletes, config).set_index("activity_id")
                self.resid[key] = table
        self.resid_columns = sorted({c for t in self.resid.values() for c in t.columns})

    def features(self, test_fold):
        pred = np.full((len(self.frame), 2), np.nan)
        extra = pd.DataFrame(np.nan, index=self.frame.index, columns=self.resid_columns)
        ids = self.frame.activity_id.to_numpy()
        for fold in range(5):
            key = stacking_key(test_fold, fold)
            mask = self.fold == fold
            pred[mask] = self.out[key][self.where[mask]]
            extra.loc[mask, self.resid_columns] = self.resid[key].reindex(ids[mask])[self.resid_columns].to_numpy()
        return pd.DataFrame(pred, columns=["p_d", "p_m"], index=self.frame.index).join(extra)

    def training_features(self):
        """For a model on all FitRec: each row uses the network that excluded its own fold."""
        pred = np.full((len(self.frame), 2), np.nan)
        extra = pd.DataFrame(np.nan, index=self.frame.index, columns=self.resid_columns)
        ids = self.frame.activity_id.to_numpy()
        for fold in range(5):
            mask = self.fold == fold
            pred[mask] = self.out[(fold,)][self.where[mask]]
            extra.loc[mask, self.resid_columns] = self.resid[(fold,)].reindex(ids[mask])[self.resid_columns].to_numpy()
        return pd.DataFrame(pred, columns=["p_d", "p_m"], index=self.frame.index).join(extra)


def _clip(values, target, config):
    name = "magnitude" if target == "target_magnitude" else "drift"
    return np.clip(values, *config["rich_early"]["label_clip"][name])


def _e29_cv(frame, stack, columns, y_raw, target, config):
    quantiles = tuple(config["runner_state"]["model"]["guard_quantiles"])
    clipped = _clip(y_raw, target, config)
    out = np.full(len(frame), np.nan)
    fold = frame.outer_fold.to_numpy()
    for test_fold in range(5):
        train, test = np.flatnonzero(fold != test_fold), np.flatnonzero(fold == test_fold)
        assert_group_split_integrity(frame, train, test, group_col="athlete_id")
        full = frame[columns].astype(float).join(stack.features(test_fold))
        a, b = guard(full.iloc[train], full.iloc[test], quantiles=quantiles)
        out[test] = _clip(_fit_predict(a, clipped[train], b, config), target, config)
    return out


def _audit(sessions, labels, config, rng):
    """Mutating anything from the target onward must not change a state, sequence or error input."""
    n = int(config["runner_state"]["audit_runs"])
    gate = sessions[sessions.gate_pass]
    sample = gate.sample(n, random_state=int(config["seed"]))
    state_changed, seq_changed, seq_bite, resid_changed, tested = 0, 0, 0, 0, 0
    frac = float(config["features"]["early_window_frac"])
    for row in sample[KEYS + ["hrmax_bpm", "cs_ms"]].itertuples():
        group = sessions[sessions.athlete_id == row.athlete_id].reset_index(drop=True)
        position = int(np.flatnonzero(group.activity_id == row.activity_id)[0])
        base = state_row(group, position, config)
        mutated = group.copy()
        later = (mutated.end_time >= mutated.start_time.iat[position]).to_numpy() & (np.arange(len(group)) != position)
        for column in config["runner_state"]["state_columns"]:
            mutated.loc[later, column] = rng.normal(size=later.sum()) * 50
        new = state_row(mutated, position, config)
        state_changed += int(set(base) != set(new) or any(k != "activity_id" and not np.isclose(float(base[k]), float(new[k]), equal_nan=True) for k in base))
        stream = reload_preprocessed_stream(config, row.athlete_id, row.activity_id)
        made = sequence_row(stream, row.hrmax_bpm, row.cs_ms, config)
        if made is not None:
            changed = stream.copy()
            elapsed = changed.elapsed_s.to_numpy(float)
            after = (elapsed - elapsed[0]) / (elapsed[-1] - elapsed[0]) > frac
            for column in ("heart_rate", "gap_speed_ms", "gradient"):
                changed[column] = changed[column].astype(float)
                changed.loc[after, column] = rng.uniform(1, 200, after.sum())
            again = sequence_row(changed, row.hrmax_bpm, row.cs_ms, config)
            seq_changed += int(not np.array_equal(made[0], again[0]))
            seq_bite += int(not np.allclose(np.array(made[2], float), np.array(again[2], float), equal_nan=True))
        idx = np.arange(len(group))
        start, end = group.start_time.values, group.end_time.values
        truth = np.column_stack([group.drift.to_numpy(float), group.magnitude.to_numpy(float)])
        pred = np.zeros_like(truth)
        base_r = residual_row(idx, position, start, end, truth, pred, row.activity_id, config)
        truth2, pred2 = truth.copy(), pred.copy()
        future = end >= start[position]
        truth2[future] = rng.normal(size=truth2[future].shape) * 50
        pred2[future] = rng.normal(size=pred2[future].shape) * 50
        new_r = residual_row(idx, position, start, end, truth2, pred2, row.activity_id, config)
        resid_changed += int(any(k != "activity_id" and not np.isclose(float(base_r[k]), float(new_r[k]), equal_nan=True) for k in base_r))
        tested += 1
    return pd.DataFrame([{"check": "state_future_mutated", "runs": tested, "runs_with_changed_inputs": state_changed},
                         {"check": "sequence_after_cutoff_mutated", "runs": tested, "runs_with_changed_inputs": seq_changed, "mutation_changed_full_run_label": seq_bite},
                         {"check": "encoder_errors_future_mutated", "runs": tested, "runs_with_changed_inputs": resid_changed}]).assign(provenance="MEASURED")


def _summary_rows(data, scope, tier):
    rows = []
    for (target, model), part in data.groupby(["target", "model"]):
        error = (part.observed - part.predicted).abs().groupby(part.athlete_id).mean()
        rows.append({"scope": scope, "tier": tier, "target": target, "model": model, **regression_metrics(part.observed, part.predicted), "mae_athlete": error.mean(), "n_athletes": len(error),
                     **within_metrics(part.observed.to_numpy(), part.predicted.to_numpy(), part.athlete_id.to_numpy()), "provenance": PROVENANCE})
    return rows


def _contrasts(predictions, config):
    settings = config["runner_state"]
    outside = predictions[predictions.scope.eq("holdout_gc")]
    inside = predictions[predictions.scope.eq("fitrec") & predictions.outer_fold.isin(settings["holdout_folds"])]
    plan = [("H1", outside, "target_drift", "e29", "e28_huber"), ("H2", outside, "target_magnitude", "e29", "e28_huber"),
            ("H3", outside, "target_drift", "e29", "production_lgbm"), ("H4", outside, "target_magnitude", "e29", "production_lgbm"),
            ("context_strict_future", outside, "strict_future_drift", "e29", "e28_huber"), ("context_strict_future", outside, "strict_future_drift", "e29", "production_lgbm")]
    for target in settings["targets"]:
        plan += [("context_fitrec_3_4", inside, target, "e29", "e28_huber"), ("context_fitrec_3_4", inside, target, "e29", "production_lgbm"),
                 ("context_fitrec_3_4_q1", inside[inside.in_q1], target, "e29", "e28_huber"), ("context_fitrec_3_4_q1", inside[inside.in_q1], target, "e29", "production_lgbm")]
    rows = []
    for hypothesis, data, target, challenger, reference in plan:
        part = data[data.target.eq(target)]
        errors = {m: (p.observed - p.predicted).abs().groupby(p.athlete_id).mean() for m, p in part.groupby("model")}
        athletes = sorted(part.athlete_id.unique())
        weights = bootstrap_weights(len(athletes), int(settings["bootstrap_n"]), int(config["seed"]))
        delta = errors[challenger] - errors[reference]
        draws = weighted_bootstrap(delta, athletes, weights)
        rows.append({"hypothesis": hypothesis, "target": target, "challenger": challenger, "reference": reference, "delta_mae_athlete": float(delta.mean()),
                     **inference(delta.mean(), draws, float(settings["confidence_level"])), "n_athletes": len(athletes), "provenance": PROVENANCE})
    table = pd.DataFrame(rows)
    table["q_bh"] = multipletests(table.p_bootstrap, method="fdr_bh")[1]
    table["meets_gain_criterion"] = table.ci_high.lt(0) & table.q_bh.le(float(settings["fdr_alpha"]))
    return table


def _learning_curve(frame, train_full, gated, test_full, config):
    """Secondary and exploratory: FitRec plus a growing share of GoldenCheetah athletes from the other folds."""
    settings = config["runner_state"]
    rules, seed = settings["learning_curve"], int(config["seed"])
    quantiles = tuple(settings["model"]["guard_quantiles"])
    cap = int(settings["run_cap_per_athlete"])
    folds = athlete_folds(gated.athlete_id, int(rules["n_folds"]), seed)
    fold = gated.athlete_id.map(folds).to_numpy()
    inside = gated.index.isin(cap_runs(gated, cap, seed).index)
    people = gated.athlete_id.to_numpy()
    rows, errors = [], {}
    for target in settings["targets"]:
        clip_all = _clip(frame[target].to_numpy(float), target, config)
        clip_gc = _clip(gated[target].to_numpy(float), target, config)
        observed = gated[target].to_numpy(float)
        for fraction in rules["fractions"]:
            predicted, used_rows = np.full(len(gated), np.nan), []
            for k in range(int(rules["n_folds"])):
                test = fold == k
                chosen = nested_subset(people[~test], float(fraction), seed + k)
                add = np.flatnonzero(np.isin(people, chosen) & inside)
                assert not set(people[add]) & set(people[test]), "learning-curve athlete overlap"
                a, b = guard(pd.concat([train_full, test_full.iloc[add]], ignore_index=True), test_full[test], quantiles=quantiles)
                predicted[test] = _clip(_fit_predict(a, np.r_[clip_all, clip_gc[add]], b, config), target, config)
                used_rows.append(len(add))
            errors[target, fraction] = pd.Series(np.abs(observed - predicted)).groupby(people).mean()
            rows.append({"target": target, "fraction_of_other_folds": fraction, "mean_gc_training_rows": float(np.mean(used_rows)), "mae_athlete": errors[target, fraction].mean(),
                         "r_squared_all": regression_metrics(observed, predicted)["r_squared"], "r_squared_capped": regression_metrics(observed[inside], predicted[inside])["r_squared"]})
            LOGGER.info("E29 learning curve %s", rows[-1])
    table = pd.DataFrame(rows)
    athletes = sorted(gated.athlete_id.unique())
    weights = bootstrap_weights(len(athletes), int(settings["bootstrap_n"]), seed)
    low, high = min(rules["fractions"]), max(rules["fractions"])
    for target in settings["targets"]:
        delta = errors[target, high] - errors[target, low]
        result = inference(delta.mean(), weighted_bootstrap(delta, athletes, weights), float(settings["confidence_level"]))
        table.loc[(table.target == target) & (table.fraction_of_other_folds == high), ["delta_vs_fitrec_only", "ci_low", "ci_high"]] = [float(delta.mean()), result["ci_low"], result["ci_high"]]
    table["provenance"] = PROVENANCE
    return table


def run_evaluate_stage(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    settings = config["runner_state"]
    work = repo_path(config, settings["outputs"]["work_dir"])
    outputs = settings["outputs"]
    frame, sessions = build_frame(config)
    sets = feature_sets(frame, config)
    state = _state_table(sessions, config)
    frame = frame.merge(state, on="activity_id", how="left", validate="one_to_one")
    state_cols = [c for c in state if c.startswith("s_")]
    columns = sets["core"] + state_cols
    rng = np.random.default_rng(int(config["seed"]))
    labels = pd.read_parquet(work / "fitrec_labels.parquet")
    audit = _audit(sessions, labels, config, rng)
    LOGGER.info("E29 mutation audit:\n%s", audit.to_string(index=False))
    if audit.runs_with_changed_inputs.sum() > 0:
        raise AssertionError("E29 halted: an input changed after future data was mutated")
    stack = _Stack(frame, config)
    in_q1 = tier_masks(frame, config)["Q1"].to_numpy()
    frame["strict_future_drift"] = frame.post_drift
    long_rows = []

    def keep(model, target, values, scope, source, flags):
        part = source[KEYS + ["outer_fold"]].copy()
        part["scope"], part["model"], part["target"], part["in_q1"] = scope, model, target, flags
        part["observed"], part["predicted"], part["provenance"] = source[target].to_numpy(float), values, PROVENANCE
        long_rows.append(part)

    e28 = pd.read_parquet(repo_path(config, config["rich_early"]["outputs"]["predictions"]))
    e28["activity_id"] = e28.activity_id.astype(str)
    checks = []
    for target in settings["targets"]:
        values = _e29_cv(frame, stack, columns, frame[target].to_numpy(float), target, config)
        keep("e29", target, values, "fitrec", frame, in_q1)
        for model, name in (("huber_lgbm", "e28_huber"), ("production_lgbm", "production_lgbm")):
            stored = e28[(e28.target == target) & (e28.model == model)].set_index("activity_id").predicted.reindex(frame.activity_id).to_numpy()
            keep(name, target, stored, "fitrec", frame, in_q1)
        r2 = regression_metrics(frame[target].to_numpy(float), values)["r_squared"]
        reference = float(settings["lab_reference_r2"][target])
        checks.append({"target": target, "e29_all_fold_r2": r2, "lab_r2": reference})
        LOGGER.info("E29 reproduction %s", checks[-1])
        if abs(r2 - reference) > float(settings["reproduction_tol"]):
            raise AssertionError(f"E29 reproduction check failed: {checks[-1]}")

    # external holdout: train on all FitRec, no recalibration
    gated = pd.read_pickle(work / "gc_gated.pkl")
    gated["target_drift"], gated["target_magnitude"] = gated.drift.astype(float), gated.magnitude.astype(float)
    gated["strict_future_drift"] = gated.post_drift.astype(float)
    store = np.load(work / "gc_out.npz", allow_pickle=True)
    gc_labels = pd.read_parquet(work / "gc_labels.parquet")
    gc_ids, gc_out = store["ids"], store["out"]
    keep_ids = gc_labels.activity_id.isin(gc_ids)
    resid = _residuals(gc_labels[keep_ids], gc_ids, gc_out, gc_labels.athlete_id.unique(), config).set_index("activity_id")
    gated = gated[gated.activity_id.isin(gc_ids)].reset_index(drop=True)
    pos = pd.Series(np.arange(len(gc_ids)), index=gc_ids).reindex(gated.activity_id).to_numpy(int)
    for column in columns + sets["manifest"]:
        if column not in gated:
            gated[column] = np.nan
    extra = pd.DataFrame(gc_out[pos], columns=["p_d", "p_m"], index=gated.index).join(resid.reindex(gated.activity_id)[stack.resid_columns].set_index(gated.index))
    train_full = frame[columns].astype(float).join(stack.training_features())
    test_full = gated[columns].astype(float).join(extra)
    quantiles = tuple(settings["model"]["guard_quantiles"])
    a, b = guard(train_full, test_full, quantiles=quantiles)
    flags = np.zeros(len(gated), dtype=bool)
    gated["outer_fold"] = -1
    seed = int(config["seed"])
    for target in [*settings["targets"], "strict_future_drift"]:
        usable = frame[target].notna().to_numpy()
        short = "m" if target == "target_magnitude" else "d"
        y = frame[target].to_numpy(float)
        clipped = _clip(y, target, config)
        keep("e29", target, _clip(_fit_predict(a[usable], clipped[usable], b, config), target, config), "holdout_gc", gated, flags)
        keep("e28_huber", target, _e28_bag(frame.loc[usable, sets["core"]].astype(float), clipped[usable], gated[sets["core"]].astype(float), config), "holdout_gc", gated, flags)
        production = make_gbdt_model("tier1_lightgbm", config, seed=seed).fit(frame.loc[usable, sets["manifest"]].to_numpy(float), y[usable])
        keep("production_lgbm", target, production.predict(gated[sets["manifest"]].to_numpy(float)), "holdout_gc", gated, flags)
        population = clipped[usable].mean()
        keep("population_mean", target, np.full(len(gated), population), "holdout_gc", gated, flags)
        count, mean = gated.h_gate_n.fillna(0).to_numpy(float), gated[f"h_gate_{short}_mean"].astype(float).fillna(population).to_numpy()
        keep("shrunk_history", target, (count * mean + float(config["rich_early"]["shrinkage_strength"]) * population) / (count + float(config["rich_early"]["shrinkage_strength"])), "holdout_gc", gated, flags)
        LOGGER.info("E29 holdout predicted %s", target)
    predictions = pd.concat(long_rows, ignore_index=True)
    assert predictions.predicted.notna().all()

    # probe
    shuffled = shuffle_targets_within_athlete(frame, list(settings["targets"]), seed=seed)
    holdout = frame.outer_fold.isin(settings["holdout_folds"]).to_numpy()
    probe = []
    for target in settings["targets"]:
        y = shuffled[target].to_numpy(float)
        estimate = _e29_cv(shuffled, stack, columns, y, target, config)
        for scope, mask in (("all_folds", np.ones(len(frame), bool)), ("fitrec_3_4", holdout)):
            metrics = regression_metrics(y[mask], estimate[mask])
            limit = float(config["evaluate"]["leakage_r2_max"])
            probe.append({"target": target, "scope": scope, "r_squared": metrics["r_squared"], **within_metrics(y[mask], estimate[mask], frame.athlete_id.to_numpy()[mask]), "limit": limit, "exceeds_limit": metrics["r_squared"] > limit, "provenance": "MEASURED WITHIN-ATHLETE SHUFFLED TARGET"})
    probe = pd.DataFrame(probe)
    LOGGER.info("E29 probe:\n%s", probe.to_string(index=False))
    status = {t: bool(p.exceeds_limit.eq(False).all()) for t, p in probe.groupby("target")}

    rows = _summary_rows(predictions[predictions.scope.eq("holdout_gc")], "holdout_gc", "all")
    clip = config["rich_early"]["label_clip"]
    outside = predictions[predictions.scope.eq("holdout_gc")]
    in_range = outside[np.where(outside.target.eq("target_magnitude"), outside.observed.between(*clip["magnitude"]), outside.observed.between(*clip["drift"]))]
    rows += _summary_rows(in_range, "holdout_gc", "in_training_label_range")
    capped_ids = set(cap_runs(gated, int(settings["run_cap_per_athlete"]), seed).activity_id)
    rows += _summary_rows(outside[outside.activity_id.isin(capped_ids)], "holdout_gc", "capped_runs_per_athlete")
    inner = predictions[predictions.scope.eq("fitrec")]
    rows += _summary_rows(inner, "fitrec_all_folds", "all") + _summary_rows(inner[inner.outer_fold.isin(settings["holdout_folds"])], "fitrec_3_4", "all")
    summary = pd.DataFrame(rows)
    contrasts = _contrasts(predictions, config)
    parent = {"target_drift": "target_drift", "target_magnitude": "target_magnitude", "strict_future_drift": "target_drift"}
    contrasts["probe_within_limit"] = contrasts.target.map(parent).map(status)
    contrasts["established_gain"] = contrasts.meets_gain_criterion & contrasts.probe_within_limit
    curve = _learning_curve(frame, train_full, gated, test_full, config)
    tables = {"predictions": predictions, "summary": summary, "contrasts": contrasts, "leakage": probe, "audit": audit, "learning_curve": curve}
    for name, table in tables.items():
        path = repo_path(config, outputs[name])
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(path, index=False) if path.suffix == ".parquet" else table.to_csv(path, index=False)
    _figure(summary, config)
    _report(summary, contrasts, probe, audit, pd.DataFrame(checks), status, len(columns), curve, config)
    return tables


def _figure(summary, config):
    order = ["population_mean", "shrunk_history", "production_lgbm", "e28_huber", "e29"]
    names = {"population_mean": "Population mean", "shrunk_history": "Shrunk history", "production_lgbm": "Production LightGBM", "e28_huber": "E28", "e29": "E29"}
    data = summary[(summary.scope == "holdout_gc") & (summary.tier == "in_training_label_range")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, (target, title) in zip(axes, [("target_drift", "Drift"), ("target_magnitude", "Magnitude")], strict=True):
        part = data[data.target == target].set_index("model").reindex(order)
        ax.barh(range(len(order)), part.r_squared.clip(lower=0), height=0.55, color=["#2a78d6" if m == "e29" else "#8c8c8c" for m in order])
        for i, value in enumerate(part.r_squared):
            ax.text(max(value, 0) + 0.008, i, f"{value:.3f}", va="center", fontsize=9, color="#404040")
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_yticks(range(len(order)), [names[m] for m in order], fontsize=9)
        ax.set_xlabel("R², runs inside training label range (negative drawn at 0)", fontsize=9)
        ax.set_xlim(0, 0.6)
        ax.set_xticks(np.arange(0, 0.61, 0.1))
        ax.grid(axis="x", color="#e5e5e5", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", length=0)
    fig.suptitle("E29 on unseen GoldenCheetah runners, PILOT MEASURED EXPLORATORY", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    path = repo_path(config, config["runner_state"]["outputs"]["figure"])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _report(summary, contrasts, probe, audit, checks, status, n_columns, curve, config):
    def table(frame, digits=4):
        return markdown_table(frame.round(digits))
    columns = ["target", "model", "mae", "mae_athlete", "r_squared", "spearman", "r2_within", "spearman_within", "n", "n_athletes"]
    view = lambda scope, tier: table(summary[(summary.scope == scope) & (summary.tier == tier)].sort_values(["target", "mae"])[columns])
    cohort = pd.read_csv(repo_path(config, config["runner_state"]["outputs"]["cohort"]))
    verdict = "\n".join(f"- {t}: shuffled probe {'within the limit' if ok else 'ABOVE the limit, so no gain may be called established'}." for t, ok in status.items())
    lines = ["# E29: Runner-state features and a pretrained sequence encoder", "",
             f"All results are **PILOT MEASURED EXPLORATORY**. Protocol: docs/E29_PROTOCOL.md. Seed {config['seed']} (CONFIGURED). Tabular inputs: {n_columns}, plus encoder outputs and errors.", "",
             "This revision replaces the first E29 run of 300 athletes, archived in `archive/E29_v1_300_athletes/`. It is a second test of the same hypotheses, on athletes not used before.", "",
             "The GoldenCheetah holdout is the only independent evidence in this report. FitRec folds were opened during design.", "",
             "## Probe status", "", verdict, "",
             "## Holdout cohort (GoldenCheetah, never seen)", "", table(cohort), "",
             "## Holdout: all gated runs", "", view("holdout_gc", "all"), "",
             "## Holdout: runs inside the training label range", "", view("holdout_gc", "in_training_label_range"), "",
             f"## Holdout: at most {config['runner_state']['run_cap_per_athlete']} runs per athlete", "", view("holdout_gc", "capped_runs_per_athlete"), "",
             "## Contrasts (athlete-weighted MAE, BH across all contrasts)", "", table(contrasts.drop(columns=["provenance"])), "",
             "## Secondary: training on GoldenCheetah too (exploratory, outside the BH family)", "", table(curve.drop(columns=["provenance"])), "",
             "## FitRec folds 3 and 4 (already opened, descriptive)", "", view("fitrec_3_4", "all"), "",
             "## FitRec all folds (descriptive)", "", view("fitrec_all_folds", "all"), "",
             "## Controls", "", table(audit), "", table(probe), "", table(checks), "",
             "![E29](../../figures/e29_runner_state.png)", "",
             "## Limits", "",
             "- The 40% window requires knowing the run length.",
             "- The cohort is selected on the outcome through the gate, so these are not deployment scores.",
             "- Labels overlap the input between 12% and 40% of the run. `strict_future_drift` removes this overlap for drift.",
             "- GoldenCheetah speed comes from recorded distance rather than GPS.",
             "- Athletes with zips of 150 MB or more are excluded, so the sample is not fully representative.", ""]
    path = repo_path(config, config["runner_state"]["outputs"]["report"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
