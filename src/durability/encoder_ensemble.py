"""E31: two-dataset encoder, frozen tree ensembles and a FiLM network. Protocol: docs/E31_PROTOCOL.md.

Every number is PILOT MEASURED EXPLORATORY. The test athletes were opened by E30, so nothing here is independent evidence.
"""
from __future__ import annotations

import logging
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from statsmodels.stats.multitest import multipletests
from xgboost import XGBRegressor

from durability import runner_state as rs
from durability.config import repo_path
from durability.evaluate import regression_metrics, shuffle_targets_within_athlete
from durability.level_change import markdown_table
from durability.rich_early import build_frame, feature_sets, huber_alpha, within_metrics
from durability.runner_state_features import athlete_folds, cap_runs, guard
from durability.two_stage_subgroups import bootstrap_weights, inference, weighted_bootstrap

LOGGER = logging.getLogger(__name__)
PROVENANCE = "PILOT MEASURED EXPLORATORY, NOT INDEPENDENT"
LABEL = {"target_drift": "drift", "target_magnitude": "magnitude", "strict_future_drift": "drift"}


def forward(values: np.ndarray, kind: str | None) -> np.ndarray:
    if kind == "sqrt":
        return np.sign(values) * np.sqrt(np.abs(values))
    return np.log(values) if kind == "log" else values


def backward(values: np.ndarray, kind: str | None) -> np.ndarray:
    if kind == "sqrt":
        return np.sign(values) * values ** 2
    return np.exp(values) if kind == "log" else values


def transform_kind(member: dict[str, Any], target: str) -> str | None:
    """A transform belongs to one label: signed root to drift, log to magnitude."""
    kind = member.get("target")
    if kind == "sqrt" and target in ("target_drift", "strict_future_drift"):
        return kind
    return kind if kind == "log" and target == "target_magnitude" else None


def blend(predictions: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    return sum(float(w) * predictions[m] for m, w in weights.items()) / sum(float(w) for w in weights.values())


def fit_member(model: str, x, y, x_new, seeds: int, config: dict[str, Any]) -> np.ndarray:
    settings, seed = config["encoder_ensemble"]["models"], int(config["seed"])
    alpha = huber_alpha(y, float(config["runner_state"]["model"]["huber_quantile"]))
    jobs = int(config["evaluate"]["n_jobs"])
    if model == "xgb":
        return np.mean([XGBRegressor(objective="reg:pseudohubererror", huber_slope=alpha, n_jobs=jobs, random_state=seed + i, tree_method="hist", base_score=float(np.median(y)), **settings["xgb"]).fit(x, y).predict(x_new) for i in range(seeds)], axis=0)
    rules = dict(config["runner_state"]["model"]["lightgbm"])
    if model != "lgbm":
        rules.update(settings[model])
    if not rules.get("linear_tree"):
        rules.pop("linear_lambda", None)
    return np.mean([LGBMRegressor(objective="huber", alpha=alpha, verbosity=-1, n_jobs=jobs, random_state=seed + i, **rules).fit(x, y).predict(x_new) for i in range(seeds)], axis=0)


def _gated(paths):
    data = pd.concat([pd.read_pickle(p) for p in paths], ignore_index=True).drop_duplicates("activity_id").reset_index(drop=True)
    data["target_drift"], data["target_magnitude"], data["strict_future_drift"] = data.drift.astype(float), data.magnitude.astype(float), data.post_drift.astype(float)
    return data


class _Data:
    def __init__(self, config):
        settings = config["encoder_ensemble"]
        self.config = config
        frame, sessions = build_frame(config)
        self.sessions = sessions
        self.sets = feature_sets(frame, config)
        state = rs._state_table(sessions, config)
        frame = frame.merge(state, on="activity_id", how="left", validate="one_to_one")
        frame["strict_future_drift"] = frame.post_drift
        self.columns = self.sets["core"] + [c for c in state if c.startswith("s_")]
        self.stack = rs._Stack(frame, config)
        self.frame = frame
        dirs = [repo_path(config, p) for p in settings["run_gc_dirs"]]
        self.gc = _gated([d / "gc_gated.pkl" for d in dirs])
        self.gc_labels = pd.concat([pd.read_parquet(d / "gc_labels.parquet") for d in dirs], ignore_index=True)
        reference = pd.read_parquet(repo_path(config, config["combined_training"]["outputs"]["predictions"]))
        self.reference = reference[reference.scope.eq("holdout_gc")].assign(activity_id=lambda d: d.activity_id.astype(str))
        test_dir = repo_path(config, settings["test_dir"])
        self.test = _gated([test_dir / "gc_gated.pkl"])
        self.test = self.test[self.test.activity_id.astype(str).isin(set(self.reference.activity_id))].reset_index(drop=True)
        self.test_labels = pd.read_parquet(test_dir / "gc_labels.parquet")
        overlap = set(self.test.athlete_id) & (set(self.gc.athlete_id) | set(frame.athlete_id))
        assert not overlap, f"{len(overlap)} test athletes are in training"
        folds = athlete_folds(self.gc.athlete_id, int(settings["n_folds"]), int(config["seed"]))
        self.gc["outer_fold"] = self.gc.athlete_id.map(folds)
        for data in (self.gc, self.test):
            for column in self.columns:
                if column not in data:
                    data[column] = np.nan
        self.people = np.r_[frame.athlete_id.astype(str).to_numpy(), self.gc.athlete_id.astype(str).to_numpy()]
        self.fold = np.r_[frame.outer_fold.to_numpy(), self.gc.outer_fold.to_numpy()]
        self.n_fitrec = len(frame)
        self._tables = {}

    def _take(self, data, ids, out, residuals):
        position = pd.Series(np.arange(len(ids)), index=ids).reindex(data.activity_id.astype(str)).to_numpy(float)
        values = np.full((len(data), 2), np.nan)
        found = np.isfinite(position)
        values[found] = out[position[found].astype(int)]
        extra = residuals.reindex(data.activity_id.astype(str))[self.stack.resid_columns].reset_index(drop=True)
        return data[self.columns].astype(float).reset_index(drop=True).join(pd.DataFrame(values, columns=["p_d", "p_m"])).join(extra)

    def tables(self, encoder):
        """(development inputs, test inputs). Each row uses a network that excluded its athlete."""
        if encoder in self._tables:
            return self._tables[encoder]
        settings = self.config["encoder_ensemble"]
        if encoder == "old":
            store = np.load(repo_path(self.config, settings["old_encoder_train"]), allow_pickle=True)
            ids, out = store["ids"].astype(str), store["out"][0]
            fitrec = self.frame[self.columns].astype(float).reset_index(drop=True).join(self.stack.training_features().reset_index(drop=True))
            labels = self.gc_labels.assign(activity_id=lambda d: d.activity_id.astype(str))
            labels = labels[labels.activity_id.isin(ids)].drop_duplicates("activity_id")
            residuals = rs._residuals(labels, ids, out, labels.athlete_id.unique(), self.config).set_index("activity_id")
            development = pd.concat([fitrec, self._take(self.gc, ids, out, residuals)], ignore_index=True)
            test_store = np.load(repo_path(self.config, settings["old_encoder_test"]), allow_pickle=True)
        else:
            work = repo_path(self.config, settings["outputs"]["work_dir"])
            store = np.load(work / "encoder_train.npz", allow_pickle=True)
            ids, fold, networks = store["ids"].astype(str), store["fold"], store["out"]
            out = np.empty((len(ids), 2), np.float32)
            for k in range(int(settings["n_folds"])):
                out[fold == k] = networks[k + 1][fold == k]
            fitrec_labels = pd.read_parquet(repo_path(self.config, self.config["runner_state"]["outputs"]["work_dir"]) / "fitrec_labels.parquet")
            labels = pd.concat([fitrec_labels, self.gc_labels], ignore_index=True).assign(activity_id=lambda d: d.activity_id.astype(str))
            labels = labels[labels.activity_id.isin(ids)].drop_duplicates("activity_id")
            residuals = rs._residuals(labels, ids, out, labels.athlete_id.unique(), self.config).set_index("activity_id")
            development = pd.concat([self._take(self.frame, ids, out, residuals), self._take(self.gc, ids, out, residuals)], ignore_index=True)
            test_store = np.load(work / "encoder_test.npz", allow_pickle=True)
        test_ids, test_out = test_store["ids"].astype(str), test_store["out"]
        labels = self.test_labels.assign(activity_id=lambda d: d.activity_id.astype(str))
        labels = labels[labels.activity_id.isin(test_ids)]
        residuals = rs._residuals(labels, test_ids, test_out, labels.athlete_id.unique(), self.config).set_index("activity_id")
        self._tables[encoder] = (development, self._take(self.test, test_ids, test_out, residuals))
        return self._tables[encoder]

    def observed(self, target):
        return np.r_[self.frame[target].to_numpy(float), self.gc[target].to_numpy(float)]


def run_export_stage(config: dict[str, Any]) -> None:
    """Arrays for the torch process. Test labels are not written."""
    data = _Data(config)
    development, test = data.tables("new")
    work = repo_path(config, config["encoder_ensemble"]["outputs"]["work_dir"])
    np.savez(work / "arrays.npz", x=development.to_numpy(np.float32), test_x=test.to_numpy(np.float32), ids=np.r_[data.frame.activity_id.astype(str).to_numpy(), data.gc.activity_id.astype(str).to_numpy()],
             test_ids=data.test.activity_id.astype(str).to_numpy(), fold=data.fold, athletes=data.people, source=np.r_[np.zeros(data.n_fitrec), np.ones(len(data.gc))],
             y=np.column_stack([data.observed("target_drift"), data.observed("target_magnitude")]))
    LOGGER.info("E31 arrays: development %s, test %s", development.shape, test.shape)


def _member_test(data, name, target, seeds, config):
    member = config["encoder_ensemble"]["members"][name]
    development, test = data.tables(member["encoder"])
    raw = data.observed(target)
    usable = np.isfinite(raw)
    clip = config["rich_early"]["label_clip"][LABEL[target]]
    kind = transform_kind(member, target)
    a, b = guard(development[usable], test, quantiles=tuple(config["runner_state"]["model"]["guard_quantiles"]))
    return np.clip(backward(fit_member(member["model"], a, forward(np.clip(raw[usable], *clip), kind), b, seeds, config), kind), *clip)


def _member_folds(data, name, target, raw, seeds, config):
    member = config["encoder_ensemble"]["members"][name]
    development, _ = data.tables(member["encoder"])
    clip = config["rich_early"]["label_clip"][LABEL[target]]
    kind = transform_kind(member, target)
    y = forward(np.clip(raw, *clip), kind)
    quantiles = tuple(config["runner_state"]["model"]["guard_quantiles"])
    out = np.full(len(raw), np.nan)
    for k in range(int(config["encoder_ensemble"]["n_folds"])):
        train, test = data.fold != k, data.fold == k
        assert not set(data.people[train]) & set(data.people[test])
        a, b = guard(development[train], development[test], quantiles=quantiles)
        out[test] = np.clip(backward(fit_member(member["model"], a, y[train], b, seeds, config), kind), *clip)
    return out


def _scores(observed, predicted, people, cap_index, clip):
    ok = np.isfinite(observed) & np.isfinite(predicted)
    o, p, who = observed[ok], predicted[ok], people[ok]
    inside = (o >= clip[0]) & (o <= clip[1])
    return {"mae": float(np.abs(o - p).mean()), "mae_athlete": float(pd.Series(np.abs(o - p)).groupby(who).mean().mean()), "r2_all": regression_metrics(o, p)["r_squared"],
            "r2_in_label_range": regression_metrics(o[inside], p[inside])["r_squared"], "r2_capped": regression_metrics(observed[cap_index & ok], predicted[cap_index & ok])["r_squared"],
            "spearman": float(pd.Series(o).corr(pd.Series(p), method="spearman")), **within_metrics(o, p, who), "n": len(o), "n_athletes": len(set(who))}


def run_evaluate_stage(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    settings = config["encoder_ensemble"]
    base = config["runner_state"]
    outputs = settings["outputs"]
    work = repo_path(config, outputs["work_dir"])
    seed = int(config["seed"])
    data = _Data(config)
    labels = pd.read_parquet(repo_path(config, base["outputs"]["work_dir"]) / "fitrec_labels.parquet")
    audit = rs._audit(data.sessions, labels, config, np.random.default_rng(seed))
    LOGGER.info("E31 mutation audit:\n%s", audit.to_string(index=False))
    if audit.runs_with_changed_inputs.sum() > 0:
        raise AssertionError("E31 halted: an input changed after future data was mutated")
    test = data.test
    who = test.athlete_id.to_numpy().astype(str)
    capped = test.index.isin(cap_runs(test, int(settings["report_run_cap"]), seed).index)
    clip = config["rich_early"]["label_clip"]
    members_test: dict[tuple[str, str], np.ndarray] = {}
    predictions: dict[tuple[str, str], np.ndarray] = {}
    for choice, spec in settings["choices"].items():
        for target, source in (("target_drift", "target_drift"), ("target_magnitude", "target_magnitude"), ("strict_future_drift", "target_drift")):
            for name in spec[source]:
                key = (name, target)
                if key not in members_test:
                    members_test[key] = _member_test(data, name, target, int(settings["bag_seeds"]), config)
                    LOGGER.info("E31 member %s %s fitted", name, target)
            predictions[choice, target] = blend({m: members_test[m, target] for m in spec[source]}, spec[source])
    for name in ("e31_deep", "e31_mlp"):
        store = np.load(work / f"{name}.npz", allow_pickle=True)
        order = pd.Series(np.arange(len(store["test_ids"])), index=store["test_ids"].astype(str)).reindex(test.activity_id.astype(str)).to_numpy(int)
        predictions[name, "target_drift"] = np.clip(store["test"][order, 0], *clip["drift"])
        predictions[name, "target_magnitude"] = np.clip(store["test"][order, 1], *clip["magnitude"])
    for target in LABEL:
        part = data.reference[data.reference.model.eq("e30") & data.reference.target.eq(target)].set_index("activity_id").predicted.reindex(test.activity_id.astype(str)).to_numpy(float)
        assert np.isfinite(part).all(), "E30 reference predictions are incomplete"
        predictions["e30", target] = part

    rows = [{"model": m, "target": t, **_scores(test[t].to_numpy(float), p, who, capped, clip[LABEL[t]]), "provenance": PROVENANCE} for (m, t), p in predictions.items()]
    summary = pd.DataFrame(rows).sort_values(["target", "mae_athlete"])
    contrasts = []
    for (model, target), values in predictions.items():
        if model == "e30":
            continue
        observed = test[target].to_numpy(float)
        ok = np.isfinite(observed)
        delta = (pd.Series(np.abs(observed[ok] - values[ok])).groupby(who[ok]).mean() - pd.Series(np.abs(observed[ok] - predictions["e30", target][ok])).groupby(who[ok]).mean())
        athletes = sorted(delta.index)
        weights = bootstrap_weights(len(athletes), int(settings["bootstrap_n"]), seed)
        contrasts.append({"model": model, "target": target, "reference": "e30", "delta_mae_athlete": float(delta.mean()), **inference(delta.mean(), weighted_bootstrap(delta, athletes, weights), float(settings["confidence_level"])), "n_athletes": len(athletes), "provenance": PROVENANCE})
    contrasts = pd.DataFrame(contrasts)
    contrasts["q_bh"] = multipletests(contrasts.p_bootstrap, method="fdr_bh")[1]
    contrasts["meets_gain_criterion"] = contrasts.ci_high.lt(0) & contrasts.q_bh.le(float(settings["fdr_alpha"]))

    # development folds and shuffled probe
    n = data.n_fitrec
    fold_rows = []
    shuffled = pd.concat([data.frame[["athlete_id", *base["targets"]]], data.gc[["athlete_id", *base["targets"]]]], ignore_index=True)
    shuffled = shuffle_targets_within_athlete(shuffled, list(base["targets"]), seed=seed)
    limit = float(config["evaluate"]["leakage_r2_max"])
    probe = []
    fitrec_3_4 = np.isin(data.fold[:n], base["holdout_folds"])
    for choice, spec in settings["choices"].items():
        for target in base["targets"]:
            real = {m: _member_folds(data, m, target, data.observed(target), int(settings["probe_bag_seeds"]), config) for m in spec[target]}
            shuffle = {m: _member_folds(data, m, target, shuffled[target].to_numpy(float), int(settings["probe_bag_seeds"]), config) for m in spec[target]}
            value, fake = blend(real, spec[target]), blend(shuffle, spec[target])
            observed = data.observed(target)
            for scope, slice_ in (("fitrec", slice(0, n)), ("goldencheetah", slice(n, None))):
                fold_rows.append({"model": choice, "target": target, "scope": scope, **_scores(observed[slice_], value[slice_], data.people[slice_], np.ones(len(observed[slice_]), bool), clip[LABEL[target]]), "provenance": PROVENANCE})
            y = shuffled[target].to_numpy(float)[:n]
            for scope, mask in (("all_folds", np.ones(n, bool)), ("fitrec_3_4", fitrec_3_4)):
                probe.append({"model": choice, "target": target, "scope": scope, "r_squared": regression_metrics(y[mask], fake[:n][mask])["r_squared"], **within_metrics(y[mask], fake[:n][mask], data.people[:n][mask]), "limit": limit, "exceeds_limit": regression_metrics(y[mask], fake[:n][mask])["r_squared"] > limit, "provenance": "MEASURED WITHIN-ATHLETE SHUFFLED TARGET"})
            LOGGER.info("E31 probe %s %s finished", choice, target)
    for name in ("e31_deep", "e31_mlp"):
        store = np.load(work / f"{name}.npz", allow_pickle=True)
        for j, target in enumerate(base["targets"]):
            value, observed = np.clip(store["development"][:, j], *clip[LABEL[target]]), data.observed(target)
            for scope, slice_ in (("fitrec", slice(0, n)), ("goldencheetah", slice(n, None))):
                fold_rows.append({"model": name, "target": target, "scope": scope, **_scores(observed[slice_], value[slice_], data.people[slice_], np.ones(len(observed[slice_]), bool), clip[LABEL[target]]), "provenance": PROVENANCE})
    development, probe = pd.DataFrame(fold_rows), pd.DataFrame(probe)
    long = pd.concat([pd.DataFrame({"activity_id": test.activity_id.astype(str), "athlete_id": who, "model": m, "target": t, "observed": test[t].to_numpy(float), "predicted": p, "provenance": PROVENANCE}) for (m, t), p in predictions.items()], ignore_index=True)
    tables = {"predictions": long, "summary": summary, "contrasts": contrasts, "leakage": probe, "audit": audit, "development": development}
    for name, table in tables.items():
        path = repo_path(config, outputs[name])
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(path, index=False) if path.suffix == ".parquet" else table.to_csv(path, index=False)
    _figure(summary, config)
    _report(summary, contrasts, probe, audit, development, config)
    return tables


def _figure(summary, config):
    order = ["e30", "e31_mlp", "e31_deep", "e31_xgb", "e31_blend"]
    names = {"e30": "E30", "e31_mlp": "MLP", "e31_deep": "FiLM TCN", "e31_xgb": "XGBoost", "e31_blend": "Blend"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True)
    for ax, (target, title) in zip(axes, [("target_drift", "Drift"), ("target_magnitude", "Magnitude")], strict=True):
        part = summary[summary.target == target].set_index("model").reindex(order)
        ax.barh(range(len(order)), part.r2_in_label_range.clip(lower=0), height=0.55, color=["#8c8c8c" if m == "e30" else "#2a78d6" for m in order])
        for i, value in enumerate(part.r2_in_label_range):
            ax.text(max(value, 0) + 0.008, i, f"{value:.3f}", va="center", fontsize=9, color="#404040")
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_yticks(range(len(order)), [names[m] for m in order], fontsize=9)
        ax.set_xlabel("R², runs inside training label range", fontsize=9)
        ax.set_xlim(0, 0.6)
        ax.grid(axis="x", color="#e5e5e5", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", length=0)
    fig.suptitle("E31 on opened E30 athletes, PILOT MEASURED EXPLORATORY, not independent", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    path = repo_path(config, config["encoder_ensemble"]["outputs"]["figure"])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _report(summary, contrasts, probe, audit, development, config):
    def table(frame, digits=4):
        return markdown_table(frame.round(digits))
    lines = ["# E31: Two-dataset encoder, tree ensembles and a FiLM sequence network", "",
             f"All results are **PILOT MEASURED EXPLORATORY**. Protocol: docs/E31_PROTOCOL.md. Seed {config['seed']} (CONFIGURED).", "",
             "**This is not independent evidence.** The test athletes were opened by E30, and `e31_blend` and `e31_xgb` were scored on them in a lab before registration.", "",
             "## Test athletes", "", table(summary.drop(columns=["provenance"])), "",
             "## Contrasts against E30 (athlete-weighted MAE, BH across all contrasts)", "", table(contrasts.drop(columns=["provenance"])), "",
             "## Development folds", "", table(development.drop(columns=["provenance"])), "",
             "## Controls", "", table(audit), "", table(probe), "",
             "![E31](../../figures/e31_encoder_ensemble.png)", ""]
    path = repo_path(config, config["encoder_ensemble"]["outputs"]["report"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
