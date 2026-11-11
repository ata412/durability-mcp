"""Reusable EAT (Expert-Attention Transformer, Tree-Deep Fusion v4) model set for local inference.

The fusion lab saved only per-fold networks; its nine expert models were refit and discarded inside
each fold. This module refits the frozen v4 recipe once on every development row, keeps the experts,
trains the EAT networks on athlete-disjoint out-of-fold expert predictions, and predicts FIT runs
through the same path as the saved E31 model set.

Every number is MEASURED EXPLORATORY: EAT beat E31 on magnitude only (1.3%, development folds) and
tied on drift. The models are derived from FitRec (academic use, no redistribution): local only.
Stages run in separate processes because tree libraries and Torch conflict on macOS OpenMP:
  python -m durability.eat_artifact experts | eat | check
"""
from __future__ import annotations

import copy
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from durability.config import load_config, repo_path

LOGGER = logging.getLogger(__name__)
PROVENANCE = "MEASURED EXPLORATORY; EAT FUSION (v4) REFIT ON ALL DEVELOPMENT ROWS; LOCAL ONLY"
TARGETS = ("target_drift", "target_magnitude")
SHORT = ("drift", "magnitude")
EXPERTS = ("lgbm_l1", "xgb_l1", "extratrees", "lgbm_huber", "catboost", "e31_m1", "e31_m2", "e31_m3", "e31_repro")


def model_dir(config: dict[str, Any]) -> Path:
    path = repo_path(config, config["eat_artifact"]["model_dir"])
    path.mkdir(parents=True, exist_ok=True)
    return path


def _inputs(config: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Exactly the rows and tables the fusion labs used (written by advanced_leakage.prepare)."""
    folder = repo_path(config, config["deep_improvement_lab"]["input_dir"])
    base = np.load(folder / "base.npz", allow_pickle=True)
    return {k: base[k] for k in base.files}, {kind: np.load(folder / f"{kind}.npy") for kind in ("old", "new")}


def _labels(config: dict[str, Any], y: np.ndarray) -> np.ndarray:
    return np.column_stack([np.clip(y[:, j], *config["rich_early"]["label_clip"][s]) for j, s in enumerate(SHORT)])


def athlete_weights(people: np.ndarray) -> np.ndarray:
    counts = pd.Series(people).value_counts()
    weights = np.array([1.0 / counts[p] for p in people])
    return weights / weights.mean()


def _lab_transform(y: np.ndarray, target: str, inverse: bool = False) -> np.ndarray:
    """The fusion v2 Huber transform: signed root for drift, log for magnitude."""
    if target == "target_drift":
        return np.sign(y) * y ** 2 if inverse else np.sign(y) * np.sqrt(np.abs(y))
    return np.exp(y) if inverse else np.log(y)


# --------------------------------------------------------------------------- tree process only
def _fit_experts(config: dict[str, Any], tables: dict[str, np.ndarray], y: np.ndarray, people: np.ndarray, rows: np.ndarray) -> dict[str, Any]:
    """Fit all nine experts on ``rows``. Returns fitted objects, never predictions."""
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.impute import SimpleImputer
    from xgboost import XGBRegressor

    from durability.e31_artifact import member_models
    from durability.encoder_ensemble import forward, transform_kind

    seed, jobs = int(config["seed"]), int(config["evaluate"]["n_jobs"])
    v1, v2, ee = config["tree_deep_fusion"], config["tree_deep_fusion_v2"], config["encoder_ensemble"]
    x = tables["new"][rows]
    weights = athlete_weights(people[rows])
    imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(x)
    xx = imputer.transform(x)
    out: dict[str, Any] = {"imputer": imputer, "trees": [], "members": {}}
    for j, target in enumerate(TARGETS):
        yy = y[rows, j]
        learners = [LGBMRegressor(**v1["lgbm"], objective="regression_l1", verbosity=-1, random_state=seed, n_jobs=jobs),
                    XGBRegressor(**v1["xgb"], objective="reg:absoluteerror", tree_method="hist", random_state=seed, n_jobs=jobs),
                    ExtraTreesRegressor(**v1["extra"], random_state=seed, n_jobs=jobs)]
        fitted = [m.fit(xx, yy, sample_weight=weights) for m in learners]
        huber = LGBMRegressor(**v2["lgbm_huber"], objective="huber", alpha=float(v2["huber_alpha"]), verbosity=-1, random_state=seed, n_jobs=jobs)
        fitted.append(huber.fit(xx, _lab_transform(yy, target), sample_weight=weights))
        fitted.append(CatBoostRegressor(**v2["catboost"], loss_function="MAE", random_seed=seed, verbose=0, thread_count=jobs, allow_writing_files=False).fit(xx, yy, sample_weight=weights))
        out["trees"].append(fitted)
        quantiles = tuple(config["runner_state"]["model"]["guard_quantiles"])
        for name in ee["choices"]["e31_blend"][target]:
            member = ee["members"][name]
            frame = pd.DataFrame(tables[member["encoder"]][rows])
            low, high = frame.quantile(quantiles[0]), frame.quantile(quantiles[1])
            kind = transform_kind(member, target)
            models = member_models(member["model"], frame.clip(low, high, axis=1), forward(yy, kind), int(config["tree_deep_fusion_v4"]["member_seeds"]), config)
            out["members"][name, target] = {"models": models, "low": low, "high": high, "kind": kind, "encoder": member["encoder"]}
    return out


def expert_predictions(config: dict[str, Any], fitted: dict[str, Any], tables: dict[str, np.ndarray]) -> np.ndarray:
    """(rows, 2 targets, 9 experts) in the EXPERTS order, clipped to the label range like the lab."""
    from durability.encoder_ensemble import backward

    ee = config["encoder_ensemble"]
    xx = fitted["imputer"].transform(tables["new"])
    out = np.empty((len(xx), 2, len(EXPERTS)))
    for j, target in enumerate(TARGETS):
        clip = config["rich_early"]["label_clip"][SHORT[j]]
        trees = fitted["trees"][j]
        for i in range(3):
            out[:, j, i] = trees[i].predict(xx)
        out[:, j, 3] = _lab_transform(trees[3].predict(xx), target, inverse=True)
        out[:, j, 4] = trees[4].predict(xx)
        names = list(ee["choices"]["e31_blend"][target])
        for i, name in enumerate(names):
            member = fitted["members"][name, target]
            frame = pd.DataFrame(tables[member["encoder"]]).clip(member["low"], member["high"], axis=1)
            out[:, j, 5 + i] = np.clip(backward(np.mean([m.predict(frame) for m in member["models"]], axis=0), member["kind"]), *clip)
        weights = np.array([ee["choices"]["e31_blend"][target][n] for n in names], float)
        out[:, j, 8] = out[:, j, 5:8] @ (weights / weights.sum())
    return out


def fit_experts(config: dict[str, Any]) -> dict[str, Any]:
    """Out-of-fold experts on every development row (athlete-disjoint), then full-data experts kept for inference."""
    from sklearn.model_selection import GroupKFold

    a, tables = _inputs(config)
    y, people = _labels(config, a["y"]), a["athletes"].astype(str)
    rows = np.arange(len(y))
    oof = np.full((len(y), 2, len(EXPERTS)), np.nan)
    for inner, (fit_rows, valid_rows) in enumerate(GroupKFold(int(config["tree_deep_fusion"]["inner_folds"])).split(rows, groups=people)):
        assert not set(people[fit_rows]) & set(people[valid_rows])
        fitted = _fit_experts(config, tables, y, people, fit_rows)
        oof[valid_rows] = expert_predictions(config, fitted, {k: v[valid_rows] for k, v in tables.items()})
        LOGGER.info("EAT artifact inner fold %s done", inner)
    assert np.isfinite(oof).all()
    full = _fit_experts(config, tables, y, people, rows)
    full.update(columns=_columns(config), feature_count=int(tables["new"].shape[1]), experts=list(EXPERTS), provenance=PROVENANCE)
    joblib.dump(full, model_dir(config) / "experts.joblib")
    np.savez(model_dir(config) / "oof_experts.npz", oof=oof, y=y, people=people, ids=a["ids"].astype(str))
    LOGGER.info("EAT artifact experts saved")
    return {"rows": len(y), "athletes": len(set(people))}


def _columns(config: dict[str, Any]) -> dict[str, list[str]]:
    """Column order of the 'old' and 'new' tables, taken from the saved E31 member set built from the same tables."""
    bundle = joblib.load(repo_path(config, config["e31_artifact"]["model_dir"]) / "trees.joblib")
    columns = {}
    for member in bundle["members"].values():
        columns.setdefault(member["encoder"], list(member["columns"]))
    return columns


# --------------------------------------------------------------------------- torch process only
def l1_stack(p: np.ndarray, y: np.ndarray, wt: np.ndarray) -> tuple[float, np.ndarray]:
    """min sum wt|y - b - p @ beta| with beta >= 0 and sum(beta) = 1 (the fusion v2 stack)."""
    from scipy import sparse
    from scipy.optimize import linprog

    n, m = p.shape
    cost = np.concatenate([[0], np.zeros(m), wt, wt])
    identity = sparse.identity(n, format="csr")
    a = sparse.hstack([sparse.csr_matrix(np.hstack([np.ones((n, 1)), p])), identity, -identity])
    eq = sparse.vstack([a, sparse.csr_matrix(np.concatenate([[0], np.ones(m), np.zeros(2 * n)])[None])]).tocsr()
    result = linprog(cost, A_eq=eq, b_eq=np.concatenate([y, [1]]), bounds=[(None, None)] + [(0, None)] * (m + 2 * n), method="highs")
    assert result.success, result.message
    return float(result.x[0]), result.x[1 : m + 1]


def eat_network(groups: list[Any], m: int, s: dict[str, Any], gate_bias: np.ndarray):
    """The v4 Expert-Attention Transformer: feature-group tokens + expert tokens + disagreement + CLS."""
    import torch
    from torch import nn

    class EAT(nn.Module):
        def __init__(self):
            super().__init__()
            d = int(s["d_model"])
            self.groups, self.M = groups, m
            self.feat = nn.ModuleList([nn.Linear(len(g), d) for g in groups])
            self.val, self.eid, self.tid, self.dis = nn.Linear(1, d), nn.Embedding(m, d), nn.Embedding(2, d), nn.Linear(2, d)
            self.cls, self.kind = nn.Parameter(torch.zeros(1, 1, d)), nn.Embedding(4, d)
            layer = nn.TransformerEncoderLayer(d, int(s["heads"]), int(s["ff"]), float(s["dropout"]), activation="gelu", batch_first=True, norm_first=True)
            self.enc, self.norm = nn.TransformerEncoder(layer, int(s["layers"]), enable_nested_tensor=False), nn.LayerNorm(d)
            self.gate, self.res = nn.Linear(d, 1), nn.Linear(d, 2)
            for layer_ in (self.gate, self.res):
                nn.init.zeros_(layer_.weight)
                nn.init.zeros_(layer_.bias)
            self.register_buffer("prior", torch.as_tensor(gate_bias, dtype=torch.float32))
            self.register_buffer("ids", torch.arange(m).repeat(2))
            self.register_buffer("tids", torch.arange(2).repeat_interleave(m))

        def forward(self, x, e, icpt):
            n = len(x)
            ft = torch.stack([f(x[:, g]) for f, g in zip(self.feat, self.groups)], 1) + self.kind.weight[0]
            et = self.val(e.reshape(n, -1, 1)) + self.eid(self.ids) + self.tid(self.tids) + self.kind.weight[1]
            dt = (self.dis(e.std(-1)) + self.kind.weight[2])[:, None]
            ct = self.cls.expand(n, -1, -1) + self.kind.weight[3]
            h = self.norm(self.enc(torch.cat([ct, dt, ft, et], 1)))
            g = torch.softmax(self.gate(h[:, -2 * self.M :]).reshape(n, 2, self.M) + self.prior, -1)
            res = self.res(h[:, 0])
            return (g * e).sum(-1) + icpt + res, res, g

    return EAT()


def fit_eat(config: dict[str, Any]) -> dict[str, Any]:
    """Train the EAT bags on all development rows from out-of-fold expert inputs, as v4 'combine' did per fold."""
    import torch
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    s, ns = config["tree_deep_fusion_v4"], config["tree_deep_fusion_v4"]["net"]
    _, tables = _inputs(config)
    x = tables["new"]
    saved = np.load(model_dir(config) / "oof_experts.npz", allow_pickle=True)
    oof, y, people = saved["oof"], saved["y"], saved["people"].astype(str)
    m = oof.shape[-1]
    wt = athlete_weights(people)
    intercept, beta = np.empty(2), np.empty((2, m))
    for j in range(2):
        intercept[j], beta[j] = l1_stack(oof[:, j], y[:, j], wt)
    mu, sd = y.mean(0), y.std(0)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(x)
    scaler = StandardScaler().fit(imputer.transform(x))
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tensor = lambda v: torch.as_tensor(np.asarray(v, dtype="float32"), device=device)
    xs, es = tensor(scaler.transform(imputer.transform(x))), tensor((oof - mu[None, :, None]) / sd[None, :, None])
    ys, ic = tensor((y - mu) / sd), tensor(intercept / sd)
    group_index = [g.tolist() for g in np.array_split(np.arange(x.shape[1]), int(s["feature_groups"]))]
    groups = [torch.as_tensor(g, device=device) for g in group_index]
    prior = np.log(np.clip(beta, 0, None) + 1e-6)
    states, logs = [], []
    rows = np.arange(len(y))
    for bag, (fit_rows, valid_rows) in enumerate(GroupKFold(int(s["bags"])).split(rows, groups=people)):
        assert not set(people[fit_rows]) & set(people[valid_rows])
        seed = int(config["seed"]) + bag
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        wv = np.zeros(len(rows))
        wv[fit_rows], wv[valid_rows] = athlete_weights(people[fit_rows]), athlete_weights(people[valid_rows])
        w = tensor(wv)
        net = eat_network(groups, m, ns, prior).to(device)
        ema = copy.deepcopy(net)
        for p in ema.parameters():
            p.requires_grad_(False)
        ema.eval()
        opt = torch.optim.AdamW(net.parameters(), lr=float(ns["learning_rate"]), weight_decay=float(ns["weight_decay"]))

        def validate() -> float:
            with torch.no_grad():
                p, _, _ = ema(xs[valid_rows], es[valid_rows], ic)
            return float(((p - ys[valid_rows]).abs().mean(-1) * w[valid_rows]).sum() / w[valid_rows].sum())

        best, stale, chosen = validate(), 0, 0
        start, state = best, copy.deepcopy(ema.state_dict())
        for epoch in range(int(ns["epochs"])):
            net.train()
            order = rng.permutation(fit_rows)
            for q in range(0, len(order), int(ns["batch_size"])):
                ix = order[q : q + int(ns["batch_size"])]
                p, res, _ = net(xs[ix], es[ix], ic)
                loss = ((p - ys[ix]).abs().mean(-1) * w[ix]).sum() / w[ix].sum() + float(ns["residual_penalty"]) * res.square().mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                with torch.no_grad():
                    for pe, pn in zip(ema.parameters(), net.parameters()):
                        pe.mul_(float(s["ema_decay"])).add_(pn.detach(), alpha=1 - float(s["ema_decay"]))
            value = validate()
            if value < best - 1e-6:
                best, stale, chosen, state = value, 0, epoch + 1, copy.deepcopy(ema.state_dict())
            else:
                stale += 1
            if stale >= int(ns["patience"]):
                break
        states.append({k: v.cpu() for k, v in state.items()})
        logs.append({"bag": bag, "best_epoch": chosen, "val_start": start, "val_best": best})
        LOGGER.info("EAT artifact bag %s: best epoch %s, validation %.4f -> %.4f", bag, chosen, start, best)
    torch.save({"states": states, "settings": dict(ns), "groups": group_index, "prior": prior, "intercept": intercept, "beta": beta,
                "mu": mu, "sd": sd, "imputer": imputer, "scaler": scaler, "experts": list(EXPERTS), "provenance": PROVENANCE},
               model_dir(config) / "eat.pt")
    summary = {"stack_intercept": intercept.round(4).tolist(), "stack_weights": dict(zip(SHORT, (dict(zip(EXPERTS, b.round(4).tolist())) for b in beta))), "bags": logs}
    (model_dir(config) / "eat_training.json").write_text(json.dumps(summary, indent=1))
    return summary


def apply_eat(config: dict[str, Any], x: np.ndarray, experts: np.ndarray) -> np.ndarray:
    """EAT predictions (rows, 2): mean over the saved bags, clipped to the label range."""
    import torch

    saved = torch.load(model_dir(config) / "eat.pt", map_location="cpu", weights_only=False)
    mu, sd = saved["mu"], saved["sd"]
    xs = torch.as_tensor(saved["scaler"].transform(saved["imputer"].transform(x)), dtype=torch.float32)
    es = torch.as_tensor((experts - mu[None, :, None]) / sd[None, :, None], dtype=torch.float32)
    ic = torch.as_tensor(saved["intercept"] / sd, dtype=torch.float32)
    groups = [torch.as_tensor(g) for g in saved["groups"]]
    preds = []
    for state in saved["states"]:
        net = eat_network(groups, experts.shape[-1], saved["settings"], saved["prior"])
        net.load_state_dict(state)
        net.eval()
        with torch.no_grad():
            preds.append(net(xs, es, ic)[0].numpy() * sd + mu)
    out = np.mean(preds, axis=0)
    for j, short in enumerate(SHORT):
        out[:, j] = np.clip(out[:, j], *config["rich_early"]["label_clip"][short])
    return out


def apply_in_subprocess(x: np.ndarray, experts: np.ndarray) -> np.ndarray:
    """Torch must never share a process with the tree libraries, so EAT runs in a child process."""
    with tempfile.TemporaryDirectory() as folder:
        source, target = Path(folder) / "in.npz", Path(folder) / "out.npy"
        np.savez(source, x=x, experts=experts)
        subprocess.run([sys.executable, "-m", "durability.eat_artifact", "apply", str(source), str(target)], check=True, stdout=sys.stderr)  # keeps an MCP stdio channel clean
        return np.load(target)


# --------------------------------------------------------------------------- inference (tree process)
def predict_tables(config: dict[str, Any], tables: dict[str, pd.DataFrame]) -> dict[str, np.ndarray]:
    """EAT and the E31 reproduction for named input tables (as ``e31_artifact.athlete_tables`` builds them)."""
    fitted = joblib.load(model_dir(config) / "experts.joblib")
    arrays = {kind: tables[kind].reindex(columns=fitted["columns"][kind]).to_numpy(float) for kind in ("old", "new")}
    assert arrays["new"].shape[1] == fitted["feature_count"]
    experts = expert_predictions(config, fitted, arrays)
    eat = apply_in_subprocess(arrays["new"], experts) if len(experts) else np.empty((0, 2))
    return {"eat": eat, "e31": experts[:, :, EXPERTS.index("e31_repro")]}


def predict_fit_folder(folder: str | Path, athlete: str, config: dict[str, Any]) -> dict[str, Any]:
    """FIT files -> E31 input path -> EAT predictions for gated runs, with E31, actual values and two baselines."""
    from durability.e31_artifact import athlete_tables, encode_in_subprocess, shrunk_history
    from durability.fit_runner_state import process_fit_folder

    bundle = joblib.load(repo_path(config, config["e31_artifact"]["model_dir"]) / "trees.joblib")
    result = process_fit_folder(folder, athlete, config)
    gated, labels = pd.DataFrame(result["gated"]), pd.DataFrame(result["labels"])
    summary = {"counts": result["counts"], "gated_runs": len(gated), "sequences": len(result["sequences"])}
    if gated.empty:
        return {"summary": summary, "runs": pd.DataFrame()}
    ids = np.array([s[0] for s in result["sequences"]], dtype=str)
    outputs = encode_in_subprocess(None, np.stack([s[1] for s in result["sequences"]]), np.array([s[2] for s in result["sequences"]], np.float32))
    preds = predict_tables(config, athlete_tables(gated, labels, ids, outputs, bundle, config))
    runs = gated[["activity_id", "start_time"]].copy()
    for j, target in enumerate(TARGETS):
        short = SHORT[j]
        runs[f"{short}_observed"] = gated[target].astype(float).to_numpy() if target in gated else gated[short].astype(float).to_numpy()
        runs[f"{short}_eat"] = preds["eat"][:, j]
        runs[f"{short}_e31"] = preds["e31"][:, j]
        runs[f"{short}_training_mean"] = bundle["population_mean"][target]
        runs[f"{short}_own_shrunk"] = shrunk_history(labels, gated, target, bundle["population_mean"][target], float(config["e31_artifact"]["shrinkage_strength"]))
    return {"summary": summary, "runs": runs}


def check(config: dict[str, Any]) -> dict[str, Any]:
    """Sanity check on development rows (in-sample, so not evidence): EAT tracks its E31 expert and the labels."""
    a, tables = _inputs(config)
    rng = np.random.default_rng(int(config["seed"]))
    rows = np.sort(rng.choice(len(a["ids"]), size=min(int(config["eat_artifact"]["check_rows"]), len(a["ids"])), replace=False))
    fitted = joblib.load(model_dir(config) / "experts.joblib")
    experts = expert_predictions(config, fitted, {k: v[rows] for k, v in tables.items()})
    eat = apply_in_subprocess(tables["new"][rows], experts)
    y = _labels(config, a["y"])[rows]
    out = {f"{s}_pearson_eat_vs_e31": float(np.corrcoef(eat[:, j], experts[:, j, 8])[0, 1]) for j, s in enumerate(SHORT)}
    out.update({f"{s}_in_sample_mae_eat": float(np.abs(eat[:, j] - y[:, j]).mean()) for j, s in enumerate(SHORT)})
    out.update({f"{s}_in_sample_mae_e31": float(np.abs(experts[:, j, 8] - y[:, j]).mean()) for j, s in enumerate(SHORT)})
    out["rows"], out["provenance"] = int(len(rows)), "IN-SAMPLE SANITY CHECK, NOT EVIDENCE"
    (model_dir(config) / "eat_check.json").write_text(json.dumps(out, indent=1))
    LOGGER.info("EAT artifact check: %s", out)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    stage = sys.argv[1]
    config = load_config()
    if stage == "apply":
        loaded = np.load(sys.argv[2])
        np.save(sys.argv[3], apply_eat(config, loaded["x"], loaded["experts"]))
        return
    if stage == "fit":
        print(predict_fit_folder(sys.argv[2], sys.argv[3], config)["runs"].drop(columns=["start_time"], errors="ignore").round(4).to_string(index=False))
        return
    print(json.dumps({"experts": fit_experts, "eat": fit_eat, "check": check}[stage](config), indent=1, default=str))


if __name__ == "__main__":
    main()
