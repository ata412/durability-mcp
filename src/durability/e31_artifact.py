"""Reusable E31 model set: saved encoders and tree members, parity checks, and inference on FIT files.

E31 itself saved only predictions. This module refits the frozen E31 recipe on the same
training rows and saves the models. Every number is PILOT MEASURED. The 472 test athletes
were opened by E30, so parity on them is a reproduction check, not new evidence.
The models are derived from FitRec, which is academic-use only: never redistribute them.
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
PROVENANCE = "PILOT MEASURED; E31 ARTIFACT; NOT INDEPENDENT"
TARGETS = ("target_drift", "target_magnitude")
LABEL = {"target_drift": "drift", "target_magnitude": "magnitude"}


def _dir(config: dict[str, Any]) -> Path:
    path = repo_path(config, config["e31_artifact"]["model_dir"])
    path.mkdir(parents=True, exist_ok=True)
    return path


# --------------------------------------------------------------------------- torch process only
def _encoder_data(config: dict[str, Any], kind: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Training data for the full network, assembled exactly as the E30 (old) or E31 (new) encoder stage."""
    fitrec = np.load(repo_path(config, config["runner_state"]["outputs"]["work_dir"]) / "fitrec_seq.npz", allow_pickle=True)
    if kind == "old":
        return {k: fitrec[k] for k in fitrec.files}, config
    from durability.runner_state_features import athlete_folds

    settings = config["encoder_ensemble"]
    clip = config["rich_early"]["label_clip"]
    derived = copy.deepcopy(config)
    derived["runner_state"]["encoder"]["hidden"] = int(settings["encoder_hidden"])
    parts = []
    for path in settings["run_gc_dirs"]:
        folder = repo_path(config, path)
        seq = np.load(folder / "gc_seq.npz", allow_pickle=True)
        labels = pd.read_parquet(folder / "gc_labels.parquet").set_index("activity_id").reindex(seq["ids"])
        y = np.full((len(seq["ids"]), fitrec["y"].shape[1]), np.nan, dtype=np.float32)
        y[:, 0], y[:, 1] = np.clip(labels.drift.astype(float), *clip["drift"]), np.clip(labels.magnitude.astype(float), *clip["magnitude"])
        parts.append((seq["ids"].astype(str), seq["x"], seq["ld"], labels.athlete_id.to_numpy().astype(str), y))
    ids, x, ld, athletes, y = (np.concatenate(values) for values in zip(*parts, strict=True))
    folds = athlete_folds(athletes, int(settings["n_folds"]), int(config["seed"]))
    data = {"x": np.concatenate([fitrec["x"], x]).astype(np.float32), "ld": np.concatenate([fitrec["ld"], ld]).astype(np.float32), "y": np.concatenate([fitrec["y"], y]).astype(np.float32),
            "fold": np.r_[fitrec["fold"].astype(float), np.array([folds[a] for a in athletes], float)], "athlete": np.concatenate([fitrec["athlete"].astype(str), athletes])}
    return data, derived


def train_encoders(config: dict[str, Any]) -> dict[str, Any]:
    """Refit both full networks, save them, and compare with the saved test outputs E31 used."""
    import torch

    from durability.runner_state_encoder import _train

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    test = np.load(repo_path(config, config["encoder_ensemble"]["test_dir"]) / "gc_seq.npz", allow_pickle=True)
    saved = {"old": repo_path(config, config["encoder_ensemble"]["old_encoder_test"]),
             "new": repo_path(config, config["encoder_ensemble"]["outputs"]["work_dir"]) / "encoder_test.npz"}
    checks = {}
    for kind in ("old", "new"):
        data, used = _encoder_data(config, kind)
        apply = _train(data, (), used, device)
        torch.save({"state_dict": apply.net.state_dict(), "norm": {k: np.asarray(v) for k, v in apply.norm.items()}, "rules": dict(used["runner_state"]["encoder"]),
                    "outputs": int(data["y"].shape[1])}, _dir(config) / f"encoder_{kind}.pt")
        reference = np.load(saved[kind], allow_pickle=True)
        assert (reference["ids"].astype(str) == test["ids"].astype(str)).all()
        again = apply(test["x"], test["ld"])
        checks[kind] = {f"pearson_r_{LABEL[t]}": float(np.corrcoef(again[:, i], reference["out"][:, i])[0, 1]) for i, t in enumerate(TARGETS)}
        checks[kind].update({f"mean_abs_diff_{LABEL[t]}": float(np.abs(again[:, i] - reference["out"][:, i]).mean()) for i, t in enumerate(TARGETS)})
        LOGGER.info("E31 artifact encoder %s: %s", kind, checks[kind])
    (_dir(config) / "encoder_checks.json").write_text(json.dumps(checks, indent=2))
    return checks


def encode(config: dict[str, Any], x: np.ndarray, ld: np.ndarray) -> dict[str, np.ndarray]:
    """Outputs of both saved full networks for sequences ``x`` and log-durations ``ld``."""
    import torch

    from durability.runner_state_encoder import Encoder

    out = {}
    for kind in ("old", "new"):
        saved = torch.load(_dir(config) / f"encoder_{kind}.pt", weights_only=False)
        net = Encoder(saved["outputs"], saved["rules"])
        net.load_state_dict(saved["state_dict"])
        net.eval()
        n = saved["norm"]
        ax = torch.from_numpy(((x - n["xm"]) / n["xs"]).astype(np.float32))
        al = torch.from_numpy(((ld - n["lm"]) / n["ls"]).astype(np.float32))
        with torch.no_grad():
            raw = np.concatenate([net(ax[b], al[b]).numpy()[:, :2] for b in torch.arange(len(ax)).split(1024)]) if len(ax) else np.empty((0, 2))
        out[kind] = raw * n["sd"] + n["mean"]
    return out


def encode_in_subprocess(config_path: str | None, x: np.ndarray, ld: np.ndarray) -> dict[str, np.ndarray]:
    """torch must never share a process with LightGBM, so encoding runs in a child process."""
    with tempfile.TemporaryDirectory() as folder:
        source, target = Path(folder) / "in.npz", Path(folder) / "out.npz"
        np.savez(source, x=x, ld=ld)
        subprocess.run([sys.executable, "-m", "durability.e31_artifact", "encode", str(source), str(target)] + ([config_path] if config_path else []), check=True, stdout=sys.stderr)  # keeps an MCP stdio channel clean
        loaded = np.load(target)
        return {k: loaded[k] for k in loaded.files}


# --------------------------------------------------------------------------- tree process only
def member_models(model: str, x: pd.DataFrame, y: np.ndarray, seeds: int, config: dict[str, Any]) -> list[Any]:
    """The models ``encoder_ensemble.fit_member`` fits, kept instead of discarded."""
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor

    from durability.rich_early import huber_alpha

    settings, seed = config["encoder_ensemble"]["models"], int(config["seed"])
    alpha = huber_alpha(y, float(config["runner_state"]["model"]["huber_quantile"]))
    jobs = int(config["evaluate"]["n_jobs"])
    if model == "xgb":
        return [XGBRegressor(objective="reg:pseudohubererror", huber_slope=alpha, n_jobs=jobs, random_state=seed + i, tree_method="hist", base_score=float(np.median(y)), **settings["xgb"]).fit(x, y) for i in range(seeds)]
    rules = dict(config["runner_state"]["model"]["lightgbm"])
    if model != "lgbm":
        rules.update(settings[model])
    if not rules.get("linear_tree"):
        rules.pop("linear_lambda", None)
    return [LGBMRegressor(objective="huber", alpha=alpha, verbosity=-1, n_jobs=jobs, random_state=seed + i, **rules).fit(x, y) for i in range(seeds)]


def predict_member(member: dict[str, Any], table: pd.DataFrame, clip: tuple[float, float]) -> np.ndarray:
    from durability.encoder_ensemble import backward

    x = table[member["columns"]].astype(float).clip(member["low"], member["high"], axis=1)
    return np.clip(backward(np.mean([m.predict(x) for m in member["models"]], axis=0), member["kind"]), *clip)


def predict(bundle: dict[str, Any], tables: dict[str, pd.DataFrame]) -> dict[str, np.ndarray]:
    """Blend of saved members. ``tables`` maps encoder kind ('old', 'new') to an input table."""
    out = {}
    for target in TARGETS:
        weights = bundle["weights"][target]
        clip = bundle["clip"][LABEL[target]]
        parts = {name: predict_member(bundle["members"][name, target], tables[bundle["members"][name, target]["encoder"]], clip) for name in weights}
        out[target] = sum(parts[n] * w for n, w in weights.items()) / sum(weights.values())
    return out


def fit_trees(config: dict[str, Any]) -> dict[str, Any]:
    """Refit the chosen blend on all development rows, save it, and check it reproduces E31's test predictions."""
    from durability.encoder_ensemble import _Data, forward, transform_kind

    settings, rules = config["encoder_ensemble"], config["e31_artifact"]
    data = _Data(config)
    spec = settings["choices"][rules["choice"]]
    clip = config["rich_early"]["label_clip"]
    quantiles = tuple(config["runner_state"]["model"]["guard_quantiles"])
    bundle: dict[str, Any] = {"choice": rules["choice"], "weights": {t: dict(spec[t]) for t in TARGETS}, "members": {}, "clip": {k: tuple(v) for k, v in clip.items()},
                              "population_mean": {}, "provenance": PROVENANCE}
    for target in TARGETS:
        raw = data.observed(target)
        usable = np.isfinite(raw)
        bundle["population_mean"][target] = float(np.clip(raw[usable], *clip[LABEL[target]]).mean())
        for name in spec[target]:
            member = settings["members"][name]
            development, _ = data.tables(member["encoder"])
            kind = transform_kind(member, target)
            train = development[usable]
            low, high = train.quantile(quantiles[0]), train.quantile(quantiles[1])
            models = member_models(member["model"], train.clip(low, high, axis=1), forward(np.clip(raw[usable], *clip[LABEL[target]]), kind), int(settings["bag_seeds"]), config)
            bundle["members"][name, target] = {"models": models, "low": low, "high": high, "kind": kind, "encoder": member["encoder"], "columns": list(development.columns)}
            LOGGER.info("E31 artifact member %s %s saved", name, target)
    tables = {kind: data.tables(kind)[1] for kind in ("old", "new")}
    mine = predict(bundle, tables)
    saved = pd.read_parquet(repo_path(config, settings["outputs"]["predictions"]))
    checks = {}
    for target in TARGETS:
        ref = saved[saved.model.eq(rules["choice"]) & saved.target.eq(target)].set_index("activity_id").predicted.reindex(data.test.activity_id.astype(str)).to_numpy(float)
        checks[f"tree_max_abs_diff_{LABEL[target]}"] = float(np.nanmax(np.abs(mine[target] - ref)))
    bundle["columns"] = {"state": list(data.columns), "resid": list(data.stack.resid_columns)}
    joblib.dump(bundle, _dir(config) / "trees.joblib")
    (_dir(config) / "tree_checks.json").write_text(json.dumps(checks, indent=2))
    LOGGER.info("E31 artifact tree checks: %s", checks)
    return checks


def athlete_tables(gated: pd.DataFrame, labels: pd.DataFrame, ids: np.ndarray, outputs: dict[str, np.ndarray], bundle: dict[str, Any], config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Input tables for gated rows, as ``encoder_ensemble._Data._take`` builds them for test athletes."""
    from durability import runner_state as rs

    columns, resid = bundle["columns"]["state"], bundle["columns"]["resid"]
    gated = gated.copy()
    for column in columns:
        if column not in gated:
            gated[column] = np.nan
    labels = labels.assign(activity_id=lambda d: d.activity_id.astype(str))
    tables = {}
    for kind, out in outputs.items():
        keep = labels[labels.activity_id.isin(ids)]
        residuals = rs._residuals(keep, ids, out, keep.athlete_id.unique(), config)
        residuals = residuals.set_index("activity_id") if len(residuals) else pd.DataFrame(columns=resid)
        position = pd.Series(np.arange(len(ids)), index=ids).reindex(gated.activity_id.astype(str)).to_numpy(float)
        values = np.full((len(gated), 2), np.nan)
        found = np.isfinite(position)
        values[found] = out[position[found].astype(int)]
        extra = residuals.reindex(gated.activity_id.astype(str)).reindex(columns=resid).reset_index(drop=True)
        tables[kind] = gated[columns].astype(float).reset_index(drop=True).join(pd.DataFrame(values, columns=["p_d", "p_m"])).join(extra)
    return tables


def whole_parity(config: dict[str, Any]) -> dict[str, Any]:
    """Saved encoders plus saved trees on the 472 E30 athletes, scored against E31's saved predictions."""
    from durability.encoder_ensemble import _Data

    settings, rules = config["encoder_ensemble"], config["e31_artifact"]
    bundle = joblib.load(_dir(config) / "trees.joblib")
    data = _Data(config)
    seq = np.load(repo_path(config, settings["test_dir"]) / "gc_seq.npz", allow_pickle=True)
    ids = seq["ids"].astype(str)
    outputs = encode_in_subprocess(None, seq["x"], seq["ld"])
    tables = athlete_tables(data.test, data.test_labels, ids, outputs, bundle, config)
    mine = predict(bundle, tables)
    saved = pd.read_parquet(repo_path(config, settings["outputs"]["predictions"]))
    people = data.test.athlete_id.astype(str).to_numpy()
    rows = []
    for target in TARGETS:
        observed = data.test[target].to_numpy(float)
        ref = saved[saved.model.eq(rules["choice"]) & saved.target.eq(target)].set_index("activity_id").predicted.reindex(data.test.activity_id.astype(str)).to_numpy(float)
        ok = np.isfinite(observed)
        score = lambda p: float(pd.Series(np.abs(observed[ok] - p[ok])).groupby(people[ok]).mean().mean())
        rows.append({"target": target, "mae_athlete_e31_saved": score(ref), "mae_athlete_artifact": score(mine[target]), "change": score(mine[target]) - score(ref),
                     "limit": float(rules["max_mae_athlete_change"][target]), "pearson_r_vs_saved": float(np.corrcoef(mine[target][ok], ref[ok])[0, 1]),
                     "n_runs": int(ok.sum()), "n_athletes": int(len(set(people[ok]))), "provenance": PROVENANCE})
    frame = pd.DataFrame(rows)
    frame["within_limit"] = frame.change.abs() <= frame.limit
    frame.to_csv(_dir(config) / "whole_parity.csv", index=False)
    LOGGER.info("E31 artifact whole parity:\n%s", frame.to_string(index=False))
    return {"table": frame}


def shrunk_history(labels: pd.DataFrame, gated: pd.DataFrame, target: str, population: float, strength: float) -> np.ndarray:
    """Own gated runs that ended before this run started, shrunk toward the training mean."""
    column = LABEL[target]
    done = labels[labels.gate_pass.astype(bool)]
    out = []
    for row in gated.itertuples():
        prior = done[(done.athlete_id == row.athlete_id) & (pd.to_datetime(done.end_time) < pd.Timestamp(row.start_time))][column].astype(float).dropna()
        out.append((prior.sum() + strength * population) / (len(prior) + strength))
    return np.asarray(out, float)


def predict_fit_folder(folder: str | Path, athlete: str, config: dict[str, Any]) -> dict[str, Any]:
    """FIT files -> E31 path -> blend predictions for gated runs, with actual values and two baselines."""
    from durability.fit_runner_state import process_fit_folder

    rules = config["e31_artifact"]
    bundle = joblib.load(_dir(config) / "trees.joblib")
    result = process_fit_folder(folder, athlete, config)
    gated, labels = pd.DataFrame(result["gated"]), pd.DataFrame(result["labels"])
    summary = {"counts": result["counts"], "gated_runs": len(gated), "sequences": len(result["sequences"])}
    if gated.empty:
        return {"summary": summary, "runs": pd.DataFrame()}
    ids = np.array([s[0] for s in result["sequences"]], dtype=str)
    outputs = encode_in_subprocess(None, np.stack([s[1] for s in result["sequences"]]), np.array([s[2] for s in result["sequences"]], np.float32))
    preds = predict(bundle, athlete_tables(gated, labels, ids, outputs, bundle, config))
    runs = gated[["activity_id", "start_time"]].copy()
    for target in TARGETS:
        runs[f"{LABEL[target]}_observed"] = gated[target].astype(float).to_numpy() if target in gated else gated[LABEL[target]].astype(float).to_numpy()
        runs[f"{LABEL[target]}_e31"] = preds[target]
        runs[f"{LABEL[target]}_training_mean"] = bundle["population_mean"][target]
        runs[f"{LABEL[target]}_own_shrunk"] = shrunk_history(labels, gated, target, bundle["population_mean"][target], float(rules["shrinkage_strength"]))
    return {"summary": summary, "runs": runs}


def report_fit_folder(folder: str | Path, athlete: str, config: dict[str, Any]) -> pd.DataFrame:
    """Run the FIT path, keep per-run rows private, and publish aggregate errors only."""
    from durability.level_change import markdown_table

    rules = config["e31_artifact"]
    result = predict_fit_folder(folder, athlete, config)
    private = repo_path(config, rules["private_dir"])
    private.mkdir(parents=True, exist_ok=True)
    runs = result["runs"]
    runs.to_parquet(private / f"{athlete}_runs.parquet", index=False)
    rows = []
    for name in ("drift", "magnitude"):
        if runs.empty:
            break
        for model in ("e31", "training_mean", "own_shrunk"):
            error = (runs[f"{name}_observed"] - runs[f"{name}_{model}"]).astype(float)
            rows.append({"target": name, "model": model, "n_runs": int(error.notna().sum()), "mae": float(error.abs().mean()), "bias": float(-error.mean()), "provenance": PROVENANCE + "; SINGLE ATHLETE"})
    table = pd.DataFrame(rows)
    counts = result["summary"]
    path = repo_path(config, rules["report"])
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# E31 model set: reproduction and FIT inference", "", f"**{PROVENANCE}.** Settings are CONFIGURED. The models are derived from FitRec and must not be redistributed.", "",
             "## Reproduction checks (472 E30 athletes, opened before)", ""]
    checks = {**json.loads((_dir(config) / "encoder_checks.json").read_text()), **{"trees": json.loads((_dir(config) / "tree_checks.json").read_text())}}
    lines += [f"- `{k}`: {v}" for k, v in checks.items()]
    lines += ["", markdown_table(pd.read_csv(_dir(config) / "whole_parity.csv").round(5)), "", "## FIT inference, one athlete (aggregate only)", "",
              f"Runs through the GoldenCheetah test path: {counts['counts']}. Gate-passing runs with E31 inputs: {counts['gated_runs']}.", ""]
    lines += [markdown_table(table.round(4)) if len(table) else "No gate-passing run, so E31 made no prediction.", "",
              "E31 was trained on gate-passing runs only, so only those runs are scored. One athlete cannot show generalisation; this checks that the path runs end to end."]
    path.write_text("\n".join(lines) + "\n")
    return table


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    stage = sys.argv[1]
    if stage == "encode":
        config = load_config(sys.argv[4]) if len(sys.argv) > 4 else load_config()
        loaded = np.load(sys.argv[2])
        np.savez(sys.argv[3], **encode(config, loaded["x"], loaded["ld"]))
        return
    config = load_config()
    if stage == "fit":
        print(report_fit_folder(sys.argv[2], sys.argv[3], config).round(4).to_string(index=False))
        return
    {"encoders": train_encoders, "trees": fit_trees, "parity": whole_parity}[stage](config)


if __name__ == "__main__":
    main()
