"""E29 encoder stage. Runs in its own process: torch must never share a process with LightGBM fits."""
from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import torch
from torch import nn

from durability.config import repo_path
from durability.runner_state_features import network_keys

LOGGER = logging.getLogger(__name__)


class _Block(nn.Module):
    def __init__(self, channels: int, dilation: int, kernel: int, dropout: float) -> None:
        super().__init__()
        pad = (kernel // 2) * dilation
        self.net = nn.Sequential(nn.Conv1d(channels, channels, kernel, padding=pad, dilation=dilation), nn.GroupNorm(8, channels), nn.GELU(), nn.Dropout(dropout),
                                 nn.Conv1d(channels, channels, kernel, padding=pad, dilation=dilation), nn.GroupNorm(8, channels), nn.GELU())

    def forward(self, x):
        return x + self.net(x)


class Encoder(nn.Module):
    def __init__(self, outputs: int, rules: dict[str, Any]) -> None:
        super().__init__()
        hidden, kernel, dropout = int(rules["hidden"]), int(rules["kernel"]), float(rules["dropout"])
        self.first = nn.Conv1d(4, hidden, 1)
        self.blocks = nn.Sequential(*[_Block(hidden, int(d), kernel, dropout) for d in rules["dilations"]])
        self.head = nn.Sequential(nn.Linear(2 * hidden + 1, int(rules["head"])), nn.GELU(), nn.Dropout(dropout), nn.Linear(int(rules["head"]), outputs))

    def forward(self, x, log_duration):
        z = self.blocks(self.first(x.transpose(1, 2)))
        return self.head(torch.cat([z.mean(2), z.amax(2), log_duration[:, None]], 1))


def _train(data, excluded, config, device):
    rules = config["runner_state"]["encoder"]
    seed = int(config["seed"]) + sum(excluded)
    x, y, ld, fold, athlete = data["x"], data["y"], data["ld"], data["fold"], data["athlete"]
    use = ~np.isin(fold, list(excluded))
    mean, sd = np.nanmean(y[use], 0), np.nanstd(y[use], 0) + 1e-6
    xm, xs = x[use].mean((0, 1)), x[use].std((0, 1)) + 1e-6
    lm, ls = ld[use].mean(), ld[use].std()
    people = np.unique(athlete[use])
    held = np.random.default_rng(seed).choice(people, int(len(people) * float(rules["val_athlete_frac"])), replace=False)
    fit = torch.from_numpy(np.flatnonzero(use & ~np.isin(athlete, held))).to(device)
    val = torch.from_numpy(np.flatnonzero(use & np.isin(athlete, held))).to(device)
    tx = torch.from_numpy(((x - xm) / xs).astype(np.float32)).to(device)
    ty = torch.from_numpy(((y - mean) / sd).astype(np.float32)).to(device)
    tl = torch.from_numpy(((ld - lm) / ls).astype(np.float32)).to(device)
    torch.manual_seed(seed)
    net = Encoder(y.shape[1], rules).to(device)
    opt = torch.optim.AdamW(net.parameters(), float(rules["learning_rate"]), weight_decay=float(rules["weight_decay"]))

    def loss(batch):
        pred, truth = net(tx[batch], tl[batch]), ty[batch]
        ok = torch.isfinite(truth)
        return nn.functional.huber_loss(pred[ok], truth[ok], delta=float(rules["huber_delta"]))
    best, state, wait, started = np.inf, None, 0, time.time()
    for epoch in range(int(rules["max_epochs"])):
        net.train()
        for batch in fit[torch.randperm(len(fit), device=device)].split(int(rules["batch_size"])):
            opt.zero_grad()
            loss(batch).backward()
            nn.utils.clip_grad_norm_(net.parameters(), 2.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            score = float(np.mean([loss(batch).item() for batch in val.split(int(rules["inference_batch"]))]))
        if score < best - 1e-4:
            best, state, wait = score, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= int(rules["patience"]):
                break
    net.load_state_dict(state)
    net.eval()
    LOGGER.info("E29 encoder excluded=%s rows=%s epochs=%s val=%.4f sec=%.0f", excluded, len(fit), epoch + 1, best, time.time() - started)
    norm = dict(mean=mean[:2], sd=sd[:2], xm=xm, xs=xs, lm=lm, ls=ls)

    def apply(px, pl):
        ax = torch.from_numpy(((px - xm) / xs).astype(np.float32)).to(device)
        al = torch.from_numpy(((pl - lm) / ls).astype(np.float32)).to(device)
        with torch.no_grad():
            out = np.concatenate([net(ax[b], al[b]).cpu().numpy()[:, :2] for b in torch.arange(len(ax), device=device).split(int(rules["inference_batch"]))])
        return out * norm["sd"] + norm["mean"]
    apply.net, apply.norm = net, norm  # exposed so a trained network can be saved; behaviour unchanged
    return apply


def run_encoder_stage(config: dict[str, Any]) -> None:
    work = repo_path(config, config["runner_state"]["outputs"]["work_dir"])
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    loaded = np.load(work / "fitrec_seq.npz", allow_pickle=True)
    data = {k: loaded[k] for k in loaded.files}
    outside = np.load(work / "gc_seq.npz", allow_pickle=True)
    if (work / "fitrec_out.npz").exists():
        apply = _train(data, (), config, device)
        np.savez(work / "gc_out.npz", ids=outside["ids"], out=apply(outside["x"], outside["ld"]))
        first = repo_path(config, config["runner_state"]["outputs"]["first_run_dir"])
        if (first / "gc_seq.npz").exists():
            old_seq, old_out = np.load(first / "gc_seq.npz", allow_pickle=True), np.load(first / "gc_out.npz", allow_pickle=True)
            again = apply(old_seq["x"], old_seq["ld"])
            LOGGER.info("E29 retrained network vs first run, Pearson r: drift %.4f, magnitude %.4f", *[np.corrcoef(again[:, i], old_out["out"][:, i])[0, 1] for i in range(2)])
        return
    keys = network_keys()
    outputs = np.empty((len(keys), len(data["x"]), 2), dtype=np.float32)
    for i, key in enumerate(keys):
        apply = _train(data, key, config, device)
        outputs[i] = apply(data["x"], data["ld"])
        if key == ():
            np.savez(work / "gc_out.npz", ids=outside["ids"], out=apply(outside["x"], outside["ld"]))
    np.savez(work / "fitrec_out.npz", ids=data["ids"], keys=np.array(["_".join(map(str, k)) for k in keys]), out=outputs)
