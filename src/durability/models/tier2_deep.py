"""Compact multi-task implementations of the four Tier-2 architectures."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from durability.config import repo_path
from durability.features import prepare_early_window
from durability.preprocess import reload_preprocessed_stream

LOGGER = logging.getLogger(__name__)
SEQUENCE_COLUMNS = (
    "sequence_heart_rate",
    "sequence_gap_speed",
    "sequence_gradient",
    "sequence_relative_time",
)


def _fill_channel(values: np.ndarray) -> np.ndarray:
    series = pd.Series(values, dtype=float)
    if not series.notna().any():
        return np.zeros(len(series), dtype=float)
    return (
        series.interpolate(limit_direction="both")
        .fillna(float(series.median()))
        .to_numpy(dtype=float)
    )


def materialize_sequence_features(
    features: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    """Downsample the allowed early window into fixed raw-channel sequences."""
    sequence_length = int(config["models"]["tier2"]["sequence_length"])
    rows: list[dict[str, Any]] = []
    progress_every = int(config["data"]["feature_progress_every"])
    for number, target in enumerate(
        features.itertuples(index=False), start=1
    ):
        stream = reload_preprocessed_stream(
            config, target.athlete_id, target.activity_id
        )
        early = prepare_early_window(
            stream,
            {
                "hrmax_bpm": target.hrmax_bpm_asof,
                "cs_ms": target.cs_ms_asof,
                "dprime_m": target.dprime_m_asof,
            },
            config,
        )
        if len(early) < sequence_length:
            raise AssertionError(
                "Configured sequence length would upsample an early window"
            )
        selected = np.linspace(
            0, len(early) - 1, sequence_length
        ).round().astype(int)
        heart_rate_raw = _fill_channel(
            early["heart_rate"].to_numpy(dtype=float)
        )
        gap_speed_raw = _fill_channel(
            early["gap_speed_ms"].to_numpy(dtype=float)
        )
        gradient_raw = _fill_channel(
            early["gradient"].to_numpy(dtype=float)
        )
        elapsed = early["elapsed_s"].to_numpy(dtype=float)
        elapsed_h = (elapsed - elapsed[0]) / 3_600.0
        finite_speed = np.isfinite(gap_speed_raw) & np.isfinite(elapsed_h)
        speed_slope = (
            float(np.polyfit(elapsed_h[finite_speed], gap_speed_raw[finite_speed], 1)[0])
            if finite_speed.sum()
            >= int(config["features"]["min_slope_observations"])
            and np.ptp(elapsed_h[finite_speed]) > 0
            else np.nan
        )
        heart_rate = heart_rate_raw / float(target.hrmax_bpm_asof)
        gap_speed = gap_speed_raw / float(target.cs_ms_asof)
        gradient = gradient_raw / float(config["gap"]["gradient_clip"])
        relative_time = (
            elapsed - elapsed[0]
        ) / float(target.early_cutoff_elapsed_s)
        rows.append(
            {
                "activity_id": str(target.activity_id),
                "athlete_id": str(target.athlete_id),
                "sequence_heart_rate": heart_rate[selected].tolist(),
                "sequence_gap_speed": gap_speed[selected].tolist(),
                "sequence_gradient": gradient[selected].tolist(),
                "sequence_relative_time": relative_time[selected].tolist(),
                "early_gap_speed_slope_ms_per_h": speed_slope,
                "source_early_samples": len(early),
                "sequence_samples": sequence_length,
                "provenance": "MEASURED",
            }
        )
        if number % progress_every == 0:
            LOGGER.info(
                "Tier-2 sequence progress: %s/%s",
                number,
                len(features),
            )
    result = pd.DataFrame(rows)
    path = repo_path(config, config["data"]["sequence_features"])
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(path, index=False)
    return result


def sequence_array(frame: pd.DataFrame) -> np.ndarray:
    """Convert Parquet list columns to [activities, time, channels]."""
    channels = [
        np.stack(frame[column].map(np.asarray).to_numpy())
        for column in SEQUENCE_COLUMNS
    ]
    return np.stack(channels, axis=2).astype(np.float32)


class TCNRegressor(nn.Module):
    def __init__(self, channels: int, hidden: int, outputs: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(
                hidden,
                hidden,
                kernel_size=3,
                padding=2,
                dilation=2,
            ),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden, outputs),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values.transpose(1, 2))


class LSTMRegressor(nn.Module):
    def __init__(self, channels: int, hidden: int, outputs: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(channels, hidden, batch_first=True)
        self.head = nn.Linear(hidden, outputs)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(values)
        return self.head(hidden[-1])


class PatchTSTRegressor(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden: int,
        outputs: int,
        patch_length: int,
        patch_stride: int,
        heads: int,
    ) -> None:
        super().__init__()
        self.patch_length = patch_length
        self.patch_stride = patch_stride
        self.embedding = nn.Linear(channels * patch_length, hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 2,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.head = nn.Linear(hidden, outputs)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        patches = values.unfold(
            dimension=1,
            size=self.patch_length,
            step=self.patch_stride,
        )
        patches = patches.reshape(patches.shape[0], patches.shape[1], -1)
        encoded = self.encoder(self.embedding(patches))
        return self.head(encoded.mean(dim=1))


class ITransformerRegressor(nn.Module):
    def __init__(
        self,
        channels: int,
        sequence_length: int,
        hidden: int,
        outputs: int,
        heads: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Linear(sequence_length, hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 2,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.head = nn.Linear(hidden, outputs)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        variable_tokens = self.embedding(values.transpose(1, 2))
        encoded = self.encoder(variable_tokens)
        return self.head(encoded.mean(dim=1))


def make_deep_model(
    model_name: str,
    *,
    channels: int,
    sequence_length: int,
    outputs: int,
    config: dict[str, Any],
) -> nn.Module:
    settings = config["models"]["tier2"]
    hidden = int(settings["hidden_size"])
    if model_name == "tier2_tcn":
        return TCNRegressor(channels, hidden, outputs)
    if model_name == "tier2_lstm":
        return LSTMRegressor(channels, hidden, outputs)
    if model_name == "tier2_patchtst":
        return PatchTSTRegressor(
            channels,
            hidden,
            outputs,
            int(settings["patch_length"]),
            int(settings["patch_stride"]),
            int(settings["attention_heads"]),
        )
    if model_name == "tier2_itransformer":
        return ITransformerRegressor(
            channels,
            sequence_length,
            hidden,
            outputs,
            int(settings["attention_heads"]),
        )
    raise ValueError(f"Unknown Tier-2 model: {model_name}")


def fit_deep_model(
    model_name: str,
    train_sequences: np.ndarray,
    train_targets: np.ndarray,
    prediction_sequences: np.ndarray,
    config: dict[str, Any],
    *,
    seed: int,
    epochs: int | None = None,
) -> tuple[np.ndarray, nn.Module, np.ndarray, np.ndarray]:
    """Fit one masked multi-task model and return predictions in raw units."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    settings = config["models"]["tier2"]
    x_train = np.asarray(train_sequences, dtype=np.float32)
    y_train = np.asarray(train_targets, dtype=np.float32)
    mask = np.isfinite(y_train)
    target_mean = np.nanmean(y_train, axis=0)
    target_std = np.nanstd(y_train, axis=0)
    target_std[target_std == 0] = 1.0
    normalised = (y_train - target_mean) / target_std
    normalised[~mask] = 0.0
    dataset = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(normalised),
        torch.from_numpy(mask.astype(np.float32)),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=int(settings["batch_size"]),
        shuffle=True,
        generator=generator,
    )
    model = make_deep_model(
        model_name,
        channels=x_train.shape[2],
        sequence_length=x_train.shape[1],
        outputs=y_train.shape[1],
        config=config,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    model.train()
    n_epochs = int(settings["epochs"] if epochs is None else epochs)
    for _ in range(n_epochs):
        for values, targets, target_mask in loader:
            optimizer.zero_grad()
            estimate = model(values)
            squared = torch.square(estimate - targets) * target_mask
            loss = squared.sum() / target_mask.sum().clamp_min(1.0)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        prediction = model(
            torch.from_numpy(
                np.asarray(prediction_sequences, dtype=np.float32)
            )
        ).numpy()
    prediction = prediction * target_std + target_mean
    return prediction, model, target_mean, target_std


def fit_deep_model_with_history(
    model_name: str,
    train_sequences: np.ndarray,
    train_targets: np.ndarray,
    validation_sequences: np.ndarray,
    validation_targets: np.ndarray,
    config: dict[str, Any],
    *,
    seed: int,
    epochs: int | None = None,
) -> tuple[pd.DataFrame, nn.Module]:
    """Run a diagnostic fit and record masked normalised MSE per epoch."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    settings = config["models"]["tier2"]
    x_train = np.asarray(train_sequences, dtype=np.float32)
    x_validation = np.asarray(validation_sequences, dtype=np.float32)
    y_train = np.asarray(train_targets, dtype=np.float32)
    y_validation = np.asarray(validation_targets, dtype=np.float32)

    train_mask = np.isfinite(y_train)
    validation_mask = np.isfinite(y_validation)
    target_mean = np.nanmean(y_train, axis=0)
    target_std = np.nanstd(y_train, axis=0)
    if not np.isfinite(target_mean).all():
        raise AssertionError("Every diagnostic target needs training labels")
    target_std[~np.isfinite(target_std) | (target_std == 0)] = 1.0
    train_normalised = (y_train - target_mean) / target_std
    validation_normalised = (y_validation - target_mean) / target_std
    train_normalised[~train_mask] = 0.0
    validation_normalised[~validation_mask] = 0.0

    def dataset(
        values: np.ndarray, targets: np.ndarray, mask: np.ndarray
    ) -> TensorDataset:
        return TensorDataset(
            torch.from_numpy(values),
            torch.from_numpy(targets.astype(np.float32)),
            torch.from_numpy(mask.astype(np.float32)),
        )

    train_dataset = dataset(x_train, train_normalised, train_mask)
    validation_dataset = dataset(
        x_validation, validation_normalised, validation_mask
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(settings["batch_size"]),
        shuffle=True,
        generator=generator,
    )
    train_evaluation_loader = DataLoader(
        train_dataset,
        batch_size=int(settings["batch_size"]),
        shuffle=False,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(settings["batch_size"]),
        shuffle=False,
    )

    model = make_deep_model(
        model_name,
        channels=x_train.shape[2],
        sequence_length=x_train.shape[1],
        outputs=y_train.shape[1],
        config=config,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )

    def masked_mse(loader: DataLoader) -> float:
        model.eval()
        squared_error = 0.0
        observed = 0.0
        with torch.no_grad():
            for values, targets, target_mask in loader:
                estimate = model(values)
                squared_error += float(
                    (torch.square(estimate - targets) * target_mask)
                    .sum()
                    .item()
                )
                observed += float(target_mask.sum().item())
        if observed == 0:
            raise AssertionError("Diagnostic split has no observed targets")
        return squared_error / observed

    rows: list[dict[str, Any]] = []
    n_epochs = int(settings["epochs"] if epochs is None else epochs)
    for epoch in range(1, n_epochs + 1):
        model.train()
        for values, targets, target_mask in train_loader:
            optimizer.zero_grad()
            estimate = model(values)
            squared = torch.square(estimate - targets) * target_mask
            loss = squared.sum() / target_mask.sum().clamp_min(1.0)
            loss.backward()
            optimizer.step()
        rows.append(
            {
                "model": model_name,
                "epoch": epoch,
                "train_normalised_mse": masked_mse(
                    train_evaluation_loader
                ),
                "validation_normalised_mse": masked_mse(
                    validation_loader
                ),
                "provenance": "MEASURED",
            }
        )
    return pd.DataFrame(rows), model
