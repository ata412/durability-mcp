"""Configuration for the packaged tool."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8") if path else resources.files("durability").joinpath("default.yaml").read_text(encoding="utf-8")
    config = yaml.safe_load(text)
    config["_repo_root"] = str(Path.cwd())
    return config


def repo_path(config: dict[str, Any], value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(config["_repo_root"]) / path
