"""Configuration loading utilities."""

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a UTF-8 YAML file and require a mapping at its root."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8-sig") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")
    return config
