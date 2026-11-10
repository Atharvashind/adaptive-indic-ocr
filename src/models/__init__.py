"""Model adapter registration and construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.models.base_adapter import BaseOCRAdapter
from src.models.crnn_adapter import CRNNAdapter

MODEL_ADAPTERS = {"crnn": CRNNAdapter}


def load_model_adapter(
    name: str,
    config_path: Path | None = None,
    project_root: Path | None = None,
) -> BaseOCRAdapter:
    """Construct a configured adapter through the shared model registry."""
    root = (project_root or Path(__file__).resolve().parents[2]).resolve()
    selected_config = config_path or root / "configs" / "models.yaml"
    if not selected_config.is_absolute():
        selected_config = root / selected_config
    if not selected_config.is_file():
        raise FileNotFoundError(f"Model configuration not found: {selected_config}")
    configuration: Any = yaml.safe_load(selected_config.read_text(encoding="utf-8"))
    models = configuration.get("models") if isinstance(configuration, dict) else None
    if not isinstance(models, dict) or name not in models:
        raise ValueError(f"Model {name!r} is not configured in {selected_config}")
    adapter_type = MODEL_ADAPTERS.get(name)
    if adapter_type is None:
        raise ValueError(f"No adapter implementation is registered for model {name!r}")
    model_config = models[name]
    if not isinstance(model_config, dict):
        raise ValueError(f"Configuration for model {name!r} must be a mapping")
    return adapter_type(model_config, root)


__all__ = ["BaseOCRAdapter", "CRNNAdapter", "MODEL_ADAPTERS", "load_model_adapter"]
