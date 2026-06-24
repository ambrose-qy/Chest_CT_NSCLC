"""
Load maintained LIDC training defaults from a standalone YAML file.
"""

from __future__ import print_function

from copy import deepcopy
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "lidc_training.yaml"


def load_training_config(config_path=None):
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError("Missing LIDC training configuration: {}".format(path))
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload.get("common"), dict):
        raise ValueError("Training configuration must contain a 'common' mapping.")
    if not isinstance(payload.get("models"), dict):
        raise ValueError("Training configuration must contain a 'models' mapping.")
    return payload


def get_model_hyperparameters(profile_name, overrides=None, config_path=None):
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    payload = load_training_config(config_path=path)
    profiles = payload["models"]
    if profile_name not in profiles:
        raise KeyError(
            "Unknown training profile '{}'. Available profiles: {}".format(
                profile_name,
                ", ".join(sorted(profiles)),
            )
        )
    params = deepcopy(payload["common"])
    params.update(deepcopy(profiles[profile_name]))
    params["training_config_path"] = str(path.resolve())
    params["training_config_profile"] = profile_name
    if overrides:
        params.update(overrides)
    return params
