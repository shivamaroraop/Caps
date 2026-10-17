"""YAML config loading and mapping onto the typed dataclasses."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import yaml

from ..env.community_env import EnvConfig
from ..marl.mappo import MAPPOConfig
from ..marl.trainer import TrainConfig


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _filter(cls, d: dict) -> dict:
    valid = set(cls.__dataclass_fields__)
    unknown = set(d) - valid
    if unknown:
        raise KeyError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    return {k: v for k, v in d.items() if k in valid}


def build_configs(cfg: dict) -> Tuple[EnvConfig, MAPPOConfig, TrainConfig]:
    """Fail loudly on unknown keys: a silently ignored hyperparameter is the
    kind of bug that makes an ablation report a difference that never existed."""
    return (EnvConfig(**_filter(EnvConfig, cfg.get("env", {}))),
            MAPPOConfig(**_filter(MAPPOConfig, cfg.get("algo", {}))),
            TrainConfig(**_filter(TrainConfig, cfg.get("train", {}))))
