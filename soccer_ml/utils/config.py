"""Configuration loader and schema validation."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any
import yaml


@dataclass
class DataConfig:
    num_players: int = 22
    node_dim: int = 7
    edge_dim: int = 1
    graph_dim: int = 4
    pitch_length: float = 105.0
    pitch_width: float = 68.0


@dataclass
class ModelConfig:
    hidden_dim: int = 64
    num_layers: int = 4
    nb_heads: int = 8
    dropout: float = 0.0
    enable_receiver_head: bool = True
    enable_shot_head: bool = True
    enable_creativity_head: bool = True
    enable_vae_head: bool = True
    vae_latent_dim: int = 16


@dataclass
class TrainingConfig:
    batch_size: int = 128
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    clip_grad_norm: float = 1.0
    checkpoint_dir: str = "checkpoints"


@dataclass
class LossWeightsConfig:
    receiver_weight: float = 1.0
    shot_weight: float = 1.0
    creativity_weight: float = 1.0
    vae_weight: float = 1.0
    vae_kl_weight: float = 0.01


@dataclass
class SoccerConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss_weights: LossWeightsConfig = field(default_factory=LossWeightsConfig)


def load_config(config_path: Optional[str] = None) -> SoccerConfig:
    """Load configuration from a YAML file, falling back to default values."""
    if config_path is None or not Path(config_path).exists():
        return SoccerConfig()

    with open(config_path, "r", encoding="utf-8") as f:
        raw_dict = yaml.safe_load(f) or {}

    data_cfg = DataConfig(**raw_dict.get("data", {}))
    model_cfg = ModelConfig(**raw_dict.get("model", {}))
    train_cfg = TrainingConfig(**raw_dict.get("training", {}))
    loss_cfg = LossWeightsConfig(**raw_dict.get("loss_weights", {}))

    return SoccerConfig(
        data=data_cfg,
        model=model_cfg,
        training=train_cfg,
        loss_weights=loss_cfg,
    )
