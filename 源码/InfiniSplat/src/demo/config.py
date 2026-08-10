from dataclasses import dataclass
from typing import Type, TypeVar

from dacite import from_dict
from omegaconf import DictConfig, OmegaConf

from src.model.decoder import DecoderCfg
from src.model.encoder import EncoderCfg


@dataclass
class ModelCfg:
    decoder: DecoderCfg
    encoder: EncoderCfg


@dataclass
class RootCfg:
    model: ModelCfg


T = TypeVar("T")


def load_typed_config(cfg: DictConfig, data_class: Type[T]) -> T:
    """Convert one resolved Hydra config into its inference dataclass."""
    return from_dict(data_class, OmegaConf.to_container(cfg, resolve=True))


def load_typed_root_config(cfg: DictConfig) -> RootCfg:
    """Load the typed root configuration used by demo inference."""
    return load_typed_config(cfg, RootCfg)
