from src.model.encoder.encoder import Encoder
from src.model.encoder.encoder_infinidepth_query import (
    EncoderInfiniDepthQuery,
    EncoderInfiniDepthQueryCfg,
)
from src.model.encoder.encoder_infinisplat import EncoderInfiniSplat, EncoderInfiniSplatCfg

ENCODERS = {
    "infinisplat": EncoderInfiniSplat,
    "infinisplat_infinidepth": EncoderInfiniDepthQuery,
}

EncoderCfg = EncoderInfiniSplatCfg | EncoderInfiniDepthQueryCfg


def get_encoder(cfg: EncoderCfg) -> Encoder:
    encoder = ENCODERS[cfg.name]
    encoder = encoder(cfg)
    return encoder
