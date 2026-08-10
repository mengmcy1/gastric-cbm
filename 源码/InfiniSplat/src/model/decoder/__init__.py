from src.model.decoder.decoder import Decoder
from src.model.decoder.decoder_gsplat import DecoderGsplat, DecoderGsplatCfg

DECODERS = {
    "gsplat": DecoderGsplat,
}

DecoderCfg = DecoderGsplatCfg


def get_decoder(cfg: DecoderCfg) -> Decoder:
    decoder = DECODERS[cfg.name]
    decoder = decoder(cfg)
    return decoder


__all__ = [
    "Decoder",
    "DecoderCfg",
    "DecoderGsplat",
    "DecoderGsplatCfg",
    "get_decoder",
]
