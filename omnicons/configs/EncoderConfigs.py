from typing import Union

from torch import nn

from omnicons.configs.Config import ConfigTemplate
from omnicons.models.encoders.SentenceEncoder import (
    EncoderDict,
    SentenceEncoder,
)
from omnicons.models.encoders.WordEncoder import WordEncoder


class SentenceEncoderConfig(ConfigTemplate):

    def __init__(
        self,
        sentence_structure: list[str],
        encoder_dicts: list[EncoderDict],
        dropout: float = 0.1,
        mlp_layers: int = 1,
    ):
        super().__init__(
            base="SentenceEncoder",
            properties={
                "sentence_structure": sentence_structure,
                "encoder_dicts": encoder_dicts,
                "dropout": dropout,
                "mlp_layers": mlp_layers,
            },
        )

    def get_model(self) -> nn.Module:
        return SentenceEncoder(**self.properties)


class WordEncoderConfig(ConfigTemplate):

    def __init__(
        self,
        num_embeddings: int = 100,
        embedding_dim: int = 128,
        extra_features: int = 0,
        dropout: float = 0.1,
        mlp_layers: int = 1,
    ):
        super().__init__(
            base="WordEncoder",
            properties={
                "num_embeddings": num_embeddings,
                "embedding_dim": embedding_dim,
                "extra_features": extra_features,
                "dropout": dropout,
                "mlp_layers": mlp_layers,
            },
        )

    def get_model(self) -> nn.Module:
        return WordEncoder(**self.properties)


EncoderConfig = Union[SentenceEncoderConfig, WordEncoderConfig]
