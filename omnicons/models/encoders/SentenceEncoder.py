from typing import List, TypedDict

import torch
from torch import nn
from torch.nn import ModuleDict

from omnicons.models.encoders.MLPEncoder import MLPEncoder


class EncoderDict(TypedDict):
    name: str
    num_embeddings: int
    embedding_dim: int


def alternative_forward_for_torchscript(self, x: torch.Tensor) -> torch.Tensor:
    new_x = torch.empty(
        (x.shape[0], self.length, self.embedding_dim), device=x.device
    )
    # process nodes in the sentence structure
    for n, key in self.keys:
        new_x[:, n] = self.call_encoder(x, n, key)
    # concatenate all embeddings
    new_x = new_x.reshape(x.shape[0], -1)
    return self.mlp(new_x)


class SentenceEncoder(nn.Module):

    def __init__(
        self,
        sentence_structure: List[str],
        encoder_dicts: List[EncoderDict],
        dropout: float = 0.1,
        mlp_layers: int = 1,
    ):
        super().__init__()
        self.sentence_structure = sentence_structure
        self.encoder_dict = ModuleDict()
        name_to_dim = {}
        for encoder in encoder_dicts:
            name = encoder["name"]
            self.encoder_dict[name] = nn.Embedding(
                num_embeddings=encoder["num_embeddings"],
                embedding_dim=encoder["embedding_dim"],
            )
            name_to_dim[name] = encoder["embedding_dim"]
        total_dim = sum([name_to_dim[name] for name in sentence_structure])
        embedding_dim = encoder_dicts[0]["embedding_dim"]
        self.embedding_dim = embedding_dim
        self.length = len(sentence_structure)
        self.mlp = MLPEncoder(
            input_dim=total_dim,
            output_dim=embedding_dim,
            dropout=dropout,
            num_layers=mlp_layers,
        )
        self.keys = [
            (n, self.sentence_structure[n]) for n in range(self.length)
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new_x = torch.empty(
            (x.shape[0], self.length, self.embedding_dim), device=x.device
        ).to(torch.float16)
        # process nodes in the sentence structure
        for n, key in self.keys:
            new_x[:, n] = self.encoder_dict[key](x[:, n])
        # concatenate all embeddings
        new_x = new_x.reshape(x.shape[0], -1)
        return self.mlp(new_x)

    def call_encoder(self, x: torch.Tensor, n: int, name: str) -> torch.Tensor:
        # only used in the torchscript version
        result = x
        for key, module in self.encoder_dict.items():
            if key == name:
                result = module(x[:, n])
        return result
