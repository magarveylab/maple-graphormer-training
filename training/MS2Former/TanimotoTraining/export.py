import argparse
import os

import torch
from DataModule import MS2DataModule
from models import get_model

from omnicons import experiment_dir
from omnicons.models.Compilers import compile_with_torchscript
from omnicons.models.encoders.SentenceEncoder import (
    alternative_forward_for_torchscript,
)


def compile_model(
    pytorch_checkpoint_fp: str = f"{experiment_dir}/MS2-tanimoto/checkpoints/last.pt",
    torchscript_dir: str = f"{experiment_dir}/MS2-tanimoto/torchscript",
    node_embedding_dim: int = 128,
    edge_embedding_dim: int = 128,
    num_gnn_heads: int = 8,
    num_transformer_heads: int = 8,
):
    # data module
    dm = MS2DataModule()
    weights = dm.calculate_weights()
    # model
    model = get_model(
        weights=weights,
        node_embedding_dim=int(node_embedding_dim),
        edge_embedding_dim=int(edge_embedding_dim),
        num_gnn_heads=int(num_gnn_heads),
        num_transformer_heads=int(num_transformer_heads),
    )
    # have to modify forward method for torchscript
    model.model.node_encoders["MZ"].forward = (
        alternative_forward_for_torchscript
    )
    model.model.node_encoders["NL"].forward = (
        alternative_forward_for_torchscript
    )
    # load weights
    states = torch.load(pytorch_checkpoint_fp)
    model.load_state_dict(states["state_dict"], strict=True)
    # export model
    os.makedirs(torchscript_dir, exist_ok=True)
    # compile models
    os.makedirs(f"{torchscript_dir}/node_encoders", exist_ok=True)
    for node_type, node_encoder in model.model.node_encoders.items():
        compile_with_torchscript(
            model=node_encoder,
            model_fp=f"{torchscript_dir}/node_encoders/{node_type}.pt",
        )
    compile_with_torchscript(
        model=model.model.edge_type_encoder,
        model_fp=f"{torchscript_dir}/edge_type_encoder.pt",
    )
    compile_with_torchscript(
        model=model.model.gnn, model_fp=f"{torchscript_dir}/gnn.pt"
    )
    compile_with_torchscript(
        model=model.model.transformer,
        model_fp=f"{torchscript_dir}/transformer.pt",
    )


parser = argparse.ArgumentParser(
    description="Convert MS2Former model (for analog embeddings) to torchscript format"
)
parser.add_argument(
    "-torchscript_dir",
    help="Directory to save torchscript models",
    default=f"{experiment_dir}/MS2-tanimoto/torchscript",
)
parser.add_argument(
    "-pytorch_checkpoint_fp",
    help="Pytorch checkpoint file path",
    default=f"{experiment_dir}/MS2-tanimoto/checkpoints/last.pt",
)

if __name__ == "__main__":
    args = parser.parse_args()
    compile_model(
        torchscript_dir=args.torchscript_dir,
        pytorch_checkpoint_fp=args.pytorch_checkpoint_fp,
    )
