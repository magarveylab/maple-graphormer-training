import json
from typing import Callable, Optional

import torch
from Maple.Embedder.graphs.MS1Graph import get_node_vocab, get_word_vocab
from torch.nn import ModuleDict

from omnicons import dataset_dir
from omnicons.lightning.HeteroGraphModelForMultiTask import (
    HeteroGraphModelForMultiTaskLightning,
)
from omnicons.metrics import ClassificationMetrics
from omnicons.optimizers.preconfigured import get_deepspeed_adamw


def get_node_encoders(
    node_vocab: dict, word_vocab: dict, embedding_dim: int = 128
):
    from omnicons.configs.EncoderConfigs import (
        SentenceEncoderConfig,
        WordEncoderConfig,
    )

    node_encoders = {}
    for node_type in node_vocab:
        node_encoders[node_type] = WordEncoderConfig(
            num_embeddings=len(node_vocab[node_type]),
            embedding_dim=embedding_dim,
            dropout=0.1,
            mlp_layers=1,
        )
    # prepare peak encoder
    encoder_dicts = [
        {
            "name": s,
            "num_embeddings": len(word_vocab[s]),
            "embedding_dim": embedding_dim,
        }
        for s in ["MZbase", "MZdec", "Charge", "Intensity"]
    ]
    sentence_structure = [
        "MZbase",
        "MZdec",
        "MZbase",
        "MZdec",
        "MZbase",
        "MZdec",
        "Charge",
        "Intensity",
    ]
    node_encoders["Peak"] = SentenceEncoderConfig(
        sentence_structure=sentence_structure,
        encoder_dicts=encoder_dicts,
        dropout=0.1,
        mlp_layers=1,
    )

    return node_encoders


def get_edge_type_encoder(edge_types: list, embedding_dim: int = 10):
    from omnicons.configs.EncoderConfigs import WordEncoderConfig

    return WordEncoderConfig(
        num_embeddings=len(edge_types),
        embedding_dim=embedding_dim,
        extra_features=0,
        dropout=0.1,
        mlp_layers=1,
    )


def get_gnn(
    node_embedding_dim: int = 128,
    edge_embedding_dim: int = 128,
    num_heads: int = 8,
):
    from omnicons.configs.GNNConfigs import GATConfig

    gnn_config = GATConfig(
        num_layers=num_heads,
        num_heads=num_heads,
        embed_dim=node_embedding_dim,
        edge_dim=edge_embedding_dim,
        dropout=0.1,
    )
    return gnn_config


def get_transformer(embedding_dim: int = 128, num_heads: int = 8):
    from omnicons.configs.TransformerConfigs import GraphormerConfig

    transformer_config = GraphormerConfig(
        num_layers=num_heads,
        num_heads=num_heads,
        embed_dim=embedding_dim,
        dropout=0.1,
        attention_dropout=0.1,
        mlp_dropout=0.1,
    )
    return transformer_config


def get_graph_pooler(embedding_dim: int = 128):
    from omnicons.configs.GraphPoolerConfigs import HeteroNodeClsPoolerConfig

    graph_pooler_config = HeteroNodeClsPoolerConfig(
        node_type="Spectra", index_selector=1, hidden_channels=embedding_dim
    )
    return graph_pooler_config


def get_heads(weights: dict, tax_class_dict: dict, embedding_dim: int = 128):
    from omnicons.configs.HeadConfigs import GraphClsTaskHeadConfig

    heads = {}
    for level in ["phylum", "class", "order", "family", "genus"]:
        heads[level] = GraphClsTaskHeadConfig(
            hidden_size=embedding_dim,
            hidden_dropout_prob=0.1,
            class_weight=weights[level].tolist(),
            num_labels=len(tax_class_dict[level]),
            analyze_inputs=["a"],
        )
    return heads


def get_model(
    weights: dict,
    node_embedding_dim: int = 128,
    edge_embedding_dim: int = 10,
    num_gnn_heads: int = 8,
    num_transformer_heads: int = 8,
    optimizer: Callable = get_deepspeed_adamw,
    pretrained_checkpoint: Optional[str] = None,
):
    edge_types = [
        ("Spectra", "spectra_to_rt", "RT"),
        ("RT", "rt_to_rt", "RT"),
        ("Peak", "peak_to_rt", "RT"),
    ]
    # get vocab
    node_vocab = get_node_vocab()
    edge_vocab = get_word_vocab()
    # get tax class dict
    class_dict_fp = f"{dataset_dir}/taxonomy_class_dicts.json"
    tax_class_dict = json.load(open(class_dict_fp, "r"))
    # model setup
    node_encoders = get_node_encoders(
        node_vocab=node_vocab,
        word_vocab=edge_vocab,
        embedding_dim=node_embedding_dim,
    )
    edge_type_encoder_config = get_edge_type_encoder(
        edge_types=edge_types,
        embedding_dim=edge_embedding_dim,
    )
    gnn_config = get_gnn(
        node_embedding_dim=node_embedding_dim,
        edge_embedding_dim=edge_embedding_dim,
        num_heads=num_gnn_heads,
    )
    transformer_config = get_transformer(
        embedding_dim=node_embedding_dim, num_heads=num_transformer_heads
    )
    graph_pooler_config = get_graph_pooler(embedding_dim=node_embedding_dim)
    heads = get_heads(
        weights=weights,
        tax_class_dict=tax_class_dict,
        embedding_dim=node_embedding_dim,
    )
    # Metrics
    train_metrics = ModuleDict()
    val_metrics = ModuleDict()
    for level in tax_class_dict:
        key = f"{level}___a"
        train_metrics[key] = ClassificationMetrics.get(
            name=f"{key}_train",
            num_classes=len(tax_class_dict[level]),
            task="multiclass",
        )
        val_metrics[key] = ClassificationMetrics.get(
            name=f"{key}_val",
            num_classes=len(tax_class_dict[level]),
            task="multiclass",
        )
    # Instantiate a PyTorch Lightning Module
    model = HeteroGraphModelForMultiTaskLightning(
        node_encoders=node_encoders,
        edge_type_encoder_config=edge_type_encoder_config,
        gnn_config=gnn_config,
        transformer_config=transformer_config,
        graph_pooler_config=graph_pooler_config,
        heads=heads,
        optimizer_fn=optimizer,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        inputs=["a"],
        edge_types=edge_types,
    )
    if pretrained_checkpoint != None:
        states = torch.load(pretrained_checkpoint)
        model.load_state_dict(states["state_dict"], strict=False)
    return model
