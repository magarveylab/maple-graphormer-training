from typing import Callable, Optional

import torch
from Maple.Embedder.graphs.MS2Graph import get_node_vocab, get_word_vocab
from torch.nn import ModuleDict

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
    # prepare mz encoder
    encoder_dicts = [
        {
            "name": s,
            "num_embeddings": len(word_vocab[s]),
            "embedding_dim": embedding_dim,
        }
        for s in ["MZbase", "MZdec", "Intensity"]
    ]
    sentence_structure = [
        "MZbase",
        "MZdec",
        "MZbase",
        "MZdec",
        "MZbase",
        "MZdec",
        "Intensity",
    ]
    node_encoders["MZ"] = SentenceEncoderConfig(
        sentence_structure=sentence_structure,
        encoder_dicts=encoder_dicts,
        dropout=0.1,
        mlp_layers=1,
    )
    # prepare nl encoder
    encoder_dicts = [
        {
            "name": s,
            "num_embeddings": len(word_vocab[s]),
            "embedding_dim": embedding_dim,
        }
        for s in ["NLbase", "NLdec"]
    ]
    sentence_structure = [
        "NLbase",
        "NLdec",
        "NLbase",
        "NLdec",
        "NLbase",
        "NLdec",
    ]
    node_encoders["NL"] = SentenceEncoderConfig(
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


def get_heads(word_vocab: dict, embedding_dim: int = 128):
    from omnicons.configs.HeadConfigs import NodeClsTaskHeadConfig

    heads = {}
    # min mz
    heads["min_mz_base_mask"] = NodeClsTaskHeadConfig(
        hidden_size=embedding_dim,
        hidden_dropout_prob=0.1,
        num_labels=len(word_vocab["MZbase"]),
        multi_label=False,
        node_type="MZ",
        analyze_inputs=["a"],
    )
    heads["min_mz_dec_mask"] = NodeClsTaskHeadConfig(
        hidden_size=embedding_dim,
        hidden_dropout_prob=0.1,
        num_labels=len(word_vocab["MZdec"]),
        multi_label=False,
        node_type="MZ",
        analyze_inputs=["a"],
    )
    # mz
    heads["mz_base_mask"] = NodeClsTaskHeadConfig(
        hidden_size=embedding_dim,
        hidden_dropout_prob=0.1,
        num_labels=len(word_vocab["MZbase"]),
        multi_label=False,
        node_type="MZ",
        analyze_inputs=["a"],
    )
    heads["mz_dec_mask"] = NodeClsTaskHeadConfig(
        hidden_size=embedding_dim,
        hidden_dropout_prob=0.1,
        num_labels=len(word_vocab["MZdec"]),
        multi_label=False,
        node_type="MZ",
        analyze_inputs=["a"],
    )
    # max mz
    heads["max_mz_base_mask"] = NodeClsTaskHeadConfig(
        hidden_size=embedding_dim,
        hidden_dropout_prob=0.1,
        num_labels=len(word_vocab["MZbase"]),
        multi_label=False,
        node_type="MZ",
        analyze_inputs=["a"],
    )
    heads["max_mz_dec_mask"] = NodeClsTaskHeadConfig(
        hidden_size=embedding_dim,
        hidden_dropout_prob=0.1,
        num_labels=len(word_vocab["MZdec"]),
        multi_label=False,
        node_type="MZ",
        analyze_inputs=["a"],
    )
    # min nl
    heads["min_nl_base_mask"] = NodeClsTaskHeadConfig(
        hidden_size=embedding_dim,
        hidden_dropout_prob=0.1,
        num_labels=len(word_vocab["NLbase"]),
        multi_label=False,
        node_type="NL",
        analyze_inputs=["a"],
    )
    heads["min_nl_dec_mask"] = NodeClsTaskHeadConfig(
        hidden_size=embedding_dim,
        hidden_dropout_prob=0.1,
        num_labels=len(word_vocab["NLdec"]),
        multi_label=False,
        node_type="NL",
        analyze_inputs=["a"],
    )
    # nl
    heads["nl_base_mask"] = NodeClsTaskHeadConfig(
        hidden_size=embedding_dim,
        hidden_dropout_prob=0.1,
        num_labels=len(word_vocab["NLbase"]),
        multi_label=False,
        node_type="NL",
        analyze_inputs=["a"],
    )
    heads["nl_dec_mask"] = NodeClsTaskHeadConfig(
        hidden_size=embedding_dim,
        hidden_dropout_prob=0.1,
        num_labels=len(word_vocab["NLdec"]),
        multi_label=False,
        node_type="NL",
        analyze_inputs=["a"],
    )
    # max nl
    heads["max_nl_base_mask"] = NodeClsTaskHeadConfig(
        hidden_size=embedding_dim,
        hidden_dropout_prob=0.1,
        num_labels=len(word_vocab["NLbase"]),
        multi_label=False,
        node_type="NL",
        analyze_inputs=["a"],
    )
    heads["max_nl_dec_mask"] = NodeClsTaskHeadConfig(
        hidden_size=embedding_dim,
        hidden_dropout_prob=0.1,
        num_labels=len(word_vocab["NLdec"]),
        multi_label=False,
        node_type="NL",
        analyze_inputs=["a"],
    )
    # intensisty
    heads["intensity_mask"] = NodeClsTaskHeadConfig(
        hidden_size=embedding_dim,
        hidden_dropout_prob=0.1,
        num_labels=len(word_vocab["Intensity"]),
        multi_label=False,
        node_type="MZ",
        analyze_inputs=["a"],
    )
    return heads


def get_model(
    node_embedding_dim: int = 128,
    edge_embedding_dim: int = 10,
    num_gnn_heads: int = 8,
    num_transformer_heads: int = 8,
    pretrained_checkpoint: Optional[str] = None,
    optimizer: Callable = get_deepspeed_adamw,
):
    edge_types = [
        ("Precursor", "precursor_to_nl", "NL"),
        ("NL", "nl_to_mz", "MZ"),
    ]
    # get vocab
    node_vocab = get_node_vocab()
    word_vocab = get_word_vocab()
    # model setup
    node_encoders = get_node_encoders(
        node_vocab=node_vocab,
        word_vocab=word_vocab,
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
    heads = get_heads(word_vocab=word_vocab, embedding_dim=node_embedding_dim)
    # Metrics
    train_metrics = ModuleDict()
    val_metrics = ModuleDict()
    num_classes = {
        # predict mz
        "min_mz_base_mask": len(word_vocab["MZbase"]),
        "min_mz_dec_mask": len(word_vocab["MZdec"]),
        "mz_base_mask": len(word_vocab["MZbase"]),
        "mz_dec_mask": len(word_vocab["MZdec"]),
        "max_mz_base_mask": len(word_vocab["MZbase"]),
        "max_mz_dec_mask": len(word_vocab["MZdec"]),
        # predict nl
        "min_nl_base_mask": len(word_vocab["NLbase"]),
        "min_nl_dec_mask": len(word_vocab["NLdec"]),
        "nl_base_mask": len(word_vocab["NLbase"]),
        "nl_dec_mask": len(word_vocab["NLdec"]),
        "max_nl_base_mask": len(word_vocab["NLbase"]),
        "max_nl_dec_mask": len(word_vocab["NLdec"]),
        # predict intensity
        "intensity_mask": len(word_vocab["Intensity"]),
    }
    for label_name in num_classes:
        key = f"{label_name}___a"
        train_metrics[key] = ClassificationMetrics.get(
            name=f"{key}_train",
            num_classes=num_classes[label_name],
            task="multiclass",
        )
        val_metrics[key] = ClassificationMetrics.get(
            name=f"{key}_val",
            num_classes=num_classes[label_name],
            task="multiclass",
        )
    # Instantiate a PyTorch Lightning Module
    model = HeteroGraphModelForMultiTaskLightning(
        node_encoders=node_encoders,
        edge_type_encoder_config=edge_type_encoder_config,
        gnn_config=gnn_config,
        transformer_config=transformer_config,
        heads=heads,
        optimizer_fn=optimizer,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        inputs=["a"],
        edge_types=edge_types,
    )
    if pretrained_checkpoint != None:
        states = torch.load(pretrained_checkpoint)
        model.load_state_dict(states["state_dict"], strict=True)
    return model
