import argparse
import os
from multiprocessing import freeze_support

from DataModule import MS1DataModule
from models import get_model

from omnicons import experiment_dir
from omnicons.trainers import get_trainer


def train(
    checkpoint_dir: str = f"{experiment_dir}/MS1-taxonomy/checkpoints",
    mlm_checkpoint_fp: str = f"{experiment_dir}/MS1-mlm/checkpoints/last.pt",
    checkpoint_name: str = "ms1tax-{epoch:02d}-{val_loss:.2f}",
    logger_entity: str = "magarvey",
    logger_name: str = "taxonomy",
    logger_project: str = "MS1Former",
    trainer_strategy: str = "deepspeed_stage_3_offload",
    node_embedding_dim: int = 128,
    edge_embedding_dim: int = 128,
    num_gnn_heads: int = 8,
    num_transformer_heads: int = 8,
):
    # setup directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    # data module
    dm = MS1DataModule()
    dm.setup(stage="fit")
    weights = dm.calculate_weights()
    # model
    model = get_model(
        weights=weights,
        node_embedding_dim=node_embedding_dim,
        edge_embedding_dim=edge_embedding_dim,
        num_gnn_heads=num_gnn_heads,
        num_transformer_heads=num_transformer_heads,
        pretrained_checkpoint=mlm_checkpoint_fp,
    )
    # trainer
    trainer = get_trainer(
        checkpoint_dir=checkpoint_dir,
        checkpoint_name=checkpoint_name,
        logger_entity=logger_entity,
        logger_name=logger_name,
        logger_project=logger_project,
        trainer_strategy=trainer_strategy,
    )
    trainer.fit(model, dm)


parser = argparse.ArgumentParser(
    description="Train MS1Former with Supervised Taxonomy Classification"
)
parser.add_argument(
    "-checkpoint_dir",
    help="Directory to save checkpoints",
    default=f"{experiment_dir}/MS1-taxonomy",
)
parser.add_argument(
    "-mlm_checkpoint_fp",
    help="pytorch checkpoint for MS1Former mlm pretrained weights",
    default=f"{experiment_dir}/MS1-mlm/checkpoints/last.pt",
)
parser.add_argument(
    "-checkpoint_name",
    help="checkpoint name for wandb",
    default="ms1tax-{epoch:02d}-{val_loss:.2f}",
)
parser.add_argument(
    "-logger_entity",
    help="wandb entity",
    default="user",
)
parser.add_argument(
    "-logger_name",
    help="wandb entity",
    default="taxonomy",
)
parser.add_argument(
    "-node_embedding_dim",
    help="node embedding dimension",
    default=128,
)
parser.add_argument(
    "-edge_embedding_dim",
    help="edge embedding dimension",
    default=128,
)
parser.add_argument(
    "-num_gnn_heads",
    help="number of gnn heads",
    default=8,
)
parser.add_argument(
    "-num_transformer_heads",
    help="number of transformer heads",
    default=8,
)

if __name__ == "__main__":
    args = parser.parse_args()
    freeze_support()
    train(
        checkpoint_dir=args.checkpoint_dir,
        mlm_checkpoint_fp=args.mlm_checkpoint_fp,
        checkpoint_name=args.checkpoint_name,
        logger_entity=args.logger_entity,
        logger_name=args.logger_name,
        node_embedding_dim=args.node_embedding_dim,
        edge_embedding_dim=args.edge_embedding_dim,
        num_gnn_heads=args.num_gnn_heads,
        num_transformer_heads=args.num_transformer_heads,
    )
