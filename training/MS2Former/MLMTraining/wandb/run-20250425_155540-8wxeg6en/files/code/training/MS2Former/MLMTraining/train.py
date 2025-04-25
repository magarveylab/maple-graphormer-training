import argparse
import os
from multiprocessing import freeze_support

from DataModule import MS2DataModule
from models import get_model

from omnicons import experiment_dir
from omnicons.trainers import get_trainer


def train(
    checkpoint_dir: str = f"{experiment_dir}/MS2-mlm/checkpoints",
    checkpoint_name: str = "ms2mlm-{epoch:02d}-{val_loss:.2f}",
    logger_entity: str = "magarvey",
    logger_name: str = "mlm",
    logger_project: str = "MS2Former",
    trainer_strategy: str = "deepspeed_stage_3_offload",
    node_embedding_dim: int = 128,
    edge_embedding_dim: int = 128,
    num_gnn_heads: int = 8,
    num_transformer_heads: int = 8,
):
    # setup directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    # data module
    dm = MS2DataModule(dataset_fp="/data/mass_spec/ms2/graphs", subset=100)
    # model
    model = get_model(
        node_embedding_dim=int(node_embedding_dim),
        edge_embedding_dim=int(edge_embedding_dim),
        num_gnn_heads=int(num_gnn_heads),
        num_transformer_heads=int(num_transformer_heads),
        pretrained_checkpoint=None,
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


parser = argparse.ArgumentParser(description="Train MS2Former with MLM")
parser.add_argument(
    "-checkpoint_dir",
    help="Directory to save checkpoints",
    default=f"{experiment_dir}/MS2-mlm/checkpoints",
)
parser.add_argument(
    "-checkpoint_name",
    help="checkpoint name for wandb",
    default="ms2mlm-{epoch:02d}-{val_loss:.2f}",
)
parser.add_argument(
    "-logger_entity",
    help="wandb entity",
    default="magarvey",
)
parser.add_argument(
    "-logger_name",
    help="wandb entity",
    default="mlm",
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
        checkpoint_name=args.checkpoint_name,
        logger_entity=args.logger_entity,
        logger_name=args.logger_name,
        node_embedding_dim=args.node_embedding_dim,
        edge_embedding_dim=args.edge_embedding_dim,
        num_gnn_heads=args.num_gnn_heads,
        num_transformer_heads=args.num_transformer_heads,
    )
