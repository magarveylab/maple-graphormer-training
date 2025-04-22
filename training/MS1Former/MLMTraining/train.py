import os
from multiprocessing import freeze_support

from MassSpecEmbedder.training.ms1_mlm import models, trainers
from MassSpecEmbedder.training.ms1_mlm.DataModule import MS1DataModule

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python train.py


def train():
    # model dir
    training_dir = (
        "/home/gunam/storage/deep_learning_experiments/neo4j/MS1-mlm"
    )
    root_dir = f"{training_dir}/p1"
    checkpoint_dir = f"{root_dir}/checkpoints"
    config_dir = f"{root_dir}/configs"
    # wandb
    checkpoint_name = "ms1mlm-{epoch:02d}-{val_loss:.2f}"
    logger_entity = "magarvey"
    logger_project = "ms1-mlm"
    logger_name = f"mlm-p1"
    trainer_strategy = "deepspeed_stage_3_offload"
    # setup directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)
    # data module
    dm = MS1DataModule()
    # model
    pretrained_checkpoint = None
    # default params for model
    param = {
        "node_embedding_dim": 128,
        "edge_embedding_dim": 128,
        "num_gnn_heads": 8,
        "num_transformer_heads": 8,
    }
    model = models.get_model(
        node_embedding_dim=param["node_embedding_dim"],
        edge_embedding_dim=param["edge_embedding_dim"],
        num_gnn_heads=param["num_gnn_heads"],
        num_transformer_heads=param["num_transformer_heads"],
        pretrained_checkpoint=pretrained_checkpoint,
    )
    # trainer
    trainer = trainers.get_trainer(
        checkpoint_dir=checkpoint_dir,
        checkpoint_name=checkpoint_name,
        logger_entity=logger_entity,
        logger_name=logger_name,
        logger_project=logger_project,
        trainer_strategy=trainer_strategy,
    )
    trainer.fit(model, dm)


if __name__ == "__main__":
    freeze_support()
    train()
