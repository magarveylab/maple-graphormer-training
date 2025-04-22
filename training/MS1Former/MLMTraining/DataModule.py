from typing import List, Optional

import pandas as pd
from Maple.Embedder.graphs.MS1Graph import get_node_vocab, get_word_vocab
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from tqdm import tqdm
from TrainingDataset import TrainingDataset

from omnicons import curdir
from omnicons.collators.MaskCollators import NodeSentenceMaskCollator
from omnicons.collators.MixedCollators import MixedCollator


class MS1DataModule(LightningDataModule):

    def __init__(
        self,
        dataset_fp: str = f"{curdir}/datasets/ms1_taxonomy.csv",
        graph_dir: str = f"{curdir}/datasets/MS1Graphs",
        batch_size: int = 36,
        num_workers: int = 0,
        persistent_workers: bool = False,
        subset: Optional[int] = None,
        node_types_to_consider: Optional[List[str]] = None,
        edge_types_to_consider: Optional[List[str]] = None,
    ):
        super().__init__()
        self.dataset_fp = dataset_fp
        self.graph_dir = graph_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.subset = subset
        # vocab
        self.node_vocab = get_node_vocab()
        self.word_vocab = get_word_vocab()
        # graph parameters to create tensors
        self.node_types_to_consider = node_types_to_consider
        self.edge_types_to_consider = edge_types_to_consider
        # collators
        self.collator = MixedCollator(
            collators=(
                NodeSentenceMaskCollator(
                    mask_id=self.word_vocab["MZbase"]["[MASK]"],
                    p=0.15,
                    mask_names=(
                        "min_mz_base_mask",
                        "min_mz_dec_mask",
                        "mz_base_mask",
                        "mz_dec_mask",
                        "max_mz_base_mask",
                        "max_mz_dec_mask",
                        "charge_mask",
                        "intensity_mask",
                    ),
                    node_types_to_consider=("Peak"),
                ),
            )
        )

    def setup(self, stage: str = "fit"):
        # load datapoints
        self.datapoints = {"train": [], "val": [], "test": []}
        for r in tqdm(pd.read_csv(self.dataset_fp).to_dict("records")):
            self.datapoints[r["split"]].append(r)
        if stage == "fit":
            self.train = TrainingDataset(
                datapoints=self.datapoints["train"],
                root=self.graph_dir,
                subset=self.subset,
                node_types_to_consider=self.node_types_to_consider,
                edge_types_to_consider=self.edge_types_to_consider,
                node_vocab=self.node_vocab,
                word_vocab=self.word_vocab,
            )
            self.val = TrainingDataset(
                datapoints=self.datapoints["val"],
                root=self.graph_dir,
                subset=self.subset,
                node_types_to_consider=self.node_types_to_consider,
                edge_types_to_consider=self.edge_types_to_consider,
                node_vocab=self.node_vocab,
                word_vocab=self.word_vocab,
            )
        if stage == "test":
            self.test = TrainingDataset(
                datapoints=self.datapoints["test"],
                root=self.graph_dir,
                subset=self.subset,
                node_types_to_consider=self.node_types_to_consider,
                edge_types_to_consider=self.edge_types_to_consider,
                node_vocab=self.node_vocab,
                word_vocab=self.word_vocab,
            )

    def train_dataloader(self):
        train_dl = DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collator,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )
        return train_dl

    def test_dataloader(self):
        test_dl = DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collator,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )
        return test_dl

    def val_dataloader(self):
        val_dl = DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collator,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )
        return val_dl
