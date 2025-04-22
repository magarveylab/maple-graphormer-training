from typing import List, Optional

import numpy as np
import pandas as pd
from Maple.Embedder.graphs.MS2Graph import get_node_vocab, get_word_vocab
from pytorch_lightning import LightningDataModule
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from tqdm import tqdm
from TrainingDataset import TrainingDataset

from omnicons import dataset_dir
from omnicons.collators.StandardCollators import StandardCollator


class MS2DataModule(LightningDataModule):

    def __init__(
        self,
        dataset_fp: str = f"{dataset_dir}/augmented_chemotype_dataset.csv",
        graph_dir: str = f"{dataset_dir}/MS2Graphs",
        batch_size: int = 1500,
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
        self.collator = StandardCollator()

    def setup(self, stage: str = "fit"):
        # load datapoints
        self.datapoints = {"train": [], "test": []}
        for r in tqdm(pd.read_csv(self.dataset_fp).to_dict("records")):
            self.datapoints[r["split"]].append(r)
        # setup dynamic datasets
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
                datapoints=self.datapoints["test"],
                root=self.graph_dir,
                subset=self.subset,
                node_types_to_consider=self.node_types_to_consider,
                edge_types_to_consider=self.edge_types_to_consider,
                node_vocab=self.node_vocab,
                word_vocab=self.word_vocab,
            )

    def calculate_weights(self):
        chemotype_dict = {
            "Alkaloid": 0,
            "NonRibosomalPeptide": 1,
            "TypeIPolyketide": 2,
            "TypeIIPolyketide": 3,
            "NRPS-IndependentSiderophore": 4,
            "Aminoglycoside": 5,
            "Nucleoside": 6,
        }
        # for every label track cls_bins
        labels_to_cls_bins = {
            label: set() for label in chemotype_dict.values()
        }
        for r in pd.read_csv(self.dataset_fp).to_dict("records"):
            split = r["split"]
            if split != "train":
                continue
            cls_bin = r["cls_bin"]
            label = chemotype_dict[r["chemotype"]]
            labels_to_cls_bins[label].add(cls_bin)
        # calculate weights
        y = []
        for label, cls_bins in labels_to_cls_bins.items():
            if len(cls_bins) > 1:
                y.extend([label] * len(cls_bins))
            else:
                y.append(label)
        return compute_class_weight(
            class_weight="balanced", classes=np.unique(y), y=y
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
