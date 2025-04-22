from typing import List, Optional

import numpy as np
import pandas as pd
from Maple.Embedder.graphs.MS2Graph import get_node_vocab, get_word_vocab
from pytorch_lightning import LightningDataModule
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from tqdm import tqdm
from TrainingDataset import ChemotypeTrainingDataset, TanimotoTrainingDataset

from omnicons import dataset_dir
from omnicons.collators.StandardCollators import StandardCollator
from omnicons.data.DatasetWrapper import MultiGraphInMemoryDataset


class MS2DataModule(LightningDataModule):

    def __init__(
        self,
        chemotype_dataset_fp: str = f"{dataset_dir}/augmented_chemotype_dataset.csv",
        chemotype_graph_dir: str = f"{dataset_dir}/MS2Graphs",
        tanimoto_dataset_fp: str = f"{dataset_dir}/tanimoto_edge_ds.csv",
        tanimoto_graph_dir: str = f"{dataset_dir}/MSDial-MS2Graphs",
        batch_size: int = 300,
        num_workers: int = 0,
        persistent_workers: bool = False,
        subset: Optional[int] = None,
        node_types_to_consider: Optional[List[str]] = None,
        edge_types_to_consider: Optional[List[str]] = None,
    ):
        super().__init__()
        self.chemotype_dataset_fp = chemotype_dataset_fp
        self.tanimoto_dataset_fp = tanimoto_dataset_fp
        self.chemotype_graph_dir = chemotype_graph_dir
        self.tanimoto_graph_dir = tanimoto_graph_dir
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
        # load chemotype datapoints
        self.chemotype_datapoints = {"train": [], "test": []}
        for r in tqdm(
            pd.read_csv(self.chemotype_dataset_fp).to_dict("records")
        ):
            self.chemotype_datapoints[r["split"]].append(r)
        # load tanimoto datapoints
        self.tanimoto_datapoints = {"train": [], "val": [], "test": []}
        for r in tqdm(
            pd.read_csv(self.tanimoto_dataset_fp).to_dict("records")
        ):
            self.tanimoto_datapoints[r["split"]].append(r)
        input_map = {
            "tanimoto": {"a": "a", "b": "b"},
            "chemotype": {"a": "c"},
        }
        # setup dynamic datasets
        if stage == "fit":
            # train
            datasets = {
                "tanimoto": TanimotoTrainingDataset(
                    datapoints=self.tanimoto_datapoints["train"],
                    root=self.tanimoto_graph_dir,
                    subset=self.subset,
                    node_types_to_consider=self.node_types_to_consider,
                    edge_types_to_consider=self.edge_types_to_consider,
                    node_vocab=self.node_vocab,
                    word_vocab=self.word_vocab,
                ),
                "chemotype": ChemotypeTrainingDataset(
                    datapoints=self.chemotype_datapoints["train"],
                    root=self.chemotype_graph_dir,
                    subset=self.subset,
                    node_types_to_consider=self.node_types_to_consider,
                    edge_types_to_consider=self.edge_types_to_consider,
                    node_vocab=self.node_vocab,
                    word_vocab=self.word_vocab,
                ),
            }
            self.train = MultiGraphInMemoryDataset(
                datasets=datasets, input_map=input_map
            )
            # val
            datasets = {
                "tanimoto": TanimotoTrainingDataset(
                    datapoints=self.tanimoto_datapoints["val"],
                    root=self.tanimoto_graph_dir,
                    subset=self.subset,
                    node_types_to_consider=self.node_types_to_consider,
                    edge_types_to_consider=self.edge_types_to_consider,
                    node_vocab=self.node_vocab,
                    word_vocab=self.word_vocab,
                ),
                "chemotype": ChemotypeTrainingDataset(
                    datapoints=self.chemotype_datapoints["test"],
                    root=self.chemotype_graph_dir,
                    subset=self.subset,
                    node_types_to_consider=self.node_types_to_consider,
                    edge_types_to_consider=self.edge_types_to_consider,
                    node_vocab=self.node_vocab,
                    word_vocab=self.word_vocab,
                ),
            }
            self.val = MultiGraphInMemoryDataset(
                datasets=datasets, input_map=input_map
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
        for r in pd.read_csv(self.chemotype_dataset_fp).to_dict("records"):
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
