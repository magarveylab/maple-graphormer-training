import json
import pickle
import random
from typing import List, Optional

import torch
from torch_geometric.data import Dataset
from tqdm import tqdm

from omnicons import dataset_dir


class TrainingDataset(Dataset):

    def __init__(
        self,
        datapoints: List[int],
        root: str,
        subset: Optional[int] = None,
        node_types_to_consider: Optional[List[str]] = None,
        edge_types_to_consider: Optional[List[str]] = None,
        node_vocab: dict = {},
        edge_vocab: dict = {},
        word_vocab: dict = {},
    ):
        self.root = root
        self.node_types_to_consider = node_types_to_consider
        self.edge_types_to_consider = edge_types_to_consider
        self.node_vocab = node_vocab
        self.edge_vocab = edge_vocab
        self.word_vocab = word_vocab
        class_dict_fp = f"{dataset_dir}/taxonomy_class_dicts.json"
        self.class_dict = json.load(open(class_dict_fp, "r"))
        self.prepare_data(datapoints=datapoints, subset=subset)
        super().__init__(
            root=root, transform=None, pre_transform=None, pre_filter=None
        )

    @property
    def processed_file_names(self):
        return []

    def prepare_data(
        self,
        datapoints: List[dict],
        subset: Optional[int] = None,
    ):
        # sort datapoints
        self.sorted_datapoints = {}
        sample_meta = {}
        for d in datapoints:
            sample_id = d["sample_id"]
            cls_bin = d["cls_bin"]
            if cls_bin not in self.sorted_datapoints:
                self.sorted_datapoints[cls_bin] = []
            self.sorted_datapoints[cls_bin].append(sample_id)
            sample_meta[sample_id] = d
        if subset:
            for cls_bin in self.sorted_datapoints:
                n = len(self.sorted_datapoints[cls_bin])
                if n > subset:
                    self.sorted_datapoints[cls_bin] = random.sample(
                        self.sorted_datapoints[cls_bin], subset
                    )
        # blanced distribution
        min_sample_size = min(
            [len(v) for v in self.sorted_datapoints.values()]
        )
        if min_sample_size < 10000:
            self.datapoints = sorted(self.sorted_datapoints) * min_sample_size
        else:
            self.datapoints = sorted(self.sorted_datapoints) * 10000
        # prepare tensors
        self.tensor_cache = {}
        datapoints = [y for x in self.sorted_datapoints.values() for y in x]
        for sample_id in tqdm(datapoints):
            self.tensor_cache[sample_id] = self.get_tensor(
                sample_meta[sample_id]
            )

    def len(self):
        return len(self.datapoints)

    def get_tensor(self, s: dict):
        mzml_id = s["mzml_id"]
        sample_id = s["sample_id"]
        graph_fp = f"{self.root}/{mzml_id}/{sample_id}.pkl"
        G = pickle.load(open(graph_fp, "rb"))
        tensor = G.get_tensor_data(
            node_vocab=self.node_vocab,
            edge_vocab=self.edge_vocab,
            word_vocab=self.word_vocab,
            node_types_to_consider=self.node_types_to_consider,
            edge_types_to_consider=self.edge_types_to_consider,
            apply_edge_attr=False,
            apply_multigraph_wrapper=True,
        )
        for level in ["phylum", "class", "order", "family", "genus"]:
            label = str(s[f"{level}_id"])
            setattr(
                tensor.graphs["a"],
                level,
                torch.LongTensor([self.class_dict[level][label]]),
            )
        return tensor

    def get(self, input_idx):
        s = random.choice(self.sorted_datapoints[self.datapoints[input_idx]])
        return self.tensor_cache[s]
