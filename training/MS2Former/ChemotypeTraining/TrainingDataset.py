import pickle
import random
from typing import List, Optional

import torch
from torch_geometric.data import Dataset
from tqdm import tqdm


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
        self.chemotype_dict = {
            "Alkaloid": 0,
            "NonRibosomalPeptide": 1,
            "TypeIPolyketide": 2,
            "TypeIIPolyketide": 3,
            "NRPS-IndependentSiderophore": 4,
            "Aminoglycoside": 5,
            "Nucleoside": 6,
        }
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
        # compile labels
        self.label_dict = {}
        self.peak_to_mzml = {}
        for d in datapoints:
            peak_id = d["ms1_peak_id"]
            mzml_id = d["mzml_id"]
            chemotype = d["chemotype"]
            self.label_dict[peak_id] = self.chemotype_dict[chemotype]
            self.peak_to_mzml[peak_id] = mzml_id
        # sort datapoints
        self.sorted_datapoints = {}
        for d in datapoints:
            peak_id = d["ms1_peak_id"]
            cls_bin = d["cls_bin"]
            if cls_bin not in self.sorted_datapoints:
                self.sorted_datapoints[cls_bin] = []
            self.sorted_datapoints[cls_bin].append(peak_id)
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
        # compile peak ids
        all_peak_ids = set()
        for cls_bin in self.sorted_datapoints:
            for peak_id in self.sorted_datapoints[cls_bin]:
                all_peak_ids.add(peak_id)
        # create tensor cache
        self.tensor_cache = {}
        for peak_id in tqdm(all_peak_ids):
            mzml_id = self.peak_to_mzml[peak_id]
            graph_fp = f"{self.root}/{mzml_id}/{peak_id}.pkl"
            G = pickle.load(open(graph_fp, "rb"))
            self.tensor_cache[peak_id] = G.get_tensor_data(
                node_vocab=self.node_vocab,
                edge_vocab=self.edge_vocab,
                word_vocab=self.word_vocab,
                node_types_to_consider=self.node_types_to_consider,
                edge_types_to_consider=self.edge_types_to_consider,
                apply_edge_attr=False,
                apply_multigraph_wrapper=True,
            )

    def len(self):
        return len(self.datapoints)

    def get(self, input_idx):
        peak_id = random.choice(
            self.sorted_datapoints[self.datapoints[input_idx]]
        )
        dp = self.tensor_cache[peak_id]
        setattr(
            dp.graphs["a"],
            "chemotype",
            torch.LongTensor([self.label_dict[peak_id]]),
        )
        return dp
