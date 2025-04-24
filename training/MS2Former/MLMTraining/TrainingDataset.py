import pickle
import random
from typing import List, Optional

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
        if subset:
            datapoints = random.sample(datapoints, subset)
        self.tensor_cache = {}
        for s in tqdm(datapoints):
            mzml_id = s["mzml_id"]
            ms1_peak_id = s["ms1_peak_id"]
            graph_fp = f"{self.root}/{mzml_id}/{ms1_peak_id}.pkl"
            G = pickle.load(open(graph_fp, "rb"))
            self.tensor_cache[ms1_peak_id] = G.get_tensor_data(
                node_vocab=self.node_vocab,
                edge_vocab=self.edge_vocab,
                word_vocab=self.word_vocab,
                node_types_to_consider=self.node_types_to_consider,
                edge_types_to_consider=self.edge_types_to_consider,
                apply_edge_attr=False,
                apply_multigraph_wrapper=True,
            )
        self.datapoints = sorted(self.tensor_cache)

    def len(self):
        return len(self.datapoints)

    def get(self, input_idx):
        return self.tensor_cache[self.datapoints[input_idx]]
