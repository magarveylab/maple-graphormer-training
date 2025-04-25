import json
import os
import pickle
from glob import glob

from Maple.Embedder.graphs.MS1Graph import MS1Graph
from Maple.Embedder.graphs.MS2Graph import MS2Graph
from tqdm import tqdm

from omnicons import curdir


def prepare_ms1_graphs():
    # check if raw data exists
    mzml_dir = f"{curdir}/datasets/raw_data"
    if os.path.exists(mzml_dir) == False:
        raise FileNotFoundError(f"{mzml_dir} does not exist")
    # create output directory
    output_dir = f"{curdir}/datasets/MS1Graphs"
    os.makedirs(output_dir, exist_ok=True)
    filenames = glob(f"{mzml_dir}/*.json")
    # create graphs
    for fp in tqdm(filenames):
        peaks = json.load(open(fp))
        mzml_id = int(fp.split("/")[-1].split(".")[0])
        os.makedirs(f"{output_dir}/{mzml_id}", exist_ok=True)
        for p in peaks:
            p["intensity"] = p["intensity_raw"]
        if len(peaks) == 0:
            continue
        out = MS1Graph.build_from_ms1_spectra(mzml_id=mzml_id, ms1_peaks=peaks)
        for graph in out:
            graph_id = graph.graph_id
            with open(f"{output_dir}/{mzml_id}/{graph_id}.pkl", "wb") as f:
                pickle.dump(graph, f)


def prepare_ms2_graphs():
    # check if raw data exists
    mzml_dir = f"{curdir}/datasets/raw_data"
    if os.path.exists(mzml_dir) == False:
        raise FileNotFoundError(f"{mzml_dir} does not exist")
    # create output directory
    output_dir = f"{curdir}/datasets/MS2Graphs"
    os.makedirs(output_dir, exist_ok=True)
    filenames = glob(f"{mzml_dir}/*.json")
    # create graphs
    for fp in filenames:
        mzml_id = int(fp.split("/")[-1].split(".")[0])
        os.makedirs(f"{output_dir}/{mzml_id}", exist_ok=True)
        peaks = json.load(open(fp))
        for p in peaks:
            if "ms2" not in p:
                continue
            peak_id = p["ms1_peak_id"]
            output_fp = f"{output_dir}/{mzml_id}/{peak_id}.pkl"
            if os.path.exists(output_fp):
                continue
            G = MS2Graph.build_from_ms2_spectra(
                spectra_id=peak_id,
                ms2_spectra=p["ms2"],
                precursor_mz=p["mz"],
            )
            with open(output_fp, "wb") as f:
                pickle.dump(G, f)


def prep_msdial_dataset():
    raw_data_fp = f"{curdir}/datasets/ms-dial.json"
    if os.path.exists(raw_data_fp) == False:
        raise FileNotFoundError(f"{raw_data_fp} does not exist")
    # create output directory
    output_dir = f"{curdir}/datasets/MSDial-MS2Graphs"
    os.makedirs(output_dir, exist_ok=True)
    # create graphs
    data = json.load(open(raw_data_fp))
    for p in data:
        spectra_id = p["spectra_id"]
        output_fp = f"{output_dir}/{spectra_id}.pkl"
        if os.path.exists(output_fp):
            return None
        G = MS2Graph.build_from_ms2_spectra(
            spectra_id=spectra_id,
            ms2_spectra=p["ms2"],
            precursor_mz=p["precursor_mz"],
        )
        with open(output_fp, "wb") as f:
            pickle.dump(G, f)
