# maple-graphormer-training
Training scripts for MAPLE Graphormer models for publication

## Installation

### Training-Only Installation
1. Install the Package via Pip Symlinks.
```
    conda env create -f maple-training.yml
    conda activate maple-training
    pip install -e .
```
2. Set Up Weights & Biases (wandb)
    - Follow the [official quickstart guide](https://docs.wandb.ai/quickstart/) to configure Weights & Biases for experiment tracking.

## Dataset Preparation
Download and extract `datasets.zip` and `raw_data.zip` from the accompanying [Zenodo repository](https://zenodo.org/doi/10.5281/zenodo.15226521) and place their contents in this [directory](https://github.com/magarveylab/maple-graphormer-training/tree/main/omnicons/datasets). Then, run the modules below to pre-generate the required graphs for training.
```python
from omnicons import datasetprep

datasetprep.prepare_ms1_graphs()
datasetprep.prepare_ms2_graphs()
datasetprep.prep_msdial_dataset()
```

## MS1Former Training
1. Masked Language Modeling (MLM) – MS<sup>1</sup> signals are randomly masked, and the model is trained to predict properties of the masked metabolites. `save.py` converts DeepSpeed checkpoints into standard PyTorch checkpoint format.
```
cd training/MS1Former/MLMTraining
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -logger_entity new_user
python save.py
```
2. Taxonomy-Supervised Classification – Predicts taxonomic ranks (from phylum to genus) from spectral embeddings using parallel classification heads. This step fine-tunes the MS1Former model previously trained with MLM. `export.py` converts the trained PyTorch model into TorchScript format for deployment. 
```
cd training/MS1Former/TaxonomyTraining
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -logger_entity new_user
python save.py
python export.py
```

## MS2Former Training
1. Masked Language Modelling (MLM) - MS<sup>2</sup> fragments and neutral losses are randomly masked, and the model is trained to predict their corresponding masses from the remaining context.
```
cd training/MS2Former/MLMTraining
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -logger_entity new_user
python save.py
```
2. Chemotype-Supervised Classification – Predicts biosynthetic classes from fragmentation embeddings by fine-tuning the MS2Former model previously trained with MLM. This step uses an augmented dataset generated via graph label propagation.
```
cd training/MS2Former/ChemotypeTraining
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -logger_entity new_user
python save.py
python export.py
```
3. Molecular Similarity-Supervised Classification – Predicts Tanimoto similarity bins for pairwise MS<sup>2</sup> spectra using an external compound dataset from MS-DIAL. Trained in parallel with the chemotype dataset to preserve underlying biochemical organization.
```
cd training/MS2Former/TanimotoTraining
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py -logger_entity new_user
python save.py
python export.py
```