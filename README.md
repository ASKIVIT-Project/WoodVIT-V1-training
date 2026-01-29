# ASKIVIT-Project-WoodVIT-V1-training

Training and inference scripts for CNN-based models on the **WoodVIT V1** dataset.

This repository focuses on:
- computing dataset statistics (channel means) for normalization,
- training models (GPU),
- evaluating trained checkpoints,
- running inference on the test split.

---

## Repository structure (overview)

- **`Checkpoint_examples/`**  
  Example checkpoints: one CNN checkpoint per modality.

- **`V1_5_train.py`**  
  Main training script.

- **`V1_5_train_cfg.py`**  
  Training configuration (paths, hyperparameters, modality selection, etc.).

- **`WoodVIT_train.yaml`**  
  Conda environment file (requirements).

- **`analysis_ASKIVIT_V1.5.py`**  
  Computes channel-wise statistics (mean per channel) for the multimodal data.  
  **Required before training.**

- **`askivit_V1.5_inference.py`**  
  Runs inference on the test data using a trained checkpoint.

- **`torchaskivit/`**  
  PyTorch utilities: models, datasets/dataloaders, evaluation, and plotting helpers.

---


## Setup

### 1) Create the environment

Using the provided environment file:

```bash
conda env create -f WoodVIT_train.yaml
conda activate askivit_env
```

## Data

Download the WoodVIT datasets:

- **WoodVIT V1** (ready-to-use patched dataset): https://doi.org/10.35097/aj4ve1c03pkan0dr  

---

## Run

### Step 1 - Compute channel means (required)

1. Set dataset paths in **`analysis_ASKIVIT_V1.5.py`**
2. Run:

```bash
python analysis_ASKIVIT_V1.5.py
```

This script computes the per-channel mean values used for normalization during training.

### Step 2 - Train (GPU)

1. Set paths and training parameters in **`V1_5_train_cfg.py`**
2. Run:

```bash
python V1_5_train.py
```

Training checkpoints and logs will be written to the output directory configured in `V1_5_train_cfg.py`.

### Step 4 - Inference (test split)

The output is a `.json` file, which can be used later for further evaluation.

1. Set the paths (dataset + checkpoint) in **`askivit_V1.5_inference.py`**
2. Run:

```bash
python askivit_V1.5_inference.py
```

### Step 3 - Evaluate

Further evaluation scripts are located in the **`torchaskivit/evalutils`** module.  
The main input is the `.json` file generated in **Step 4 (Inference)**.

---

## Checkpoints

- `Checkpoint_examples/` contains example checkpoints (one per modality).

## Citing

If you use this code or the WoodVIT datasets in academic work, please cite the paper and the dataset.

### WoodVIT paper

BibTeX:
```bibtex
IN REVIEW
```

### Dataset

**WoodVIT V1 patch**:
```bibtex
@misc{WoodVIT_V1,
    author = {Manuel Bihler and Lukas Roming and Dovilė Čibiraitė-Lukenskienė and Jochen Aderhold and Andreas Keil and Friedrich Schlüter and Robin Gruna and Michael Heizmann},
    doi = {10.35097/aj4ve1c03pkan0dr},
    howpublished = {RADAR4KIT},
    publisher = {Karlsruhe Institute of Technology},
    title = {WoodVIT V1},
    url = {https://doi.org/10.35097/aj4ve1c03pkan0dr},
    year = {2023}
}
```

Further information about the acquisition, registration, and processing can be found in the paper cited above.

---

## License

This repository is published under the **CC0-1.0** license. See [`LICENSE`](./LICENSE).
