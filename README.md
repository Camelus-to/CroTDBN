# CroMDBN: Dynamic Brain Network Analysis for Neuropsychiatric Diseases Diagnosing via Multi-Knowledge Integration

[![License](https://img.shields.io/github/license/Camelus-to/CroTDBN)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.11.0-red.svg)](https://pytorch.org/)

CroMDBN is a PyTorch-based implementation for dynamic brain network analysis, targeting neuropsychiatric disease diagnosis via multi-knowledge integration. This repository provides all source code for data preprocessing, dynamic graph construction, model training, and evaluation.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Hyperparameters & Experimental Setup](#hyperparameters--experimental-setup)
- [Dependencies](#dependencies)
- [Citation](#citation)
- [License](#license)

---

## Features

- **Full Pipeline:** Preprocessing, dynamic graph construction, training, and evaluation scripts provided.
- **Dynamic Brain Network Modeling:** Employs multi-knowledge integration for dynamic analysis.
- **Cross-Validation:** 5-fold cross-validation with 100 runs under different random seeds.
- **GPU Support:** Optimized for NVIDIA GPUs (tested on RTX 4090, CUDA 11.3).
- **Comprehensive Evaluation:** Detailed evaluation metrics and reproducibility.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Camelus-to/CroTDBN.git
   cd CroTDBN
   ```

2. **Install dependencies:**
   It is recommended to use a Python 3.9+ virtual environment.
   ```bash
   pip install -r requirements.txt
   ```
   If `requirements.txt` is not present, install core packages manually (see [Dependencies](#dependencies)).

## Usage

1. **Data Preparation:**  
   Prepare your datasets (ABIDE, ADNI, MDD, etc.) as described in the appendix or dataset folder.

2. **Preprocessing:**  
   Run the preprocessing script to generate input graphs and features.

3. **Training:**
   ```bash
   python train.py
   ```
   Adjust hyperparameters in the configuration files or directly in the script as needed.

4. **Evaluation:**  
   After training, use the evaluation script to validate results and generate metrics.

## Model Details

- **Backbone:** PyTorch 1.11.0, Python 3.9, CUDA 11.3
- **Dynamic Graph Construction:** Customized for each dataset
- **Position Encoding:** 116x116 matrix based on AAL brain regions
- **Sliding Window:**  
  - ABIDE: 9 windows  
  - ADNI & MDD: 8 windows

- **Pre-trained Models:**  
  Parameters for BERT and Tokenizer are provided in the appendix.

## Hyperparameters & Experimental Setup

- **Hardware:** NVIDIA RTX 4090 GPU
- **Optimizer:** Adam
- **Learning Rate:** 1.0e-4 (reduced by factor 0.1 during training)
- **Epochs:** 200
- **Batch Size:** 32
- **Validation Split:** 10% (with 70% train, 20% test)
- **Cross-Validation:** 5-fold, 100 runs with different seeds

## Dependencies

Below is a list of the main Python packages required for this project. A full list is available in `requirements.txt`. Key dependencies:

- torch==2.0.0
- torch_geometric==2.5.0
- einops==0.8.1
- scikit-learn==1.6.1
- numpy==1.24.4
- pandas==2.3.0
- matplotlib==3.7.5
- seaborn==0.13.2
- omegaconf==2.3.0
- hydra-core==1.3.2
- nibabel==5.3.2
- nilearn==0.12.0
- tqdm==4.67.1
- wandb==0.19.8
- optuna==4.4.0
- dhg==0.9.4

**Note:** For GPU support, ensure the corresponding CUDA toolkit (11.3+) is installed.

## Citation

If you use this codebase or CroMDBN in your research, please cite the following paper:

```
@article{CroMDBN2025,
  title={CroMDBN: Dynamic Brain Network Analysis for Neuropsychiatric Diseases Diagnosing via Multi-Knowledge Integration},
  author={Your Author List},
  journal={Journal/Conference},
  year={2025},
  url={https://github.com/Camelus-to/CroTDBN}
}
```

## License

This repository is licensed under the MIT License. See [LICENSE](./LICENSE) for details.

---

For detailed hyperparameter settings, sliding window details, and pre-trained model parameters, please refer to the appendix in the original paper or supplementary files in this repository.

For any questions or issues, please open an [issue](https://github.com/Camelus-to/CroTDBN/issues).
