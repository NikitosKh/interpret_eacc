# Interpret AI

Welcome to the Interpret AI repository. This project focuses on developing a model merging tool that allows for the combination of pretrained models into a single, efficient model with negligible additional training cost compared to pretraining. For more detailed information about our methods, please visit our [website](link-to-the-site).

## Table of Contents

1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Getting Started](#getting-started)
4. [Training the Joint Sparse Autoencoder](#training-the-joint-sparse-autoencoder)

## Introduction

The main product of Interpret AI is a model merging tool. The idea is to take a set of pretrained models, possibly with some constraints, and combine them into a single model with a cost that is negligible compared to pretraining each model individually.

## Project Structure

The project has the following directory structure:

```
InterpretAI/
├── data/
│   └── [datasets]
├── results/
│   └── [checkpoints]
├── train_sae.py
├── README.md
└── [other scripts and files]
```

- **data/**: This directory contains the datasets used for training and evaluation.
- **results/**: This is where you need to place checkpoints. These can be downloaded from the provided link (https://drive.google.com/file/d/10vfy3_iWVEj2nF4r7FumN4_OWt_gDyxT/view?usp=sharing).
- **train_sae.py**: Script to train the joint sparse autoencoder.

## Getting Started

### Prerequisites

To get started, you need to have the following installed:

- Python 3.7+
- PyTorch
- NumPy
- einops
- wandb

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/NikitosKh/interpret_eacc
    cd interpret-ai
    ```

2. Download the necessary checkpoints and place them in the `results/` directory. You can download them from [this link](link-to-checkpoints).

## Training the Joint Sparse Autoencoder

TODO: CONFIGS!!!!!!!

To train the joint sparse autoencoder, run the `train_sae.py` script:

```bash
python train_sae.py
```

Make sure the datasets are placed in the `data/` directory and checkpoints in the `results/` directory as mentioned in the project structure.
