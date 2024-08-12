---

# Interpret AI

Welcome to the Interpret AI repository. This project focuses on developing a model merging tool that allows for the combination of pretrained models into a single, efficient model with negligible additional training cost compared to pretraining. For more detailed information about our methods, please visit our [website](https://interpretaieacc.github.io).

## Table of Contents

1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Getting Started](#getting-started)
4. [Training the Joint Sparse Autoencoder](#training-the-joint-sparse-autoencoder)
5. [Evaluating the Merged Model](#evaluating-the-merged-model)

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
├── evaluate_merged_model.py
├── README.md
├── requirements.txt
├── evaluate.py
├── eval_merged.yaml
└── [other scripts and files]
```

- **data/**: This directory contains the datasets used for training and evaluation.
- **results/**: This is where you need to place checkpoints. These can be downloaded from the provided link (https://drive.google.com/file/d/10vfy3_iWVEj2nF4r7FumN4_OWt_gDyxT/view?usp=sharing).
- **train_sae.py**: Script to train the joint sparse autoencoder.
- **evaluate_merged_model.py**: Script to evaluate the performance of the merged model.
- **requirements.txt**: All the necessary dependencies.
- **evaluate.py**: script for merged model evaluation

## Getting Started

### Prerequisites

To get started, you need to have the following installed:

- Python 3.7+
- PyTorch
- NumPy
- einops
- wandb
- you can find all the dependencies in the requirements.txt file

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/NikitosKh/interpret_eacc
    cd interpret-ai
    ```

2. Download the necessary checkpoints and place them in the `results/` directory. You can download them from [this link](https://drive.google.com/file/d/10vfy3_iWVEj2nF4r7FumN4_OWt_gDyxT/view).

## Training the Joint Sparse Autoencoder

To train the joint sparse autoencoder, run the `train_sae.py` script (you can check and change the config in the `training_sae.yaml` file):

```bash
python train_sae.py
```

Make sure the datasets are placed in the `data/` directory and checkpoints in the `results/` directory as mentioned in the project structure.

## Evaluating the Merged Model

Once the joint sparse autoencoder (SAE) has been trained using the `train_sae.py` script, the next crucial step is to evaluate the performance of the resulting merged model. This section outlines the evaluation process to ensure the model meets the desired performance criteria and provides insights into its effectiveness.

### Evaluation Process

1. **Preparation**:
   Ensure the evaluation datasets are correctly placed in the `data/` directory and that the necessary checkpoints are available in the `results/` directory.

2. **Running Evaluations**:
   Use the provided evaluation script `evaluate_merged_model.py` to run the evaluation. This script loads the merged model, processes the evaluation dataset, and computes the aforementioned metrics.

   ```bash
   python evaluate.py
   ```

3. **Configuration**:
   The evaluation script uses a configuration file `eval_merged.yaml`. Ensure this file is correctly set up with paths to the dataset, model checkpoints, and other parameters.

4. **Analyzing Results**:
   TODO (now it only outputs average loss)
