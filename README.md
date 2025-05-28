# MNIST Pattern Recognition

A comprehensive exploration of various machine learning approaches for MNIST digit recognition.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Documentation](#documentation)
- [License](#license)

## Overview

This project implements and compares multiple machine learning algorithms for the MNIST handwritten digit recognition task. It includes CNN, MLP, SVM, Random Forest, and KNN approaches with consistent evaluation metrics and visualization capabilities.

## Installation

```console
hatch build
pip install dist/mnist-nn-<version>-py3-none-any.whl
```

Or for development:

```console
git clone https://github.com/CheesyChocolate/mnist-nn.git
cd mnist-nn
pip install -e .
```

## Usage

Run a specific model:

```console
python -m src.main --model cnn --epochs 10
```

Available models:
- `cnn`: Convolutional Neural Network
- `mlp`: Multi-layer Perceptron
- `svm`: Support Vector Machine
- `rf`: Random Forest
- `knn`: K-Nearest Neighbors
- `all`: Run all models and compare results

Additional options:
- `--epochs`: Number of epochs for neural network training (default: 10)
- `--batch-size`: Batch size for training (default: 128)
- `--no-reshape`: Do not reshape data for CNN (default: False)
- `--sample-size`: Sample size for training (default: None - use all data)

## Models

The project includes the following models:

1. **CNN (Convolutional Neural Network)**: A deep learning approach with convolutional layers optimized for image recognition.
2. **MLP (Multi-layer Perceptron)**: A standard neural network with fully connected layers.
3. **SVM (Support Vector Machine)**: A traditional machine learning approach for classification.
4. **Random Forest**: An ensemble method based on decision trees.
5. **KNN (K-Nearest Neighbors)**: A simple instance-based learning method.

## Documentation

The results of model training and evaluation are saved in:
- `doc/fig/`: Visualizations including confusion matrices, training histories, and sample predictions
- `doc/out/`: Metrics and model performance comparisons

A detailed report and presentation are available in LaTeX format:
- `doc/report.tex`: Academic article describing the methodology and results
- `doc/presentation.tex`: Presentation slides summarizing the project

## License

`mnist-nn` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
