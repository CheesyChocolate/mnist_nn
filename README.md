# MNIST Pattern Recognition

This project implements various machine learning approaches for handwritten digit recognition using the MNIST dataset. It provides a comprehensive comparison of different algorithms including traditional machine learning and deep learning techniques.

## Features

- Multiple model implementations:
  - Support Vector Machine (SVM)
  - Random Forest
  - K-Nearest Neighbors (KNN)
  - Vision Transformer (ViT)
- Comprehensive model evaluation
- Performance visualization (confusion matrices, sample predictions)
- Modular code structure

## Project Structure

```
mnist_nn/
├── doc/               # Documentation and results
│   ├── fig/           # Generated figures
│   └── out/           # Model metrics and outputs
├── src/               # Source code
│   ├── modules/       # Core modules
│   │   ├── models/    # Model implementations
│   │   └── ...        # Other modules
│   └── ...            # Entry points and utilities
└── tests/             # Unit tests
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/cheesychocolate/mnist_nn.git
   cd mnist_nn
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install the package and dependencies:
   ```
   pip install -e .
   ```

## Usage

To generate results for all models:

```
python src/generate_samples.py
```

This will:
1. Load and preprocess the MNIST dataset
2. Train all models
3. Generate evaluation metrics
4. Create visualizations in `doc/fig/`
5. Save metrics in `doc/out/`

## Models

### Support Vector Machine (SVM)

The SVM model uses an RBF kernel to capture non-linear relationships in the data. It is effective for image classification tasks but can be computationally expensive for large datasets.

### Random Forest

The Random Forest model uses an ensemble of decision trees to improve accuracy and reduce overfitting. It provides good interpretability through feature importance measures.

### K-Nearest Neighbors (KNN)

The KNN model is a simple yet effective approach that classifies digits based on the majority class of their nearest neighbors. It requires no training time but needs to store the entire training set.

### Vision Transformer (ViT)

The Vision Transformer model treats image classification as a sequence modeling task by splitting the image into patches and applying transformer-based self-attention mechanisms. This approach leverages the power of attention to capture global relationships in the image.

## Results

The models are evaluated using standard metrics:
- Accuracy
- Precision, Recall, F1-score
- Confusion matrices
- Sample predictions

Results are generated in `doc/fig/` and `doc/out/` directories.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Requirements

- Python 3.8+
- NumPy
- SciPy
- Matplotlib
- scikit-learn
- PyTorch (for transformer model)
- pandas
