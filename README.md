# MNIST Pattern Recognition

This project implements various machine learning approaches for handwritten digit recognition using the MNIST dataset. It provides a comprehensive comparison of different algorithms including traditional machine learning and deep learning techniques.

## Features

- Multiple model implementations:
  - Support Vector Machine (SVM)
  - Random Forest
  - K-Nearest Neighbors (KNN)
  - Convolutional Neural Network (CNN)
  - Multi-Layer Perceptron (MLP)
  - CycleGAN (for semi-supervised learning)
  - Vision Transformer (ViT)
- Support for limited labeled data scenarios (semi-supervised learning)
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

### Example Commands

You can run specific models using the `src/main.py` script with various options:

```bash
# Run all models with default settings
python src/main.py --model all

# Run SVM model only
python src/main.py --model svm --sample-size 10000 --svm-c 1.0 --svm-gamma scale

# Run Random Forest with 200 trees
python src/main.py --model rf --n-estimators 200 --sample-size 10000

# Run KNN with 7 neighbors
python src/main.py --model knn --n-neighbors 7 --sample-size 10000

# Run CNN model with custom epochs and batch size
python src/main.py --model cnn --epochs 10 --batch-size 64 --sample-size 5000

# Run MLP model with custom settings
python src/main.py --model mlp --epochs 15 --batch-size 128 --sample-size 5000

# Run CycleGAN model with limited labeled data
python src/main.py --model cyclegan --gan-epochs 50 --classifier-epochs 10 --batch-size 32

# Run Transformer model with custom settings
python src/main.py --model transformer --epochs 5 --batch-size 32 --sample-size 5000
```

## Models

### Support Vector Machine (SVM)

The SVM model uses an RBF kernel to capture non-linear relationships in the data. It is effective for image classification tasks but can be computationally expensive for large datasets.

### Random Forest

The Random Forest model uses an ensemble of decision trees to improve accuracy and reduce overfitting. It provides good interpretability through feature importance measures.

### K-Nearest Neighbors (KNN)

The KNN model is a simple yet effective approach that classifies digits based on the majority class of their nearest neighbors. It requires no training time but needs to store the entire training set.

### Convolutional Neural Network (CNN)

The CNN model utilizes convolutional layers to extract spatial features from the digit images. It is particularly effective for image classification tasks as it can learn hierarchical features automatically.

### Multi-Layer Perceptron (MLP)

The MLP model is a fully-connected neural network that learns to classify digits by adjusting weights through backpropagation. It provides a good baseline for neural network approaches.

### CycleGAN with Limited Data

The CycleGAN approach demonstrates semi-supervised learning when labeled data is scarce. It utilizes a CycleGAN architecture to learn meaningful representations from unlabeled data, which are then leveraged for classification with a small amount of labeled examples.

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
