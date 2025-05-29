# Model Implementations

This directory contains implementations of various machine learning models for the MNIST digit recognition task.

## Available Models

### Traditional Machine Learning Models

- **SVM (Support Vector Machine)**: `svm_model.py`
  - Uses an RBF kernel to capture non-linear relationships
  - Strong for high-dimensional data
  - Scikit-learn implementation

- **Random Forest**: `random_forest_model.py`
  - Ensemble of decision trees
  - Good balance of accuracy and interpretability
  - Provides feature importance information

- **KNN (K-Nearest Neighbors)**: `knn_model.py`
  - Instance-based learning approach
  - No training phase required
  - Simple but effective for this task

### Deep Learning Models

- **CNN (Convolutional Neural Network)**: `cnn_model.py`
  - Specialized for image data
  - Captures spatial hierarchies in images
  - Highly effective for digit recognition

- **MLP (Multi-layer Perceptron)**: `mlp_model.py`
  - Standard feedforward neural network
  - Fully connected layers
  - Simple baseline deep learning approach

- **Vision Transformer (ViT)**: `transformer_model.py`
  - Treats images as sequences of patches
  - Applies self-attention mechanisms
  - Captures global relationships in images
  - Based on the transformer architecture

## Model Structure

Each model is implemented with a similar interface for consistency, including:

- Initialization with configurable hyperparameters
- `fit(X, y)` method for training
- `predict(X)` method for making predictions

This allows for easy comparison between different approaches. 