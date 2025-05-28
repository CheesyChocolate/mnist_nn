# MNIST Model Implementations

This directory contains different model implementations for MNIST digit recognition:

## Model Implementations

- **cnn_model.py**: Convolutional Neural Network model implementation
- **mlp_model.py**: Multi-layer Perceptron (fully connected neural network) implementation
- **svm_model.py**: Support Vector Machine classifier implementation
- **random_forest_model.py**: Random Forest classifier implementation
- **knn_model.py**: K-Nearest Neighbors classifier implementation

## Common Interface

Each model implements a common interface with the following methods:

- `__init__()`: Constructor with model-specific parameters
- `train(x_train, y_train, x_val=None, y_val=None, ...)`: Training method with consistent parameters
- `predict(x)`: Prediction method
- `predict_proba(x)`: Probability estimates (when applicable)

This modular structure allows for easy comparison between different models and makes it simple to add new model implementations in the future. 