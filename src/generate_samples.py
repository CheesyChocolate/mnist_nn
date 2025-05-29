#!/usr/bin/env python
"""Generate sample results for the MNIST dataset using scikit-learn models."""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import torch
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

# Import transformer model
from src.modules.models.transformer_model import MNISTTransformerModel
from src.modules.models.cnn_model import CNNModel
from src.modules.models.mlp_model import MLPModel

# Ensure output directories exist
os.makedirs("doc/fig", exist_ok=True)
os.makedirs("doc/out", exist_ok=True)

# Constants
RANDOM_STATE = 42
TEST_SIZE = 0.2
SAMPLE_SIZE = 5000  # Use a smaller subset for faster processing


def load_data():
    """Load MNIST data from scikit-learn's datasets."""
    print("Loading MNIST data...")
    mnist = fetch_openml("mnist_784", version=1, parser="auto")
    X = mnist.data.astype("float32").to_numpy()
    y = mnist.target.astype("int").to_numpy()

    # Normalize the data
    X = X / 255.0

    # Use a subset of the data for faster processing
    if SAMPLE_SIZE and SAMPLE_SIZE < len(X):
        indices = np.random.RandomState(RANDOM_STATE).choice(
            len(X), SAMPLE_SIZE, replace=False
        )
        X = X[indices]
        y = y[indices]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    print(
        f"Data loaded: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples"
    )
    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate a model and generate reports."""
    print(f"Evaluating {model_name}...")

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Save metrics to file
    with open(f"doc/out/{model_name}_metrics.txt", "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                f.write(f"  Class {label}:\n")
                for metric_name, value in metrics.items():
                    f.write(f"    {metric_name}: {value:.4f}\n")

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.colorbar()
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, range(10))
    plt.yticks(tick_marks, range(10))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    plt.savefig(f"doc/fig/{model_name}_confusion_matrix.png")
    plt.close()

    # Plot sample predictions
    indices = np.random.choice(len(X_test), 10, replace=False)
    plt.figure(figsize=(20, 4))
    for i, idx in enumerate(indices):
        plt.subplot(1, 10, i + 1)
        img = X_test[idx].reshape(28, 28)
        plt.imshow(img, cmap="gray")
        plt.title(f"True: {y_test[idx]}\nPred: {y_pred[idx]}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(f"doc/fig/{model_name}_sample_predictions.png")
    plt.close()

    # Generate sample MNIST digits for the report
    if model_name == "SVM":  # Only do this once
        plt.figure(figsize=(8, 2))
        for i in range(10):
            # Find an example of each digit
            idx = np.where(y_test == i)[0][0]
            plt.subplot(1, 10, i + 1)
            img = X_test[idx].reshape(28, 28)
            plt.imshow(img, cmap="gray")
            plt.title(f"{i}")
            plt.axis("off")
        plt.tight_layout()
        plt.savefig("doc/fig/mnist_samples.png")
        plt.close()

    return accuracy


def train_svm(X_train, y_train, X_test, y_test):
    """Train and evaluate an SVM model."""
    print("Training SVM model...")
    model = SVC(kernel="rbf", gamma="scale", C=10.0, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return evaluate_model(model, X_test, y_test, "SVM")


def train_random_forest(X_train, y_train, X_test, y_test):
    """Train and evaluate a Random Forest model."""
    print("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
    )
    model.fit(X_train, y_train)
    return evaluate_model(model, X_test, y_test, "RandomForest")


def train_knn(X_train, y_train, X_test, y_test):
    """Train and evaluate a KNN model."""
    print("Training KNN model...")
    model = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
    model.fit(X_train, y_train)
    return evaluate_model(model, X_test, y_test, "KNN")


def train_transformer(X_train, y_train, X_test, y_test):
    """Train and evaluate a Vision Transformer model."""
    print("Training Vision Transformer model...")

    # Use a more CPU-friendly configuration with 5 epochs for better results
    model = MNISTTransformerModel(
        img_size=28,
        patch_size=7,
        embed_dim=64,
        num_heads=4,
        num_layers=4,
        batch_size=32,
        num_epochs=5,  # Increased from 3 to 5 for better accuracy
        learning_rate=0.001,
        device=torch.device("cpu"),  # Explicitly use CPU
    )

    model.fit(X_train, y_train)
    return evaluate_model(model, X_test, y_test, "Transformer")


def train_cnn(X_train, y_train, X_test, y_test):
    """Train and evaluate a CNN model."""
    print("Training CNN model...")

    # Reshape data for CNN input (NHWC format)
    X_train_reshaped = X_train.reshape(-1, 28, 28, 1)
    X_test_reshaped = X_test.reshape(-1, 28, 28, 1)

    # Convert labels to one-hot encoding
    y_train_cat = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)

    # Initialize and build the model
    model = CNNModel(input_shape=(28, 28, 1), num_classes=10)
    model.build_model()

    # Train the model
    history = model.train(
        X_train_reshaped,
        y_train_cat,
        x_val=X_test_reshaped,
        y_val=y_test_cat,
        batch_size=128,
        epochs=5,
    )

    # Evaluate
    y_pred_prob = model.model.predict(X_test_reshaped)
    y_pred = np.argmax(y_pred_prob, axis=1)

    # Create scikit-learn compatible predictor
    class CNNPredictor:
        def predict(self, X):
            X_reshaped = X.reshape(-1, 28, 28, 1)
            y_pred_prob = model.model.predict(X_reshaped)
            return np.argmax(y_pred_prob, axis=1)

    return evaluate_model(CNNPredictor(), X_test, y_test, "CNN")


def train_mlp(X_train, y_train, X_test, y_test):
    """Train and evaluate an MLP model."""
    print("Training MLP model...")

    # Reshape data for MLP input
    X_train_reshaped = X_train.reshape(-1, 28, 28, 1)
    X_test_reshaped = X_test.reshape(-1, 28, 28, 1)

    # Convert labels to one-hot encoding
    y_train_cat = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)

    # Initialize and build the model
    model = MLPModel(input_shape=(28, 28, 1), num_classes=10)
    model.build_model()

    # Train the model
    history = model.train(
        X_train_reshaped,
        y_train_cat,
        x_val=X_test_reshaped,
        y_val=y_test_cat,
        batch_size=128,
        epochs=5,
    )

    # Evaluate
    y_pred_prob = model.model.predict(X_test_reshaped)
    y_pred = np.argmax(y_pred_prob, axis=1)

    # Create scikit-learn compatible predictor
    class MLPPredictor:
        def predict(self, X):
            X_reshaped = X.reshape(-1, 28, 28, 1)
            y_pred_prob = model.model.predict(X_reshaped)
            return np.argmax(y_pred_prob, axis=1)

    return evaluate_model(MLPPredictor(), X_test, y_test, "MLP")


def compare_models(results):
    """Compare accuracies of different models."""
    print("Comparing model performances...")
    models = list(results.keys())
    accuracies = [results[model] for model in models]

    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.bar(models, accuracies)
    plt.ylim(0.9, 1.0)  # Adjust y-axis for better visualization
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy Comparison")
    plt.savefig("doc/fig/model_comparison.png")
    plt.close()

    # Save comparison to file
    with open("doc/out/model_comparison.txt", "w") as f:
        f.write("Model Accuracy Comparison\n")
        f.write("========================\n\n")
        for model in models:
            f.write(f"{model}: {results[model]:.4f}\n")


def main():
    """Main function to run the workflow."""
    X_train, X_test, y_train, y_test = load_data()

    # Dictionary to store results
    results = {}

    # Train and evaluate models
    results["SVM"] = train_svm(X_train, y_train, X_test, y_test)
    results["RandomForest"] = train_random_forest(X_train, y_train, X_test, y_test)
    results["KNN"] = train_knn(X_train, y_train, X_test, y_test)

    # Deep learning models
    results["CNN"] = train_cnn(X_train, y_train, X_test, y_test)
    results["MLP"] = train_mlp(X_train, y_train, X_test, y_test)

    # Always train the transformer model on CPU
    print("Training transformer model on CPU...")
    results["Transformer"] = train_transformer(X_train, y_train, X_test, y_test)

    # Compare models
    compare_models(results)

    print("Done! Results are saved in doc/fig and doc/out directories.")


if __name__ == "__main__":
    main()
