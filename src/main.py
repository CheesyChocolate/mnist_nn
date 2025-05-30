#!/usr/bin/env python
"""Main entry point for MNIST pattern recognition using various models."""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import torch
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

# Import the models
from src.modules.models.transformer_model import MNISTTransformerModel
from src.modules.models.cnn_model import CNNModel
from src.modules.models.mlp_model import MLPModel
from src.modules.models.cyclegan_model import CycleGANModel
from src.modules.models.diffusion_model import DiffusionModel

# Ensure output directories exist
os.makedirs("doc/fig", exist_ok=True)
os.makedirs("doc/out", exist_ok=True)

# Constants
RANDOM_STATE = 42
TEST_SIZE = 0.2
SAMPLE_SIZE = 5000  # Default sample size
LIMITED_DATA_SIZE = 500  # Small number of labeled examples for CycleGAN/Diffusion mode


def load_data(sample_size=SAMPLE_SIZE, limited_data=False):
    """Load MNIST data from scikit-learn's datasets."""
    print("Loading MNIST data...")
    mnist = fetch_openml("mnist_784", version=1, parser="auto")
    X = mnist.data.astype("float32").to_numpy()
    y = mnist.target.astype("int").to_numpy()

    # Normalize the data
    X = X / 255.0

    # For limited data mode, we'll separate into labeled and unlabeled
    if limited_data:
        # First get a subset of the data
        if sample_size and sample_size < len(X):
            subset_indices = np.random.RandomState(RANDOM_STATE).choice(
                len(X), sample_size, replace=False
            )
            X_subset = X[subset_indices]
            y_subset = y[subset_indices]
        else:
            X_subset = X
            y_subset = y

        # Split into train and test
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X_subset, y_subset, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

        # Take a small subset as labeled data
        labeled_size = LIMITED_DATA_SIZE
        unlabeled_size = len(X_train_full) - labeled_size

        indices = np.random.RandomState(RANDOM_STATE).choice(
            len(X_train_full), len(X_train_full), replace=False
        )

        # Split into labeled and unlabeled sets
        labeled_indices = indices[:labeled_size]
        unlabeled_indices = indices[labeled_size : labeled_size + unlabeled_size]

        X_labeled = X_train_full[labeled_indices]
        y_labeled = y_train_full[labeled_indices]
        X_unlabeled = X_train_full[unlabeled_indices]

        print(f"Data loaded in limited data mode:")
        print(f"  - {X_labeled.shape[0]} labeled training samples")
        print(f"  - {X_unlabeled.shape[0]} unlabeled training samples")
        print(f"  - {X_test.shape[0]} test samples")

        return X_labeled, X_unlabeled, X_test, y_labeled, y_test

    # Regular mode: Use a subset of the data for faster processing
    if sample_size and sample_size < len(X):
        indices = np.random.RandomState(RANDOM_STATE).choice(
            len(X), sample_size, replace=False
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


def train_svm(X_train, y_train, X_test, y_test, C=1.0, gamma="scale"):
    """Train and evaluate an SVM model."""
    print("Training SVM model...")
    model = SVC(kernel="rbf", gamma=gamma, C=C, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"SVM accuracy: {accuracy:.4f}")
    return accuracy


def train_random_forest(X_train, y_train, X_test, y_test, n_estimators=100):
    """Train and evaluate a Random Forest model."""
    print("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=n_estimators, random_state=RANDOM_STATE, n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"Random Forest accuracy: {accuracy:.4f}")
    return accuracy


def train_knn(X_train, y_train, X_test, y_test, n_neighbors=5):
    """Train and evaluate a KNN model."""
    print("Training KNN model...")
    model = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"KNN accuracy: {accuracy:.4f}")
    return accuracy


def train_transformer(X_train, y_train, X_test, y_test, epochs=3, batch_size=32):
    """Train and evaluate a Vision Transformer model."""
    print("Training Vision Transformer model...")

    # Use CPU explicitly
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # Initialize and train the model
    model = MNISTTransformerModel(
        img_size=28,
        patch_size=7,
        embed_dim=64,
        num_heads=4,
        num_layers=4,
        batch_size=batch_size,
        num_epochs=epochs,
        learning_rate=0.001,
        device=device,
    )

    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"Transformer accuracy: {accuracy:.4f}")
    return accuracy


def train_cnn(X_train, y_train, X_test, y_test, epochs=10, batch_size=128):
    """Train and evaluate a Convolutional Neural Network model."""
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
        batch_size=batch_size,
        epochs=epochs,
    )

    # Evaluate
    _, accuracy = model.model.evaluate(X_test_reshaped, y_test_cat, verbose=0)
    print(f"CNN accuracy: {accuracy:.4f}")
    return accuracy


def train_mlp(X_train, y_train, X_test, y_test, epochs=20, batch_size=128):
    """Train and evaluate a Multi-Layer Perceptron model."""
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
        batch_size=batch_size,
        epochs=epochs,
    )

    # Evaluate
    _, accuracy = model.model.evaluate(X_test_reshaped, y_test_cat, verbose=0)
    print(f"MLP accuracy: {accuracy:.4f}")
    return accuracy


def train_cyclegan(
    X_labeled,
    y_labeled,
    X_unlabeled,
    X_test,
    y_test,
    gan_epochs=50,
    classifier_epochs=10,
    batch_size=32,
):
    """Train and evaluate a CycleGAN-based model with limited labeled data."""
    print("Training CycleGAN model...")

    # Reshape data for CNN input
    X_labeled_reshaped = X_labeled.reshape(-1, 28, 28, 1)
    X_unlabeled_reshaped = X_unlabeled.reshape(-1, 28, 28, 1)
    X_test_reshaped = X_test.reshape(-1, 28, 28, 1)

    # Initialize the model
    model = CycleGANModel(input_shape=(28, 28, 1), num_classes=10)

    # Train the model with limited labeled data and additional unlabeled data
    model.train_with_limited_data(
        X_labeled_reshaped,
        y_labeled,
        X_unlabeled_reshaped,
        X_test_reshaped,
        y_test,
        gan_epochs=gan_epochs,
        classifier_epochs=classifier_epochs,
        batch_size=batch_size,
    )

    # Evaluate
    y_pred = model.predict(X_test_reshaped)
    accuracy = np.mean(y_pred == y_test)
    print(f"CycleGAN accuracy: {accuracy:.4f}")
    return accuracy


def train_diffusion(
    X_labeled,
    y_labeled,
    X_unlabeled,
    X_test,
    y_test,
    diffusion_epochs=20,
    classifier_epochs=10,
    batch_size=32,
):
    """Train and evaluate a diffusion-based model with limited labeled data."""
    print("Training Diffusion model...")

    # Reshape data for diffusion model input
    X_labeled_reshaped = X_labeled.reshape(-1, 28, 28, 1)
    X_unlabeled_reshaped = X_unlabeled.reshape(-1, 28, 28, 1)
    X_test_reshaped = X_test.reshape(-1, 28, 28, 1)

    # Initialize the model with smaller diffusion steps for faster training
    model = DiffusionModel(input_shape=(28, 28, 1), num_classes=10, diffusion_steps=100)

    # Train the model with limited labeled data and additional unlabeled data
    model.train_with_limited_data(
        X_labeled_reshaped,
        y_labeled,
        X_unlabeled_reshaped,
        X_test_reshaped,
        y_test,
        diffusion_epochs=diffusion_epochs,
        classifier_epochs=classifier_epochs,
        batch_size=batch_size,
    )

    # Evaluate
    y_pred = model.predict(X_test_reshaped)
    accuracy = np.mean(y_pred == y_test)
    print(f"Diffusion model accuracy: {accuracy:.4f}")
    return accuracy


def train_cnn_limited_data(
    X_labeled, y_labeled, X_test, y_test, epochs=20, batch_size=32
):
    """Train a regular CNN model with only limited labeled data for comparison."""
    print("Training CNN model with limited data...")

    # Reshape data for CNN input
    X_labeled_reshaped = X_labeled.reshape(-1, 28, 28, 1)
    X_test_reshaped = X_test.reshape(-1, 28, 28, 1)

    # Convert labels to one-hot encoding
    y_labeled_cat = to_categorical(y_labeled, 10)
    y_test_cat = to_categorical(y_test, 10)

    # Initialize and build the model
    model = CNNModel(input_shape=(28, 28, 1), num_classes=10)
    model.build_model()

    # Train the model
    history = model.train(
        X_labeled_reshaped,
        y_labeled_cat,
        x_val=X_test_reshaped,
        y_val=y_test_cat,
        batch_size=batch_size,
        epochs=epochs,
    )

    # Evaluate
    _, accuracy = model.model.evaluate(X_test_reshaped, y_test_cat, verbose=0)
    print(f"CNN with limited data accuracy: {accuracy:.4f}")
    return accuracy


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="MNIST Pattern Recognition")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["svm", "rf", "knn", "transformer", "cnn", "mlp", "cyclegan", "diffusion", "all"],
        help="Model to train (default: all)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=SAMPLE_SIZE,
        help=f"Sample size for training (default: {SAMPLE_SIZE})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of epochs for neural network training (default: 3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for neural network training (default: 32)",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of trees for Random Forest (default: 100)",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=5,
        help="Number of neighbors for KNN (default: 5)",
    )
    parser.add_argument(
        "--svm-c", type=float, default=1.0, help="C parameter for SVM (default: 1.0)"
    )
    parser.add_argument(
        "--svm-gamma",
        type=str,
        default="scale",
        help="Gamma parameter for SVM (default: 'scale')",
    )
    parser.add_argument(
        "--gan-epochs",
        type=int,
        default=50,
        help="Number of epochs for GAN training (default: 50)",
    )
    parser.add_argument(
        "--classifier-epochs",
        type=int,
        default=10,
        help="Number of epochs for classifier fine-tuning (default: 10)",
    )
    parser.add_argument(
        "--diffusion-epochs",
        type=int,
        default=20,
        help="Number of epochs for diffusion model training (default: 20)",
    )
    parser.add_argument(
        "--limited-data",
        action="store_true",
        help="Use limited labeled data mode for training (required for cyclegan and diffusion)",
    )

    args = parser.parse_args()

    # Force limited data mode if cyclegan or diffusion model is selected
    if args.model in ["cyclegan", "diffusion"]:
        args.limited_data = True

    # Load data based on mode (limited or full)
    if args.limited_data:
        X_labeled, X_unlabeled, X_test, y_labeled, y_test = load_data(
            sample_size=args.sample_size, limited_data=True
        )
    else:
        X_train, X_test, y_train, y_test = load_data(sample_size=args.sample_size)

    results = {}

    # Train selected model(s)
    if not args.limited_data:
        # Standard training with full labeled data
        if args.model in ["svm", "all"]:
            results["SVM"] = train_svm(
                X_train, y_train, X_test, y_test, C=args.svm_c, gamma=args.svm_gamma
            )

        if args.model in ["rf", "all"]:
            results["Random Forest"] = train_random_forest(
                X_train, y_train, X_test, y_test, n_estimators=args.n_estimators
            )

        if args.model in ["knn", "all"]:
            results["KNN"] = train_knn(
                X_train, y_train, X_test, y_test, n_neighbors=args.n_neighbors
            )

        if args.model in ["transformer", "all"]:
            results["Transformer"] = train_transformer(
                X_train,
                y_train,
                X_test,
                y_test,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )

        if args.model in ["cnn", "all"]:
            results["CNN"] = train_cnn(
                X_train,
                y_train,
                X_test,
                y_test,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )

        if args.model in ["mlp", "all"]:
            results["MLP"] = train_mlp(
                X_train,
                y_train,
                X_test,
                y_test,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )
    else:
        # Limited data mode - semi-supervised learning models
        if args.model in ["cyclegan", "all"]:
            results["CycleGAN"] = train_cyclegan(
                X_labeled,
                y_labeled,
                X_unlabeled,
                X_test,
                y_test,
                gan_epochs=args.gan_epochs,
                classifier_epochs=args.classifier_epochs,
                batch_size=args.batch_size,
            )
            
        if args.model in ["diffusion", "all"]:
            results["Diffusion"] = train_diffusion(
                X_labeled,
                y_labeled,
                X_unlabeled,
                X_test,
                y_test,
                diffusion_epochs=args.diffusion_epochs,
                classifier_epochs=args.classifier_epochs,
                batch_size=args.batch_size,
            )
        elif args.model not in ["cyclegan", "diffusion", "all"]:
            print("Limited data mode is only supported for CycleGAN and Diffusion models")
            print("Use --model cyclegan or --model diffusion with --limited-data flag")

    # Print comparison if multiple models were trained
    if len(results) > 1:
        print("\nModel Comparison:")
        print("----------------")
        for model, accuracy in results.items():
            print(f"{model}: {accuracy:.4f}")

    return results


if __name__ == "__main__":
    main()
