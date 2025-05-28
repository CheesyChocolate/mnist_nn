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

# Ensure output directories exist
os.makedirs('doc/fig', exist_ok=True)
os.makedirs('doc/out', exist_ok=True)

# Constants
RANDOM_STATE = 42
TEST_SIZE = 0.2
SAMPLE_SIZE = 5000  # Use a smaller subset for faster processing

def load_data():
    """Load MNIST data from scikit-learn's datasets."""
    print("Loading MNIST data...")
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    X = mnist.data.astype('float32').to_numpy()
    y = mnist.target.astype('int').to_numpy()
    
    # Normalize the data
    X = X / 255.0
    
    # Use a subset of the data for faster processing
    if SAMPLE_SIZE and SAMPLE_SIZE < len(X):
        indices = np.random.RandomState(RANDOM_STATE).choice(len(X), SAMPLE_SIZE, replace=False)
        X = X[indices]
        y = y[indices]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    print(f"Data loaded: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
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
    with open(f'doc/out/{model_name}_metrics.txt', 'w') as f:
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
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.colorbar()
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, range(10))
    plt.yticks(tick_marks, range(10))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.savefig(f'doc/fig/{model_name}_confusion_matrix.png')
    plt.close()
    
    # Plot sample predictions
    indices = np.random.choice(len(X_test), 10, replace=False)
    plt.figure(figsize=(20, 4))
    for i, idx in enumerate(indices):
        plt.subplot(1, 10, i+1)
        img = X_test[idx].reshape(28, 28)
        plt.imshow(img, cmap='gray')
        plt.title(f'True: {y_test[idx]}\nPred: {y_pred[idx]}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'doc/fig/{model_name}_sample_predictions.png')
    plt.close()
    
    # Generate sample MNIST digits for the report
    if model_name == "SVM":  # Only do this once
        plt.figure(figsize=(8, 2))
        for i in range(10):
            # Find an example of each digit
            idx = np.where(y_test == i)[0][0]
            plt.subplot(1, 10, i+1)
            img = X_test[idx].reshape(28, 28)
            plt.imshow(img, cmap='gray')
            plt.title(f'{i}')
            plt.axis('off')
        plt.tight_layout()
        plt.savefig('doc/fig/mnist_samples.png')
        plt.close()
    
    return accuracy

def train_svm(X_train, y_train, X_test, y_test):
    """Train and evaluate an SVM model."""
    print("Training SVM model...")
    model = SVC(kernel='rbf', gamma='scale', C=1.0, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return evaluate_model(model, X_test, y_test, "SVM")

def train_random_forest(X_train, y_train, X_test, y_test):
    """Train and evaluate a Random Forest model."""
    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)
    return evaluate_model(model, X_test, y_test, "RandomForest")

def train_knn(X_train, y_train, X_test, y_test):
    """Train and evaluate a KNN model."""
    print("Training KNN model...")
    model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    model.fit(X_train, y_train)
    return evaluate_model(model, X_test, y_test, "KNN")

def compare_models(results):
    """Compare accuracies of different models."""
    print("Comparing model performances...")
    models = list(results.keys())
    accuracies = [results[model] for model in models]
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.bar(models, accuracies)
    plt.ylim(0, 1.0)
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.savefig('doc/fig/model_comparison.png')
    plt.close()
    
    # Save comparison to file
    with open('doc/out/model_comparison.txt', 'w') as f:
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
    
    # Compare models
    compare_models(results)
    
    print("Done! Results are saved in doc/fig and doc/out directories.")

if __name__ == "__main__":
    main() 