import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import os
import tensorflow as tf


class ModelEvaluator:
    """Evaluates model performance and generates visualizations."""

    def __init__(self, model_name, fig_dir="../../doc/fig", out_dir="../../doc/out"):
        """
        Initialize the model evaluator.

        Args:
            model_name: Name of the model being evaluated
            fig_dir: Directory for saving figures
            out_dir: Directory for saving output metrics
        """
        self.model_name = model_name
        self.fig_dir = fig_dir
        self.out_dir = out_dir

        # Create directories if they don't exist
        os.makedirs(self.fig_dir, exist_ok=True)
        os.makedirs(self.out_dir, exist_ok=True)

    def evaluate_model(self, model, x_test, y_test, batch_size=128):
        """Evaluate model performance on test data."""
        if (
            isinstance(y_test, np.ndarray)
            and len(y_test.shape) > 1
            and y_test.shape[1] > 1
        ):
            # One-hot encoded labels
            y_true = np.argmax(y_test, axis=1)
            is_one_hot = True
        else:
            y_true = y_test
            is_one_hot = False

        # For Keras/TF models
        if hasattr(model, "predict"):
            y_pred_prob = model.predict(x_test, batch_size=batch_size)
            if is_one_hot or len(y_pred_prob.shape) > 1 and y_pred_prob.shape[1] > 1:
                y_pred = np.argmax(y_pred_prob, axis=1)
            else:
                y_pred = y_pred_prob
        else:
            # For scikit-learn models
            y_pred = model.predict(x_test)

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="weighted")
        recall = recall_score(y_true, y_pred, average="weighted")
        f1 = f1_score(y_true, y_pred, average="weighted")

        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Save metrics
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": cm,
        }

        # Save detailed classification report
        report = classification_report(y_true, y_pred, output_dict=True)
        metrics["classification_report"] = report

        return metrics

    def plot_confusion_matrix(self, cm, labels=range(10)):
        """Plot confusion matrix and save to figure directory."""
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix - {self.model_name}")
        plt.colorbar()
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels)
        plt.yticks(tick_marks, labels)
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
        plt.savefig(f"{self.fig_dir}/{self.model_name}_confusion_matrix.png")
        plt.close()

    def plot_training_history(self, history):
        """Plot training history for Keras models."""
        # Plot accuracy
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history["accuracy"])
        plt.plot(history.history["val_accuracy"])
        plt.title(f"Model Accuracy - {self.model_name}")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Validation"], loc="upper left")

        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.title(f"Model Loss - {self.model_name}")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Validation"], loc="upper left")

        plt.tight_layout()
        plt.savefig(f"{self.fig_dir}/{self.model_name}_training_history.png")
        plt.close()

    def plot_sample_predictions(self, model, x_test, y_test, num_samples=10):
        """Plot sample predictions."""
        # Get predictions
        if (
            isinstance(y_test, np.ndarray)
            and len(y_test.shape) > 1
            and y_test.shape[1] > 1
        ):
            y_true = np.argmax(y_test, axis=1)
        else:
            y_true = y_test

        y_pred_prob = model.predict(x_test)
        y_pred = np.argmax(y_pred_prob, axis=1)

        # Select random samples
        indices = np.random.choice(range(len(x_test)), num_samples, replace=False)

        # Plot samples
        plt.figure(figsize=(20, 4))
        for i, idx in enumerate(indices):
            plt.subplot(1, num_samples, i + 1)

            # Reshape if needed
            if len(x_test.shape) == 4:
                img = x_test[idx].reshape(28, 28)
            else:
                img = x_test[idx]

            plt.imshow(img, cmap="gray")
            plt.title(f"True: {y_true[idx]}\nPred: {y_pred[idx]}")
            plt.axis("off")

        plt.tight_layout()
        plt.savefig(f"{self.fig_dir}/{self.model_name}_sample_predictions.png")
        plt.close()

    def save_metrics(self, metrics):
        """Save metrics to output directory."""
        # Save metrics as text file
        with open(f"{self.out_dir}/{self.model_name}_metrics.txt", "w") as f:
            f.write(f"Model: {self.model_name}\n")
            f.write(f'Accuracy: {metrics["accuracy"]:.4f}\n')
            f.write(f'Precision: {metrics["precision"]:.4f}\n')
            f.write(f'Recall: {metrics["recall"]:.4f}\n')
            f.write(f'F1 Score: {metrics["f1_score"]:.4f}\n')

            f.write("\nClassification Report:\n")
            report = metrics["classification_report"]
            for label, metrics_dict in report.items():
                if isinstance(metrics_dict, dict):
                    f.write(f"  Class {label}:\n")
                    for metric_name, value in metrics_dict.items():
                        f.write(f"    {metric_name}: {value:.4f}\n")

        # Plot confusion matrix
        self.plot_confusion_matrix(metrics["confusion_matrix"])
