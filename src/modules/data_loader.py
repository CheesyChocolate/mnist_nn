import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


class MNISTDataLoader:
    """MNIST dataset loader with preprocessing functionality."""

    def __init__(self, normalize=True, reshape=True, one_hot=True):
        """
        Initialize the MNIST data loader.

        Args:
            normalize: Whether to normalize the data to [0,1]
            reshape: Whether to reshape the data for CNN input (add channel dimension)
            one_hot: Whether to convert labels to one-hot encoding
        """
        self.normalize = normalize
        self.reshape = reshape
        self.one_hot = one_hot
        self.num_classes = 10
        self.img_rows, self.img_cols = 28, 28

    def load_data(self):
        """Load and preprocess the MNIST dataset."""
        # Load the MNIST dataset
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # Convert to float32
        x_train = x_train.astype("float32")
        x_test = x_test.astype("float32")

        # Normalize if requested
        if self.normalize:
            x_train /= 255.0
            x_test /= 255.0

        # Reshape for CNN if requested
        if self.reshape:
            x_train = x_train.reshape(x_train.shape[0], self.img_rows, self.img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], self.img_rows, self.img_cols, 1)

        # Convert to one-hot encoding if requested
        if self.one_hot:
            y_train = to_categorical(y_train, self.num_classes)
            y_test = to_categorical(y_test, self.num_classes)

        return (x_train, y_train), (x_test, y_test)

    def get_sample(self, index=0, dataset="train"):
        """Get a single sample from the dataset."""
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        if dataset == "train":
            return x_train[index], y_train[index]
        else:
            return x_test[index], y_test[index]
