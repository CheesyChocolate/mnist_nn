import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam


class CNNModel:
    """Convolutional Neural Network model for MNIST classification."""

    def __init__(self, input_shape=(28, 28, 1), num_classes=10):
        """
        Initialize the CNN model.

        Args:
            input_shape: Shape of input images
            num_classes: Number of output classes
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None

    def build_model(self):
        """Build and compile the CNN model."""
        model = Sequential()

        # First convolutional layer
        model.add(
            Conv2D(
                32, kernel_size=(3, 3), activation="relu", input_shape=self.input_shape
            )
        )

        # Second convolutional layer
        model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))

        # Max pooling
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # Flatten and dense layers
        model.add(Flatten())
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.5))

        # Output layer
        model.add(Dense(self.num_classes, activation="softmax"))

        # Compile the model
        model.compile(
            loss=tf.keras.losses.categorical_crossentropy,
            optimizer=Adam(),
            metrics=["accuracy"],
        )

        self.model = model
        return model

    def train(
        self, x_train, y_train, x_val=None, y_val=None, batch_size=128, epochs=20
    ):
        """
        Train the CNN model.

        Args:
            x_train: Training data
            y_train: Training labels
            x_val: Validation data
            y_val: Validation labels
            batch_size: Batch size for training
            epochs: Number of epochs to train

        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()

        validation_data = None
        if x_val is not None and y_val is not None:
            validation_data = (x_val, y_val)

        history = self.model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=validation_data,
        )

        return history
