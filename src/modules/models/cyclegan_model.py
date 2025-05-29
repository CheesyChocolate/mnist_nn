"""CycleGAN-based model for pattern recognition in limited data scenarios."""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Reshape
from tensorflow.keras.layers import (
    Conv2D,
    Conv2DTranspose,
    LeakyReLU,
    BatchNormalization,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt


class CycleGANModel:
    """CycleGAN-based model for pattern recognition with discriminator transfer learning."""

    def __init__(self, input_shape=(28, 28, 1), num_classes=10, latent_dim=100):
        """
        Initialize the CycleGAN model for pattern recognition.

        Args:
            input_shape: Shape of input images
            num_classes: Number of output classes
            latent_dim: Dimension of the latent space for generator
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.latent_dim = latent_dim

        # Initialize model components
        self.generator_AB = None
        self.generator_BA = None
        self.discriminator_A = None
        self.discriminator_B = None
        self.classifier = None
        self.cycle_gan_model = None

        # Training parameters
        self.d_optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
        self.g_optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
        self.c_optimizer = Adam(learning_rate=0.0001)

    def build_generator(self):
        """Build the generator model for CycleGAN."""
        model = Sequential()

        # Encoder
        model.add(
            Conv2D(
                32,
                kernel_size=3,
                strides=2,
                padding="same",
                input_shape=self.input_shape,
            )
        )
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))

        # Bottleneck
        model.add(Conv2D(256, kernel_size=3, padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))

        # Decoder
        model.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2DTranspose(64, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2DTranspose(32, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(
            Conv2D(
                self.input_shape[2], kernel_size=3, padding="same", activation="tanh"
            )
        )

        return model

    def build_discriminator(self):
        """Build the discriminator model for CycleGAN."""
        model = Sequential()

        # Feature extraction layers
        model.add(
            Conv2D(
                32,
                kernel_size=3,
                strides=2,
                padding="same",
                input_shape=self.input_shape,
            )
        )
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Conv2D(256, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))

        # Output layer for real/fake prediction
        model.add(Flatten())
        model.add(Dense(1, activation="sigmoid"))

        return model

    def build_classifier_from_discriminator(self, discriminator):
        """
        Build a classifier using the discriminator's feature extraction layers.

        Args:
            discriminator: The trained discriminator model

        Returns:
            A classifier model for pattern recognition
        """
        # Extract the feature layers from the discriminator (all but the last layer)
        feature_extractor = Model(
            inputs=discriminator.inputs, outputs=discriminator.layers[-2].output
        )

        # We'll leave the feature extractor trainable for better adaptation to the task
        feature_extractor.trainable = True

        # Create a new classifier model
        inputs = Input(shape=self.input_shape)
        features = feature_extractor(inputs)

        # Add a more complex classification head
        x = Dense(128, activation="relu")(features)
        x = Dropout(0.4)(x)
        outputs = Dense(self.num_classes, activation="softmax")(x)

        classifier = Model(inputs=inputs, outputs=outputs)
        classifier.compile(
            loss="categorical_crossentropy",
            optimizer=self.c_optimizer,
            metrics=["accuracy"],
        )

        return classifier

    def build_models(self):
        """Build and compile all models needed for CycleGAN-based pattern recognition."""
        # Build generators
        self.generator_AB = self.build_generator()
        self.generator_BA = self.build_generator()

        # Build discriminators
        self.discriminator_A = self.build_discriminator()
        self.discriminator_B = self.build_discriminator()

        # Compile discriminators
        self.discriminator_A.compile(
            loss="binary_crossentropy", optimizer=self.d_optimizer, metrics=["accuracy"]
        )
        self.discriminator_B.compile(
            loss="binary_crossentropy", optimizer=self.d_optimizer, metrics=["accuracy"]
        )

        # Build the CycleGAN model
        # (Implementation simplified for this example)

        return self.discriminator_A, self.discriminator_B

    def train_gan(self, data_A, data_B, epochs=100, batch_size=128, sample_interval=50):
        """
        Train the CycleGAN model.

        Args:
            data_A: Domain A data (e.g., real MNIST images)
            data_B: Domain B data (e.g., a different style of digits)
            epochs: Number of training epochs
            batch_size: Batch size for training
            sample_interval: Interval for sampling and visualization

        Returns:
            Training history
        """
        # Build and compile models if they don't exist
        if self.discriminator_A is None:
            self.build_models()

        # Training logic for CycleGAN - Simplified implementation
        # In a full implementation, we would train the complete CycleGAN
        # Here we'll just train the discriminator for feature learning

        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        d_losses = []

        for epoch in range(epochs):
            # Select a random batch of images from each domain
            idx = np.random.randint(0, data_A.shape[0], batch_size)
            imgs_A = data_A[idx]
            imgs_B = data_B[idx]

            # Generate fake samples
            noise = np.random.normal(0, 1, (batch_size,) + self.input_shape)
            fake_A = imgs_A + 0.1 * noise  # Simple way to generate fake samples

            # Train the discriminators
            d_loss_real = self.discriminator_A.train_on_batch(imgs_A, real)
            d_loss_fake = self.discriminator_A.train_on_batch(fake_A, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            d_losses.append(d_loss[0])

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - D loss: {d_loss[0]:.4f}")

        # Plot discriminator training loss
        plt.figure(figsize=(10, 5))
        plt.plot(d_losses)
        plt.title("Discriminator Loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.savefig("doc/fig/cyclegan_discriminator_loss.png")
        plt.close()

        print(f"CycleGAN training completed for {epochs} epochs")

        # Return the trained discriminator for classification use
        return self.discriminator_A

    def create_classifier(self, train_discriminator=True, domain="A"):
        """
        Create a classifier from the trained discriminator.

        Args:
            train_discriminator: Whether to train the GAN first
            domain: Which domain's discriminator to use ('A' or 'B')

        Returns:
            A classifier model
        """
        # Build models if they don't exist
        if self.discriminator_A is None or self.discriminator_B is None:
            self.build_models()

        # Select the appropriate discriminator
        discriminator = self.discriminator_A if domain == "A" else self.discriminator_B

        # Build the classifier from the discriminator
        self.classifier = self.build_classifier_from_discriminator(discriminator)

        return self.classifier

    def train_classifier(
        self, x_train, y_train, x_val=None, y_val=None, batch_size=128, epochs=20
    ):
        """
        Train the classifier (after CycleGAN pre-training).

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
        # Create classifier if it doesn't exist
        if self.classifier is None:
            self.create_classifier(train_discriminator=False)

        # Convert labels to categorical
        y_train_cat = to_categorical(y_train, self.num_classes)
        if y_val is not None:
            y_val_cat = to_categorical(y_val, self.num_classes)
        else:
            y_val_cat = None

        # Train the classifier
        validation_data = None
        if x_val is not None and y_val_cat is not None:
            validation_data = (x_val, y_val_cat)

        history = self.classifier.fit(
            x_train,
            y_train_cat,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=validation_data,
        )

        return history

    def predict(self, x):
        """
        Predict classes for the input data.

        Args:
            x: Input data

        Returns:
            Predicted class labels
        """
        if self.classifier is None:
            raise ValueError("Classifier has not been trained yet")

        # Get predicted probabilities
        y_pred_prob = self.classifier.predict(x)

        # Convert to class labels
        y_pred = np.argmax(y_pred_prob, axis=1)

        return y_pred

    def train_with_limited_data(
        self,
        x_labeled,
        y_labeled,
        x_unlabeled,
        x_val=None,
        y_val=None,
        gan_epochs=100,
        classifier_epochs=20,
        batch_size=128,
    ):
        """
        Train the model with limited labeled data and additional unlabeled data.

        Args:
            x_labeled: Labeled training data
            y_labeled: Labels for the labeled data
            x_unlabeled: Unlabeled data for GAN training
            x_val: Validation data
            y_val: Validation labels
            gan_epochs: Number of epochs for GAN training
            classifier_epochs: Number of epochs for classifier training
            batch_size: Batch size for training

        Returns:
            The trained classifier
        """
        # Step 1: Train the CycleGAN using both labeled and unlabeled data
        all_real_data = np.concatenate([x_labeled, x_unlabeled], axis=0)

        # For simplicity, we'll create domain B by adding some noise or transformations to domain A
        # In a real application, domain B could be a different style of digits or transformed images
        noise_factor = 0.3
        all_transformed_data = all_real_data + noise_factor * np.random.normal(
            loc=0.0, scale=1.0, size=all_real_data.shape
        )
        all_transformed_data = np.clip(all_transformed_data, 0.0, 1.0)

        print("Training CycleGAN with combined labeled and unlabeled data...")
        self.train_gan(
            all_real_data,
            all_transformed_data,
            epochs=gan_epochs,
            batch_size=batch_size,
        )

        # Step 2: Create a classifier from the discriminator
        print("Creating classifier from discriminator...")
        self.create_classifier(train_discriminator=False, domain="A")

        # Step 3: Pre-train the classifier with labeled data to initialize it better
        print("Pre-training classifier with labeled data...")
        # Convert labels to categorical
        y_labeled_cat = to_categorical(y_labeled, self.num_classes)
        if y_val is not None:
            y_val_cat = to_categorical(y_val, self.num_classes)
        else:
            y_val_cat = None

        # Prepare validation data
        validation_data = None
        if x_val is not None and y_val_cat is not None:
            validation_data = (x_val, y_val_cat)

        # Use a higher learning rate for initial training
        self.classifier.optimizer.learning_rate.assign(0.001)

        # Pre-train with a higher learning rate for quick initial adaptation
        self.classifier.fit(
            x_labeled,
            y_labeled_cat,
            batch_size=batch_size,
            epochs=min(5, classifier_epochs),  # Short pre-training phase
            verbose=1,
            validation_data=validation_data,
        )

        # Step 4: Fine-tune the classifier with labeled data
        print("Fine-tuning classifier with labeled data...")
        # Lower learning rate for fine-tuning
        self.classifier.optimizer.learning_rate.assign(0.0001)

        history = self.classifier.fit(
            x_labeled,
            y_labeled_cat,
            batch_size=batch_size,
            epochs=classifier_epochs,
            verbose=1,
            validation_data=validation_data,
        )

        # Plot training history
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history.history["accuracy"])
        if "val_accuracy" in history.history:
            plt.plot(history.history["val_accuracy"])
        plt.title("Classifier Accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Validation"], loc="upper left")

        plt.subplot(1, 2, 2)
        plt.plot(history.history["loss"])
        if "val_loss" in history.history:
            plt.plot(history.history["val_loss"])
        plt.title("Classifier Loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Validation"], loc="upper left")

        plt.tight_layout()
        plt.savefig("doc/fig/cyclegan_classifier_training.png")
        plt.close()

        return self.classifier
